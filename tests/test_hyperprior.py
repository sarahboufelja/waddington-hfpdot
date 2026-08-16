"""Math-correctness tests for the HFPD-OT hyperprior (balanced + unbalanced).

The hyperprior has two regimes:
  - balanced   : log p(pi) = -(l1+lI1) KL(mu,mu_0) - (l2+lI2) KL(nu,nu_0) - KL(pi,pi_I)
                 with the shifted KL; sampled on the SIMPLEX (pi is a coupling).
  - unbalanced : same with the GENERALIZED KL (mass not conserved -- apoptosis /
                 proliferation); sampled on the POSITIVE ORTHANT.

A single ``hyperprior_score_fun`` (the generalized-KL gradient, no +1) serves both:
it is the exact score on the orthant, and remains exact on the simplex because the
simplex lift only uses the score through the centering F*(s - <F,s>), which (since
sum(F)=1) annihilates the constant +1 that distinguishes the two gradients. We test
this by composing the score THROUGH each support and comparing to jax.grad of the
matching latent log-density.

float64 for tight tolerances.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from langevin_sampler import HFPDOTHyperprior, MetropolisAdjustedLangevinSampler
from supports import Simplex, PositiveOrthant


def _make_prior(II=3, JJ=4, seed=0, **kw):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    mu_0 = jax.random.uniform(k1, (II,), minval=0.1, maxval=1.0)
    mu_0 = mu_0 / mu_0.sum()
    nu_0 = jax.random.uniform(k2, (JJ,), minval=0.1, maxval=1.0)
    nu_0 = nu_0 / nu_0.sum()
    cost = jax.random.uniform(k3, (II, JJ), minval=0.0, maxval=2.0)
    prior = HFPDOTHyperprior(
        mu_0=mu_0, nu_0=nu_0,
        lambda_1=0.5, lambda_2=0.3, lambda_I_1=0.01, lambda_I_2=0.02,
        cost_fn=cost, epsilon=0.1, **kw,
    )
    return prior, II, JJ


def _assert_score_matches_grad(prior, II, JJ, support, log_prob_fn, seed=7, atol=1e-7):
    """The analytic score, lifted through `support`, equals grad of the lifted log-prob."""
    y = jax.random.normal(jax.random.key(seed), (1, II * JJ))
    grad = jax.grad(lambda yy: jnp.sum(support.latent_log_prob(log_prob_fn, yy)))(y)
    score = support.latent_score(prior.hyperprior_score_fun, y)
    assert score.shape == y.shape, score.shape
    assert jnp.allclose(score, grad, atol=atol), float(jnp.max(jnp.abs(score - grad)))


# --------------------------------------------------------------------------- #
# The single score is exact through BOTH regimes' supports.
# --------------------------------------------------------------------------- #
def test_score_exact_balanced_on_simplex():
    prior, II, JJ = _make_prior()
    _assert_score_matches_grad(prior, II, JJ, Simplex(), prior.balanced_hyperprior_log_prob_fun)


def test_score_exact_unbalanced_on_orthant():
    prior, II, JJ = _make_prior()
    _assert_score_matches_grad(prior, II, JJ, PositiveOrthant(), prior.unbalanced_hyperprior_log_prob_fun)


def test_lambda_pi_scales_only_the_ideal_term():
    """lambda_pi = 1 is the paper's hyperprior (Def 2) exactly; lambda_pi = 0 removes
    precisely gKL(pi || pi_I) from the log-prob, and the score stays exact at any weight."""
    full, II, JJ = _make_prior()
    off, _, _ = _make_prior(lambda_pi=0.0)
    pi = jnp.abs(jax.random.normal(jax.random.key(5), (1, II * JJ))) + 0.1
    gap = (off.unbalanced_hyperprior_log_prob_fun(pi)
           - full.unbalanced_hyperprior_log_prob_fun(pi))
    assert jnp.allclose(gap, HFPDOTHyperprior.generalized_kl_div(pi, full.pi_I))
    half, _, _ = _make_prior(lambda_pi=0.5)
    _assert_score_matches_grad(half, II, JJ, PositiveOrthant(),
                               half.unbalanced_hyperprior_log_prob_fun)
    _assert_score_matches_grad(half, II, JJ, Simplex(),
                               half.balanced_hyperprior_log_prob_fun)


def test_score_exact_nonsquare():
    """II != JJ to catch row/column index mix-ups, in both regimes."""
    prior, II, JJ = _make_prior(II=5, JJ=2, seed=3)
    _assert_score_matches_grad(prior, II, JJ, Simplex(), prior.balanced_hyperprior_log_prob_fun)
    _assert_score_matches_grad(prior, II, JJ, PositiveOrthant(), prior.unbalanced_hyperprior_log_prob_fun)


def test_hyperprior_score_is_unclipped():
    """Regression guard: tiny pi -> large gradient; the score must NOT be norm-clipped."""
    prior, II, JJ = _make_prior()
    pi = jnp.full((1, II * JJ), 1e-3)
    score = prior.hyperprior_score_fun(pi)
    assert jnp.linalg.norm(score) > 1.0, float(jnp.linalg.norm(score))


# --------------------------------------------------------------------------- #
# End-to-end sampling in each regime.
# --------------------------------------------------------------------------- #
def _sample(prior, dim, support, log_prob_fn, step_size):
    sampler = MetropolisAdjustedLangevinSampler(
        target_log_prob_fn=log_prob_fn,
        target_score_fn=prior.hyperprior_score_fun,
        shape=dim, support=support,
        num_parallel_chains=2, num_samples=500, num_burnin=300,
        warm_up_steps=200, step_size=step_size,
    )
    sampler.init_key = jax.random.key(0)
    out = sampler.sample(with_diagnostics=False).samples.reshape(-1, dim)  # (C, N, dim) -> (C*N, dim)
    return out


def test_balanced_hfpdot_samples_are_couplings_on_simplex():
    prior, II, JJ = _make_prior(II=2, JJ=2, seed=2)
    out = _sample(prior, II * JJ, "simplex", prior.balanced_hyperprior_log_prob_fun, step_size=0.1)
    assert out.shape == (1000, II * JJ)
    assert jnp.all(out > 0) and jnp.all(jnp.isfinite(out))
    # Simplex projection => each sampled plan is a normalized coupling.
    assert jnp.allclose(out.sum(axis=1), 1.0, atol=1e-5)


def test_unbalanced_hfpdot_samples_are_positive_on_orthant():
    prior, II, JJ = _make_prior(II=2, JJ=2, seed=2)
    out = _sample(prior, II * JJ, "positive_orthant", prior.unbalanced_hyperprior_log_prob_fun, step_size=0.05)
    assert out.shape == (1000, II * JJ)
    assert jnp.all(out > 0) and jnp.all(jnp.isfinite(out))


# --------------------------------------------------------------------------- #
# The log-density IS the HFPD-OT hyperprior.
#
# The tests above check internal consistency (score == grad, samples on support).
# These pin the target itself to its mathematical definition, so it cannot drift:
#
#   S(pi) ∝ exp[-l1 KL(mu||mu_0)] · exp[-l2 KL(nu||nu_0)] · Sbase(pi)
#
# with mu, nu the first/second marginals of pi, and Sbase the entropic-OT Gibbs
# measure with reference plan pi_I = exp(-C/eps). Unbalanced uses the generalized
# KL for positive measures, KL(p||q) = Σ p log(p/q) - Σ p + Σ q; balanced uses the
# shifted KL (same, without the mass terms).
# --------------------------------------------------------------------------- #

SMOOTH = 1e-9  # smoothing used inside the divergences


def _gen_kl_ref(p, q, e=SMOOTH):
    """Reference generalized KL for positive measures: Σ p log(p/q) - Σ p + Σ q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum((p + e) * np.log((p + e) / (q + e))) - np.sum(p) + np.sum(q))


def _shifted_kl_ref(p, q, e=SMOOTH):
    """Reference shifted KL: Σ p log(p/q) (no mass-balancing terms)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(np.sum((p + e) * np.log((p + e) / (q + e))))


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


def test_generalized_kl_matches_positive_measure_definition():
    rng = np.random.default_rng(1)
    p, q = np.abs(rng.normal(size=8)), np.abs(rng.normal(size=8))
    got = _scalar(HFPDOTHyperprior.generalized_kl_div(jnp.asarray(p), jnp.asarray(q)))
    assert np.isclose(got, _gen_kl_ref(p, q))


def test_generalized_kl_vanishes_at_equality_and_penalises_mass_mismatch():
    """The -Σp + Σq terms are what make it valid for unnormalised measures."""
    rng = np.random.default_rng(2)
    p = np.abs(rng.normal(size=8)) + 0.1
    assert np.isclose(_scalar(HFPDOTHyperprior.generalized_kl_div(jnp.asarray(p), jnp.asarray(p))),
                      0.0, atol=1e-6)
    assert _scalar(HFPDOTHyperprior.generalized_kl_div(jnp.asarray(p), jnp.asarray(2.0 * p))) > 0.0


def test_shifted_kl_is_generalized_kl_without_mass_terms():
    rng = np.random.default_rng(3)
    p, q = np.abs(rng.normal(size=8)), np.abs(rng.normal(size=8))
    shifted = _scalar(HFPDOTHyperprior.shifted_kl_div(jnp.asarray(p), jnp.asarray(q)))
    generalized = _scalar(HFPDOTHyperprior.generalized_kl_div(jnp.asarray(p), jnp.asarray(q)))
    assert np.isclose(shifted, _shifted_kl_ref(p, q))
    assert np.isclose(shifted, generalized + np.sum(p) - np.sum(q))


def test_reference_plan_is_the_entropic_ot_kernel():
    """pi_I must be the Gibbs kernel exp(-C/eps) -- the base design of the hyperprior."""
    prior, II, JJ = _make_prior()
    cost = np.asarray(prior.cost_fn).reshape(II, JJ)
    assert np.allclose(np.asarray(prior.pi_I).reshape(II, JJ), np.exp(-cost / prior.epsilon))


def test_base_term_encodes_transport_cost_and_entropy():
    """-KL(pi||pi_I) == H(pi) - (1/eps)<C,pi> + Σpi - Σpi_I: the entropic-OT objective."""
    k1, k2 = jax.random.split(jax.random.key(11))
    II = JJ = 4
    eps = 1.0                                    # large enough that SMOOTH is negligible
    cost = np.asarray(jax.random.uniform(k1, (II, JJ), minval=0.0, maxval=2.0))
    prior = HFPDOTHyperprior(mu_0=np.full(II, 1 / II), nu_0=np.full(JJ, 1 / JJ),
                             lambda_1=0.0, lambda_2=0.0, lambda_I_1=0.0, lambda_I_2=0.0,
                             cost_fn=cost, epsilon=eps)
    pi = np.abs(np.asarray(jax.random.normal(k2, (1, II * JJ)))) * 0.05
    got = -_scalar(HFPDOTHyperprior.generalized_kl_div(jnp.asarray(pi), prior.pi_I))
    entropy = -np.sum(pi * np.log(pi))                        # H(pi)
    transport = np.sum(pi * cost.reshape(1, -1)) / eps        # (1/eps) <C, pi>
    expected = entropy - transport + np.sum(pi) - np.sum(np.asarray(prior.pi_I))
    assert np.isclose(got, expected, rtol=1e-4)


@pytest.mark.parametrize("balanced", [True, False])
def test_log_density_matches_the_hyperprior_definition(balanced):
    """Reconstruct S(pi) term-by-term from the definition and compare with the implementation."""
    prior, II, JJ = _make_prior()
    rng = np.random.default_rng(4)
    pi = np.abs(rng.normal(size=(1, II * JJ))) * 0.05
    pi_mat = pi.reshape(II, JJ)
    div = _shifted_kl_ref if balanced else _gen_kl_ref

    mu, nu = pi_mat.sum(axis=1), pi_mat.sum(axis=0)           # first / second marginals
    expected = (
        -(prior.lambda_1 + prior.lambda_I_1) * div(mu, np.asarray(prior.mu_0))
        - (prior.lambda_2 + prior.lambda_I_2) * div(nu, np.asarray(prior.nu_0))
        - div(pi, np.asarray(prior.pi_I))
    )
    fn = prior.balanced_hyperprior_log_prob_fun if balanced else prior.unbalanced_hyperprior_log_prob_fun
    assert np.isclose(_scalar(fn(jnp.asarray(pi))), expected, rtol=1e-5)


def test_constrained_marginals_are_row_and_column_sums():
    """Redistributing mass within rows preserves mu, so the first marginal term is unchanged."""
    prior, II, JJ = _make_prior()
    rng = np.random.default_rng(5)
    pi_mat = np.abs(rng.normal(size=(II, JJ))) * 0.05
    shuffled = np.stack([rng.permutation(row) for row in pi_mat])   # same row sums, different cols
    assert np.allclose(shuffled.sum(axis=1), pi_mat.sum(axis=1))
    assert not np.allclose(shuffled.sum(axis=0), pi_mat.sum(axis=0))
    mu_0 = np.asarray(prior.mu_0)
    assert np.isclose(_gen_kl_ref(pi_mat.sum(axis=1), mu_0), _gen_kl_ref(shuffled.sum(axis=1), mu_0))


def test_base_design_mode_is_the_reference_plan():
    """With the marginal terms switched off, S is maximised exactly at pi = pi_I = exp(-C/eps)."""
    II = JJ = 3
    cost = np.asarray(jax.random.uniform(jax.random.key(13), (II, JJ), minval=0.0, maxval=2.0))
    prior = HFPDOTHyperprior(mu_0=np.full(II, 1 / II), nu_0=np.full(JJ, 1 / JJ),
                             lambda_1=0.0, lambda_2=0.0, lambda_I_1=0.0, lambda_I_2=0.0,
                             cost_fn=cost, epsilon=0.5, support="positive_orthant")
    pi_I = np.asarray(prior.pi_I)
    lp_mode = _scalar(prior.hyperprior_log_prob_fun(jnp.asarray(pi_I)))
    assert np.isclose(lp_mode, 0.0, atol=1e-6)               # KL(p||p) = 0

    rng = np.random.default_rng(6)
    for _ in range(50):
        pert = np.abs(pi_I + rng.normal(scale=0.1 * pi_I.std(), size=pi_I.shape))
        assert _scalar(prior.hyperprior_log_prob_fun(jnp.asarray(pert))) <= lp_mode + 1e-9


def test_unsupported_support_raises():
    prior, II, JJ = _make_prior()
    prior.support = "klein_bottle"
    with pytest.raises(ValueError):
        prior.hyperprior_log_prob_fun(jnp.zeros((1, II * JJ)))
