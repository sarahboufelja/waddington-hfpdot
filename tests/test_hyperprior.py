"""Math-correctness tests for the HFPD-OT hyperprior (issue #2c).

The hyperprior log-density lives in constrained (positive-orthant) pi-space:
    log p(pi) = -(l1+lI1) D(mu, mu_0) - (l2+lI2) D(nu, nu_0) - D(pi, pi_I),
with mu/nu the row/column marginals of pi and D the shifted KL divergence.

The analytic ``hyperprior_score_fun`` must equal grad of ``hyperprior_log_prob_fun``;
we check it against ``jax.grad`` (the oracle), and guard that the score is the
*true* gradient (no norm-clipping baked in).

float64 for tight tolerances.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from langevin_sampler import HFPDOTHyperprior


def _make_prior(II=3, JJ=4, seed=0):
    k1, k2, k3 = jax.random.split(jax.random.key(seed), 3)
    mu_0 = jax.random.uniform(k1, (II,), minval=0.1, maxval=1.0)
    mu_0 = mu_0 / mu_0.sum()
    nu_0 = jax.random.uniform(k2, (JJ,), minval=0.1, maxval=1.0)
    nu_0 = nu_0 / nu_0.sum()
    cost = jax.random.uniform(k3, (II, JJ), minval=0.0, maxval=2.0)
    prior = HFPDOTHyperprior(
        mu_0=mu_0, nu_0=nu_0,
        lambda_1=0.5, lambda_2=0.3, lambda_I_1=0.01, lambda_I_2=0.02,
        cost_fn=cost, epsilon=0.1,
    )
    return prior, II, JJ


def _random_pi(II, JJ, seed=1):
    return jax.random.uniform(jax.random.key(seed), (1, II * JJ), minval=0.05, maxval=1.0)


def test_hyperprior_score_matches_grad():
    prior, II, JJ = _make_prior()
    pi = _random_pi(II, JJ)

    analytic = prior.hyperprior_score_fun(pi)
    # log_prob returns a (1, 1) array; sum to a scalar for grad.
    grad = jax.grad(lambda p: jnp.sum(prior.hyperprior_log_prob_fun(p)))(pi)

    assert analytic.shape == pi.shape, analytic.shape
    assert jnp.allclose(analytic, grad, atol=1e-7), float(jnp.max(jnp.abs(analytic - grad)))


def test_hyperprior_score_matches_grad_other_dims():
    """Non-square II != JJ to catch row/column index mix-ups."""
    prior, II, JJ = _make_prior(II=5, JJ=2, seed=3)
    pi = _random_pi(II, JJ, seed=4)

    analytic = prior.hyperprior_score_fun(pi)
    grad = jax.grad(lambda p: jnp.sum(prior.hyperprior_log_prob_fun(p)))(pi)
    assert jnp.allclose(analytic, grad, atol=1e-7), float(jnp.max(jnp.abs(analytic - grad)))


def test_hyperprior_score_is_unclipped():
    """Regression guard: tiny pi -> large gradient; the score must NOT be clipped to norm 1."""
    prior, II, JJ = _make_prior()
    pi = jnp.full((1, II * JJ), 1e-3)
    score = prior.hyperprior_score_fun(pi)
    assert jnp.linalg.norm(score) > 1.0, float(jnp.linalg.norm(score))


def test_positive_orthant_hfpdot_end_to_end():
    """The HFPD-OT hyperprior samples through the positive_orthant support:
    the lift (Jacobian + score) must produce finite, strictly positive plans."""
    from langevin_sampler import MetropolisAdjustedLangevinSampler

    prior, II, JJ = _make_prior(II=2, JJ=2, seed=2)
    dim = II * JJ
    sampler = MetropolisAdjustedLangevinSampler(
        target_log_prob_fn=prior.hyperprior_log_prob_fun,
        target_score_fn=prior.hyperprior_score_fun,
        shape=dim,
        support="positive_orthant",
        num_parallel_chains=2,
        num_samples=500,
        num_burnin=300,
        warm_up_steps=200,
        step_size=0.05,
    )
    sampler.init_key = jax.random.key(0)
    out, num_accepted, _, _ = sampler.sample(with_diagnostics=False)

    assert out.shape == (2 * 500, dim)
    assert jnp.all(jnp.isfinite(out)), "non-finite plan entries"
    assert jnp.all(out > 0), "positive-orthant samples must be strictly positive"
