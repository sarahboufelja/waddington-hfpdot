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
import pytest

from langevin_sampler import HFPDOTHyperprior, MetropolisAdjustedLangevinSampler
from supports import Simplex, PositiveOrthant


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
