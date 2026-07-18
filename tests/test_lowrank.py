"""Correctness tests for the log-low-rank Support (``supports.LowRank``, Option B).

Covers, in order:
  group 1 - the (U, V) coordinates: pack/unpack round-trip, batch-awareness, num_free,
  group 2 - reconstruction: to_constrained batches, and the simplex normalizes globally,
  group 3 - GL(r) gauge invariance of the plan,
  group 4 - the warm start: to_unconstrained -> to_constrained round-trip (+ batched),
  group 5 - the analytic latent_score against jax.grad (orthant, simplex, higher rank, ridge).

Run in float64 for tight tolerances.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from supports import LowRank


def _lr(II=6, JJ=5, rank=2, support="positive_orthant", ridge=0.0, seed=0):
    cost = jax.random.uniform(jax.random.key(seed), (II, JJ))
    return LowRank(II, JJ, rank, cost, epsilon=0.3, support=support, ridge=ridge)


def _plan_of(lr, U, V):
    """Reconstruct pi from factors via the public API (LowRank has no plan(U, V))."""
    return lr.to_constrained(lr.pack(U, V))


def _synthetic(n, seed):
    """An arbitrary smooth plan-space density with a known analytic score."""
    target = jax.random.uniform(jax.random.key(seed), (1, n)) + 0.5
    return (lambda pi: -0.5 * jnp.sum((pi - target) ** 2)), (lambda pi: target - pi)


def _check_score(lr, seed):
    n = lr.II * lr.JJ
    log_prob_fn, score_fn = _synthetic(n, seed)
    theta = 0.1 * jax.random.normal(jax.random.key(seed + 1), (lr.num_free,))
    analytic = lr.latent_score(score_fn, theta)
    autodiff = jax.grad(lambda th: lr.latent_log_prob(log_prob_fn, th))(theta)
    assert jnp.allclose(analytic, autodiff, rtol=1e-6, atol=1e-8)


# =========================================================================== #
# group 1 - the (U, V) coordinates
# =========================================================================== #
def test_pack_unpack_roundtrip():
    lr = _lr()
    theta = jax.random.normal(jax.random.key(1), (lr.num_free,))
    U, V = lr.unpack(theta)
    assert U.shape == (lr.II, lr.rank) and V.shape == (lr.JJ, lr.rank)
    assert jnp.allclose(lr.pack(U, V), theta, atol=1e-12)


def test_num_free_formula():
    lr = _lr()
    assert lr.num_free == lr.rank * (lr.II + lr.JJ)  # fully free U, V


def test_unpack_pack_batched():
    lr = _lr()
    theta = jax.random.normal(jax.random.key(2), (4, 3, lr.num_free))
    U, V = lr.unpack(theta)
    assert U.shape == (4, 3, lr.II, lr.rank) and V.shape == (4, 3, lr.JJ, lr.rank)
    assert jnp.allclose(lr.pack(U, V), theta, atol=1e-12)


# =========================================================================== #
# group 2 - reconstruction (batch-aware; simplex normalizes globally)
# =========================================================================== #
def test_to_constrained_batches_consistently():
    lr = _lr()
    thetas = jax.random.normal(jax.random.key(3), (4, 3, lr.num_free))
    pi = lr.to_constrained(thetas)
    assert pi.shape == (4, 3, lr.II * lr.JJ)
    assert jnp.allclose(pi[0, 0], lr.to_constrained(thetas[0, 0]), atol=1e-10)


def test_simplex_is_globally_normalized():
    lr = _lr(support="simplex")
    theta = jax.random.normal(jax.random.key(4), (lr.num_free,))
    pi = lr.to_constrained(theta)
    assert jnp.allclose(jnp.sum(pi), 1.0, atol=1e-8)  # global softmax, total mass 1 (not row-wise)


# =========================================================================== #
# group 3 - gauge
# =========================================================================== #
def test_gl_r_orbit_leaves_plan_invariant():
    lr = _lr()
    U = jax.random.normal(jax.random.key(7), (lr.II, lr.rank))
    V = jax.random.normal(jax.random.key(8), (lr.JJ, lr.rank))
    R = jax.random.normal(jax.random.key(9), (lr.rank, lr.rank)) + lr.rank * jnp.eye(lr.rank)
    assert jnp.allclose(_plan_of(lr, U @ R, V @ jnp.linalg.inv(R).T), _plan_of(lr, U, V), atol=1e-8)


# =========================================================================== #
# group 4 - warm start: to_unconstrained -> to_constrained round-trip
# =========================================================================== #
def test_roundtrip_orthant_rank2():
    # A Sinkhorn-form plan: log pi + C/eps = (f + g)/eps is exactly rank-2 -> exact round-trip.
    lr = _lr(II=7, JJ=6, rank=2, support="positive_orthant")
    f = jax.random.normal(jax.random.key(10), (lr.II,))
    g = jax.random.normal(jax.random.key(11), (lr.JJ,))
    pi = jnp.exp((f[:, None] + g[None, :]) / lr.epsilon + lr.log_K).reshape(-1)
    theta = lr.to_unconstrained(pi)
    assert jnp.allclose(lr.to_constrained(theta), pi, rtol=1e-6, atol=1e-8)


def test_to_unconstrained_batched():
    lr = _lr(II=7, JJ=6, rank=2, support="positive_orthant")
    f = jax.random.normal(jax.random.key(12), (2, lr.II))
    g = jax.random.normal(jax.random.key(13), (2, lr.JJ))
    pis = jnp.exp((f[:, :, None] + g[:, None, :]) / lr.epsilon + lr.log_K).reshape(2, -1)
    thetas = lr.to_unconstrained(pis)
    assert thetas.shape == (2, lr.num_free)
    assert jnp.allclose(lr.to_constrained(thetas), pis, rtol=1e-5, atol=1e-7)


def test_roundtrip_truncates_at_low_rank():
    # A rank-3 (in log) plan cannot be reproduced at r=2: round-trip must differ.
    lr = _lr(II=7, JJ=6, rank=2, support="positive_orthant")
    k1, k2 = jax.random.split(jax.random.key(14))
    U3 = jax.random.normal(k1, (lr.II, 3))
    V3 = jax.random.normal(k2, (lr.JJ, 3))
    pi = jnp.exp(U3 @ V3.T + lr.log_K).reshape(-1)
    theta = lr.to_unconstrained(pi)
    assert not jnp.allclose(lr.to_constrained(theta), pi, rtol=1e-3)


# =========================================================================== #
# group 5 - analytic latent_score vs jax.grad
# =========================================================================== #
def test_score_orthant():
    _check_score(_lr(support="positive_orthant"), seed=100)


def test_score_simplex():
    _check_score(_lr(support="simplex"), seed=200)  # exercises the softmax <pi,s> centering


def test_score_higher_rank():
    _check_score(_lr(II=8, JJ=7, rank=3), seed=300)


def test_score_with_ridge_orthant():
    _check_score(_lr(support="positive_orthant", ridge=0.1), seed=400)


def test_score_with_ridge_simplex():
    _check_score(_lr(support="simplex", ridge=0.1), seed=500)
