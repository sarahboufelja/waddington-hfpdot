"""Math-correctness tests for the constrained-support reparametrizations.

In-depth verification of issue #1. For each support we check, in order:
  group 1 - the bijection round-trips,
  group 2 - the log-Jacobian against autodiff,
  group 3 - the analytic score against jax.grad of the lifted log-density,
  group 4 - that the lifted density integrates to one (end-to-end Jacobian check),
  group 5 - the factory.

Run in float64 for tight numerical tolerances.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

import supports as S


# =========================================================================== #
# group 1 - bijection round-trips
# =========================================================================== #
def test_positive_orthant_roundtrip():
    sup = S.PositiveOrthant()
    y = jnp.array([-2.0, 0.3, 1.7, 4.0])
    assert jnp.allclose(sup.to_unconstrained(sup.to_constrained(y)), y, atol=1e-12)
    x = jnp.array([0.1, 1.0, 5.0])
    assert jnp.allclose(sup.to_constrained(sup.to_unconstrained(x)), x, atol=1e-12)


def test_simplex_inverse_after_forward_is_identity_on_Rn():
    """inverse o forward == id on all of R^n (the diffeomorphism claim)."""
    sup = S.Simplex()
    y = jnp.array([0.5, -1.2, 2.0, 0.1, -3.0])
    F, r = sup.forward(y)
    y_rec = sup.to_unconstrained(F, r)
    assert jnp.allclose(y_rec, y, atol=1e-12)


def test_simplex_forward_after_inverse_is_identity_on_simplex():
    """forward o inverse == id on Delta x R."""
    sup = S.Simplex()
    F = jnp.array([0.1, 0.2, 0.05, 0.6, 0.05])
    r = 1.7
    F2, r2 = sup.forward(sup.to_unconstrained(F, r))
    assert jnp.allclose(F2, F, atol=1e-12)
    assert jnp.allclose(r2, r, atol=1e-12)


def test_simplex_default_gauge_is_centered_and_recovers_F():
    """Default r=0 lands on the sum(y)=0 slice and still recovers F."""
    sup = S.Simplex()
    F = jnp.array([0.1, 0.2, 0.05, 0.6, 0.05])
    y = sup.to_unconstrained(F)
    assert jnp.allclose(jnp.sum(y), 0.0, atol=1e-10)
    assert jnp.allclose(jax.nn.softmax(y), F, atol=1e-12)


def test_simplex_to_constrained_is_on_simplex():
    sup = S.Simplex()
    y = jnp.array([3.0, -1.0, 0.0, 2.5, -4.0])
    F = sup.to_constrained(y)
    assert jnp.allclose(jnp.sum(F), 1.0, atol=1e-12)
    assert jnp.all(F > 0)


# =========================================================================== #
# group 2 - log-det-Jacobian vs autodiff
# =========================================================================== #
def test_unconstrained_logdet_is_zero():
    sup = S.Unconstrained()
    y = jnp.array([1.0, -2.0, 0.5])
    assert jnp.allclose(sup.log_det_jacobian(y), 0.0, atol=1e-12)


def test_positive_orthant_logdet_matches_autodiff():
    sup = S.PositiveOrthant()
    y = jnp.array([-1.0, 0.5, 2.0])
    J = jax.jacfwd(sup.to_constrained)(y)
    _, logdet = jnp.linalg.slogdet(J)
    assert jnp.allclose(logdet, sup.log_det_jacobian(y), atol=1e-10)


def test_simplex_logdet_matches_autodiff():
    """Differentiate the chart map y -> (F_1..F_{n-1}, r) and compare slogdet."""
    sup = S.Simplex()
    y = jnp.array([0.4, -1.1, 2.3, 0.7])

    def chart(yy):
        F = jax.nn.softmax(yy)
        r = jnp.sum(yy)
        return jnp.concatenate([F[:-1], r[None]])

    J = jax.jacfwd(chart)(y)
    _, logdet = jnp.linalg.slogdet(J)
    assert jnp.allclose(logdet, sup.log_det_jacobian(y), atol=1e-9)


# =========================================================================== #
# Reference constrained-space targets with HAND-DERIVED scores.
# The scores are first checked against jax.grad of the log-pdf (so the reference
# itself is trustworthy), then used to exercise the support score lift.
# =========================================================================== #
from jax.scipy.special import gammaln


def dirichlet_logpdf(alpha):
    log_norm = jnp.sum(gammaln(alpha)) - gammaln(jnp.sum(alpha))
    return lambda F: jnp.sum((alpha - 1.0) * jnp.log(F)) - log_norm


def dirichlet_score(alpha):
    # d/dF_i [ (alpha_i - 1) log F_i ] = (alpha_i - 1) / F_i
    return lambda F: (alpha - 1.0) / F


def gamma_logpdf(a, b):
    log_norm = jnp.sum(gammaln(a) - a * jnp.log(b))
    return lambda x: jnp.sum((a - 1.0) * jnp.log(x) - b * x) - log_norm


def gamma_score(a, b):
    # d/dx_i [ (a_i - 1) log x_i - b_i x_i ] = (a_i - 1)/x_i - b_i
    return lambda x: (a - 1.0) / x - b


def test_reference_scores_match_grad_of_logpdf():
    """Sanity-check the hand-derived reference scores before relying on them."""
    alpha = jnp.array([2.0, 3.0, 1.5, 4.0])
    F = jax.nn.softmax(jnp.array([0.3, -1.0, 2.0, 0.6]))
    assert jnp.allclose(dirichlet_score(alpha)(F), jax.grad(dirichlet_logpdf(alpha))(F), atol=1e-9)

    a, b = jnp.array([2.0, 0.7, 3.5]), jnp.array([1.0, 2.0, 0.5])
    x = jnp.array([0.6, 1.2, 3.3])
    assert jnp.allclose(gamma_score(a, b)(x), jax.grad(gamma_logpdf(a, b))(x), atol=1e-9)


# =========================================================================== #
# group 3 - analytic latent_score == grad of latent_log_prob
# =========================================================================== #
def test_unconstrained_score_matches_grad():
    sup = S.Unconstrained()
    tlp = lambda x: -0.5 * jnp.sum(x ** 2)
    tscore = lambda x: -x
    y = jnp.array([1.0, -2.0, 0.5])
    grad = jax.grad(lambda yy: sup.latent_log_prob(tlp, yy))(y)
    assert jnp.allclose(sup.latent_score(tscore, y), grad, atol=1e-10)


def test_positive_orthant_score_matches_grad():
    sup = S.PositiveOrthant()
    a, b = jnp.array([2.0, 0.7, 3.5]), jnp.array([1.0, 2.0, 0.5])
    tlp, tscore = gamma_logpdf(a, b), gamma_score(a, b)
    y = jnp.array([-0.5, 0.8, 1.2])
    grad = jax.grad(lambda yy: sup.latent_log_prob(tlp, yy))(y)
    assert jnp.allclose(sup.latent_score(tscore, y), grad, atol=1e-9)


def test_simplex_score_matches_grad():
    sup = S.Simplex()
    alpha = jnp.array([2.0, 3.0, 1.5, 4.0])
    tlp, tscore = dirichlet_logpdf(alpha), dirichlet_score(alpha)
    y = jnp.array([0.3, -1.0, 2.0, 0.6])
    grad = jax.grad(lambda yy: sup.latent_log_prob(tlp, yy))(y)
    assert jnp.allclose(sup.latent_score(tscore, y), grad, atol=1e-9)


def test_simplex_score_differs_from_naive_drop_of_jacobian():
    """Regression guard: the corrected score is NOT the naive target_score(F)."""
    alpha = jnp.array([2.0, 3.0, 1.5, 4.0])
    tscore = dirichlet_score(alpha)
    F = jax.nn.softmax(jnp.array([0.3, -1.0, 2.0, 0.6]))
    naive = tscore(F)
    corrected = F * (tscore(F) - jnp.sum(F * tscore(F)))
    assert not jnp.allclose(naive, corrected, atol=1e-3)

# =========================================================================== #
# group 4 -  the lifted density integrates to one (end-to-end Jacobian check)
# =========================================================================== #
def test_simplex_lifted_density_integrates_to_one():
    """
    Dirichlet density on the simplex + Gaussian radial prior --> Integral over R^n of the lifted density should be 1
    Only possible if the log_det_jacobian is correct.
    """
    sup = S.Simplex()
    alpha = jnp.array([2.0, 3.0])
    tlp = dirichlet_logpdf(alpha)

    # * Build the 2d grid
    grid = jnp.arange(-16, 16, 0.05)
    dy = float(grid[1] - grid[0])
    Y1, Y2 = jnp.meshgrid(grid, grid, indexing="ij")
    ys = jnp.stack([Y1.ravel(), Y2.ravel()], axis=-1)

    dens = jnp.exp(jax.vmap(lambda yy: sup.latent_log_prob(tlp, yy))(ys))
    integral = jnp.sum(dens) * dy ** 2
    assert jnp.allclose(integral, 1.0, atol=2e-3)

    edge = jnp.concatenate([dens.reshape(Y1.shape)[0], dens.reshape(Y1.shape)[-1],
                            dens.reshape(Y1.shape)[:, 0], dens.reshape(Y1.shape)[:, -1]])
    assert jnp.max(edge) < 1e-6, "density should be negligible at the edges of the grid"

def test_positive_orthant_lifted_density_integrates_to_one():
    """
    Gamma distribution --> Integral over R of the lifted density should be 1
    Only possible if the log_det_jacobian is correct.
    """
    sup = S.PositiveOrthant()
    a, b = jnp.array([2.0]), jnp.array([1.0])
    tlp = gamma_logpdf(a, b)

    # * Build the 1d linear grid
    grid = jnp.arange(-16, 16, 0.05)
    dy = float(grid[1] - grid[0])
    ys = grid.reshape(-1, 1)

    dens = jnp.exp(jax.vmap(lambda yy: sup.latent_log_prob(tlp, yy))(ys))
    integral = jnp.sum(dens) * dy
    assert jnp.allclose(integral, 1.0, atol=2e-3)

    edge = jnp.stack([dens[0], dens[-1]])
    assert jnp.max(edge) < 1e-6, "density should be negligible at the edges of the line"

# group 5 - factory
def test_make_support_factory_and_errors():
    assert isinstance(S.make_support("simplex"), S.Simplex)
    assert isinstance(S.make_support("positive_orthant"), S.PositiveOrthant)
    assert isinstance(S.make_support("unconstrained"), S.Unconstrained)
    with pytest.raises(ValueError):
        S.make_support("hyperbolic")


