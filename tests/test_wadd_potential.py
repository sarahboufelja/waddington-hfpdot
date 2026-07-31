"""Damped stochastic Newton for the HFPD-OT potentials lambda(eta) -- wadd_potential.solve_lambda.

The solver's identities (grad = theta - E[R], Hessian = Cov[R]) are **base-measure-independent**:
tilting any base density of the statistic R by exp(-<lambda, R>) yields an exponential family in
-lambda, so the moment relations hold whatever the base is. The oracle therefore does not model the
real hyperprior's base at all -- it only has to be a tractable exponential-family generator with a
known optimum.

We use independent Gammas: R_i ~ Gamma(shape=k_i, rate=1+lambda_i). Chosen purely because the moments
and optimum are closed-form,

    E[R_i]   = k_i / (1 + lambda_i),
    Var[R_i] = k_i / (1 + lambda_i)^2   (= -dE[R_i]/dlambda_i, the identity the Hessian rides on),
    lambda_i^* = k_i / theta_i - 1      (clipped at 0 when k_i <= theta_i: the constraint is already
                                         slack, so complementary slackness sends lambda_i -> 0).

A fixed internal seed makes each oracle deterministic per lambda, so the Newton trajectory is
reproducible. A separate guard checks the FULL covariance (off-diagonals) is what gets inverted.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wadd_potential import solve_lambda, LambdaResult  # noqa: E402


def _gamma_oracle(k, fixed_seed=None):
    """Independent-Gamma oracle: R_i ~ Gamma(k_i, rate=1+lambda_i). fixed_seed -> deterministic."""
    k = np.asarray(k, dtype=float)

    def oracle(lam, n, rng):
        r = np.random.default_rng(fixed_seed) if fixed_seed is not None else rng
        rate = 1.0 + np.asarray(lam, dtype=float)
        return r.gamma(shape=k, scale=1.0 / rate, size=(n, k.size))

    return oracle


def _analytic_mean(k, lam):
    return np.asarray(k, float) / (1.0 + np.asarray(lam, float))


# ---- recovers the closed-form optimum ------------------------------------------------------------

def test_recovers_the_analytic_potentials():
    k, theta = np.array([2.0, 3.0]), np.array([0.5, 0.5])
    lam_star = k / theta - 1.0                                  # = [3, 5]
    res = solve_lambda(theta, _gamma_oracle(k, fixed_seed=0), n_samples=6000, max_iter=80)
    assert isinstance(res, LambdaResult) and res.converged
    assert np.allclose(res.lam, lam_star, atol=0.2), f"{res.lam} vs {lam_star}"


def test_drives_expected_R_to_theta():
    """The load-bearing property: at the returned lambda, the marginal-KL matches the target radius."""
    k, theta = np.array([4.0, 1.5]), np.array([0.8, 0.3])
    res = solve_lambda(theta, _gamma_oracle(k, fixed_seed=1), n_samples=6000, max_iter=80)
    assert np.allclose(_analytic_mean(k, res.lam), theta, atol=0.03)


def test_asymmetric_radii_give_asymmetric_potentials():
    """A tighter radius on one marginal must yield a larger potential there."""
    k = np.array([3.0, 3.0])
    res = solve_lambda(np.array([0.3, 1.0]), _gamma_oracle(k, fixed_seed=2), n_samples=6000)
    assert res.lam[0] > res.lam[1]                              # tighter target -> stronger penalty


# ---- boundary / complementary slackness ----------------------------------------------------------

def test_slack_constraint_drives_lambda_to_zero():
    """k <= theta: the marginal already sits inside the KL ball, so lambda^* = 0 (KKT boundary)."""
    k, theta = np.array([0.4]), np.array([1.0])                 # E[R] at lambda=0 is 0.4 < 1.0
    res = solve_lambda(theta, _gamma_oracle(k, fixed_seed=0), n_samples=4000)
    assert res.converged and res.iterations <= 2               # stays at 0 from the first step
    assert np.allclose(res.lam, 0.0, atol=1e-9)


def test_lambda_stays_non_negative_throughout():
    k, theta = np.array([1.0, 0.5]), np.array([0.9, 0.9])       # both near/inside the ball
    res = solve_lambda(theta, _gamma_oracle(k, fixed_seed=3), n_samples=4000)
    assert np.all(res.history >= 0.0)


# ---- the Hessian is the full covariance ----------------------------------------------------------

def test_hessian_uses_the_full_covariance_not_just_the_diagonal():
    """Guard against an accidentally diagonal Hessian (np.var instead of np.cov): with correlated R,
    one Newton step under the full covariance differs from the diagonal-only one, and the solver must
    match the full-covariance step. The oracle ignores lambda so a single step is isolated."""
    R = np.array([[1.0, 2.0], [2.0, 1.0], [1.6, 1.7], [0.4, 3.1]])   # anti-correlated columns
    fixed = lambda lam, n, rng: R
    theta, lam0, delta = np.array([0.5, 0.5]), np.array([10.0, 10.0]), 1e-3

    grad = theta - R.mean(axis=0)
    cov = np.cov(R, rowvar=False)
    assert abs(cov[0, 1]) > 0.1                                      # the off-diagonal is real
    full = np.maximum(0.0, lam0 - np.linalg.solve(cov + delta * np.eye(2), grad))
    diag = np.maximum(0.0, lam0 - np.linalg.solve(np.diag(np.diag(cov)) + delta * np.eye(2), grad))

    res = solve_lambda(theta, fixed, n_samples=len(R), delta=delta, max_iter=1, lam0=lam0)
    assert np.allclose(res.history[1], full)                         # matches the full-covariance step
    assert not np.allclose(full, diag)                              # and full != diagonal (discriminating)


# ---- convergence bookkeeping ---------------------------------------------------------------------

def test_history_and_iteration_count_are_consistent():
    res = solve_lambda(np.array([0.5, 0.5]), _gamma_oracle([2.0, 2.0], fixed_seed=0),
                       n_samples=4000, max_iter=50)
    assert res.history.shape == (res.iterations + 1, 2)
    assert np.allclose(res.history[0], 0.0)                     # default lam0 = 0


def test_warm_start_reduces_iterations():
    k, theta = np.array([2.0, 3.0]), np.array([0.5, 0.5])
    cold = solve_lambda(theta, _gamma_oracle(k, fixed_seed=0), n_samples=6000, max_iter=80)
    warm = solve_lambda(theta, _gamma_oracle(k, fixed_seed=0), n_samples=6000, max_iter=80,
                        lam0=cold.lam)
    assert warm.iterations < cold.iterations


# ---- input validation ----------------------------------------------------------------------------

def test_negative_radius_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        solve_lambda(np.array([-0.1, 0.5]), _gamma_oracle([2.0, 2.0]))


def test_non_1d_theta_rejected():
    with pytest.raises(ValueError, match="1-D"):
        solve_lambda(np.array([[0.5, 0.5]]), _gamma_oracle([2.0, 2.0]))


def test_lam0_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="does not match"):
        solve_lambda(np.array([0.5, 0.5]), _gamma_oracle([2.0, 2.0]), lam0=np.array([0.0]))


def test_oracle_dimension_mismatch_rejected():
    bad = lambda lam, n, rng: np.zeros((n, 3))                  # returns d=3 for a d=2 theta
    with pytest.raises(ValueError, match="expected 2"):
        solve_lambda(np.array([0.5, 0.5]), bad, n_samples=100)
