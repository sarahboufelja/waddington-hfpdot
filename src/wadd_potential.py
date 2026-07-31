"""wadd_potential -- inference of the HFPD-OT Kantorovich potentials, lambda(eta).

Given target marginal KL-radii ``theta = [eta, zeta]`` and a way to sample the marginal-KL statistic
``R(pi) = [KL(mu||mu0), KL(nu||nu0)]`` from the hyperprior ``S^o_lambda``, we solve the paper's convex
dual (Eq 20/39):

    lambda^o = argmin_{lambda >= 0}  lambda^T theta - log N(lambda),     grad = theta - E_S[R].

Two facts collapse this to a one-batch Newton step. The dual is only 2-D, and ``S^o_lambda`` is an
exponential family in ``-lambda`` with sufficient statistic ``R``, so the EXACT Hessian is the
covariance of ``R`` -- available from the same sample batch as the gradient:

    grad(lambda)   = theta - E_S[R]
    Hessian(lambda)= Cov_S[R]          (2x2, PSD -> the dual is convex)

Hence a damped (Levenberg-Marquardt) stochastic Newton step, one oracle batch per iteration:

    lambda <- clip_>=0(  lambda - (Cov_S[R] + delta I)^{-1} (theta - E_S[R]) )

This replaces Algorithm 1 of the HFPD-OT paper, which approximates the curvature with a SECOND nested
sample per step and carries a BFGS inverse-Hessian -- both unnecessary once the Hessian is recognised
as the sample covariance. (A simplification worth an erratum, but not load-bearing here.)

The sampler is hidden behind ``MarginalKLOracle`` so the Newton logic is framework-free and unit
testable; the real MALA-backed oracle (which builds an ``HFPDOTHyperprior`` at ``lambda`` and samples
plans) is a thin adapter supplied by the pipeline layer.

The ``theta`` fed here are marginal-simplex KL radii, NOT the raw latent diversity radius from
``wadd_dim_reduction`` (which is ~10^3): mapping the latent radius into a marginal radius is a separate
step and is deliberately not done here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

Array = np.ndarray


class MarginalKLOracle(Protocol):
    """Samples the marginal-KL statistic ``R(pi) = [KL(mu||mu0), KL(nu||nu0)]`` under ``S^o_lambda``.

    Returns an ``(n_samples, d)`` array (``d = 2`` for a source/target pair). The whole cost of
    ``lambda(eta)`` lives here -- one call per Newton iteration -- so an implementation may be the true
    hyperprior sampler or a cheap surrogate (Sinkhorn/Laplace), swapped without touching the solver.
    """

    def __call__(self, lam: Array, n_samples: int, rng: np.random.Generator) -> Array: ...


@dataclass(frozen=True)
class LambdaResult:
    """Outcome of a ``solve_lambda`` run."""
    lam: Array                    # (d,) the inferred potentials lambda^o >= 0
    converged: bool
    iterations: int
    newton_decrement: float       # grad^T H^{-1} grad at the last step (the Eq-49 stopping quantity)
    expected_R: Array             # (d,) E_S[R] at lambda^o -- should match theta on active constraints
    history: Array                # (iterations+1, d) the lambda trace, for diagnostics


def solve_lambda(theta: Array, oracle: MarginalKLOracle, *, n_samples: int = 1024,
                 delta: float = 1e-3, max_iter: int = 100, tol: float = 1e-4,
                 lam0: Optional[Array] = None, rng: Optional[np.random.Generator] = None,
                 verbose: bool = False) -> LambdaResult:
    """Damped stochastic Newton for the potentials ``lambda^o`` (see module docstring).

    ``theta`` are the target marginal KL radii (length ``d``); ``delta`` is the Levenberg-Marquardt
    damping that keeps the step finite where the hyperprior concentrates and ``Cov_S[R] -> 0`` (the
    ``eta -> 0`` regime, where the paper also notes the sampler mixes poorly). ``lambda`` is projected
    onto the non-negative orthant each step (complementary slackness: a slack marginal constraint
    drives its ``lambda`` to 0).

    Convergence is on the **projected** step ``||max(0, lambda - H^-1 grad) - lambda||`` -- the actual
    change in ``lambda`` after the clip -- not the raw Newton step ``||H^-1 grad||``. The two coincide
    at an interior optimum (nothing is clipped; both vanish as grad -> 0), and differ only at an active
    boundary: there a slack constraint makes the unconstrained direction point permanently off the
    orthant, so the raw step (and equally the Newton decrement, Eq 49) stays positive forever while the
    projected step is exactly zero because ``lambda_i`` is pinned at 0. The projected step is therefore
    the KKT-stationarity signal; the Newton decrement is kept only as a reported diagnostic.
    """
    theta = np.asarray(theta, dtype=float)
    d = theta.shape[0]
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D radii, got shape {theta.shape}")
    if np.any(theta < 0):
        raise ValueError("KL radii theta must be non-negative")
    rng = rng if rng is not None else np.random.default_rng(0)
    lam = np.zeros(d) if lam0 is None else np.asarray(lam0, dtype=float).copy()
    if lam.shape != (d,):
        raise ValueError(f"lam0 shape {lam.shape} does not match theta ({d},)")

    history = [lam.copy()]
    newton_dec = np.inf
    expected_R = np.full(d, np.nan)
    converged = False
    t = 0
    while t < max_iter:
        R = np.asarray(oracle(lam, n_samples, rng), dtype=float)
        if R.shape[1] != d:
            raise ValueError(f"oracle returned R with {R.shape[1]} columns, expected {d}")
        expected_R = R.mean(axis=0)
        grad = theta - expected_R                                  # theta - E_S[R]
        cov = np.cov(R, rowvar=False)                              # exact Hessian, (d,d)
        H = np.atleast_2d(cov) + delta * np.eye(d)                 # Levenberg-Marquardt damping
        step = np.linalg.solve(H, grad)                            # H^{-1} grad
        newton_dec = float(grad @ step)                            # grad^T H^{-1} grad (Eq 49), diag.
        lam_new = np.maximum(0.0, lam - step)                      # Newton descent + project >= 0
        move = float(np.linalg.norm(lam_new - lam))                # projected-step norm
        lam = lam_new
        history.append(lam.copy())
        t += 1
        if verbose:
            print(f"[lambda] iter {t:3d}  lambda={np.array2string(lam, precision=4)}  "
                  f"E[R]={np.array2string(expected_R, precision=4)}  move={move:.3e}")
        if move < tol:
            converged = True
            break

    return LambdaResult(lam=lam, converged=converged, iterations=t, newton_decrement=newton_dec,
                        expected_R=expected_R, history=np.asarray(history))
