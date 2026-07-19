"""wadd_ot -- optimal-transport layer for HierWOT.

Key design (single-source uncertainty): a cell is a latent **Gaussian** ``N(μ_k, σ²_k)``, not a point.
Cost metrics therefore operate on a ``CellCloud(means, stds)`` and the **baseline cost is GaussianW2**
(closed-form 2-Wasserstein between the cell Gaussians), which *uses* σ² instead of collapsing it.

- Uses log-domain / stabilized Sinkhorn for both balanced and unbalanced; ``reg``/``reg_m`` are explicit.
- Growth estimation is separated from plan computation behind a ``GrowthModel`` interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np
import ot

Array = np.ndarray

# --------------------------------------------------------------------------------------------------
# Latent cell representation
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CellCloud:
    """A cloud of cells in latent space. Each cell is a diagonal Gaussian ``N(means_k, diag(stds_k²))``.

    ``stds`` is optional: point-cloud metrics (sq-Euclidean) use ``means`` only; Gaussian-aware metrics
    (GaussianW2, GaussianKL) require ``stds`` (per-cell latent standard deviations).
    """
    means: Array                      # (n, d)
    stds: Array | None = None         # (n, d) standard deviations; None => point cloud

    def __post_init__(self):
        if self.means.ndim != 2:
            raise ValueError(f"means must be (n, d), got {self.means.shape}")
        if self.stds is not None and self.stds.shape != self.means.shape:
            raise ValueError(f"stds {self.stds.shape} must match means {self.means.shape}")

    def __len__(self) -> int:
        return self.means.shape[0]

    def require_stds(self, who: str) -> Array:
        if self.stds is None:
            raise ValueError(f"{who} needs per-cell stds (σ), but the CellCloud carries means only.")
        return self.stds


def _pairwise_sq_dists(A: Array, B: Array) -> Array:
    """Squared Euclidean distances between rows of A (n,d) and B (m,d) -> (n,m). numpy-only, no sklearn."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0)        # clip tiny negatives from round-off


def _median_normalize(C: Array) -> Array:
    med = np.median(C)
    return C / med if med > 0 else C


class CostMetric(Protocol):
    def __call__(self, source: CellCloud, target: CellCloud) -> Array: ...


@dataclass(frozen=True)
class SqEuclideanLatent:
    """Point cost: sq-Euclidean between latent means. Collapses σ² -- kept for A/B only."""
    normalize: bool = True

    def __call__(self, source: CellCloud, target: CellCloud) -> Array:
        C = _pairwise_sq_dists(source.means, target.means)
        return _median_normalize(C) if self.normalize else C


@dataclass(frozen=True)
class GaussianW2:
    """Baseline cost: squared 2-Wasserstein between diagonal cell Gaussians.

    For diagonal covariances, ``W₂²(N(μ_k,σ_k²), N(μ_l,σ_l²)) = ‖μ_k-μ_l‖² + ‖σ_k-σ_l‖²`` -- i.e.
    sq-Euclidean in the augmented space ``[μ ; σ]``. Uses σ instead of discarding it (single-source).
    """
    normalize: bool = True

    def __call__(self, source: CellCloud, target: CellCloud) -> Array:
        s_src, s_tgt = source.require_stds("GaussianW2"), target.require_stds("GaussianW2")
        C = _pairwise_sq_dists(source.means, target.means) + _pairwise_sq_dists(s_src, s_tgt)
        return _median_normalize(C) if self.normalize else C


@dataclass(frozen=True)
class GaussianKL:
    """Symmetric-KL cost between diagonal cell Gaussians (A/B option). Closed form, O(n·m·d)."""
    normalize: bool = True
    eps: float = 1e-12

    def __call__(self, source: CellCloud, target: CellCloud) -> Array:
        mu1 = np.asarray(source.means, np.float64)
        mu2 = np.asarray(target.means, np.float64)
        v1 = np.asarray(source.require_stds("GaussianKL"), np.float64) ** 2 + self.eps   # (n,d)
        v2 = np.asarray(target.require_stds("GaussianKL"), np.float64) ** 2 + self.eps   # (m,d)
        # KL(N1||N2) = 0.5 Σ_d [ v1/v2 + (μ2-μ1)²/v2 - 1 + log(v2/v1) ], summed over d.
        # Symmetric KL = KL(1||2) + KL(2||1). Broadcast over pairs (n,m,d).
        dmu2 = (mu1[:, None, :] - mu2[None, :, :]) ** 2          # (n,m,d)
        v1b, v2b = v1[:, None, :], v2[None, :, :]                # (n,1,d),(1,m,d)
        kl_12 = 0.5 * np.sum(v1b / v2b + dmu2 / v2b - 1.0 + np.log(v2b / v1b), axis=-1)
        kl_21 = 0.5 * np.sum(v2b / v1b + dmu2 / v1b - 1.0 + np.log(v1b / v2b), axis=-1)
        C = kl_12 + kl_21
        return _median_normalize(C) if self.normalize else C


@dataclass(frozen=True)
class Mahalanobis:
    """Pooled-precision sq-distance between means (A/B option): sq-Euclidean weighted by 1/pooled-var."""
    normalize: bool = True
    eps: float = 1e-12

    def __call__(self, source: CellCloud, target: CellCloud) -> Array:
        mu1, mu2 = np.asarray(source.means, np.float64), np.asarray(target.means, np.float64)
        # pooled per-dim variance from whatever σ we have (fallback: uniform)
        parts = [c.stds ** 2 for c in (source, target) if c.stds is not None]
        pooled = np.mean(np.concatenate(parts, axis=0), axis=0) + self.eps if parts else np.ones(mu1.shape[1])
        w = 1.0 / pooled                                        # (d,)
        C = _pairwise_sq_dists(mu1 * np.sqrt(w), mu2 * np.sqrt(w))
        return _median_normalize(C) if self.normalize else C


# --------------------------------------------------------------------------------------------------
# Marginals
# --------------------------------------------------------------------------------------------------

def as_marginal(weights: Array | None, n: int) -> Array:
    """Normalized probability marginal of length n (uniform if weights is None)."""
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"marginal must have shape ({n},), got {w.shape}")
    s = w.sum()
    if s <= 0:
        raise ValueError("marginal sums to <= 0")
    return w / s

# --------------------------------------------------------------------------------------------------
# Transport plans (balanced EOT / unbalanced UOT) and Wasserstein distance
# --------------------------------------------------------------------------------------------------

def eot_plan(source: CellCloud, target: CellCloud, *, a: Array | None = None, b: Array | None = None,
             cost: CostMetric = GaussianW2(), reg: float = 1e-2, num_iter_max: int = 20_000) -> Array:
    """Balanced entropic-OT plan via log-domain Sinkhorn. Marginals a, b are used (uniform if None)."""
    C = cost(source, target)
    p = as_marginal(a, len(source))
    q = as_marginal(b, len(target))
    return ot.bregman.sinkhorn(p, q, C, reg=reg, method="sinkhorn_log", numItermax=num_iter_max)


def uot_plan(source: CellCloud, target: CellCloud, *, a: Array | None = None, b: Array | None = None,
             cost: CostMetric = GaussianW2(), reg: float = 1e-2,
             reg_m: float | Sequence[float] = (1.0, 50.0), num_iter_max: int = 20_000) -> Array:
    """Unbalanced OT plan via **stabilized** (log-domain) Sinkhorn.

    The caller's marginals a, b are honoured (uniform if None). reg_m = (source, target) KL
    relaxations, asymmetric by design: source loose, target tighter (cells may divide/die).
    """
    C = cost(source, target)
    p = as_marginal(a, len(source))
    q = as_marginal(b, len(target))
    reg_m_arg = list(reg_m) if isinstance(reg_m, (tuple, list, np.ndarray)) else reg_m
    return ot.sinkhorn_unbalanced(p, q, C, reg=reg, reg_m=reg_m_arg,
                                  method="sinkhorn_stabilized", numItermax=num_iter_max)


def wasserstein_distance(source: CellCloud, target: CellCloud, *, a: Array | None = None,
                         b: Array | None = None, cost: CostMetric = GaussianW2(),
                         reg: float = 1e-2) -> float:
    """Entropic 2-Wasserstein distance (for the §2.3 interpolation-accuracy result)."""
    C = cost(source, target)
    p = as_marginal(a, len(source))
    q = as_marginal(b, len(target))
    return float(ot.sinkhorn2(p, q, C, reg=reg, method="sinkhorn_log"))

# --------------------------------------------------------------------------------------------------
# Growth model (swappable). WOTIteration = Schiebinger's fixed point (baseline to stress-test).
# --------------------------------------------------------------------------------------------------

class GrowthModel(Protocol):
    def __call__(self, source: CellCloud, target: CellCloud, cost: CostMetric) -> tuple[Array, Array]:
        """Returns (plan, growth) where growth[k] is the estimated mass multiplier of source cell k."""
        ...


@dataclass(frozen=True)
class WOTIteration:
    """Schiebinger WOT growth: iterate UOT with source mass = current growth; growth = plan row-sums.

    Yields a point estimate (no UQ). Intended to be A/B-tested against an HFPD-OT-native growth model
    that reads the row-marginals off the sampled random plans, and so carries uncertainty.
    """
    reg: float = 1e-2
    reg_m: Tuple[float, float] = (1.0, 50.0)
    n_iters: int = 5

    def __call__(self, source: CellCloud, target: CellCloud, cost: CostMetric) -> Tuple[Array, Array]:
        C = cost(source, target)
        growth = np.ones(len(source), dtype=np.float64)
        plan = None
        for _ in range(self.n_iters):
            a = growth
            b = np.full(len(target), float(growth.mean()))
            plan = ot.sinkhorn_unbalanced(a, b, C, reg=self.reg, reg_m=list(self.reg_m),
                                          method="sinkhorn_stabilized")
            growth = plan.sum(axis=1)
        return plan, growth
