"""wadd_dim_reduction -- latent representation of cells and the uncertainty radii derived from it.

A cell is represented by its **latent posterior**, not by a point: an embedding model maps raw counts
to a diagonal Gaussian ``N(mu_k, sigma_k^2)`` over latent space plus a categorical distribution over
mixture components. Keeping the whole posterior is what lets the transport cost use ``sigma`` and the
population membership use the categorical, instead of collapsing either to a point estimate.

Order of operations (deliberate):

    embed ALL cells   ->  CellPosterior{mu, sigma^2, prob_cat}
    radii on ALL      ->  eta                      <- computed before any subsampling
    subsample latent  ->  the transport budget

The radii are computed on the full population because random subsampling biases ``eta`` **downward**
(self-inclusion: each component is itself part of the mixture it is compared against) while a
coverage-preserving subsampler biases it **upward** (it deliberately favours extremes). Computing on
everything makes ``eta`` independent of which subsampling strategy is in play, so the two can be
compared without the radius moving underneath them.

The embedding model sits behind ``Embedder`` so this module -- and its tests -- do not depend on the
modelling framework.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Protocol

import numpy as np

from wadd_data_ingest import CellAxis, ExpressionMatrix

Array = np.ndarray

_VAR_FLOOR = 1e-12          # keeps log/division finite for a component with vanishing variance


# --------------------------------------------------------------------------------------------------
# The latent representation
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CellPosterior:
    """Per-cell latent posterior: a diagonal Gaussian plus a categorical over mixture components.

    ``variances`` (not standard deviations) is the stored quantity because that is what a Gaussian
    encoder emits and what the closed-form divergences below are written in; ``stds`` is exposed for
    consumers that want the scale, so the square root happens in exactly one place.

    ``prob_cat`` is the categorical posterior over the model's components. It is the natural source
    of a *soft* population membership: a cell sitting between components is genuinely ambiguous, and
    the argmax of this vector would throw that away.
    """
    means: Array                  # (n_cells, latent_dim)
    variances: Array              # (n_cells, latent_dim), positive
    prob_cat: Array               # (n_cells, n_components), rows sum to 1
    cells: CellAxis

    def __post_init__(self):
        n, q = self.means.shape
        if self.variances.shape != (n, q):
            raise ValueError(f"variances {self.variances.shape} must match means {self.means.shape}")
        if self.prob_cat.ndim != 2 or self.prob_cat.shape[0] != n:
            raise ValueError(f"prob_cat {self.prob_cat.shape} must be (n_cells={n}, n_components)")
        if len(self.cells) != n:
            raise ValueError(f"{len(self.cells)} cells for {n} rows")
        if np.any(self.variances < 0):
            raise ValueError("variances must be non-negative")
        if np.any(self.prob_cat < 0) or not np.allclose(self.prob_cat.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("prob_cat rows must be non-negative and sum to 1")

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def stds(self) -> Array:
        return np.sqrt(self.variances)

    @property
    def latent_dim(self) -> int:
        return self.means.shape[1]

    @property
    def n_components(self) -> int:
        return self.prob_cat.shape[1]

    @property
    def cell_ids(self) -> List[str]:
        return self.cells.ids

    @property
    def day(self) -> float:
        return self.cells.day

    def select(self, idx) -> CellPosterior:
        """Row-subset, keeping every array and the cell axis aligned."""
        idx = np.asarray(idx, dtype=int)
        return CellPosterior(means=self.means[idx], variances=self.variances[idx],
                             prob_cat=self.prob_cat[idx], cells=self.cells.select(idx))


class Embedder(Protocol):
    """Maps raw counts to latent posteriors. The seam that keeps the modelling framework out."""

    def embed(self, expr: ExpressionMatrix) -> CellPosterior: ...


# --------------------------------------------------------------------------------------------------
# Uncertainty radii
# --------------------------------------------------------------------------------------------------

def _kl_diagonal_gaussians(mu_a: Array, var_a: Array, mu_b: Array, var_b: Array) -> Array:
    """KL(N(mu_a, var_a) || N(mu_b, var_b)) for diagonal Gaussians, summed over dimensions.

    Broadcasts, so it serves both the one-to-many and pairwise cases.
    """
    var_a = np.maximum(var_a, _VAR_FLOOR)
    var_b = np.maximum(var_b, _VAR_FLOOR)
    return 0.5 * np.sum(var_a / var_b + (mu_b - mu_a) ** 2 / var_b - 1.0
                        + np.log(var_b / var_a), axis=-1)


def _symmetrised_kl(mu_a: Array, var_a: Array, mu_b: Array, var_b: Array) -> Array:
    """Jeffreys divergence between diagonal Gaussians: KL(a||b) + KL(b||a), summed over dims.

    Symmetrised because farthest-point coverage needs a symmetric ``d(a, b) == d(b, a)``; plain KL is
    directional. Zero iff the two posteriors coincide, but NOT a metric (no triangle inequality) --
    so it forfeits the k-center 2-approximation, which is acceptable here because stratified seeding,
    not the FPS walk, carries the per-population coverage guarantee.

    Reuses the broadcasting ``_kl_diagonal_gaussians``, so a single representative ``(q,)`` against
    all remaining cells ``(n, q)`` returns ``(n,)`` -- exactly the one-to-many call the FPS min-dist
    update makes each step. The variance floor lives in ``_kl_diagonal_gaussians``, so a
    near-degenerate posterior stays finite.
    """
    return (_kl_diagonal_gaussians(mu_a, var_a, mu_b, var_b)
            + _kl_diagonal_gaussians(mu_b, var_b, mu_a, var_a))


class RadiusEstimator(Protocol):
    """Estimates the uncertainty radius of a population of latent posteriors.

    The radius is the mean divergence from the population's mixture to each individual cell,
    ``eta = (1/n) sum_k KL(psi_0 || psi_k)`` with ``psi_0 = (1/n) sum_j psi_j`` -- a measure of how
    diverse the population is, which is what sets how far the transport plan's marginals may wander.
    """

    def __call__(self, posterior: CellPosterior) -> float: ...


@dataclass(frozen=True)
class MomentMatchedGaussian:
    """Default estimator. Approximates the mixture by a single moment-matched Gaussian: O(n).

    The mixture's first two moments follow from the law of total variance,

        mu_bar    = mean_k(mu_k)
        Sigma_bar = mean_k(var_k)  +  var_k(mu_k)
                    \\____________/    \\_________/
                    within-cell         between-cell

    after which every ``KL(N(mu_bar, Sigma_bar) || psi_k)`` is closed-form. Linear in the number of
    cells and deterministic, so it carries none of the sampling noise of a Monte-Carlo estimate.

    The approximation is exact when the cells are identical and degrades as the population becomes
    strongly multimodal -- the mixture is then poorly described by one Gaussian. ``MonteCarloMixture``
    is the reference to check it against.
    """

    def __call__(self, posterior: CellPosterior) -> float:
        mu, var = posterior.means, posterior.variances
        mu_bar = mu.mean(axis=0)
        sigma_bar = var.mean(axis=0) + ((mu - mu_bar) ** 2).mean(axis=0)
        return float(np.mean(_kl_diagonal_gaussians(mu_bar, sigma_bar, mu, var)))

def _log_gaussian_all_pairs(x: Array, mu: Array, var: Array) -> Array:
    """log psi_j(x_s) for every sample s and component j -> (S, n). Diagonal Gaussians.

    x is (S, q): S points in latent space. mu, var are (n, q): the n components (one per cell).

    The quadratic ``sum_d (x_sd - mu_jd)^2 / sigma_jd^2`` is expanded into matrix products so the
    (S, n, q) intermediate is never formed -- at S=2048, n=7000, q=100 that intermediate alone is
    11.5 GB, which is what made the naive form unusable at population scale.

    Note which indices are shared. ``x`` carries s and ``P`` carries j, so contracting them over d is
    a genuine matrix product; but ``mu`` and ``P`` BOTH carry j, so they must be multiplied
    elementwise before contracting -- a matmul there would pair component j with a different
    component k and produce an (n, n) array.
    """
    sigma_2 = np.maximum(var, _VAR_FLOOR)
    precision = 1.0 / sigma_2                                       # (n, q)
    quadratic = ((x ** 2) @ precision.T                             # sum_d x_sd^2 P_jd
                 - 2.0 * (x @ (mu * precision).T)                   # -2 sum_d x_sd mu_jd P_jd
                 + np.sum(mu * mu * precision, axis=1)[None, :])    # sum_d mu_jd^2 P_jd
    # Algebraically a sum of squares over positive variances, so non-negative; the expanded form can
    # dip slightly below zero through cancellation when x is close to mu.
    quadratic = np.maximum(quadratic, 0.0)
    log_det = np.sum(np.log(2.0 * np.pi * sigma_2), axis=1)[None, :]
    return -0.5 * (log_det + quadratic)

@dataclass(frozen=True)
class MonteCarloMixture:
    """Reference estimator: samples the mixture directly. O(n_samples * n), no Gaussian assumption.

    Slower and noisier than the moment-matched form, but it makes no claim about the shape of the
    mixture, so it is the yardstick for deciding when that claim is safe.

    Draws ``n_samples`` cells uniformly, one latent sample each (which is exactly a draw from the
    equal-weight mixture), then evaluates ``log psi_0(x) - (1/n) sum_k log psi_k(x)`` at those
    points. The mixture density is evaluated against every component, so cost is O(n_samples * n).
    """
    n_samples: int = 2048
    seed: int = 0

    def __call__(self, posterior: CellPosterior) -> float:
        mu, var = posterior.means, posterior.variances
        n, q = mu.shape
        var = np.maximum(var, _VAR_FLOOR)
        rng = np.random.default_rng(self.seed)

        which = rng.integers(0, n, size=self.n_samples)                       # pick components
        x = mu[which] + rng.normal(size=(self.n_samples, q)) * np.sqrt(var[which])

        log_comp = _log_gaussian_all_pairs(x, mu, var)                        # Shape (n_samples, n)
        log_mixture = _logsumexp(log_comp, axis=1) - np.log(n)                # log psi_0(x_s)
        mean_log_component = log_comp.mean(axis=1)                            # (1/n) sum_k log psi_k
        return float(np.mean(log_mixture - mean_log_component))

def _logsumexp(a: Array, axis: int) -> Array:
    peak = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(a - peak), axis=axis))


def uncertainty_radius(posterior: CellPosterior,
                       estimator: RadiusEstimator | None = None) -> float:
    """The population's uncertainty radius, computed on **all** cells given (see module docstring)."""
    return (estimator or MomentMatchedGaussian())(posterior)


# --------------------------------------------------------------------------------------------------
# Subsampling to a transport budget
# --------------------------------------------------------------------------------------------------

class Subsampler(Protocol):
    """Selects a transport-budget subset of a population.

    Returns row indices into the ``CellPosterior`` -- indices are the currency, because the *same*
    array subsets the plan, the ``ExpressionMatrix``, and the ``Membership`` through their ``select``
    methods, keeping every cell-indexed object aligned. A subsampler never builds the subset itself;
    the caller does ``posterior.select(idx)`` (or ``cells.select(idx)``).
    """

    def indices(self, posterior: CellPosterior, k: int) -> Array: ...


@dataclass(frozen=True)
class RandomSubsampler:
    """Uniform subsample without replacement -- the baseline.

    Biases the uncertainty radius ``eta`` *downward* through self-inclusion (each drawn cell is part
    of the mixture it is compared against), which is exactly why ``eta`` is computed on the full
    population *before* any subsampling (see the module docstring).

    The returned indices are sorted, so the subset preserves the population's original relative cell
    order -- ``CellAxis`` order is meaningful and comparisons against it are explicit.
    """

    seed: int = 0

    def indices(self, posterior: CellPosterior, k: int) -> Array:
        n = len(posterior)
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        if k >= n:
            return np.arange(n)                          # budget exceeds the population: keep all
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(n, size=k, replace=False))


@dataclass(frozen=True)
class CoverageSubsampler:
    """Stratified farthest-point coverage. Guarantees every population present is represented, then
    fills the budget with cells maximally spread in posterior space (symmetrised KL).

    Strata are the MAP populations ``argmax(prob_cat)``: after population-seeded training the
    components ARE the populations, so no external labels are needed. Each covered stratum contributes
    its highest-confidence (most prototypical) cell as a guaranteed seed; the rest of the budget is
    filled by greedy farthest-point sampling over the WHOLE population, which front-loads extremes and
    rare transition cells -- biasing ``eta`` *upward*, the opposite of ``RandomSubsampler``.

    Deterministic: the seeds (``argmax`` confidence) and the FPS walk (``argmax`` min-distance, ties
    broken by first index) are fixed by the data, so there is no seed parameter.
    """

    def indices(self, posterior: CellPosterior, k: int) -> Array:
        n = len(posterior)
        if k <= 0:
            raise ValueError(f"k must be positive; got {k}")
        if k >= n:
            return np.arange(n)                                     # budget exceeds population

        mu, var = posterior.means, posterior.variances
        strata = np.argmax(posterior.prob_cat, axis=1)             # MAP population per cell
        labels, counts = np.unique(strata, return_counts=True)

        covered = labels                                           # every present population
        if k < len(labels):                                       # budget can't seat them all
            warnings.warn(f"k={k} < {len(labels)} populations present; the per-population guarantee "
                          f"cannot hold. Covering the {k} most populous.", stacklevel=2)
            covered = labels[np.argsort(counts)[::-1][:k]]

        # seed: the highest-confidence (most prototypical) cell of each covered stratum
        seeds = [np.where(strata == c)[0][np.argmax(posterior.prob_cat[strata == c, c])]
                 for c in covered]

        min_dist = np.full(n, np.inf)                             # divergence to nearest selected
        for s in seeds:
            min_dist = np.minimum(min_dist, _symmetrised_kl(mu[s], var[s], mu, var))
        selected = list(seeds)
        min_dist[selected] = -np.inf                             # never reselect a chosen cell

        while len(selected) < k:
            nxt = int(np.argmax(min_dist))                       # farthest from the current set
            selected.append(nxt)
            min_dist = np.minimum(min_dist, _symmetrised_kl(mu[nxt], var[nxt], mu, var))
            min_dist[nxt] = -np.inf
        return np.sort(np.asarray(selected))

