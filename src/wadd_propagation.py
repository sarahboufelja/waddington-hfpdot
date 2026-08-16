"""wadd_propagation -- the marginal particle filter (module 5).

Propagates marginal uncertainty through a chain of HFPD-OT couplings without the exponential
explosion that naive composition creates: sampling n plans per pair turns one source marginal into
n futures, each future seeds its own sampling at the next pair, and after T steps there are n^T
trajectories. The filter holds a fixed budget of P *marginal particles* per timepoint instead.

This is the exact filter, not a moment-matched collapse: **each particle conditions its own
hyperprior** (``mu_0 :=`` the particle's marginal), so the distribution over futures at step t+1 is
the genuine mixture over the step-t ensemble, and what compounds over time is the model's own
coupling uncertainty -- the quantity the propagation is supposed to measure.

The trimming primitive is coverage-first, not resampling: multinomial resampling collapses the
ensemble onto its mode and quietly narrows every downstream credible band, which is the one failure
mode a UQ pipeline must not have. Instead the pooled futures are farthest-point subsampled
(symmetrised KL on the simplex) and every pruned future donates its ensemble mass to its nearest
survivor -- the same select-then-absorb construction as the space-level landmark quantisation, one
level up. The ensemble stays a weighted, unbiased particle set; extremes survive; bands stay
conservative. Parent indices are recorded so multi-step trajectory statements remain
reconstructible.

The sampler enters only through the ``PlanSampler`` callable, injected by the driver: swapping the
MALA kernel for its successor (or a mock in tests) touches nothing here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np
from numpy.typing import NDArray as Array

_EPS = 1e-12

# (mu_0, nu_0) -> (draws, II*JJ) plan samples plus a diagnostics dict for the run record.
PlanSampler = Callable[[Array, Array], tuple[Array, dict]]


@dataclass(frozen=True)
class MarginalParticle:
    """One point of the marginal ensemble at a timepoint.

    ``weights`` is a distribution over the day's support (sums to 1); ``log_mass`` is the particle's
    log ensemble weight, including everything absorbed from trimmed siblings; ``parent`` indexes the
    previous day's ensemble (``None`` at the root).
    """
    weights: Array
    log_mass: float
    parent: int | None

    def __post_init__(self):
        w = np.asarray(self.weights)
        if w.ndim != 1 or np.any(w < 0):
            raise ValueError("weights must be a 1-D non-negative vector")
        # Weights are MASSES (intrinsic growth): total mass is the growth signal, so it is
        # required finite and positive, not normalised. Shape statistics normalise locally.
        tot = float(w.sum())
        if not np.isfinite(tot) or tot <= 0:
            raise ValueError(f"weights must carry finite positive total mass, got {tot}")


def initial_ensemble(m: int) -> List[MarginalParticle]:
    """The root ensemble: one particle, uniform over the day's ``m``-cell support."""
    return [MarginalParticle(weights=np.full(m, 1.0 / m), log_mass=0.0, parent=None)]


def push_forward_marginal(plan: Array, II: int, JJ: int) -> Array:
    """Descendant marginal of one plan draw: column sums of ``pi``, as MASSES.

    No normalisation: on the orthant the total mass of a draw is its growth signal, and the
    marginal ensemble carries it forward so growth and apoptosis compound across pairs
    (intrinsic growth -- no exogenous rates). Balanced (simplex) draws sum to 1, so their
    behaviour is unchanged. Shape statistics downstream normalise locally where a
    distribution is required.
    """
    nu = np.asarray(plan).reshape(II, JJ).sum(axis=0)
    tot = nu.sum()
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError(f"plan draw has non-positive total mass {tot}")
    return nu


def _sym_kl(p: Array, q: Array) -> float:
    """Symmetrised generalized KL, guarded at the boundary; valid on the positive orthant.

    ``sum (p-q)(log p - log q)`` equals ``gKL(p||q) + gKL(q||p)`` -- the linear mass terms
    cancel in the symmetrisation -- so it is a proper divergence for unnormalised mass
    vectors (intrinsic growth) and coincides with the symmetrised plain KL on the simplex.
    Shape-identical futures with different total mass are separated, so the coverage trim
    retains growth spread.
    """
    p = np.maximum(p, _EPS); q = np.maximum(q, _EPS)
    return float(np.sum((p - q) * (np.log(p) - np.log(q))))


def fps_trim(futures: Sequence[MarginalParticle], budget: int) -> List[MarginalParticle]:
    """Coverage-trim a pooled future set down to ``budget`` particles, conserving ensemble mass.

    Selection is greedy farthest-point under symmetrised KL, seeded at the highest-mass future --
    the ensemble's principal mode is always retained, then the budget buys the most spread-out
    remainder (extremes first: conservative bands). Every pruned future adds its mass to its
    nearest survivor (log-sum-exp), so the returned set is a reweighting, never a subsample: total
    mass is conserved exactly and the estimator stays unbiased for ensemble averages.
    """
    futures = list(futures)
    if budget <= 0:
        raise ValueError(f"budget must be positive; got {budget}")
    if len(futures) <= budget:
        return futures

    n = len(futures)
    log_mass = np.array([f.log_mass for f in futures])
    selected = [int(np.argmax(log_mass))]
    min_dist = np.array([_sym_kl(futures[selected[0]].weights, f.weights) for f in futures])
    while len(selected) < budget:
        cand = int(np.argmax(min_dist))
        selected.append(cand)
        d = np.array([_sym_kl(futures[cand].weights, f.weights) for f in futures])
        min_dist = np.minimum(min_dist, d)

    # nearest-survivor assignment for every future (survivors map to themselves)
    dist_to_sel = np.stack([[_sym_kl(futures[s].weights, f.weights) for f in futures]
                            for s in selected])                       # (budget, n)
    owner = np.argmin(dist_to_sel, axis=0)                            # (n,) index into `selected`
    out = []
    for k, s in enumerate(selected):
        absorbed = log_mass[owner == k]
        out.append(MarginalParticle(weights=futures[s].weights,
                                    log_mass=float(_logsumexp(absorbed)),
                                    parent=futures[s].parent))
    out.sort(key=lambda p: -p.log_mass)
    return out


def _logsumexp(x: Array) -> float:
    x = np.asarray(x, dtype=float)
    hi = np.max(x)
    return float(hi + np.log(np.sum(np.exp(x - hi))))


@dataclass
class PairRecord:
    """Everything one pair-step produces, for bands and audit.

    ``tables`` stacks every retained draw's transition table across all particles;
    ``table_log_mass`` aligns with it (parent particle's mass split evenly over its draws), so any
    downstream statistic is a weighted functional of the draws. ``growth`` aligns the same way:
    per draw, the transported source mass ``(pi 1)_i`` -- the raw growth signal; the per-cell
    growth factor is its ratio to the parent particle's input mass (recoverable via
    ``pool[i].parent`` -> ``source``), so growth and apoptosis are posterior random variables
    read from the plans themselves. ``pool`` is the FULL pre-trim future set (one marginal per
    retained draw, parent-tagged) -- the trim step's input, kept so selection and absorption
    remain reconstructible. ``diagnostics`` holds one dict per source particle, straight from
    the injected sampler. ``extras`` is the optional second per-draw functional channel
    (``extra_of_plan``), aligned with ``tables`` draw for draw.
    """
    day_from: float
    day_to: float
    source: List[MarginalParticle]
    pool: List[MarginalParticle]             # all futures before the trim
    futures: List[MarginalParticle]          # trimmed ensemble at day_to
    tables: Array                            # (n_draws_total, K, K)
    table_log_mass: Array                    # (n_draws_total,)
    growth: Array                            # (n_draws_total, II) transported source mass per draw
    diagnostics: List[dict]
    extras: Array | None = None              # (n_draws_total, ...) optional per-draw functional


def propagate_pair(
    particles: Sequence[MarginalParticle],
    sampler: PlanSampler,
    nu_0: Array,
    II: int,
    JJ: int,
    budget: int,
    table_of_plan: Callable[[Array], Array],
    extra_of_plan: Callable[[Array], Array] | None = None,
    day_from: float = 0.0,
    day_to: float = 1.0,
    max_futures_per_particle: int | None = None,
) -> PairRecord:
    """One exact filter step: every particle conditions its own hyperprior and samples plans.

    ``sampler(mu_0, nu_0)`` returns ``(draws, diag)`` with draws of shape ``(D, II*JJ)``;
    ``table_of_plan`` maps one flat plan draw to a ``(K, K)`` transition table. Each draw becomes a
    future carrying ``parent`` = its source particle's index and ``1/D`` of that particle's mass;
    the pooled futures are then coverage-trimmed back to ``budget``. ``extra_of_plan``
    (optional) maps the same flat draw to any additional per-draw functional (e.g. ancestor
    participation ratios); results stack into ``extras`` aligned with ``tables``.
    """
    futures: List[MarginalParticle] = []
    tables, table_log_mass, growth, diags, extras = [], [], [], [], []
    for p_idx, p in enumerate(particles):
        draws, diag = sampler(np.asarray(p.weights), np.asarray(nu_0))
        draws = np.asarray(draws)
        if draws.ndim != 2 or draws.shape[1] != II * JJ:
            raise ValueError(f"sampler returned shape {draws.shape}, expected (D, {II * JJ})")
        if max_futures_per_particle is not None and len(draws) > max_futures_per_particle:
            keep = np.linspace(0, len(draws) - 1, max_futures_per_particle).astype(int)
            draws = draws[keep]
        diags.append(diag)
        share = p.log_mass - np.log(len(draws))
        for d in draws:
            futures.append(MarginalParticle(weights=push_forward_marginal(d, II, JJ),
                                            log_mass=share, parent=p_idx))
            tables.append(table_of_plan(d))
            table_log_mass.append(share)
            growth.append(np.asarray(d).reshape(II, JJ).sum(axis=1))
            if extra_of_plan is not None:
                extras.append(extra_of_plan(d))
    trimmed = fps_trim(futures, budget)
    return PairRecord(day_from=day_from, day_to=day_to, source=list(particles),
                      pool=futures, futures=trimmed, tables=np.stack(tables),
                      table_log_mass=np.array(table_log_mass), growth=np.stack(growth),
                      diagnostics=diags,
                      extras=np.stack(extras) if extra_of_plan is not None else None)


def _profile_pr(A: Array) -> Array:
    den = np.square(A).sum(axis=0)
    return np.square(A.sum(axis=0)) / np.maximum(den, 1e-300)


def ancestor_pr(pi: Array, W_t: Array) -> Array:
    """Effective ancestor count per target fate: the q=2 Hill number (inverse Simpson) of
    each fate's ancestor-mass profile ``A = pi @ W_t`` over source cells.

    Scale-invariant in the plan's global mass, so immune to a uniform mass-scale bias;
    the q=2 weighting discounts the entropic blur floor that saturates hard support
    counts. Returns ``(K,)`` values in ``[1, II]``; a fate receiving no mass reads 0.
    """
    return _profile_pr(np.asarray(pi) @ np.asarray(W_t))


def dominant_destination_hist(A: Array) -> Array:
    """Mass-weighted histogram of dominant destinations over source cells, ``(K,)``.

    Cell ``i`` votes for ``argmax_c A[i, c]`` with weight = its transported mass
    ``sum_c A[i, c]``; normalised to sum 1 (scale-invariant). W1 reads the
    {stromal, MET} share of it.
    """
    A = np.asarray(A)
    hist = np.zeros(A.shape[1])
    np.add.at(hist, np.argmax(A, axis=1), A.sum(axis=1))
    tot = hist.sum()
    return hist / tot if tot > 0 else hist


def ancestor_overlap(A: Array) -> Array:
    """Bhattacharyya overlap between fate ancestor profiles, ``(K, K)`` symmetric.

    Column ``c`` of ``A`` normalised to a distribution over source cells; entry
    ``(c, c') = sum_i sqrt(p_c p_c')`` in ``[0, 1]`` -- 1 on the diagonal of occupied
    fates, 0 rows/columns for fates receiving no mass. W2c reads the Neural vs
    trophoblast/iPSC entries along days (ancestor divergence).
    """
    A = np.asarray(A)
    tot = A.sum(axis=0)
    R = np.sqrt(np.divide(A, tot, out=np.zeros_like(A), where=tot > 0))
    return R.T @ R


def identity_readouts(pi: Array, Ws_a: Array, Ws_b: Array) -> tuple[Array, Array]:
    """Mean and variance over M identity resamples of the SAME plan's fate table (E14).

    ``Ws_a (M, II, K)``, ``Ws_b (M, JJ, K)`` are Rao-Blackwellised responsibility
    resamples (z redrawn from the encoder posterior, c integrated out analytically).
    ``T^(m) = Ws_a[m].T @ pi @ Ws_b[m]``; returns ``(E_m[T], Var_m[T])``, each ``(K, K)``.
    ``Var_m`` is the per-draw identity term of the law-of-total-variance split; the cost
    (and hence the plan) stays frozen at the mean embedding by design.
    """
    pi, Ws_a, Ws_b = np.asarray(pi), np.asarray(Ws_a), np.asarray(Ws_b)
    M, JJ, K = Ws_b.shape
    right = (pi @ Ws_b.transpose(1, 0, 2).reshape(JJ, M * K)).reshape(-1, M, K)
    T = np.einsum("mia,imb->mab", Ws_a, right)
    return T.mean(axis=0), T.var(axis=0)


def extras_slices(K: int) -> dict:
    """Layout of the flattened per-draw W-series/identity channel (``w_series_extras``)."""
    edges = np.cumsum([0, K, K, K * K, K * K, K * K])
    names = ("ancestor_pr", "dom_dest", "ancestor_overlap",
             "table_identity_mean", "table_identity_var")
    return {n: slice(int(a), int(b)) for n, a, b in zip(names, edges, edges[1:])}


def w_series_extras(pi: Array, W_b: Array, Ws_a: Array, Ws_b: Array) -> Array:
    """The full per-draw channel, flattened to ``(2K + 3K^2,)`` for ``PairRecord.extras``.

    Components in ``extras_slices`` order; (K, K) blocks flattened row-major. One
    ``A = pi @ W_b`` serves the three W-series reductions; the identity pair
    ``(Ws_a, Ws_b)`` serves the two E14 tables.
    """
    A = np.asarray(pi) @ np.asarray(W_b)
    id_mean, id_var = identity_readouts(pi, Ws_a, Ws_b)
    return np.concatenate([_profile_pr(A), dominant_destination_hist(A),
                           ancestor_overlap(A).ravel(), id_mean.ravel(), id_var.ravel()])


def weighted_quantiles(values: Array, log_mass: Array, qs: Sequence[float]) -> Array:
    """Mass-weighted quantiles along axis 0 of ``values`` (any trailing shape).

    The estimator is the weighted empirical CDF inverse -- exact for the particle representation,
    no interpolation assumptions beyond the step CDF.
    """
    values = np.asarray(values)
    w = np.exp(np.asarray(log_mass) - _logsumexp(np.asarray(log_mass)))
    order = np.argsort(values, axis=0)
    flat = values.reshape(len(values), -1)
    out = np.empty((len(qs),) + values.shape[1:])
    out_flat = out.reshape(len(qs), -1)
    for j in range(flat.shape[1]):
        idx = order.reshape(len(values), -1)[:, j]
        cdf = np.cumsum(w[idx])
        for qi, q in enumerate(qs):
            k = int(np.searchsorted(cdf, q * cdf[-1], side="left"))
            out_flat[qi, j] = flat[idx[min(k, len(idx) - 1)], j]
    return out


def _masses(particles: Sequence[MarginalParticle]) -> tuple[Array, Array]:
    W = np.stack([p.weights for p in particles])
    m = np.exp(np.array([p.log_mass for p in particles]))
    return W, m / m.sum()


def _shapes(particles: Sequence[MarginalParticle]) -> tuple[Array, Array]:
    """Per-particle weights normalised to distributions, ensemble masses normalised to sum 1.

    Shape statistics (KL ball, TV) live on the simplex regardless of whether the weights
    carry biological mass (intrinsic growth); mass statistics read the raw weights instead.
    """
    W, m = _masses(particles)
    tot = W.sum(axis=1, keepdims=True)
    return np.divide(W, tot, out=np.full_like(W, 1.0 / W.shape[1]), where=tot > 0), m


def kl_ball(particles: Sequence[MarginalParticle]) -> tuple[float, float]:
    """The ensemble's KL ball around its barycentre: ``(mean_radius, max_radius)`` in nats.

    The mass-weighted mean forward KL is the hyperprior's own moment condition
    ``E[KL(mu || mu_0)] = eta`` evaluated on the empirical ensemble (with ``mu_0`` the barycentre):
    the tube width in exactly the units and functional form of the elicited radius, so the measured
    value at a timepoint is the *propagated* radius and can seed the next pair's ``eta`` directly.
    The max is the covering-ball radius. Forward KL to the barycentre needs no boundary floor: the
    barycentre vanishes only where every particle does, so every term is well-defined.
    A SHAPE statistic: weights are normalised per particle (mass is read separately).
    """
    W, m = _shapes(particles)
    bary = (m[:, None] * W).sum(axis=0)
    kls = np.array([float(np.sum(np.where(w > _EPS,
                                          w * (np.log(np.maximum(w, _EPS)) -
                                               np.log(np.maximum(bary, _EPS))), 0.0)))
                    for w in W])
    return float(np.sum(m * kls)), float(np.max(kls))


def pairwise_tv(particles: Sequence[MarginalParticle]) -> tuple[float, float]:
    """Mass-weighted mean and maximum pairwise total variation: ``(mean, diameter)`` in [0, 1].

    The Dobrushin-compatible readout: a kernel ``K`` contracts the ensemble's TV diameter by its
    ergodicity coefficient, ``diam_TV(ensemble . K) <= delta(K) . diam_TV(ensemble)``, so the
    diameter measured across filter steps is an empirical contraction ladder, directly comparable
    with the ``delta_eff`` of the Chapman-Kolmogorov analysis. Bounded, floor-free.
    A SHAPE statistic: weights are normalised per particle (mass is read separately).
    """
    W, m = _shapes(particles)
    n = len(W)
    if n < 2:
        return 0.0, 0.0
    mean_num = mean_den = 0.0
    diam = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            tv = 0.5 * float(np.abs(W[i] - W[j]).sum())
            w = m[i] * m[j]
            mean_num += w * tv
            mean_den += w
            diam = max(diam, tv)
    return (mean_num / mean_den if mean_den > 0 else 0.0), diam


def project_particles(particles: Sequence[MarginalParticle], W: Array) -> List[MarginalParticle]:
    """Push every particle through an aggregation matrix (e.g. cell -> fate membership).

    Rows of ``W`` are per-atom distributions over the coarse axis. The projection AGGREGATES
    mass (no renormalisation): with responsibility rows summing to 1 it is exactly
    mass-preserving, so per-fate masses stay readable on the projected ensemble. Shape
    statistics downstream normalise internally. Ensemble masses and parents carry over
    unchanged -- the projection is a change of readout, not of the ensemble.
    """
    W = np.asarray(W)
    return [MarginalParticle(weights=np.asarray(p.weights) @ W,
                             log_mass=p.log_mass, parent=p.parent)
            for p in particles]
