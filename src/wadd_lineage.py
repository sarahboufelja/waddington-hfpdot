"""wadd_lineage -- transport plans as lineage operators: kernels, push/pull, transition tables.

A transport plan between two timepoints is a joint mass distribution; reading lineage off it means
conditioning that joint, and every question is a choice of which axis to condition on:

    plan  pi  (n_s, n_t):   pi[i, j] = mass moving from source cell i to target cell j
    row marginal  mu_i = sum_j pi[i, j]        how much mass leaves source cell i
    col marginal  nu_j = sum_i pi[i, j]        how much mass arrives at target cell j

    descendant kernel  K = pi / mu[:, None]    (n_s, n_t), rows sum to 1 -- P(target | source)
    ancestor kernel    B = pi.T / nu[:, None]  (n_t, n_s), rows sum to 1 -- P(source | target)

    push forward   q = K.T @ p                 a source distribution -> its descendants
    pull back      p = B.T @ q                 a target distribution -> its ancestors

The two kernels are *different* conditionings of one plan, not transposes of each other: `K` divides
by the row marginal and `B` by the column marginal. Keeping both explicit here is deliberate -- the
axis and normalisation conventions are exactly where near-duplicate lineage code drifts into silent
sign/axis bugs, so they are stated once and reused.

The population-level summary is the **transition table**

    T[c, d] = fraction of population c's descendant mass that lands in population d,

built by pushing each source population's cell distribution forward and reading its overlap with each
target population. Populations enter only through membership matrices (`Membership.matrix`), so this
module needs no notion of what a population *is* -- and works for any granularity of the fate axis.

Framework-free numpy throughout: the plan comes from `wadd_ot` (or from a sampled HFPD-OT hyperprior),
the memberships from `wadd_data_ingest`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

Array = np.ndarray

_MASS_FLOOR = 1e-300        # guards the divisions; far below any meaningful transported mass


def _row_normalize(M: Array) -> Array:
    """Rows to sum 1, leaving all-zero rows as zero (no mass to condition on -> no distribution)."""
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > _MASS_FLOOR)


def descendant_kernel(plan: Array) -> Array:
    """``P(target | source)``: the plan conditioned on the source, i.e. row-normalised. (n_s, n_t)."""
    return _row_normalize(np.asarray(plan, dtype=float))


def ancestor_kernel(plan: Array) -> Array:
    """``P(source | target)``: the plan conditioned on the target, i.e. column-normalised, transposed.

    Not the transpose of ``descendant_kernel`` -- that divides by the row marginal, this by the column
    marginal. They coincide only when both marginals are uniform.
    """
    return _row_normalize(np.asarray(plan, dtype=float).T)


def push_forward(p: Array, plan: Array) -> Array:
    """Send a distribution over SOURCE cells to its descendants over target cells.

    ``q = K.T @ p`` with ``K`` the descendant kernel. Mass-preserving when ``p`` is a distribution and
    every source cell carrying ``p``-mass transports somewhere.
    """
    p = np.asarray(p, dtype=float)
    K = descendant_kernel(plan)
    if p.shape[-1] != K.shape[0]:
        raise ValueError(f"distribution over {p.shape[-1]} cells for a plan with {K.shape[0]} sources")
    return p @ K


def pull_back(q: Array, plan: Array) -> Array:
    """Send a distribution over TARGET cells back to its ancestors over source cells (``B.T @ q``)."""
    q = np.asarray(q, dtype=float)
    B = ancestor_kernel(plan)
    if q.shape[-1] != B.shape[0]:
        raise ValueError(f"distribution over {q.shape[-1]} cells for a plan with {B.shape[0]} targets")
    return q @ B


def population_distributions(membership: Array) -> Array:
    """Per-population distributions over cells: column-normalised membership, as (n_pops, n_cells).

    Population ``c``'s row is its membership column normalised to sum 1 -- the distribution over cells
    that *are* that population, soft memberships included. A population with no mass gives a zero row.
    """
    W = np.asarray(membership, dtype=float)
    return _row_normalize(W.T)


@dataclass(frozen=True)
class TransitionTable:
    """Population-to-population lineage flow between two timepoints.

    ``matrix[c, d]`` is the fraction of source population ``c``'s descendant mass that lands in target
    population ``d``. Rows sum to 1 where the source population has mass and its descendants reach
    *some* labelled target population; a row sums to less than 1 when descendants land on unlabelled
    cells, and to 0 when the population is absent at the source timepoint.
    """
    matrix: Array                     # (n_source_pops, n_target_pops)
    source_populations: List[str]
    target_populations: List[str]
    source_day: float
    target_day: float

    def __post_init__(self):
        M = np.asarray(self.matrix, dtype=float)
        object.__setattr__(self, "matrix", M)
        if M.shape != (len(self.source_populations), len(self.target_populations)):
            raise ValueError(f"matrix {M.shape} does not match "
                             f"({len(self.source_populations)}, {len(self.target_populations)})")

    @property
    def unlabelled_leak(self) -> Array:
        """Per source population, the descendant mass that reached no labelled target population."""
        return 1.0 - self.matrix.sum(axis=1)

    def top_destinations(self, k: int = 3) -> dict[str, List[tuple[str, float]]]:
        """The ``k`` largest destinations per source population -- the table's readable summary."""
        out = {}
        for c, name in enumerate(self.source_populations):
            order = np.argsort(self.matrix[c])[::-1][:k]
            out[name] = [(self.target_populations[j], float(self.matrix[c, j])) for j in order
                         if self.matrix[c, j] > 0]
        return out


def transition_table(plan: Array, source_membership: Array, target_membership: Array,
                     source_populations: Sequence[str], target_populations: Sequence[str],
                     source_day: float = 0.0, target_day: float = 1.0,
                     normalize: bool = True) -> TransitionTable:
    """Population-level lineage flow implied by one transport plan.

    Each source population's cell distribution is pushed forward through the descendant kernel and
    read against the target memberships:

        T = P_source @ K @ W_target,     P_source = column-normalised source membership (n_pops, n_s)

    With ``normalize`` the rows are rescaled to sum 1 over the *labelled* target populations, so an
    entry reads as "the fraction of c's labelled descendants that are d" and rows are comparable
    across populations of very different size; without it, the row deficit is the mass that landed on
    unlabelled cells (recoverable either way via ``unlabelled_leak`` on the un-normalised table).

    Both memberships must already be aligned to their population axes (see
    ``Membership.align_populations``); the axes may differ between source and target.
    """
    plan = np.asarray(plan, dtype=float)
    Ws = np.asarray(source_membership, dtype=float)
    Wt = np.asarray(target_membership, dtype=float)
    if Ws.shape[0] != plan.shape[0]:
        raise ValueError(f"source membership has {Ws.shape[0]} rows for {plan.shape[0]} plan rows")
    if Wt.shape[0] != plan.shape[1]:
        raise ValueError(f"target membership has {Wt.shape[0]} rows for {plan.shape[1]} plan columns")

    P = population_distributions(Ws)                       # (n_pops_s, n_s), rows sum to 1
    T = P @ descendant_kernel(plan) @ Wt                   # (n_pops_s, n_pops_t)
    return TransitionTable(matrix=_row_normalize(T) if normalize else T,
                           source_populations=list(source_populations),
                           target_populations=list(target_populations),
                           source_day=source_day, target_day=target_day)


def compose_plans(plans: Sequence[Array]) -> Array:
    """Chain consecutive plans at the CELL level: the product of their descendant kernels.

    Returns a row-stochastic ``(n_first, n_last)`` kernel giving ``P(final cell | initial cell)``.

    **This is a Markov assumption, not a free lunch.** Chaining kernels sums over the intermediate
    timepoint,

        P(x_{t+2} | x_t) = sum_j P(x_{t+2} | x_{t+1}=j, x_t) P(x_{t+1}=j | x_t),

    and collapsing the first factor to ``P(x_{t+2} | x_{t+1}=j)`` -- i.e. Chapman-Kolmogorov -- holds
    only if the cell process is Markov in the latent state: where a cell goes next depends on where it
    is now, not on how it got there. That is the standard trajectory-inference assumption, but it is
    an assumption.

    It is nonetheless the *weaker* of the two compositions available here. ``compose`` chains tables on
    the coarse population axis, which needs this same cell-level Markov property **plus** lumpability
    of the fate partition (a function of a Markov chain is in general not itself Markov -- it is only
    when the Kemeny-Snell condition holds, which coarse fate labels have no reason to satisfy). So
    composing plans assumes strictly less than composing tables.

    It is also the only composition that survives unlabelled timepoints. A table for a day with no
    labelled cells is all-zero and annihilates a population-level chain; the underlying plans are
    unaffected, since transport does not depend on labels.
    """
    if not plans:
        raise ValueError("no plans to compose")
    K = descendant_kernel(plans[0])
    for p in plans[1:]:
        Kn = descendant_kernel(p)
        if K.shape[1] != Kn.shape[0]:
            raise ValueError(f"cannot chain: plan ends with {K.shape[1]} cells, "
                             f"next starts with {Kn.shape[0]}")
        K = K @ Kn
    return K


def compose(tables: Sequence[TransitionTable]) -> TransitionTable:
    """Chain consecutive transition tables into one source->final table (matrix product).

    Each table is row-stochastic on the population axis, so the product is well-formed; whether it
    *means* anything is a modelling question. Chaining requires fate flow to be Markov on the coarse
    population axis, which is strictly stronger than the cell-level Markov property assumed by
    ``compose_plans``: it additionally needs the fate partition to be lumpable (Kemeny-Snell), since a
    function of a Markov chain is generally not Markov. Populations are coarse-grainings of the latent
    state with no reason to satisfy that, so this is an approximation of the cell-level composition,
    not an identity.

    It also fails outright across a timepoint with no labelled cells: that table is all-zero and
    annihilates the product. Prefer ``compose_plans`` unless the cell-level plans are unavailable.
    """
    if not tables:
        raise ValueError("no tables to compose")
    for a, b in zip(tables, tables[1:]):
        if a.target_populations != b.source_populations:
            raise ValueError(f"cannot compose: {a.target_day} targets {a.target_populations[:3]}... "
                             f"but {b.source_day} sources {b.source_populations[:3]}...")
    M = tables[0].matrix
    for t in tables[1:]:
        M = M @ t.matrix
    return TransitionTable(matrix=M, source_populations=tables[0].source_populations,
                           target_populations=tables[-1].target_populations,
                           source_day=tables[0].source_day, target_day=tables[-1].target_day)
