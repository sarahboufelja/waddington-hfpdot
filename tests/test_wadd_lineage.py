"""Transport plans as lineage operators -- kernels, push/pull, transition tables.

The conventions are the whole point: a plan is a joint, and every lineage question is a choice of
which axis to condition on. The descendant kernel divides by the ROW marginal, the ancestor kernel by
the COLUMN marginal, and they are not transposes of each other. These tests pin that against
hand-computed values on deliberately asymmetric plans, where an axis slip changes the answer.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wadd_lineage import (  # noqa: E402
    descendant_kernel, ancestor_kernel, push_forward, pull_back, population_distributions,
    transition_table, compose, compose_plans, TransitionTable,
)


# A deliberately asymmetric plan: unequal row and column marginals, so row- and column-normalisation
# genuinely differ and a transposed-kernel bug cannot pass.
PLAN = np.array([[0.4, 0.1],
                 [0.1, 0.4]])
SKEW = np.array([[0.5, 0.1],          # row marginals (0.6, 0.4); col marginals (0.7, 0.3)
                 [0.2, 0.2]])


# ---- the two kernels are different conditionings --------------------------------------------------

def test_descendant_kernel_is_row_normalised():
    K = descendant_kernel(SKEW)
    assert np.allclose(K.sum(axis=1), 1.0)
    assert np.allclose(K, [[0.5 / 0.6, 0.1 / 0.6],
                           [0.2 / 0.4, 0.2 / 0.4]])


def test_ancestor_kernel_is_column_normalised_and_transposed():
    B = ancestor_kernel(SKEW)
    assert B.shape == (2, 2) and np.allclose(B.sum(axis=1), 1.0)
    assert np.allclose(B, [[0.5 / 0.7, 0.2 / 0.7],       # target 0 <- sources, col marginal 0.7
                           [0.1 / 0.3, 0.2 / 0.3]])


def test_ancestor_kernel_is_not_the_transpose_of_the_descendant_kernel():
    """They coincide only when both marginals are uniform -- the bug this guards against."""
    assert not np.allclose(ancestor_kernel(SKEW), descendant_kernel(SKEW).T)
    balanced = np.array([[0.3, 0.2], [0.2, 0.3]])        # both marginals uniform
    assert np.allclose(ancestor_kernel(balanced), descendant_kernel(balanced).T)


def test_zero_mass_rows_stay_zero():
    """A source cell that transports nothing has no conditional distribution -- not a NaN."""
    plan = np.array([[0.5, 0.5], [0.0, 0.0]])
    K = descendant_kernel(plan)
    assert np.all(np.isfinite(K)) and np.allclose(K[1], 0.0)


# ---- push / pull ----------------------------------------------------------------------------------

def test_push_forward_moves_mass_through_the_descendant_kernel():
    p = np.array([1.0, 0.0])                              # all mass on source 0
    q = push_forward(p, SKEW)
    assert np.allclose(q, [0.5 / 0.6, 0.1 / 0.6])         # source 0's own conditional
    assert np.isclose(q.sum(), 1.0)


def test_pull_back_moves_mass_through_the_ancestor_kernel():
    q = np.array([1.0, 0.0])                              # all mass on target 0
    p = pull_back(q, SKEW)
    assert np.allclose(p, [0.5 / 0.7, 0.2 / 0.7])
    assert np.isclose(p.sum(), 1.0)


def test_push_forward_preserves_total_mass():
    rng = np.random.default_rng(0)
    plan = rng.uniform(0.05, 1.0, size=(5, 4))
    p = rng.dirichlet(np.ones(5))
    assert np.isclose(push_forward(p, plan).sum(), 1.0)


def test_push_then_pull_is_not_the_identity():
    """Lineage is lossy: descendants of a source population have many ancestors, so the round trip
    diffuses rather than returning the starting distribution."""
    p = np.array([1.0, 0.0])
    assert not np.allclose(pull_back(push_forward(p, SKEW), SKEW), p)


def test_push_forward_rejects_a_mismatched_distribution():
    with pytest.raises(ValueError, match="3 cells"):
        push_forward(np.ones(3) / 3, SKEW)


# ---- population distributions ---------------------------------------------------------------------

def test_population_distributions_normalise_membership_columns():
    W = np.array([[1.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0]])
    P = population_distributions(W)
    assert P.shape == (2, 3)
    assert np.allclose(P[0], [0.5, 0.5, 0.0])            # population 0 is uniform on its two cells
    assert np.allclose(P[1], [0.0, 0.0, 1.0])


def test_population_with_no_mass_gives_a_zero_row():
    W = np.array([[1.0, 0.0], [1.0, 0.0]])               # population 1 absent
    assert np.allclose(population_distributions(W)[1], 0.0)


# ---- transition tables ----------------------------------------------------------------------------

def _two_by_two(plan, Ws, Wt, **kw):
    return transition_table(plan, Ws, Wt, ["A", "B"], ["A", "B"], 1.0, 2.0, **kw)


def test_transition_table_reads_a_clean_split():
    """Source A's two cells go entirely to target cells labelled A and B in a 3:1 ratio."""
    plan = np.array([[0.75, 0.25],                        # source cell 0 -> 75% target 0, 25% target 1
                     [0.75, 0.25]])
    W = np.array([[1.0, 0.0], [0.0, 1.0]])                # cell 0 is A, cell 1 is B
    T = _two_by_two(plan, np.array([[1.0, 0.0], [1.0, 0.0]]), W)
    assert np.allclose(T.matrix[0], [0.75, 0.25])         # A -> 75% A, 25% B
    assert np.allclose(T.matrix[1], 0.0)                  # B absent at source


def test_transition_table_rows_sum_to_one_when_normalised():
    rng = np.random.default_rng(1)
    plan = rng.uniform(0.05, 1.0, size=(6, 5))
    Ws = np.eye(6)[:, :2]; Ws[2:, :] = 0; Ws[2:4, 0] = 1.0; Ws[4:, 1] = 1.0
    Wt = np.zeros((5, 2)); Wt[:3, 0] = 1.0; Wt[3:, 1] = 1.0
    T = transition_table(plan, Ws, Wt, ["A", "B"], ["A", "B"])
    rows = T.matrix.sum(axis=1)
    assert np.allclose(rows[rows > 0], 1.0)


def test_unnormalised_table_exposes_the_unlabelled_leak():
    """Descendants landing on unlabelled cells are missing mass, and the un-normalised table keeps
    that visible instead of silently rescaling it away."""
    plan = np.array([[0.5, 0.5]])                         # one source cell, two targets
    Ws = np.array([[1.0]])                                # source population A
    Wt = np.array([[1.0], [0.0]])                         # target cell 1 is UNLABELLED
    T = transition_table(plan, Ws, Wt, ["A"], ["A"], normalize=False)
    assert np.allclose(T.matrix, [[0.5]])
    assert np.allclose(T.unlabelled_leak, [0.5])


def test_soft_membership_flows_proportionally():
    """A cell that is half A and half B contributes half its descendants to each source population."""
    plan = np.array([[1.0, 0.0],                          # cell 0 -> target 0
                     [0.0, 1.0]])                         # cell 1 -> target 1
    Ws = np.array([[0.5, 0.5], [0.0, 1.0]])               # cell 0 is 50/50, cell 1 is pure B
    Wt = np.array([[1.0, 0.0], [0.0, 1.0]])
    T = _two_by_two(plan, Ws, Wt)
    assert np.allclose(T.matrix[0], [1.0, 0.0])           # A lives only on cell 0 -> target 0
    assert np.allclose(T.matrix[1], [1 / 3, 2 / 3])       # B is 1/3 on cell 0, 2/3 on cell 1


def test_top_destinations_ranks_by_mass():
    T = TransitionTable(np.array([[0.1, 0.7, 0.2]]), ["A"], ["X", "Y", "Z"], 1.0, 2.0)
    assert T.top_destinations(2)["A"] == [("Y", 0.7), ("Z", 0.2)]


def test_transition_table_rejects_mismatched_memberships():
    with pytest.raises(ValueError, match="source membership"):
        transition_table(np.ones((3, 2)), np.ones((4, 2)), np.ones((2, 2)), ["A", "B"], ["A", "B"])
    with pytest.raises(ValueError, match="target membership"):
        transition_table(np.ones((3, 2)), np.ones((3, 2)), np.ones((5, 2)), ["A", "B"], ["A", "B"])


# ---- composition ----------------------------------------------------------------------------------

def test_compose_chains_consecutive_tables():
    t1 = TransitionTable(np.array([[0.5, 0.5], [0.0, 1.0]]), ["A", "B"], ["A", "B"], 1.0, 2.0)
    t2 = TransitionTable(np.array([[1.0, 0.0], [0.0, 1.0]]), ["A", "B"], ["A", "B"], 2.0, 3.0)
    c = compose([t1, t2])
    assert np.allclose(c.matrix, t1.matrix @ t2.matrix)
    assert c.source_day == 1.0 and c.target_day == 3.0


def test_compose_preserves_row_stochasticity():
    t1 = TransitionTable(np.array([[0.3, 0.7], [0.6, 0.4]]), ["A", "B"], ["A", "B"], 1.0, 2.0)
    t2 = TransitionTable(np.array([[0.9, 0.1], [0.2, 0.8]]), ["A", "B"], ["A", "B"], 2.0, 3.0)
    assert np.allclose(compose([t1, t2]).matrix.sum(axis=1), 1.0)


def test_compose_rejects_a_broken_chain():
    t1 = TransitionTable(np.array([[1.0]]), ["A"], ["A"], 1.0, 2.0)
    t2 = TransitionTable(np.array([[1.0]]), ["B"], ["B"], 2.0, 3.0)
    with pytest.raises(ValueError, match="cannot compose"):
        compose([t1, t2])


# ---- cell-level composition -----------------------------------------------------------------------

def test_compose_plans_chains_descendant_kernels():
    p1 = np.array([[0.5, 0.5], [0.2, 0.8]])
    p2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = compose_plans([p1, p2])
    assert np.allclose(K, descendant_kernel(p1) @ descendant_kernel(p2))
    assert np.allclose(K.sum(axis=1), 1.0)


def test_compose_plans_survives_an_unlabelled_timepoint():
    """The reason cell-level composition exists: a day with no labelled cells gives an all-zero
    population table that annihilates a table-level chain, while the plans are unaffected."""
    plans = [np.array([[0.6, 0.4], [0.3, 0.7]])] * 3
    K = compose_plans(plans)
    assert np.allclose(K.sum(axis=1), 1.0) and np.all(K > 0)

    W_unlabelled = np.zeros((2, 2))                        # the middle day carries no labels
    dead = TransitionTable(np.zeros((2, 2)), ["A", "B"], ["A", "B"], 1.0, 2.0)
    assert np.allclose(compose([dead, dead]).matrix, 0.0)  # table chain collapses...
    assert np.all(K > 0)                                   # ...but the plan chain does not


def test_compose_plans_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="cannot chain"):
        compose_plans([np.ones((3, 4)), np.ones((5, 2))])


def test_compose_plans_rejects_empty():
    with pytest.raises(ValueError, match="no plans"):
        compose_plans([])
