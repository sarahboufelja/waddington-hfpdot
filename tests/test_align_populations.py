"""Stress tests for Membership.align_populations -- the column-axis counterpart of reindex_to.

align_populations makes the population axis canonical so memberships from different days (any fate
order, some fates absent that day) share one comparable column layout. The load-bearing invariant is
mass conservation: a valid alignment only ever reorders an existing column or inserts a zero one, so
no cell's total labelled mass may change, and an input fate missing from the canonical axis is an
error rather than a silent drop.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wadd_data_ingest import CellAxis, Membership  # noqa: E402


def _memb(matrix, names, day=9.0):
    n = len(matrix)
    cells = CellAxis(ids=[f"D{day}_c{i}" for i in range(n)], day=day)
    return Membership(np.asarray(matrix, dtype=float), cells, list(names))


# ---- reorder: same set, permuted -----------------------------------------------------------------

def test_reorder_places_each_column_at_its_canonical_index():
    m = _memb([[0.2, 0.5, 0.3],
               [0.1, 0.0, 0.9]], ["B", "A", "C"])
    out = m.align_populations(["A", "B", "C"])
    assert out.population_names == ["A", "B", "C"]
    # A was column 1, B column 0, C column 2 in the input.
    assert np.allclose(out.matrix[:, 0], [0.5, 0.0])   # A
    assert np.allclose(out.matrix[:, 1], [0.2, 0.1])   # B
    assert np.allclose(out.matrix[:, 2], [0.3, 0.9])   # C


def test_reorder_preserves_row_sums():
    m = _memb([[0.2, 0.5, 0.3], [0.1, 0.0, 0.4]], ["B", "A", "C"])
    out = m.align_populations(["A", "B", "C"])
    assert np.allclose(out.matrix.sum(1), m.matrix.sum(1))


# ---- subset: canonical has fates the input lacks -> zero columns ---------------------------------

def test_absent_canonical_fate_becomes_a_zero_column():
    m = _memb([[0.6, 0.4], [1.0, 0.0]], ["A", "C"])
    out = m.align_populations(["A", "B", "C", "D"])
    assert out.population_names == ["A", "B", "C", "D"]
    assert np.allclose(out.matrix[:, 0], [0.6, 1.0])   # A
    assert np.allclose(out.matrix[:, 1], [0.0, 0.0])   # B absent -> zeros
    assert np.allclose(out.matrix[:, 2], [0.4, 0.0])   # C
    assert np.allclose(out.matrix[:, 3], [0.0, 0.0])   # D absent -> zeros


def test_subset_preserves_row_sums():
    m = _memb([[0.6, 0.4], [0.3, 0.0]], ["A", "C"])
    out = m.align_populations(["A", "B", "C", "D"])
    assert np.allclose(out.matrix.sum(1), m.matrix.sum(1))   # zero-fill adds no mass


# ---- unknown fate in the input -> hard error -----------------------------------------------------

def test_input_fate_absent_from_canonical_raises():
    m = _memb([[0.5, 0.5]], ["A", "Z"])
    with pytest.raises(ValueError, match="absent from canonical_names"):
        m.align_populations(["A", "B", "C"])


def test_error_names_the_offending_population():
    m = _memb([[0.5, 0.5]], ["A", "Ghost"])
    with pytest.raises(ValueError, match="Ghost"):
        m.align_populations(["A", "B"])


# ---- degenerate canonical axis -------------------------------------------------------------------

def test_duplicate_canonical_names_raise():
    m = _memb([[1.0, 0.0]], ["A", "B"])
    with pytest.raises(ValueError, match="duplicate"):
        m.align_populations(["A", "B", "A"])


# ---- idempotence / no-op -------------------------------------------------------------------------

def test_already_canonical_is_returned_unchanged():
    m = _memb([[0.3, 0.7], [1.0, 0.0]], ["A", "B"])
    out = m.align_populations(["A", "B"])
    assert out is m


def test_alignment_is_idempotent():
    m = _memb([[0.2, 0.5, 0.3]], ["C", "A", "B"])
    once = m.align_populations(["A", "B", "C", "D"])
    twice = once.align_populations(["A", "B", "C", "D"])
    assert twice.population_names == once.population_names
    assert np.allclose(twice.matrix, once.matrix)


# ---- result is a valid Membership ----------------------------------------------------------------

def test_result_still_satisfies_the_row_sum_invariant():
    m = _memb([[0.6, 0.4], [0.5, 0.5]], ["B", "A"])
    out = m.align_populations(["A", "B", "C"])          # Membership.__post_init__ re-checks rows<=1
    assert np.all(out.matrix.sum(1) <= 1.0 + 1e-6)


def test_cell_axis_passes_through_untouched():
    m = _memb([[0.6, 0.4], [0.5, 0.5]], ["B", "A"], day=12.5)
    out = m.align_populations(["A", "B", "C"])
    assert out.cell_ids == m.cell_ids and out.day == 12.5


# ---- late-fate skew: a whole day with no labelled populations present ----------------------------

def test_all_zero_membership_aligns_to_all_zero_columns():
    """An early day where none of the canonical (late) fates are present: every column stays zero."""
    m = _memb([[0.0, 0.0], [0.0, 0.0]], ["A", "B"])
    out = m.align_populations(["A", "B", "C"])
    assert np.allclose(out.matrix, 0.0)
    assert out.matrix.shape == (2, 3)


def test_single_population_and_single_cell():
    m = _memb([[1.0]], ["A"])
    out = m.align_populations(["A", "B"])
    assert out.matrix.shape == (1, 2)
    assert np.allclose(out.matrix, [[1.0, 0.0]])


# ---- composes with build_targets (the real caller) -----------------------------------------------

def test_aligned_memberships_stack_across_days_with_disjoint_fates():
    """Two days whose fate sets differ: after aligning both to the union order, their matrices are
    column-compatible and vstack to a single (cells, canonical) target block."""
    canonical = ["A", "B", "C"]
    day9 = _memb([[1.0, 0.0]], ["A", "B"], day=9.0)
    day10 = _memb([[0.4, 0.6]], ["B", "C"], day=10.0)
    a9 = day9.align_populations(canonical).matrix
    a10 = day10.align_populations(canonical).matrix
    stacked = np.vstack([a9, a10])
    assert stacked.shape == (2, 3)
    assert np.allclose(stacked[0], [1.0, 0.0, 0.0])    # A,B day
    assert np.allclose(stacked[1], [0.0, 0.4, 0.6])    # B,C day
