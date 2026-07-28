"""Stress tests for the GMT cell-set reader and per-day membership builder.

read_cell_sets_gmt is the single source of truth for the canonical fate axis (file order).
membership_from_cell_sets projects one day's ExpressionMatrix cell axis onto that fate axis: the row
axis is the counts' cells (so it is aligned by construction), unlabelled cells are zero rows, and a
cell tagged with several fates has unit mass split equally -- the invariant forbids a multi-hot row,
so the split is mandatory, not cosmetic. Off-arm / other-day ids in the GMT are silently ignored.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wadd_data_ingest import (CellAxis, Membership, read_cell_sets_gmt,   # noqa: E402
                              membership_from_cell_sets)


def _write_gmt(tmp_path, lines):
    p = tmp_path / "cell_sets.gmt"
    p.write_text("\n".join(lines) + "\n")
    return p


def _axis(ids, day=9.0):
    return CellAxis(ids=list(ids), day=day)


# ---- the parser ----------------------------------------------------------------------------------

def test_parses_name_description_and_members_in_file_order(tmp_path):
    p = _write_gmt(tmp_path, ["A\t-\tc1\tc2\tc3",
                              "B\t-\tc4\tc5"])
    sets = read_cell_sets_gmt(p)
    assert list(sets) == ["A", "B"]                     # file order preserved -> canonical axis
    assert sets["A"] == ["c1", "c2", "c3"]
    assert sets["B"] == ["c4", "c5"]


def test_blank_lines_and_trailing_empty_fields_are_ignored(tmp_path):
    p = _write_gmt(tmp_path, ["A\t-\tc1\t\tc2\t", "", "   ", "B\t-\tc3"])
    sets = read_cell_sets_gmt(p)
    assert sets == {"A": ["c1", "c2"], "B": ["c3"]}     # empty member fields dropped, blank lines skipped


def test_a_population_with_no_members_is_kept_as_empty(tmp_path):
    p = _write_gmt(tmp_path, ["A\t-", "B\t-\tc1"])
    sets = read_cell_sets_gmt(p)
    assert sets["A"] == [] and sets["B"] == ["c1"]


def test_duplicate_population_line_raises(tmp_path):
    p = _write_gmt(tmp_path, ["A\t-\tc1", "B\t-\tc2", "A\t-\tc3"])
    with pytest.raises(ValueError, match="more than one line"):
        read_cell_sets_gmt(p)


def test_reads_the_real_cell_sets_gmt():
    """The shipped file: 13 fates in the documented order, non-empty, with prefixed cell ids."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "cell_sets.gmt")
    if not os.path.exists(path):
        pytest.skip("data/cell_sets.gmt not present")
    sets = read_cell_sets_gmt(path)
    assert len(sets) == 13
    assert list(sets)[0] == "IPS" and list(sets)[-1] == "RadialGlia"
    assert all(len(v) > 0 for v in sets.values())
    assert sets["IPS"][0].startswith("D")               # day-prefixed cell id


# ---- membership: row axis is the day's cells -----------------------------------------------------

def test_row_axis_is_the_expression_cells_in_order():
    cells = _axis(["cX", "cA", "cB"])                   # arbitrary order -> inherited verbatim
    sets = {"A": ["cA"], "B": ["cB"]}
    m = membership_from_cell_sets(sets, cells)
    assert m.cell_ids == ["cX", "cA", "cB"]
    assert m.population_names == ["A", "B"]
    assert np.allclose(m.matrix, [[0.0, 0.0],           # cX unlabelled
                                  [1.0, 0.0],           # cA
                                  [0.0, 1.0]])          # cB


def test_unlabelled_cell_is_a_zero_row():
    cells = _axis(["c1", "c2"])
    m = membership_from_cell_sets({"A": ["c1"], "B": []}, cells)
    assert np.allclose(m.matrix[1], [0.0, 0.0])         # c2 in no set
    assert m.matrix[1].sum() == 0.0


def test_overlapping_cell_splits_mass_equally():
    cells = _axis(["c1"])
    m = membership_from_cell_sets({"A": ["c1"], "B": ["c1"], "C": ["c1"]}, cells)
    assert np.allclose(m.matrix[0], [1 / 3, 1 / 3, 1 / 3])
    assert np.isclose(m.matrix[0].sum(), 1.0)           # valid distribution, respects rows<=1


def test_overlap_is_two_way():
    cells = _axis(["c1", "c2"])
    m = membership_from_cell_sets({"A": ["c1", "c2"], "B": ["c1"]}, cells)
    assert np.allclose(m.matrix, [[0.5, 0.5],           # c1 in A and B
                                  [1.0, 0.0]])          # c2 in A only


# ---- off-axis ids and column axis ----------------------------------------------------------------

def test_gmt_ids_absent_from_the_day_are_ignored():
    """A cell-set spans all days/arms; ids not in this day's axis contribute no rows, silently."""
    cells = _axis(["c1"])
    m = membership_from_cell_sets({"A": ["c1", "otherday_c9", "2i_cell"]}, cells)
    assert m.matrix.shape == (1, 1)
    assert np.allclose(m.matrix, [[1.0]])


def test_column_axis_is_the_gmt_order():
    """Columns come out in GMT file order; reordering to a target axis is align_populations' job."""
    cells = _axis(["c1", "c2"])
    sets = {"B": ["c1"], "A": ["c2"]}                   # file order B, A
    m = membership_from_cell_sets(sets, cells)
    assert m.population_names == ["B", "A"]
    assert np.allclose(m.matrix, [[1.0, 0.0], [0.0, 1.0]])


# ---- result is a valid Membership ----------------------------------------------------------------

def test_result_is_a_valid_membership_and_carries_the_day():
    cells = _axis(["c1", "c2"], day=12.5)
    m = membership_from_cell_sets({"A": ["c1"], "B": ["c2"]}, cells)
    assert isinstance(m, Membership) and m.day == 12.5
    assert np.all(m.matrix.sum(1) <= 1.0 + 1e-6)


def test_empty_day_all_zero_membership():
    """An early day where none of the (late-fate) canonical populations are present."""
    cells = _axis(["c1", "c2", "c3"])
    m = membership_from_cell_sets({"A": [], "B": []}, cells)
    assert np.allclose(m.matrix, 0.0)
    assert m.matrix.shape == (3, 2)
