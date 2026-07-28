"""Unit tests for wadd_data_ingest. Pure numpy -- no HDF5 files or modelling stack required."""
import os
import sys
import warnings

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wadd_data_ingest  # noqa: E402  (module handle, for monkeypatching the densify threshold)
from wadd_data_ingest import (  # noqa: E402
    CellAxis, ExpressionMatrix, H5Reader, InMemoryReader, Membership, Population,
    membership_from_labels, populations_from_membership,
)

POPS = ["MEF", "Epithelial", "IPS"]


def _axis(n_cells=10, day=2.0):
    return CellAxis(ids=[f"cell_{i}" for i in range(n_cells)], day=day)


def _expr(n_cells=10, n_genes=6, day=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return ExpressionMatrix(
        counts=rng.integers(0, 50, size=(n_cells, n_genes)).astype(float),
        cells=_axis(n_cells, day),
        gene_names=[f"gene_{j}" for j in range(n_genes)],
    )


# ---- CellAxis ----------------------------------------------------------------------------------

def test_cell_axis_rejects_duplicate_ids():
    """Duplicates make alignment ambiguous: a lookup would silently keep only one row."""
    with pytest.raises(ValueError, match="duplicate cell ids"):
        CellAxis(ids=["a", "b", "a"], day=1.0)


def test_cell_axis_length_and_selection():
    axis = _axis(5)
    assert len(axis) == 5
    sub = axis.select([3, 1])
    assert sub.ids == ["cell_3", "cell_1"] and sub.day == axis.day


def test_require_same_accepts_identical_axes():
    _axis(4).require_same(_axis(4), "ctx")          # must not raise


def test_require_same_rejects_different_timepoints():
    with pytest.raises(ValueError, match="different timepoints"):
        _axis(4, day=2.0).require_same(_axis(4, day=4.0), "ctx")


def test_require_same_rejects_different_counts():
    with pytest.raises(ValueError, match="expected 3 cells, got 4"):
        _axis(4).require_same(_axis(3), "ctx")


def test_require_same_rejects_reordering_with_matching_count():
    """The dangerous case: same cells, same day, permuted -- invisible in the shapes."""
    a = _axis(3)
    permuted = CellAxis(ids=[a.ids[i] for i in (2, 0, 1)], day=a.day)
    with pytest.raises(ValueError, match="different cells or order"):
        permuted.require_same(a, "ctx")


def test_require_same_names_the_context_in_the_error():
    with pytest.raises(ValueError, match="push-forward"):
        _axis(4).require_same(_axis(3), "push-forward")


# ---- ExpressionMatrix --------------------------------------------------------------------------

def test_expression_matrix_validates_labels():
    with pytest.raises(ValueError):                                  # cells vs rows
        ExpressionMatrix(np.zeros((3, 2)), _axis(2), ["g1", "g2"])
    with pytest.raises(ValueError):                                  # gene_names vs columns
        ExpressionMatrix(np.zeros((3, 2)), _axis(3), ["g1"])


def test_day_is_metadata_not_a_gene_column():
    """The timepoint lives on the cell axis, never inside the feature matrix."""
    expr = _expr(n_genes=6)
    assert expr.counts.shape[1] == len(expr.gene_names) == expr.n_genes == 6
    assert "day" not in expr.gene_names
    assert expr.day == 2.0 and expr.cells.day == 2.0


def test_expression_matrix_exposes_axis_labels():
    expr = _expr(n_cells=4)
    assert expr.cell_ids == expr.cells.ids
    assert len(expr) == 4


# ---- sparse storage / explicit densification ---------------------------------------------------

def test_counts_are_stored_sparse_even_when_built_from_dense():
    dense = np.array([[0.0, 3.0], [0.0, 0.0]])
    expr = ExpressionMatrix(dense, _axis(2), ["g0", "g1"])
    assert sparse.issparse(expr.counts) and expr.counts.format == "csr"
    assert np.allclose(expr.densify(), dense)                        # values survive the round trip


def test_sparse_input_is_preserved_without_densifying():
    m = sparse.random(20, 50, density=0.05, format="coo", random_state=0)
    expr = ExpressionMatrix(m, _axis(20), [f"g{j}" for j in range(50)])
    assert expr.counts.format == "csr"
    assert expr.counts.nnz == m.nnz                                  # no fill-in
    assert 0.0 < expr.density < 0.1


def test_densify_selected_rows_only():
    """The model boundary: densify a batch, not the whole timepoint."""
    expr = _expr(n_cells=8, n_genes=6)
    batch = expr.densify([1, 3])
    assert batch.shape == (2, 6)
    assert np.allclose(batch, expr.densify()[[1, 3]])


def test_densify_honours_requested_dtype():
    expr = _expr(n_cells=4, n_genes=3)
    assert expr.densify(dtype=np.float32).dtype == np.float32


def test_densify_warns_only_when_materialising_something_large(monkeypatch):
    expr = _expr(n_cells=8, n_genes=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                               # small densify: must be silent
        expr.densify()
    monkeypatch.setattr(wadd_data_ingest, "_DENSIFY_WARN_BYTES", 1)  # pretend it is huge
    with pytest.warns(UserWarning, match="densifying"):
        expr.densify()


def test_select_subsets_rows_and_axis_together_and_stays_sparse():
    """Rows and their labels must move as one; drifting apart is the bug this prevents."""
    expr = _expr(n_cells=8)
    before = expr.densify()
    sub = expr.select([1, 3, 5])
    assert sparse.issparse(sub.counts)                               # subsetting must not densify
    assert sub.counts.shape == (3, expr.n_genes)
    assert sub.cell_ids == ["cell_1", "cell_3", "cell_5"]
    assert np.allclose(sub.densify(), before[[1, 3, 5]])
    assert np.allclose(expr.densify(), before)                       # original untouched
    assert sub.day == expr.day


# ---- reader seam -------------------------------------------------------------------------------

def test_in_memory_reader_round_trip_and_missing_day():
    expr = _expr(day=4.0)
    reader = InMemoryReader({4.0: expr})
    assert reader.read(4.0) is expr
    with pytest.raises(KeyError):
        reader.read(9.0)


# ---- day parsing / file discovery ---------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("GSM1_D9_serum_C1_gene_bc_mat.h5", 9.0),
    ("GSM2_D9.0_serum_C1_x.h5", 9.0),                       # trailing .0 spelling
    ("GSM3_D09_serum_C1_x.h5", 9.0),                        # zero padded
    ("GSM4_D1_serum_C1_x.h5", 1.0),
    ("GSM5_D1.5_serum_C1_x.h5", 1.5),                       # half days exist in this dataset
    ("D9_serum_C1_GCTTGAAAGCTGTCTA-1", 9.0),                # subset-file line: day at the start
    ("D18_serum_C2_ACGC-1", 18.0),
    ("GSM6_metadata_serum.h5", None),                       # no day at all
    ("", None),
])
def test_extract_day_parses_every_spelling(text, expected):
    assert H5Reader._extract_day(text) == expected


def _touch(directory, names):
    for n in names:
        (directory / n).write_text("")


def test_paths_for_day_matches_all_spellings_of_the_same_day(tmp_path):
    _touch(tmp_path, ["GSM1_D9_serum_C1_gene_bc_mat.h5",
                      "GSM2_D9.0_serum_C2_gene_bc_mat.h5",
                      "GSM3_D09_serum_C3_gene_bc_mat.h5",
                      "GSM4_D1_serum_C1_gene_bc_mat.h5"])
    reader = H5Reader(matrix_dir=tmp_path)
    for requested in (9, 9.0):                              # int or float must behave identically
        found = {p.name.split("_")[1] for p in reader._paths_for_day(requested)}
        assert found == {"D9", "D9.0", "D09"}


def test_paths_for_day_does_not_confuse_adjacent_days(tmp_path):
    _touch(tmp_path, ["GSM1_D1_serum_C1_x.h5", "GSM2_D1.5_serum_C1_x.h5",
                      "GSM3_D11_serum_C1_x.h5"])
    reader = H5Reader(matrix_dir=tmp_path)
    assert [p.name for p in reader._paths_for_day(1)] == ["GSM1_D1_serum_C1_x.h5"]
    assert [p.name for p in reader._paths_for_day(1.5)] == ["GSM2_D1.5_serum_C1_x.h5"]


def test_paths_for_day_applies_the_experiment_tag_filter(tmp_path):
    _touch(tmp_path, ["GSM1_D9_serum_C1_x.h5", "GSM2_D9_2i_C1_x.h5"])
    reader = H5Reader(matrix_dir=tmp_path)                  # tags default to serum / Dox_C1 / Dox_C2
    assert [p.name for p in reader._paths_for_day(9)] == ["GSM1_D9_serum_C1_x.h5"]


def test_a_tagged_file_without_a_day_does_not_crash_discovery(tmp_path):
    """An unparseable name must be skipped (and ideally reported), never raise."""
    _touch(tmp_path, ["GSM1_D9_serum_C1_x.h5", "serum_metadata.h5"])
    reader = H5Reader(matrix_dir=tmp_path)
    assert [p.name for p in reader._paths_for_day(9)] == ["GSM1_D9_serum_C1_x.h5"]


def test_day_is_read_from_the_filename_not_the_directory(tmp_path):
    """A directory named like a day must not hijack the match."""
    nested = tmp_path / "run_D3_batch"
    nested.mkdir()
    _touch(nested, ["GSM1_D9_serum_C1_x.h5"])
    reader = H5Reader(matrix_dir=nested)
    assert [p.name for p in reader._paths_for_day(9)] == ["GSM1_D9_serum_C1_x.h5"]
    assert reader._paths_for_day(3) == []                   # the directory's D3 is not a cell day


def test_empty_directory_yields_no_files_rather_than_raising(tmp_path):
    assert H5Reader(matrix_dir=tmp_path)._paths_for_day(9) == []


def test_missing_matrix_dir_is_reported_as_a_configuration_fault(tmp_path):
    """glob() is silent for a missing dir, an empty dir and a plain file alike -- so a wrong path
    would otherwise surface as the misleading 'no matrices for day N'."""
    with pytest.raises(NotADirectoryError, match="not an existing directory"):
        H5Reader(matrix_dir=tmp_path / "does_not_exist")._paths_for_day(9)


def test_matrix_dir_pointing_at_a_file_is_also_a_configuration_fault(tmp_path):
    a_file = tmp_path / "matrices.h5"
    a_file.write_text("")
    with pytest.raises(NotADirectoryError):
        H5Reader(matrix_dir=a_file)._paths_for_day(9)


def test_cell_subset_selects_the_requested_day(tmp_path):
    subset = tmp_path / "serum_cells.txt"
    subset.write_text("D9_serum_C1_AAAC-1\nD9.5_serum_C1_TTTG-1\nD18_serum_C2_CCCG-1\n")
    reader = H5Reader(matrix_dir=tmp_path, cell_subset_path=subset)
    assert reader._cell_subset(9) == {"D9_serum_C1_AAAC-1"}
    assert reader._cell_subset(9.5) == {"D9.5_serum_C1_TTTG-1"}


def test_cell_subset_tolerates_header_and_blank_lines(tmp_path):
    """Real text files carry headers and trailing newlines; neither is a cell."""
    subset = tmp_path / "serum_cells.txt"
    subset.write_text("cell_id\nD9_serum_C1_AAAC-1\n\nD18_serum_C2_CCCG-1\n")
    reader = H5Reader(matrix_dir=tmp_path, cell_subset_path=subset)
    assert reader._cell_subset(9) == {"D9_serum_C1_AAAC-1"}


def test_cell_subset_is_none_when_not_configured(tmp_path):
    assert H5Reader(matrix_dir=tmp_path)._cell_subset(9) is None


# ---- membership --------------------------------------------------------------------------------

def test_membership_from_labels_is_one_hot():
    m = membership_from_labels(["MEF", "IPS", "MEF"], POPS, _axis(3))
    assert m.matrix.shape == (3, 3)
    assert np.allclose(m.matrix.sum(axis=1), 1.0)
    assert m.matrix[0, POPS.index("MEF")] == 1.0 and m.matrix[1, POPS.index("IPS")] == 1.0


def test_unknown_label_leaves_cell_unassigned():
    m = membership_from_labels(["MEF", "Martian"], POPS, _axis(2))
    assert np.allclose(m.matrix[1], 0.0)                             # zero row, not an error
    assert np.isclose(m.matrix.sum(), 1.0)


def test_membership_rejects_bad_matrices():
    with pytest.raises(ValueError):                                  # shape vs labels
        Membership(np.zeros((2, 2)), _axis(3), POPS)
    with pytest.raises(ValueError):                                  # negative
        Membership(np.array([[-0.5, 0.5]]), _axis(1), ["p", "q"])
    with pytest.raises(ValueError):                                  # rows sum > 1
        Membership(np.array([[0.7, 0.7]]), _axis(1), ["p", "q"])
    with pytest.raises(ValueError):                                  # labels vs cells
        membership_from_labels(["MEF", "IPS"], POPS, _axis(1))


def test_membership_carries_its_cell_axis():
    m = membership_from_labels(["MEF"] * 3, POPS, _axis(3, day=6.0))
    assert m.cell_ids == _axis(3).ids and m.day == 6.0


# ---- the alignment contract is enforced, not assumed -------------------------------------------

def test_for_cells_rejects_a_foreign_axis():
    """A membership computed on *other* cells must be refused, not quietly reused.

    Real axes carry actual barcodes (``H5Reader`` builds e.g. "D9_serum_C1_GCTTGAAAGCTGTCTA-1");
    the ids below are deliberately from a different set of cells to stand in for that mistake.
    """
    other_cells = CellAxis(["D9_serum_C2_AAACCTGAGCGT-1",
                            "D9_serum_C2_AAACCTGCATGC-1",
                            "D9_serum_C2_AAACCTGTCCAG-1"], day=2.0)
    m = membership_from_labels(["MEF", "IPS", "MEF"], POPS, other_cells)
    with pytest.raises(ValueError, match="different cells or order"):
        m.for_cells(_axis(3), "populations")


def test_for_cells_rejects_a_different_timepoint():
    m = membership_from_labels(["MEF"] * 3, POPS, _axis(3, day=2.0))
    with pytest.raises(ValueError, match="different timepoints"):
        m.for_cells(_axis(3, day=4.0), "populations")


def test_for_cells_returns_the_matrix_when_aligned():
    axis = _axis(3)
    m = membership_from_labels(["MEF", "IPS", "MEF"], POPS, axis)
    assert m.for_cells(axis, "populations") is m.matrix


def test_reindex_to_repairs_a_permuted_membership():
    axis = _axis(3)
    shuffled = CellAxis([axis.ids[i] for i in (2, 0, 1)], day=axis.day)
    labels = ["IPS", "MEF", "MEF"]                                   # labels follow `shuffled`
    m = membership_from_labels(labels, POPS, shuffled).reindex_to(axis)
    pops = {p.name: p for p in populations_from_membership(m)}
    # cell_2 was first in the shuffled order and is the IPS cell
    assert np.allclose(pops["IPS"].weights, [0.0, 0.0, 1.0])
    assert np.allclose(pops["MEF"].weights, [0.5, 0.5, 0.0])


def test_reindex_to_can_subset():
    """Subsampling downstream must be able to narrow a membership to the surviving cells."""
    axis = _axis(5)
    m = membership_from_labels(["MEF", "IPS", "MEF", "IPS", "MEF"], POPS, axis)
    kept = axis.select([0, 2])
    narrowed = m.reindex_to(kept)
    assert narrowed.cell_ids == ["cell_0", "cell_2"]
    assert narrowed.matrix.shape == (2, 3)


def test_reindex_to_reports_a_missing_cell():
    m = membership_from_labels(["MEF", "IPS"], POPS, CellAxis(["cell_0", "cell_1"], day=2.0))
    with pytest.raises(ValueError, match="no membership row"):
        m.reindex_to(_axis(3))


def test_reindex_to_refuses_to_cross_timepoints():
    m = membership_from_labels(["MEF"] * 3, POPS, _axis(3, day=2.0))
    with pytest.raises(ValueError, match="day"):
        m.reindex_to(_axis(3, day=4.0))


# ---- populations -----------------------------------------------------------------------------

def test_hard_membership_gives_uniform_population_distributions():
    m = membership_from_labels(["MEF", "MEF", "IPS", "MEF"], POPS, _axis(4))
    pops = {p.name: p for p in populations_from_membership(m)}
    assert set(pops) == {"MEF", "IPS"}                               # Epithelial absent -> omitted
    assert np.allclose(pops["MEF"].weights, [1 / 3, 1 / 3, 0.0, 1 / 3])
    assert np.allclose(pops["IPS"].weights, [0.0, 0.0, 1.0, 0.0])


def test_soft_membership_spreads_a_cell_across_populations():
    """A 55/45 cell must contribute to BOTH populations -- the point of soft membership."""
    m = Membership(np.array([[0.55, 0.45, 0.0],
                             [1.00, 0.00, 0.0]]), _axis(2), POPS)
    pops = {p.name: p for p in populations_from_membership(m)}
    assert np.allclose(pops["MEF"].weights, [0.55 / 1.55, 1.0 / 1.55])
    assert np.allclose(pops["Epithelial"].weights, [1.0, 0.0])       # only cell 0 has Epi mass


def test_populations_inherit_the_membership_cell_axis():
    """No second argument is needed -- and populations cannot drift from their cells."""
    axis = _axis(5, day=8.0)
    rng = np.random.default_rng(1)
    W = rng.uniform(size=(5, 3))
    W /= W.sum(axis=1, keepdims=True)
    for pop in populations_from_membership(Membership(W, axis, POPS)):
        assert isinstance(pop.weights, np.ndarray)
        assert np.isclose(pop.weights.sum(), 1.0)
        assert pop.cells is axis and pop.cell_ids == axis.ids and pop.day == 8.0


def test_population_validates_weight_length():
    with pytest.raises(ValueError):
        Population(name="X", weights=np.ones(3), cells=_axis(2))


# ---- cell-id prefix ------------------------------------------------------------------------------

@pytest.mark.parametrize("stem", [
    "GSM3195648_D9_serum_C1_gene_bc_mat",                   # canonical shape
    "GSM3195648_D9_serum_C1_filtered_matrix",               # two trailing tokens
    "GSM3195648_D9_serum_C1_filtered_gene_bc_matrices_h5",  # four trailing tokens
    "D9_serum_C1_gene_bc_mat",                              # no accession prefix
])
def test_cell_prefix_is_independent_of_surrounding_tokens(tmp_path, stem):
    """Positional slicing was correct only for one filename shape; the structure is what matters."""
    reader = H5Reader(matrix_dir=tmp_path)
    assert reader._cell_prefix(tmp_path / f"{stem}.h5") == "D9_serum_C1"


def test_cell_prefix_handles_half_days_and_other_arms(tmp_path):
    reader = H5Reader(matrix_dir=tmp_path)
    assert reader._cell_prefix(tmp_path / "GSM_D9.5_serum_C2_x.h5") == "D9.5_serum_C2"
    assert reader._cell_prefix(tmp_path / "GSM_D18_Dox_C2_x.h5") == "D18_Dox_C2"


def test_unrecognised_filename_raises_instead_of_guessing_a_prefix(tmp_path):
    """A wrong prefix builds cell ids that match nothing -- fail loudly rather than silently."""
    reader = H5Reader(matrix_dir=tmp_path)
    with pytest.raises(ValueError, match="cannot determine the cell-id prefix"):
        reader._cell_prefix(tmp_path / "random_file_without_structure.h5")


def test_cell_prefix_pattern_is_configurable(tmp_path):
    reader = H5Reader(matrix_dir=tmp_path, cell_prefix_pattern=r"(lane\d+)")
    assert reader._cell_prefix(tmp_path / "experiment_lane7_counts.h5") == "lane7"


def test_available_days_enumerates_the_archive(tmp_path):
    """Replaces the legacy dependency on an external cell_days.txt listing."""
    _touch(tmp_path, ["GSM1_D0_Dox_C1_x.h5", "GSM2_D0_Dox_C2_x.h5",     # same day, two lanes
                      "GSM3_D8.25_serum_C1_x.h5", "GSM4_D18_serum_C2_x.h5"])
    assert H5Reader(matrix_dir=tmp_path).available_days() == [0.0, 8.25, 18.0]


def test_available_days_excludes_out_of_scope_files(tmp_path):
    _touch(tmp_path, ["GSM1_D9_serum_C1_x.h5",
                      "GSM2_D15_serum_GDF9_exp_R1_C1_x.h5",             # perturbation series
                      "GSM3_DiPSC_serum_C1_x.h5",                       # established line
                      "GSM4_D9_2i_C1_x.h5"])                            # arm not in experiment_tags
    assert H5Reader(matrix_dir=tmp_path).available_days() == [9.0]


def test_available_days_is_empty_for_an_empty_archive(tmp_path):
    assert H5Reader(matrix_dir=tmp_path).available_days() == []


# ---- reading real 10x HDF5 files -----------------------------------------------------------------

import h5py                                   # a declared dev dependency: required, not optional


def _write_10x(path, gene_ids, barcodes, counts_genes_by_cells, genome="mm10", symbols=None):
    """Write a minimal file in the 10x layout: a CSC matrix stored GENES-by-CELLS under /<genome>/.

    Mirrors the real archive (byte-string barcodes, int32 shape, and BOTH gene datasets -- ``genes``
    holding the unique Ensembl ids the reader keys the axis on, and ``gene_names`` the symbols), so
    tests exercise the same decoding and orientation handling as the actual data. ``symbols`` defaults
    to ``gene_ids`` when a test does not care to distinguish the two.
    """
    m = sparse.csc_matrix(np.asarray(counts_genes_by_cells, dtype=np.int32))
    symbols = gene_ids if symbols is None else symbols
    with h5py.File(path, "w") as fh:
        g = fh.create_group(genome)
        g["data"], g["indices"], g["indptr"] = m.data, m.indices, m.indptr
        g["shape"] = np.array(m.shape, dtype=np.int32)
        g["barcodes"] = np.array([b.encode() for b in barcodes])
        g["genes"] = np.array([n.encode() for n in gene_ids])          # Ensembl ids: the axis key
        g["gene_names"] = np.array([n.encode() for n in symbols])      # symbols (unused by the reader)
    return path


def test_read_transposes_to_cells_by_genes_and_decodes_labels(tmp_path):
    """10x stores genes-by-cells; the reader must emit cells-by-genes, with str (not bytes) labels.

    The 3x2 shape is deliberately non-square: a transpose bug would still produce the right shape
    for a square matrix, so orientation is checked against hand-computed values.
    """
    genes, barcodes = ["Xkr4", "Sox2", "Actb"], ["AAAC-1", "AAAG-1"]
    counts_gxc = [[0, 5],       # Xkr4: cell0=0, cell1=5
                  [3, 0],       # Sox2: cell0=3, cell1=0
                  [1, 2]]       # Actb: cell0=1, cell1=2
    _write_10x(tmp_path / "GSM1_D9_serum_C1_gene_bc_mat.h5", genes, barcodes, counts_gxc)

    expr = H5Reader(matrix_dir=tmp_path).read(9)

    assert expr.gene_names == genes and expr.n_genes == 3          # decoded to str
    assert expr.cell_ids == ["D9_serum_C1_AAAC-1", "D9_serum_C1_AAAG-1"]
    assert expr.densify().tolist() == [[0, 3, 1],                  # cell0 across genes
                                       [5, 0, 2]]                  # cell1 across genes
    assert sparse.issparse(expr.counts)                            # never densified on the way in
    assert expr.day == 9.0


def test_read_concatenates_lanes_and_keeps_shared_barcodes_distinct(tmp_path):
    """Lanes of one day are concatenated, and the prefix is what keeps colliding barcodes apart.

    10x draws barcodes from a fixed whitelist, so the same barcode recurs across lanes (in this
    archive D9 C1/C2 share four). Without the lane prefix those would be duplicate cell ids -- which
    CellAxis rejects outright, since a duplicate silently corrupts every later alignment.
    """
    genes = ["Xkr4", "Sox2"]
    _write_10x(tmp_path / "GSM1_D9_serum_C1_gene_bc_mat.h5", genes,
               ["SHARED-1", "ONLY_C1-1"], [[1, 2],          # Xkr4 across the two cells
                                           [3, 4]])         # Sox2 across the two cells
    _write_10x(tmp_path / "GSM2_D9_serum_C2_gene_bc_mat.h5", genes,
               ["SHARED-1"], [[5],                          # same barcode as lane C1
                              [6]])

    expr = H5Reader(matrix_dir=tmp_path).read(9)

    assert len(expr) == 3                                   # 2 + 1: nothing merged or dropped
    assert expr.cell_ids == ["D9_serum_C1_SHARED-1",        # file order, then within-file order
                             "D9_serum_C1_ONLY_C1-1",
                             "D9_serum_C2_SHARED-1"]
    assert len(set(expr.cell_ids)) == 3                     # the collision is resolved by prefixing
    assert expr.densify().tolist() == [[1, 3],              # rows follow the id order above
                                       [2, 4],
                                       [5, 6]]
    assert expr.gene_names == genes                         # genes agree, so the axis is shared


def test_read_applies_the_cell_subset_filter(tmp_path):
    """Only listed cells survive, and the counts follow the surviving rows."""
    genes = ["Xkr4", "Sox2"]
    _write_10x(tmp_path / "GSM1_D9_serum_C1_gene_bc_mat.h5", genes,
               ["KEEP-1", "DROP-1", "ALSO_KEEP-1"], [[1, 2, 3],       # Xkr4 across three cells
                                                     [4, 5, 6]])      # Sox2 across three cells
    subset = tmp_path / "serum_cells.txt"
    subset.write_text("D9_serum_C1_KEEP-1\nD9_serum_C1_ALSO_KEEP-1\nD18_serum_C1_OTHER_DAY-1\n")

    expr = H5Reader(matrix_dir=tmp_path, cell_subset_path=subset).read(9)

    assert expr.cell_ids == ["D9_serum_C1_KEEP-1", "D9_serum_C1_ALSO_KEEP-1"]
    assert expr.densify().tolist() == [[1, 4], [3, 6]]     # DROP-1's column (2, 5) is gone
    assert len(expr) == 2


def test_read_without_a_subset_keeps_every_cell(tmp_path):
    genes = ["Xkr4", "Sox2"]
    _write_10x(tmp_path / "GSM1_D9_serum_C1_gene_bc_mat.h5", genes,
               ["A-1", "B-1"], [[1, 2], [3, 4]])
    assert len(H5Reader(matrix_dir=tmp_path).read(9)) == 2


def test_read_warns_when_the_subset_selects_nothing_for_a_day(tmp_path):
    """Zero selected cells is legitimate (a day may have no cells in the listed arm) but looks
    identical to a prefix/format mismatch -- so it must be reported, with both forms shown."""
    genes = ["Xkr4"]
    _write_10x(tmp_path / "GSM1_D9_serum_C1_gene_bc_mat.h5", genes, ["AAAC-1"], [[7]])
    subset = tmp_path / "serum_cells.txt"
    subset.write_text("D9_WRONGFORMAT_AAAC-1\n")           # right day, incompatible id shape

    with pytest.warns(UserWarning, match="none of the 1 barcodes"):
        expr = H5Reader(matrix_dir=tmp_path, cell_subset_path=subset).read(9)

    assert len(expr) == 0 
