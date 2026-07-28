"""wadd_data_ingest -- raw single-cell data ingest for HierWOT.

Reads raw expression matrices and builds cell populations. Deliberately knows nothing about the
embedding model: it emits **raw counts**, and the latent representation is produced downstream. This
keeps data loading usable (and testable) without the modelling stack installed.

Layout:
- ``CellAxis``          -- the ordered cells labelling an array axis, plus their timepoint. Arrays
  here are cell-indexed and a permutation is invisible in the shapes, so the labels travel with the
  data and alignment is compared explicitly rather than assumed.
- ``ExpressionMatrix``  -- raw counts (sparse; ~90-95% of single-cell entries are zero) plus
  metadata kept *beside* the matrix, never as a gene column. ``densify(idx)`` is the one explicit
  place counts become dense, so models consume batches rather than whole timepoints.
- ``ExpressionReader``  -- the I/O seam. ``H5Reader`` reads the real 10x-style HDF5 files (imports
  ``h5py`` lazily); ``InMemoryReader`` serves arrays, for tests and synthetic runs.
- ``Membership``        -- cell -> population assignment, (n_cells, n_pops), carrying the labels of
  both axes so alignment is checked rather than assumed. Soft membership is the general case; hard
  labels are the one-hot special case.

Reducing a timepoint to a cell budget is **not** done here -- see the note below.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
import re, math
from typing import Any, Iterator, List, Protocol, Sequence

import numpy as np
from scipy import sparse
import logging

from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

Array = np.ndarray

# A model forward pass densifies a BATCH of cells (512 x 28k genes x 4B ~ 57 MB), whereas a whole
# timepoint is ~7k x 28k x 4B ~ 784 MB. 256 MiB sits between the two: generous batches pass quietly,
# an unintended full-timepoint materialisation gets flagged.
_DENSIFY_WARN_BYTES = 256 * 2**20

# The day encoded in a filename (``..._D9_serum_...``) or a cell id (``D9_serum_C1_AAAC-1``):
# an integer or decimal after an uppercase D, bounded by start-of-string/underscore and underscore.
# Compiled once -- the cell-subset file has one line per sequenced cell (~250k for this dataset).
_DAY_IN_TEXT = re.compile(r"(?:^|_)D(\d+(?:\.\d+)?)_")

# --------------------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CellAxis:
    """The ordered set of cells labelling one array axis, at one timepoint.

    Arrays throughout the pipeline are indexed by cells -- an expression matrix's rows, a
    membership's rows, a transport plan's rows *and* columns. Order is meaningful and a permutation
    is invisible in the shapes, so the labels travel with the data and comparisons are explicit.

    ``day`` lives here because it describes the cells, and so is stored exactly once per axis rather
    than duplicated onto every object that happens to carry them.
    """
    ids: List[str]
    day: float

    def __post_init__(self):
        if len(set(self.ids)) != len(self.ids):
            seen, dupes = set(), []
            for cid in self.ids:
                if cid in seen:
                    dupes.append(cid)
                seen.add(cid)
            raise ValueError(f"duplicate cell ids would make alignment ambiguous: {dupes[:5]}")

    def __len__(self) -> int:
        return len(self.ids)

    def select(self, idx: Sequence[int] | Array) -> CellAxis:
        idx = np.asarray(idx, dtype=int)
        return CellAxis(ids=[self.ids[i] for i in idx], day=self.day)

    def positions(self) -> dict[str, int]:
        return {cid: i for i, cid in enumerate(self.ids)}

    def require_same(self, other: CellAxis, what: str) -> None:
        """Raise unless these are the same cells, in the same order, at the same timepoint."""
        if self.day != other.day:
            raise ValueError(f"{what}: cells are from different timepoints "
                             f"(day {self.day} vs {other.day})")
        if len(self.ids) != len(other.ids):
            raise ValueError(f"{what}: expected {len(other.ids)} cells, got {len(self.ids)}")
        if list(self.ids) != list(other.ids):
            raise ValueError(f"{what}: same cell count and timepoint, but different cells or order")


@dataclass(frozen=True)
class ExpressionMatrix:
    """Raw gene counts for one timepoint, with metadata held separately from the feature matrix.

    Counts are stored **sparse** (CSR): single-cell expression is ~90-95% zeros, so a timepoint of
    ~7k cells x ~28k genes is ~80 MB sparse against ~780 MB dense, and several files are
    concatenated per day. A dense array passed to the constructor is converted.

    Densifying is therefore an explicit, deliberate act -- ``densify(idx)`` -- because that is where
    memory is actually spent. Models that need dense input (the embedding network) should densify a
    **batch at a time**: peak memory is then ``batch x genes``, not ``n_cells x genes``.
    """
    counts: Any                   # (n_cells, n_genes), scipy.sparse CSR
    cells: CellAxis               # labels the rows
    gene_names: List[str]         # length n_genes

    def __post_init__(self):
        counts = self.counts
        if not sparse.issparse(counts):
            counts = sparse.csr_matrix(np.asarray(counts))
        elif not isinstance(counts, sparse.csr_matrix):
            counts = counts.tocsr()
        object.__setattr__(self, "counts", counts)

        n, g = counts.shape
        if len(self.cells) != n:
            raise ValueError(f"{len(self.cells)} cells for {n} rows")
        if len(self.gene_names) != g:
            raise ValueError(f"{len(self.gene_names)} gene_names for {g} columns")

    def __len__(self) -> int:
        return self.counts.shape[0]

    @property
    def cell_ids(self) -> List[str]:
        return self.cells.ids

    @property
    def day(self) -> float:
        return self.cells.day

    @property
    def genes(self) -> List[str]:
        return self.gene_names

    @property
    def n_genes(self) -> int:
        return self.counts.shape[1]

    @property
    def density(self) -> float:
        """Fraction of non-zero entries -- the reason this is stored sparse."""
        n, g = self.counts.shape
        return self.counts.nnz / (n * g) if n and g else 0.0

    def select(self, idx: Sequence[int] | Array) -> ExpressionMatrix:
        """Row-subset, staying sparse (returns a new matrix; never mutates)."""
        idx = np.asarray(idx, dtype=int)
        return ExpressionMatrix(counts=self.counts[idx], cells=self.cells.select(idx),
                                gene_names=list(self.gene_names))

    def densify(self, idx: Sequence[int] | Array | None = None,
                dtype: np.dtype | str | None = None) -> Array:
        """Dense array of the given rows (all rows if ``idx`` is None).

        The one place sparse counts become dense. Prefer passing a batch of row indices; densifying
        a whole timepoint is legitimate but costs ``n_cells x n_genes`` and is warned about when
        large, since an unintended full materialisation is easy to write and expensive to discover.
        """
        rows = self.counts if idx is None else self.counts[np.asarray(idx, dtype=int)]
        n_bytes = rows.shape[0] * rows.shape[1] * np.dtype(dtype or self.counts.dtype).itemsize
        if n_bytes > _DENSIFY_WARN_BYTES:
            warnings.warn(
                f"densifying {rows.shape[0]}x{rows.shape[1]} counts "
                f"(~{n_bytes / 1e9:.1f} GB); densify a batch of rows instead if this is a model "
                f"forward pass", stacklevel=2)
        dense = np.asarray(rows.todense())
        return dense.astype(dtype) if dtype is not None else dense


@dataclass(frozen=True)
class Population:
    """A distribution over cells (numpy, framework-agnostic).

    ``weights`` is supported on ``cell_ids`` and normalised to sum to 1. Downstream layers convert to
    their own array type at the compute boundary.
    """
    name: str
    weights: Array                # (n_cells,)
    cells: CellAxis

    def __post_init__(self):
        if self.weights.shape != (len(self.cells),):
            raise ValueError(f"weights {self.weights.shape} vs {len(self.cells)} cells")

    @property
    def cell_ids(self) -> List[str]:
        return self.cells.ids

    @property
    def day(self) -> float:
        return self.cells.day


# --------------------------------------------------------------------------------------------------
# I/O seam
# --------------------------------------------------------------------------------------------------

class ExpressionReader(Protocol):
    def read(self, day: float) -> ExpressionMatrix: ...


@dataclass
class InMemoryReader:
    """Serves pre-built matrices. Used for tests and synthetic experiments."""
    matrices: dict[float, ExpressionMatrix]

    def read(self, day: float) -> ExpressionMatrix:
        if day not in self.matrices:
            raise KeyError(f"no expression matrix for day {day}; have {sorted(self.matrices)}")
        return self.matrices[day]


@dataclass
class H5Reader:
    """Reads 10x-style HDF5 count matrices for a given day.

    A file is in scope when its name has the canonical ``D<day>_<arm>_C<lane>`` shape *and* its arm
    is among ``experiment_tags``. That combination already selects a trajectory: in this archive the
    Dox arm covers days 0-8 (induction) and the serum/2i arms split from day 8.25, so the default
    tags trace the serum path and exclude the 2i branch. Anything non-canonical -- perturbation
    series, established ``DiPSC`` lines -- is skipped.

    ``cell_subset_path`` is an OPTIONAL further restriction to an explicit list of cell ids, carried
    over from the original pipeline (which pointed it at a ``serum_cell_ids.txt``). It is off by
    default and the trajectory does not depend on it. Whether such a list is *needed* is unresolved:
    if it only recorded arm membership it is now redundant with the structural filter above, but if
    it also encoded quality-control curation then omitting it changes which cells we keep relative to
    the published analysis -- and Schiebinger et al. are not explicit on the point. Left as a seam so
    a curated list can be plugged in without reworking the reader.

    ``h5py`` is imported lazily so this module stays importable without it.
    """
    matrix_dir: Path | str
    cell_subset_path: Path | str | None = None
    genome: str = "mm10"
    experiment_tags: tuple[str, ...] = ("serum", "Dox_C1", "Dox_C2")
    # Structure of the cell-id prefix: D<day>_<arm>_C<lane>, captured from the file name. Kept
    # configurable alongside `genome`/`experiment_tags` so a differently-named dataset does not
    # require editing this class.
    cell_prefix_pattern: str = r"(D\d+(?:\.\d+)?_[A-Za-z0-9]+_C\d+)"

    @staticmethod
    def _extract_day(text: str | Path) -> float | None:
        """The day encoded in a filename or a cell id, as a float, or None if there isn't one.

        Matches an integer or decimal following an uppercase ``D`` preceded by the start of the
        string or an underscore, and followed by an underscore -- covering both filenames
        (``..._D9_serum_...``) and cell ids (``D9_serum_C1_AAAC-1``).

        Parsing to a number rather than formatting the day into a search string means ``D9``,
        ``D9.0`` and ``D09`` all denote day 9, and a caller passing ``9`` or ``9.0`` gets the same
        answer. For a ``Path`` only the **file name** is examined, so a directory called something
        like ``run_D3_batch`` cannot masquerade as the cell day.
        """
        name = text.name if isinstance(text, Path) else str(text)
        m = _DAY_IN_TEXT.search(name)
        return float(m.group(1)) if m else None

    def _in_scope(self) -> Iterator[tuple[Path, float]]:
        """In-scope matrix files with their parsed day, in a stable order.

        The single definition of "in scope", shared by day lookup and day enumeration: the canonical
        ``D<day>_<arm>_C<lane>`` shape, plus an arm among ``experiment_tags``. Everything else in the
        archive -- perturbation series (``..._serum_GDF9_exp_R1_C1_...``), established lines
        (``DiPSC_...``) -- is skipped here rather than admitted and then failing later when its
        cell-id prefix cannot be determined.

        Tag and day are read from the file NAME, never the full path, so a directory that happens to
        contain "serum" or "_D3_" cannot influence the selection.

        A missing or non-directory ``matrix_dir`` is a *configuration* fault and is reported as such:
        ``glob`` yields nothing for an empty directory, a missing one, and a plain file alike, so
        without this check all three would surface later as the misleading "no matrices for day N".
        """
        root = Path(self.matrix_dir)
        if not root.is_dir():
            raise NotADirectoryError(
                f"matrix_dir is not an existing directory: {root!s} "
                "(check the configured path rather than the requested day)")

        for path in sorted(root.glob("*.h5")):
            if self._match_prefix(path) is None:
                continue
            if not any(tag in path.name for tag in self.experiment_tags):
                continue
            day = self._extract_day(path)
            if day is not None:
                yield path, day

    def available_days(self) -> list[float]:
        """Timepoints present in the archive, ascending.

        The archive is self-describing, so the valid timepoints are simply whichever in-scope files
        exist. This replaces the original pipeline's dependency on the external ``cell_days.txt``
        listing.
        """
        return sorted({day for _, day in self._in_scope()})

    def _paths_for_day(self, day: float) -> list[Path]:
        """The in-scope files for one day."""
        return [path for path, parsed in self._in_scope() if math.isclose(parsed, day)]

    def _cell_subset(self, day: float) -> set[str] | None:
        """Cell ids listed for this day, or None when no subset file is configured.

        Lines without a parseable day (headers, blank lines) are skipped rather than crashing.
        """
        if self.cell_subset_path is None:
            return None
        keep = set()
        with open(self.cell_subset_path) as fh:
            for line in fh:
                cell_id = line.strip()
                if not cell_id:
                    continue
                parsed = self._extract_day(cell_id)
                if parsed is not None and math.isclose(parsed, day):
                    keep.add(cell_id)
        return keep

    def _read_one(self, path: Path) -> tuple[Any, list[str], list[str]]:
        """Read one file as (sparse counts (cells, genes), cell ids, gene names).

        The 10x HDF5 layout stores a CSC matrix genes-by-cells under ``/<genome>/``; we transpose to
        cells-by-genes and keep it sparse -- densifying here would cost ~780 MB per file for a real
        timepoint. ``h5py`` is imported lazily so the module stays importable without it (and it
        replaces PyTables, whose wheels lag new Python/numpy and pull a heavy transitive tree).
        """
        import h5py                          # lazy: only needed for real files

        with h5py.File(path, "r") as fh:
            group = fh[self.genome]
            barcodes = group["barcodes"][:]
            matrix = sparse.csc_matrix((group["data"][:], group["indices"][:],
                                        group["indptr"][:]), shape=group["shape"][:])
            genes = group["gene_names"][:]
        counts = matrix.T.tocsr()                        # (cells, genes), still sparse
        cell_ids = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in barcodes]
        gene_names = [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in genes]
        return counts, cell_ids, gene_names

    def _match_prefix(self, path: Path) -> re.Match | None:
        """The canonical-prefix match for this file, or None if it is not a timecourse sample."""
        return re.search(self.cell_prefix_pattern, path.name)

    def _cell_prefix(self, path: Path) -> str:
        """The ``D<day>_<arm>_C<lane>`` prefix that qualifies this file's barcodes.

        10x barcodes are unique only *within* a lane, so the same barcode recurs across files; this
        prefix is what keeps cell ids distinct, and it must reproduce the convention used by the
        cell-subset file (``D9_serum_C1_GCTTGAAAGCTGTCTA-1``).

        Matched structurally rather than by counting underscores: a positional slice is correct only
        for filenames with exactly the expected number of leading/trailing tokens, and silently
        returns a plausible-but-wrong prefix for anything else.
        """
        m = self._match_prefix(path)
        if m is None:
            raise ValueError(
                f"cannot determine the cell-id prefix for {path.name!r}: it does not match "
                f"{self.cell_prefix_pattern!r}. Set `cell_prefix_pattern` for this dataset's "
                "naming convention.")
        return m.group(1)

    def read(self, day: float) -> ExpressionMatrix:
        paths = self._paths_for_day(day)
        if not paths:
            raise FileNotFoundError(f"no HDF5 matrices for day {day} under {self.matrix_dir} for experiments: {self.experiment_tags}")
        keep = self._cell_subset(day)

        blocks, ids, genes = [], [], None
        n_seen, examples = 0, []
        for path in paths:
            counts, cell_ids, gene_names = self._read_one(path)
            # Cells are uniquely identified via their arm/lane + barcode appended.
            cell_ids = [f"{self._cell_prefix(path)}_{cid}" for cid in cell_ids]
            n_seen += len(cell_ids)
            if cell_ids:
                examples.append(cell_ids[0])
            if keep is not None:
                # A file can legitimately contribute nothing -- e.g. a Dox lane read alongside a
                # serum-only subset list. Only a day yielding no cells at all is suspicious.
                sel = [i for i, cid in enumerate(cell_ids) if cid in keep]
                counts, cell_ids = counts[sel], [cell_ids[i] for i in sel]
            if genes is None:
                genes = gene_names
            elif gene_names != genes:
                raise ValueError(f"gene names differ between files for day {day}")
            blocks.append(counts)
            ids.extend(cell_ids)

        if keep is not None and not ids:
            expected = next(iter(sorted(keep)), "<subset file has no ids for this day>")
            warnings.warn(
                f"day {day}: none of the {n_seen} barcodes across {len(paths)} file(s) appear in "
                f"the cell-subset list. This is expected if the day has no cells in the listed "
                f"arm, but a prefix/format mismatch looks the same -- compare\n"
                f"  built:    {examples[0] if examples else '<no barcodes read>'}\n"
                f"  expected: {expected}", stacklevel=2)

        return ExpressionMatrix(counts=sparse.vstack(blocks, format="csr"),
                                cells=CellAxis(ids=ids, day=day),
                                gene_names=list(genes or []))


# NOTE: reducing a timepoint to a workable cell budget deliberately does NOT live here. The budget
# must be applied in the *latent* space (that is where the transport cost is computed, where the
# Voronoi quantizer is defined, and where coverage is meaningful -- raw counts are ~28k-dim
# and sparse, so distances concentrate). Ingest reads every cell; the embedding layer computes the
# uncertainty radii on the full population and only then subsamples.


# --------------------------------------------------------------------------------------------------
# Cell -> population membership
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Membership:
    """Cell -> population assignment, carrying the labels of both of its axes.

    ``matrix[i, j]`` is how much cell ``cell_ids[i]`` belongs to population ``population_names[j]``.
    Rows sum to at most 1 (a zero row means unassigned). Soft membership is the general case; hard
    labels are the one-hot special case.

    The axis labels are part of the object precisely so alignment can be **checked** rather than
    assumed: a matrix built against a different cell ordering is a silent, invisible error, so
    ``for_cells`` refuses it instead of producing confidently wrong populations.
    """
    matrix: Array                 # (n_cells, n_pops)
    cells: CellAxis               # labels the rows
    population_names: List[str]   # labels the columns

    def __post_init__(self):
        M = np.asarray(self.matrix, dtype=np.float64)
        object.__setattr__(self, "matrix", M)
        if M.ndim != 2:
            raise ValueError(f"membership matrix must be 2-D, got {M.shape}")
        if M.shape != (len(self.cells), len(self.population_names)):
            raise ValueError(
                f"membership matrix {M.shape} does not match "
                f"({len(self.cells)} cells, {len(self.population_names)} populations)")
        if np.any(M < 0):
            raise ValueError("membership has negative entries")
        if np.any(M.sum(axis=1) > 1.0 + 1e-6):
            raise ValueError("membership rows must sum to at most 1")

    @property
    def cell_ids(self) -> List[str]:
        return self.cells.ids

    @property
    def day(self) -> float:
        return self.cells.day

    def for_cells(self, cells: CellAxis, what: str = "membership") -> Array:
        """Return the matrix, having verified its rows are the given cells in the given order."""
        self.cells.require_same(cells, what)
        return self.matrix

    def reindex_to(self, cells: CellAxis) -> Membership:
        """Reorder (and/or subset) ROWS to the given cell axis; raises if a cell is missing.

        Only the cell axis. The column axis is aligned by ``align_populations``.
        """
        if self.cells.day != cells.day:
            raise ValueError(f"cannot reindex membership from day {self.cells.day} "
                             f"onto cells from day {cells.day}")
        position = self.cells.positions()
        try:
            order = [position[cid] for cid in cells.ids]
        except KeyError as exc:
            raise ValueError(f"cell {exc.args[0]!r} has no membership row") from exc
        return Membership(matrix=self.matrix[order], cells=cells,
                          population_names=list(self.population_names))

    def align_populations(self, canonical_names: List[str]) -> Membership:
        """Reorder COLUMNS to align exactly with the order in ``canonical_names``.

        The column counterpart of ``reindex_to``: it makes the population axis canonical so
        memberships from different days (which may list fates in any order, or omit fates absent that
        day) share one comparable column layout.

        Design contract:

            - A population present in the input lands at its canonical index with its column intact.
            - A canonical population absent from the input gets an all-zero column at its index.
            - A population in the input but absent from ``canonical_names`` is a hard error: aligning
              it away would silently drop labelled mass. Row sums are otherwise preserved exactly.
            - Cells and ``CellAxis`` (ids and day) pass through untouched.
        """
        canonical = list(canonical_names)

        dupes = [name for name, n in Counter(canonical).items() if n > 1]
        if dupes:
            raise ValueError(f"canonical_names has duplicate populations: {dupes[:5]}")

        known = set(canonical)
        unknown = [p for p in self.population_names if p not in known]
        if unknown:
            raise ValueError(
                f"membership has populations absent from canonical_names: {unknown[:5]}; "
                "aligning them away would drop labelled mass")

        if canonical == list(self.population_names):
            return self

        source = {name: j for j, name in enumerate(self.population_names)}
        ordered = np.zeros((self.matrix.shape[0], len(canonical)), dtype=self.matrix.dtype)
        for out_j, name in enumerate(canonical):
            src_j = source.get(name)
            if src_j is not None:
                ordered[:, out_j] = self.matrix[:, src_j]

        return Membership(matrix=ordered, cells=self.cells, population_names=canonical)


def read_cell_sets_gmt(path: Path | str) -> dict[str, List[str]]:
    """Parse a GMT cell-set file into ``{population_name: [cell_id, ...]}``, preserving file order.

    GMT is one population per line, tab-separated: ``name <tab> description <tab> id <tab> id ...``.
    The description column (often ``-``) is ignored; the remaining fields are member cell ids. Blank
    lines and lines without at least a name are skipped; a repeated population name is an error, since
    it would make the fate axis ambiguous.

    This file is the single source of truth for the canonical fate axis: the returned insertion order
    is the canonical population order passed to ``membership_from_cell_sets`` and, downstream, to
    ``Membership.align_populations``.
    """
    sets: dict[str, List[str]] = {}
    with open(path) as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            name = fields[0].strip()
            if not name:
                continue                                   # blank line
            if name in sets:
                raise ValueError(f"population {name!r} appears on more than one line in {path!s}")
            sets[name] = [cid for cid in fields[2:] if cid]
    return sets


def membership_from_cell_sets(cell_sets: dict[str, Sequence[str]], cells: CellAxis) -> Membership:
    """Build one timepoint's ``Membership`` from a parsed GMT and that day's cell axis.

    Columns are every fate in the GMT, in file order -- the canonical fate axis. Reordering to some
    other target order is not this function's concern: call ``align_populations`` on the result.

    The row axis is ``cells`` -- the ExpressionMatrix's ``CellAxis`` for this day -- so the result is
    aligned to the counts by construction (same cells, same order): no reindex is needed and
    ``build_targets``/``seed_prior`` accept it directly. Every one of those cells gets a row:

    - a cell in no set gets an all-zero (unassigned) row -- the common case, especially early, where
      it contributes only to the unsupervised ELBO and nothing to the supervised anchor;
    - a cell listed under several fates (the ~5% overlap here) has unit mass split **equally** across
      those fates, so the row sums to 1 and the ambiguity is a soft label rather than dropped or
      double-counted. The equal split is not optional: the ``Membership`` invariant forbids rows that
      sum above 1, so a multi-hot row could not be constructed in the first place.

    A cell-set spans all days and both arms, so most of its ids belong to other timepoints (or the 2i
    arm) and are simply absent from ``cells`` -- silently, since off-arm absence is expected every day
    and arm selection is the reader's concern, not this function's.
    """
    names = list(cell_sets)
    col_of = {name: j for j, name in enumerate(names)}
    cols_for_cell: dict[str, List[int]] = {}
    for name, members in cell_sets.items():
        j = col_of[name]
        for cid in members:
            cols_for_cell.setdefault(cid, []).append(j)

    M = np.zeros((len(cells), len(names)), dtype=np.float64)
    for i, cid in enumerate(cells.ids):
        cols = cols_for_cell.get(cid)
        if cols:
            M[i, cols] = 1.0 / len(cols)                   # equal split across the cell's fates
    return Membership(matrix=M, cells=cells, population_names=names)


def membership_from_labels(labels: Sequence[str], population_names: Sequence[str],
                           cells: CellAxis) -> Membership:
    """One-hot ``Membership`` from hard labels -- the crisp special case.

    Cells whose label is not in ``population_names`` get an all-zero row (unassigned).
    """
    if len(labels) != len(cells):
        raise ValueError(f"{len(labels)} labels for {len(cells)} cells")
    index = {name: j for j, name in enumerate(population_names)}
    M = np.zeros((len(labels), len(population_names)), dtype=np.float64)
    for i, lab in enumerate(labels):
        j = index.get(lab)
        if j is not None:
            M[i, j] = 1.0
    return Membership(matrix=M, cells=cells, population_names=list(population_names))


def populations_from_membership(membership: Membership) -> List[Population]:
    """Turn a membership into per-population distributions over cells.

    Population ``p``'s distribution is its **column of the membership matrix, normalised**: a cell
    that is 55% population A and 45% population B contributes to both, weighted. With one-hot
    membership this reduces to the uniform distribution over that population's cells. Populations
    with no mass at this timepoint are omitted.

    Needs nothing but the membership: it already carries the cell axis (ids and timepoint), so
    passing an expression matrix alongside would only re-supply information that is here already --
    and create the misalignment risk that the extra argument would then have to be checked for.
    """
    M = membership.matrix
    populations = []
    for j, name in enumerate(membership.population_names):
        column = M[:, j]
        total = column.sum()
        if total <= 0:
            continue                                  # population absent at this timepoint
        populations.append(Population(name=name, weights=column / total, cells=membership.cells))
    return populations
