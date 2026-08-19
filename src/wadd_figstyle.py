"""The paper's figure system -- one source of truth for matplotlib style and sizes.

Figures are designed AT FINAL SIZE: ``FULL_W`` is the double-column width (183 mm),
``COL_W`` the single column (89 mm). A FULL_W figure included at ``\\textwidth``
renders text 1:1, so the point sizes below are what the reader actually sees. The
historical failure mode -- 16-inch canvases shrunk 2.3x into the column -- printed
9 pt fonts at 4 pt.

Readers call ``apply()`` once before creating figures and DROP their explicit
per-call fontsize arguments rather than duplicating them, so the rc governs the
whole set and a style change stays a one-file edit.
"""
import matplotlib

FULL_W = 7.2   # inches, ~183 mm: full text width
COL_W = 3.5    # inches, ~89 mm: single column

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,           # editable text if we ever export vector
    "ps.fonttype": 42,
}


def apply():
    matplotlib.rcParams.update(RC)
