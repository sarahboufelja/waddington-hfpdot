"""Plan-uncertainty bands on the FLE over the fate territories -- the L2 overlay.

The title states the kernel provenance read from the record itself: records carrying a
positive ridge scale ``gamma`` were produced under the certified fixed kernel; records
without one predate the ratification and are labelled UNCERTIFIED.

Takes a particle-filter record (``run_particle_filter.py``) and paints, for each day of the window,
the filter's support cells on the published layout coloured by how undetermined their descendant
mass is across the marginal ensemble: the mass-weighted standard deviation of the cell's ensemble
weight relative to its ensemble-mean weight. A cell the couplings pin has sd/mean near 0; a cell
whose future the transport polytope leaves open sits near or above 1.

The base is the full landscape from ``fle_uncertainty.npz`` (the L1 companion overlay), tinted by
each cell's MAP fate at the COARSE level of the fate tree (six groups -- thirteen tints would drown
the sequential scale), with territory labels at the fate centroids. So the figure answers directly
*which populations the plan uncertainty concerns*; the per-fate summary printed alongside gives the
same answer numerically.

Everything inherits the run's sampler diagnostics (stamped per panel): this validates the join of
the identity and plan overlays; it does not publish plan uncertainty.

The filter's support cells are rebuilt deterministically (same run, seed and budget =>
``RandomSubsampler`` returns the same indices), which is what lets a record that stores only
weight vectors be placed back onto named, located, fate-labeled cells.

Run from the repo root:
    python scripts/fle_plan_bands.py assets/gmvae_runs/<run>/particle_filter_<window>.npz
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_artifacts import latest_artifact
from wadd_dim_reduction import RandomSubsampler
from wadd_data_ingest import read_fle_coords
from wadd_figstyle import FULL_W, apply as _style

_INK, _MUTED = "#1e293b", "#64748b"

# the 2-level fate tree (100% containment in cell_sets.gmt): coarse group per fine fate
_SUBTYPES = {"OPC": "Neural", "Astrocyte": "Neural", "Neuron": "Neural", "RadialGlia": "Neural",
             "SpongioTropho": "Trophoblast", "ProgenitorTropho": "Trophoblast",
             "SpiralArteryTrophoGiant": "Trophoblast"}
_COARSE_ORDER = ["Stromal", "MET", "Epithelial", "IPS", "Neural", "Trophoblast"]
# recessive territory tints: distinguishable at low alpha, none competing with the YlOrRd scale
_TINTS = {"Stromal": "#a8a29e", "MET": "#86b8a2", "Epithelial": "#c9a8c9",
          "IPS": "#7c96c9", "Neural": "#74add1", "Trophoblast": "#c9b874"}


def coarse_of(name):
    for k, v in _SUBTYPES.items():
        if k.lower() in name.lower():
            return v
    for g in _COARSE_ORDER:
        if g.lower() in name.lower():
            return g
    return name


def load_identity_base(run_dir):
    """Full-landscape base from the L1 overlay record: coords + coarse MAP fate per cell
    (argmax of q(c|z) at the mean z). Territory TINT only -- no statistic reads it; every
    uncertainty field on these overlays is a mass-weighted ensemble statistic."""
    d = np.load(latest_artifact(run_dir, "fle_uncertainty") / "fle_uncertainty.npz",
                allow_pickle=True)
    pops = [str(p) for p in d["populations"]]
    coarse = np.array([_COARSE_ORDER.index(coarse_of(p)) for p in pops])
    return d["x"], d["y"], coarse[d["map_fate"]]


def draw_identity_base(ax, x, y, coarse_idx, label_territories=True):
    """Tinted territories + centroid labels; recessive by construction (small, translucent)."""
    for gi, g in enumerate(_COARSE_ORDER):
        sel = coarse_idx == gi
        if not sel.any():
            continue
        ax.scatter(x[sel], y[sel], s=0.5, c=_TINTS[g], alpha=0.28, lw=0, rasterized=True,
                   zorder=1)
        if label_territories:
            ax.annotate(g, (np.median(x[sel]), np.median(y[sel])), color=_INK, fontsize=8,
                        fontweight="bold", ha="center", alpha=0.75, zorder=2,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.5))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def rebuild_support(run_dir, days, budget, seed, coords):
    """Deterministically re-derive each window day's support cells: coords + coarse MAP fate.

    Drawing data only (dot positions and territory colours). Statistics come from the
    record; in particular the tube is stored in-record, computed through q(c|z)."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrices, _, pops2 = R.assemble(days)
    assert pops2 == pops
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))
    sub = RandomSubsampler(seed=seed)
    coarse_pop = np.array([_COARSE_ORDER.index(coarse_of(p)) for p in pops])
    support = {}
    for exp in matrices:
        post = embedder.embed(exp)
        idx = sub.indices(post, min(budget, len(post)))
        ids = [post.cells.ids[i] for i in idx]
        xy = np.array([coords.get(cid, (np.nan, np.nan)) for cid in ids])
        fate = coarse_pop[np.argmax(np.asarray(post.prob_cat)[idx], axis=1)]
        support[exp.day] = {"xy": xy, "fate": fate}
    return support


def ensemble_cell_spread(weights, log_mass):
    """Relative descendant-mass spread per support cell: weighted sd / weighted mean.

    Cells whose mean weight is numerically zero return 0 -- no mass anywhere in the ensemble is
    certainty of absence at this resolution, not uncertainty.
    """
    m = np.exp(log_mass - log_mass.max())
    m = m / m.sum()
    mean = (m[:, None] * weights).sum(axis=0)
    var = (m[:, None] * (weights - mean) ** 2).sum(axis=0)
    sd = np.sqrt(var)
    out = np.zeros_like(mean)
    live = mean > 1e-12
    out[live] = sd[live] / mean[live]
    return out, mean


def main(record_path, run_dir, budget, seed, coords_path):
    rec = np.load(record_path, allow_pickle=True)
    days = rec["days"].tolist()
    run_dir = Path(run_dir)
    coords = read_fle_coords(coords_path)
    print(f"record: {Path(record_path).name} | window {days}")
    bx, by, bfate = load_identity_base(run_dir)
    support = rebuild_support(run_dir, days, budget, seed, coords)

    pairs = list(zip(days[:-1], days[1:]))
    _style()
    ncols = 2 if len(pairs) <= 6 else 4
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(FULL_W, 3.3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for spare in axes[len(pairs):]:
        spare.set_visible(False)
    last = None
    print(f"\nper-fate concern (mass share of the day's ensemble mean, and its spread):")
    for ax, (a, b) in zip(axes, pairs):
        tag = f"{a:g}->{b:g}"
        spread, mean = ensemble_cell_spread(rec[f"ensemble_{tag}"],
                                            rec[f"ensemble_mass_{tag}"])
        sup = support[b]
        xy, fate = sup["xy"], sup["fate"]
        ok = np.isfinite(xy[:, 0])
        draw_identity_base(ax, bx, by, bfate, label_territories=(ax is axes[0]))
        last = ax.scatter(xy[ok, 0], xy[ok, 1], s=22.0 + 170.0 * mean[ok] / max(mean.max(), 1e-12),
                          c=spread[ok], cmap="YlOrRd", vmin=0.0, vmax=2.0, lw=0.7,
                          edgecolors="white", zorder=3)
        diag = json.loads(str(rec[f"diag_{tag}"]))
        dmed = float(np.median([g["rhat_med"] for g in diag]))
        ax.set_title(f"D{b:g}   (draws from D{a:g})", color=_INK)
        ax.text(0.02, 0.02, f"R-hat_med {dmed:.2f}", transform=ax.transAxes, color=_MUTED)
        rows = []
        for gi, g in enumerate(_COARSE_ORDER):
            sel = fate == gi
            if sel.any() and mean[sel].sum() > 0:
                rows.append(f"{g}: {mean[sel].sum():.2f} mass, sd/mean "
                            f"{np.average(spread[sel], weights=np.maximum(mean[sel], 1e-12)):.2f}")
        print(f"  D{b:g}: " + " | ".join(rows))
    cb = fig.colorbar(last, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("arriving-mass spread  sd/mean  (ensemble)", color=_MUTED)
    cb.ax.tick_params(colors=_MUTED, labelsize=8)
    cb.outline.set_visible(False)
    gam = rec.get("gamma", None)
    prov = (f"certified fixed kernel (gamma = {float(gam):g}, gates stamped per panel)"
            if gam is not None and float(gam) > 0 else
            "UNCERTIFIED KERNEL (pre-ratification record; gates stamped per panel)")
    fig.suptitle("Plan-uncertainty bands over the fate territories  —  " + prov +
                 "; size = mean descendant mass", color=_INK,
                 fontsize=10, fontweight="bold", x=0.01, ha="left")
    out = Path(record_path).with_suffix("").as_posix() + "_fle.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", help="particle_filter_*.npz from run_particle_filter.py")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--budget", type=int, default=30, help="must match the record's run")
    p.add_argument("--seed", type=int, default=0, help="must match the record's run")
    p.add_argument("--coords", default=str(ROOT / "data" / "fle_coords.txt"))
    a = p.parse_args()
    main(a.record, a.run_dir or latest_run(), a.budget, a.seed, a.coords)
