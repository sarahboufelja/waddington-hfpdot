"""The FLE landscape with the identity-uncertainty overlay -- the identity half of the E2E result.

Places every embedded cell on the published Waddington-OT force-directed layout (so the axis is the
one readers already know; ``read_fle_coords`` documents the curation this adopts) and paints the two
per-cell fields whose day-averages are exactly the two arms of the adopted radius (section 4.5):

    ambiguity        a_i = H(q(c|z_i))            -- mean over a day  = H(C|Z), the residual arm
    distinctiveness  d_i = KL(q(c|z_i) || pi_bar) -- mean over a day  = I(C;Z), the resolved arm

so the landscape is the radius, disaggregated: eta(day) = mean_i[a_i] + mean_i[d_i] over that day's
cells. Nothing new is defined here -- single-source, the same ``prob_cat`` everywhere.

Rendering: a light-grey base of all placed cells, then a hex-binned MEAN of each field (bins under
``--mincnt`` cells stay grey rather than showing single-cell noise). Fields are heavy-tailed, so the
colour scale is clipped at the 99.5th percentile; the clip value is printed. One panel per field --
two measures never share one colour axis.

The per-cell table (coordinates, day, both fields, MAP fate and its confidence) is saved to
``fle_uncertainty.npz`` for downstream figures; the plan-uncertainty overlay will join on it once
the sampler lands.

Run from the repo root (newest run, its own days):
    python scripts/fle_uncertainty.py
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
from wadd_data_ingest import read_fle_coords

_INK, _MUTED, _BASE = "#1e293b", "#64748b", "#dbe1ea"
_EPS = 1e-12
_ANNOTATE_DAYS = (0.0, 6.5, 9.0, 12.0, 18.0)     # orientation marks along the temporal sweep


def _entropy(p, axis=-1):
    return -np.sum(np.where(p > _EPS, p * np.log(np.maximum(p, _EPS)), 0.0), axis=axis)


def per_cell_fields(prob_cat):
    """``(ambiguity_i, distinctiveness_i)`` for one day -- the summands of the section-4.5 chain rule."""
    pi_bar = prob_cat.mean(axis=0)
    amb = _entropy(prob_cat, axis=1)
    dist = np.sum(np.where(prob_cat > _EPS,
                           prob_cat * (np.log(np.maximum(prob_cat, _EPS)) -
                                       np.log(np.maximum(pi_bar, _EPS))), 0.0), axis=1)
    return amb, np.maximum(dist, 0.0)


def _panel(ax, x, y, field, cmap, label, mincnt, gridsize):
    """Grey base + hexbin mean of the field. FLE units are meaningless, so the frame is silent."""
    ax.scatter(x, y, s=0.6, c=_BASE, lw=0, rasterized=True, zorder=1)
    vmax = float(np.quantile(field, 0.995))
    hb = ax.hexbin(x, y, C=field, gridsize=gridsize, reduce_C_function=np.mean,
                   mincnt=mincnt, cmap=cmap, vmin=0.0, vmax=vmax, lw=0.0, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(label, color=_INK, fontsize=11, fontweight="bold", loc="left", pad=8)
    return hb, vmax


def main(run_dir, coords_path, mincnt, gridsize):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    coords = read_fle_coords(coords_path)
    print(f"run: {run_dir.name} | days: {len(days)} | layout: {len(coords)} published coords | "
          f"device: {device}")

    matrices, _, pops2 = R.assemble(days)
    assert pops2 == pops
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))

    xs, ys, day_col, ambs, dists, fate, conf = [], [], [], [], [], [], []
    n_total = n_placed = 0
    for exp in matrices:
        post = embedder.embed(exp)
        p = np.asarray(post.prob_cat, dtype=np.float64)
        amb, dist = per_cell_fields(p)
        placed = np.array([coords.get(cid, (np.nan, np.nan)) for cid in post.cells.ids])
        ok = np.isfinite(placed[:, 0])
        n_total += len(p); n_placed += int(ok.sum())
        xs.append(placed[ok, 0]); ys.append(placed[ok, 1])
        day_col.append(np.full(int(ok.sum()), exp.day))
        ambs.append(amb[ok]); dists.append(dist[ok])
        fate.append(np.argmax(p[ok], axis=1)); conf.append(np.max(p[ok], axis=1))
    x = np.concatenate(xs); y = np.concatenate(ys); day_arr = np.concatenate(day_col)
    amb = np.concatenate(ambs); dist = np.concatenate(dists)
    fate = np.concatenate(fate); conf = np.concatenate(conf)
    print(f"placed {n_placed}/{n_total} cells ({100 * n_placed / n_total:.1f}%) on the published "
          f"layout; {n_total - n_placed} filtered by its curation")

    # consistency: day-means of the fields must reassemble the section-4.5 decomposition
    rc = run_dir / "radius_candidates.npz"
    if rc.exists():
        ref = np.load(rc)
        r_amb = np.array([amb[day_arr == d].mean() for d in ref["day"]])
        worst = float(np.nanmax(np.abs(r_amb - ref["ambiguity"])))
        print(f"day-mean ambiguity vs H(C|Z) series: max |delta| = {worst:.4f} nats "
              f"(placement curation only)")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    specs = [(amb, "YlOrBr", "identity ambiguity   $H(q(c|z_i))$   —  the $H(C|Z)$ arm"),
             (dist, "Blues", r"fate distinctiveness   $\mathrm{KL}(q(c|z_i)\,\|\,\bar\pi_{day})$"
                             "   —  the $I(C;Z)$ arm")]
    for ax, (field, cmap, label) in zip(axes, specs):
        hb, vmax = _panel(ax, x, y, field, cmap, label, mincnt, gridsize)
        cb = fig.colorbar(hb, ax=ax, shrink=0.82, pad=0.015)
        cb.set_label("nats  (hex mean)", color=_MUTED, fontsize=9)
        cb.ax.tick_params(colors=_MUTED, labelsize=8)
        cb.outline.set_visible(False)
        print(f"colour scale '{label.split()[1]}': 0 .. {vmax:.3f} nats (99.5th pct clip)")
        for d in _ANNOTATE_DAYS:
            sel = day_arr == d
            if sel.any():
                ax.annotate(f"D{d:g}", (np.median(x[sel]), np.median(y[sel])), color=_INK,
                            fontsize=8.5, fontweight="bold", ha="center",
                            path_effects=None, zorder=4,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55))
    fig.suptitle("Identity uncertainty on the reprogramming landscape  —  the two arms of  "
                 r"$\eta = H(\bar\pi)$,  per cell", color=_INK, fontsize=12.5,
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = run_dir / "fle_uncertainty.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)

    np.savez(run_dir / "fle_uncertainty.npz", x=x, y=y, day=day_arr, ambiguity=amb,
             distinctiveness=dist, map_fate=fate, map_confidence=conf,
             populations=np.array(pops))
    print(f"table -> {run_dir/'fle_uncertainty.npz'}   figure -> {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--coords", default=str(ROOT / "data" / "fle_coords.txt"))
    p.add_argument("--mincnt", type=int, default=4)
    p.add_argument("--gridsize", type=int, default=220)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.coords, a.mincnt, a.gridsize)
