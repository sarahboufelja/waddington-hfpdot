"""Uncertainty radii (eta, Eq 6) on the real trained embedding -- module 3b validation.

Per timepoint marginal: embed the day's cells and compute the diversity radius
    eta = (1/n) sum_k KL(psi_0 || psi_k),   psi_0 = (1/n) sum_j psi_j
with the moment-matched estimator (default, O(n)) and the Monte-Carlo reference (O(S*n)). The three
checks the audit only ever ran on synthetic data, now on the real posteriors:

  (1) it runs at real scale (39 days, up to ~10k cells/day, latent-10);
  (2) moment-matched ~ MC -- the single-Gaussian approximation of psi_0 holds on real posteriors;
  (3) eta is intensive -- eta(subsample) / eta(full) ~ 1, so a full-population radius is a coherent
      constraint on a subsampled marginal (audit reported 0.97 @ m=50, 1.00 @ m=500 on synthetic).

Also prints the eta(day) trajectory: the diversity radius should rise where fates coexist
(mid-reprogramming) and fall at homogeneous endpoints. eta is per-timepoint because it is the
marginal radius that lambda(eta) feeds to the sampler.

Run from the repo root (newest run, its own days):
    python scripts/gmvae_radii.py
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
from wadd_dim_reduction import MomentMatchedGaussian, MonteCarloMixture, RandomSubsampler
from gmvae_confusion import latest_run


_INK, _MUTED, _GRID, _LINE = "#1e293b", "#64748b", "#e2e8f0", "#2b6cb0"


def _plot_eta(days, eta, out):
    """Single-series line: the identity-diversity radius over the reprogramming timecourse.

    One series -> no legend (the title names it), one hue, recessive grid/spines, thin 2px line with
    small markers, the peak direct-labelled, a single y-axis. A light guide marks the Dox->serum
    regime change (~D8.25), which is where the curve's sustained rise begins.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)

    ax.axvline(8.25, color=_MUTED, lw=1.0, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.text(8.05, 0.97, "Dox → serum", color=_MUTED, fontsize=8, va="top", ha="right",
            transform=ax.get_xaxis_transform())            # x in data, y in axes fraction

    ax.plot(days, eta, color=_LINE, lw=2.0, marker="o", ms=4.5, mfc=_LINE, mec="white",
            mew=0.6, zorder=3)

    ipk = int(np.argmax(eta))
    ax.annotate(f"peak  D{days[ipk]:g}", (days[ipk], eta[ipk]), textcoords="offset points",
                xytext=(6, 8), fontsize=8.5, color=_INK, fontweight="bold")

    ax.set_xlabel("day  (reprogramming timecourse)", color=_INK, fontsize=10)
    ax.set_ylabel(r"diversity radius  $\eta$  (nats)", color=_INK, fontsize=10)
    ax.set_title(r"Identity–diversity radius  $\eta(t)$  —  GSE122662 serum/Dox reprogramming",
                 color=_INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(run_dir, days_spec=None, intensivity_days=(2.0, 9.0, 18.0)):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = R._parse_days(days_spec) if days_spec else cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run: {run_dir.name} | days: {len(days)} | device: {device}")

    matrices, _, pops2 = R.assemble(days)
    assert pops2 == pops
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))

    mm, mc = MomentMatchedGaussian(), MonteCarloMixture(n_samples=2048, seed=0)

    posts = {}                                              # day -> CellPosterior (kept for intensivity)
    print(f"\n{'day':>6} {'n':>7} {'eta_mm':>10} {'eta_mc':>10} {'mc/mm':>7}")
    rows = []
    for exp in matrices:
        post = embedder.embed(exp)
        e_mm, e_mc = mm(post), mc(post)
        posts[exp.day] = post
        rows.append((exp.day, len(post), e_mm, e_mc))
        print(f"{exp.day:>6} {len(post):>7} {e_mm:>10.3f} {e_mc:>10.3f} {e_mc / e_mm:>7.3f}")

    ratios = np.array([r[3] / r[2] for r in rows])
    print(f"\nmoment-matched vs MC: median mc/mm = {np.median(ratios):.3f}  "
          f"[{ratios.min():.3f}, {ratios.max():.3f}]")

    # persist the series (for later overlay against the propagated fate-uncertainty curve) + figure
    day_arr = np.array([r[0] for r in rows]); n_arr = np.array([r[1] for r in rows])
    mm_arr = np.array([r[2] for r in rows]); mc_arr = np.array([r[3] for r in rows])
    order = np.argsort(day_arr)
    day_arr, n_arr, mm_arr, mc_arr = day_arr[order], n_arr[order], mm_arr[order], mc_arr[order]
    np.savez(run_dir / "eta_by_day.npz", day=day_arr, n=n_arr, eta_mm=mm_arr, eta_mc=mc_arr)
    header = "day,n_cells,eta_moment_matched,eta_monte_carlo"
    np.savetxt(run_dir / "eta_by_day.csv", np.c_[day_arr, n_arr, mm_arr, mc_arr],
               delimiter=",", header=header, comments="", fmt=["%.2f", "%d", "%.4f", "%.4f"])
    _plot_eta(day_arr, mm_arr, run_dir / "eta_by_day.png")
    print(f"series -> {run_dir/'eta_by_day.npz'} (+ .csv)   figure -> {run_dir/'eta_by_day.png'}")

    # (3) intensivity: eta on random subsamples vs the full population
    print("\nintensivity  eta(subsample)/eta(full), moment-matched:")
    print(f"{'day':>6} {'full n':>7} {'eta_full':>10} " + " ".join(f"m={m:<6}" for m in (50, 500, 2000)))
    for d in intensivity_days:
        if d not in posts:
            continue
        post = posts[d]
        e_full = mm(post)
        cells_line = f"{d:>6} {len(post):>7} {e_full:>10.3f} "
        for m in (50, 500, 2000):
            if m >= len(post):
                cells_line += f"{'--':>8}"
                continue
            idx = RandomSubsampler(seed=0).indices(post, m)
            cells_line += f"{mm(post.select(idx)) / e_full:>8.3f}"
        print(cells_line)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None, help="run dir (default: newest)")
    p.add_argument("--days", default=None, help="override day set (default: the run's own days)")
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.days)
