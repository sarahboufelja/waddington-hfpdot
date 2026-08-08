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
from wadd_dim_reduction import (MomentMatchedGaussian, MonteCarloMixture, ReverseMomentMatched,
                                RandomSubsampler)
from gmvae_confusion import latest_run


_INK, _MUTED, _GRID, _LINE = "#1e293b", "#64748b", "#e2e8f0", "#2b6cb0"
_ACCENT = "#b45309"


def _plot_eta(days, eta, cap, out):
    """The identity radius over the reprogramming timecourse, against its information ceiling.

    Two series only, so a legend plus direct labels: the radius itself (solid) and the ``log n`` cap
    (dashed, a bound not a measurement -- it moves with the day's cell count). Recessive grid/spines,
    thin 2px line, the peak direct-labelled, one y-axis. A light guide marks the Dox->serum regime
    change (~D8.25).
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

    ax.plot(days, cap, color=_MUTED, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.plot(days, eta, color=_LINE, lw=2.0, marker="o", ms=4.5, mfc=_LINE, mec="white",
            mew=0.6, zorder=3)
    ax.text(days[-1], cap[-1], "  ceiling  log n", color=_MUTED, fontsize=8.5, va="center")
    ax.text(days[-1], eta[-1], r"  $\eta = I(\mathrm{cell};z)$", color=_LINE, fontsize=8.5,
            va="center", fontweight="bold")

    ipk = int(np.argmax(eta))
    ax.annotate(f"peak  D{days[ipk]:g}", (days[ipk], eta[ipk]), textcoords="offset points",
                xytext=(6, 8), fontsize=8.5, color=_INK, fontweight="bold")

    ax.set_xlim(days[0] - 0.4, days[-1] + 2.6)                 # room for the direct labels
    ax.set_ylim(bottom=0)
    ax.set_xlabel("day  (reprogramming timecourse)", color=_INK, fontsize=10)
    ax.set_ylabel("identity radius  (nats)", color=_INK, fontsize=10)
    ax.set_title(r"Identity radius  $\eta(t) = I(\mathrm{cell};\,z)$  —  GSE122662 serum/Dox "
                 "reprogramming", color=_INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_gap(days, gap, out, split=11.0):
    """The multimodality gap KL(psi_bar || psi_0) over the timecourse.

    This is the part of the moment-matched radius that carries biology: the mutual information
    saturates at its log-n ceiling and so only tracks the day's cell count, leaving the departure of
    the latent mixture from a single Gaussian as the quantity that moves with the landscape.

    One series -> no legend. Regime means are drawn as short flat segments because the curve is a
    step-and-plateau, not a monotone rise: stating that visually is more honest than a trend line.
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
    ax.text(8.05, 0.05, "Dox → serum", color=_MUTED, fontsize=8, va="bottom", ha="right",
            transform=ax.get_xaxis_transform())

    early, late = days < split, days >= split
    for mask, label in ((early, "early"), (late, "late")):
        m = float(gap[mask].mean())
        lo, hi = days[mask].min(), days[mask].max()
        ax.plot([lo, hi], [m, m], color=_ACCENT, lw=1.6, ls=(0, (6, 3)), zorder=2)
        ax.text(lo + 0.15, m - 0.6, f"{label} mean {m:.1f}", color=_ACCENT, fontsize=8.5,
                ha="left", va="top", fontweight="bold")       # below the line, clear of the dashes

    ax.plot(days, gap, color=_LINE, lw=2.0, marker="o", ms=4.5, mfc=_LINE, mec="white",
            mew=0.6, zorder=3)

    ipk = int(np.argmax(gap))
    ax.annotate(f"D{days[ipk]:g}", (days[ipk], gap[ipk]), textcoords="offset points",
                xytext=(0, 9), fontsize=8.5, color=_INK, fontweight="bold", ha="center")

    ax.set_ylim(0, max(gap) * 1.18)
    ax.set_xlabel("day  (reprogramming timecourse)", color=_INK, fontsize=10)
    ax.set_ylabel(r"$\mathrm{KL}(\bar\psi\,\|\,\psi_0)$  (nats)", color=_INK, fontsize=10)
    ax.set_title("Multimodality of the latent population  —  GSE122662 serum/Dox reprogramming",
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

    mm, mc, rev = MomentMatchedGaussian(), MonteCarloMixture(n_samples=2048, seed=0), \
        ReverseMomentMatched()

    posts = {}                                              # day -> CellPosterior (kept for intensivity)
    print(f"\n{'day':>6} {'n':>7} {'MI (mc)':>9} {'bound(mm)':>10} {'gap':>8} {'log n':>7} "
          f"{'MI/log n':>9} {'reverse':>10}")
    rows = []
    for exp in matrices:
        post = embedder.embed(exp)
        e_mm, e_mc, e_rev = mm(post), mc(post), rev(post)
        cap = float(np.log(len(post)))
        posts[exp.day] = post
        rows.append((exp.day, len(post), e_mm, e_mc, e_mc - 0.0, e_mm - e_mc, cap, e_rev))
        print(f"{exp.day:>6} {len(post):>7} {e_mc:>9.3f} {e_mm:>10.3f} {e_mm - e_mc:>8.3f} "
              f"{cap:>7.3f} {e_mc / cap:>9.3f} {e_rev:>10.1f}")

    mi = np.array([r[3] for r in rows]); caps = np.array([r[6] for r in rows])
    gaps = np.array([r[5] for r in rows])
    print(f"\nMI (the radius):   median {np.median(mi):.3f} nats  [{mi.min():.3f}, {mi.max():.3f}]")
    print(f"cap log n:         median {np.median(caps):.3f}      -> MI/cap median "
          f"{np.median(mi / caps):.3f}  (max {np.max(mi / caps):.3f})")
    print(f"Gaussianity gap:   median {np.median(gaps):.3f}      [{gaps.min():.3f}, {gaps.max():.3f}]"
          f"   <- multimodality of the latent mixture")
    print(f"retired reverse:   median {np.median([r[7] for r in rows]):.1f} nats (the artifact)")

    # persist the series (for later overlay against the propagated fate-uncertainty curve) + figure
    order = np.argsort([r[0] for r in rows])
    day_arr = np.array([rows[i][0] for i in order]); n_arr = np.array([rows[i][1] for i in order])
    mm_arr = np.array([rows[i][2] for i in order]); mc_arr = np.array([rows[i][3] for i in order])
    cap_arr = np.array([rows[i][6] for i in order]); rev_arr = np.array([rows[i][7] for i in order])
    np.savez(run_dir / "eta_by_day.npz", day=day_arr, n=n_arr, eta=mc_arr, eta_bound=mm_arr,
             cap_log_n=cap_arr, gap=mm_arr - mc_arr, eta_reverse_retired=rev_arr)
    header = "day,n_cells,eta_mutual_information,eta_moment_matched_bound,cap_log_n,gaussianity_gap"
    np.savetxt(run_dir / "eta_by_day.csv",
               np.c_[day_arr, n_arr, mc_arr, mm_arr, cap_arr, mm_arr - mc_arr], delimiter=",",
               header=header, comments="", fmt=["%.2f", "%d", "%.4f", "%.4f", "%.4f", "%.4f"])
    _plot_eta(day_arr, mc_arr, cap_arr, run_dir / "eta_by_day.png")
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
