"""Animated uncertainty propagation on the FLE -- self-contained video of the particle filter.

One animated figure from a particle-filter record, written for a reader meeting the method cold:
an intro card states the idea in plain language (one transport map is a single hypothesis; HFPD-OT
samples many; their disagreement IS the uncertainty of the lineage map), a persistent "how to read"
rail explains the visual grammar, and each day carries an auto-generated caption naming where the
candidate futures agree and where they disagree, derived from the same per-fate statistics the
static overlay prints.

Visual grammar: every candidate future (a marginal particle of the filter) is a set of rings over
the day's support cells, ring area proportional to the descendant mass that future places on the
cell -- agreement collapses a cell's stack to one ring, disagreement fans it out. Ring colour is
inherited through parent pointers, so the trim visibly absorbs hypothesis lineages. Days
CROSS-FADE rather than move: consecutive supports are different cells, and animating trajectories
between them would fabricate a correspondence the model does not assert. The rail tracks the
propagated radius (KL-ball on the fate simplex -- the same currency as the identity radius) as a
growing sparkline on the day timeline.

Placeholder status is inherited from the record (sampler R-hat_med in the footer). Output is a GIF
(no ffmpeg on this host); hold frames carry a slight alpha pulse so the GIF encoder cannot
collapse them (identical frames get merged without extending dwell time).

Run from the repo root:
    python scripts/fle_plan_video.py assets/gmvae_runs/<run>/particle_filter_<window>.npz
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from gmvae_confusion import latest_run
from wadd_data_ingest import read_fle_coords
from fle_plan_bands import (_COARSE_ORDER, _INK, _MUTED, draw_identity_base,
                            ensemble_cell_spread, load_identity_base, rebuild_support)

# fixed lineage palette (dataviz categorical order); the root is neutral ink
_LINEAGE = ["#2b6cb0", "#b45309", "#0f766e", "#7c3aed", "#be185d", "#4d7c0f"]

_INTRO = ("Where do reprogramming cells go?\n\n"
          "A single transport map is one hypothesis\n"
          "about how cells flow between timepoints.\n\n"
          "HFPD-OT samples MANY maps consistent\n"
          "with the data. Watching them disagree\n"
          "shows the uncertainty of the lineage map.")


def _caption(state):
    """One plain-language sentence per day: where the candidate futures agree and disagree."""
    fate, W, mass = state["fate"], state["W"], state["mass"]
    m = mass / mass.sum()
    spread, mean = ensemble_cell_spread(W, np.log(np.maximum(mass, 1e-300)))
    rows = []
    for gi, g in enumerate(_COARSE_ORDER):
        sel = fate == gi
        if sel.any() and mean[sel].sum() > 1e-6:
            rows.append((g, float(mean[sel].sum()),
                         float(np.average(spread[sel], weights=np.maximum(mean[sel], 1e-12)))))
    if not rows:
        return ""
    top = max(rows, key=lambda r: r[1])
    worst = max(rows, key=lambda r: r[2])
    return (f"Most mass → {top[0]} ({top[1]:.0%})\n"
            f"Biggest disagreement: {worst[0]} (x{worst[2]:.1f})")


def _states(rec, run_dir, budget, seed, coords):
    days = rec["days"].tolist()
    support = rebuild_support(run_dir, days, budget, seed, coords)
    # The tube (eta_prop, TV diameter) is READ from the record: the filter computed it
    # through q(c|z), where no cell's mass can be dropped. Recomputing it here through
    # the drawing membership duplicated the derivation and lost unannotated cells.
    eta = np.asarray(rec["tube_eta_prop"])
    tv = np.asarray(rec["tube_tv_diam"])
    m0 = len(support[days[0]]["xy"])
    states = [{"day": days[0], "W": np.full((1, m0), 1.0 / m0), "mass": np.array([1.0]),
               "colour": [_INK], "xy": support[days[0]]["xy"],
               "fate": support[days[0]]["fate"], "eta": float(eta[0]), "tv": float(tv[0]),
               "caption": "Start: one map, no disagreement yet."}]
    colours = None
    for i, (a, b) in enumerate(zip(days[:-1], days[1:])):
        tag = f"{a:g}->{b:g}"
        W, lm = rec[f"ensemble_{tag}"], rec[f"ensemble_mass_{tag}"]
        parents = rec[f"parents_{tag}"]
        if colours is None:
            colours = [_LINEAGE[j % len(_LINEAGE)] for j in range(len(W))]
        else:
            colours = [prev_colours[p] if p >= 0 else _INK for p in parents]
        prev_colours = colours
        state = {"day": b, "W": W, "mass": np.exp(lm - np.max(lm)), "colour": list(colours),
                 "xy": support[b]["xy"], "fate": support[b]["fate"],
                 "eta": float(eta[i + 1]), "tv": float(tv[i + 1])}
        state["caption"] = _caption(state)
        states.append(state)
    return states


def _draw_rail(rail, states):
    """The persistent right rail: how-to-read glyph, day timeline, radius sparkline."""
    rail.set_xlim(0, 1); rail.set_ylim(0, 1)
    rail.axis("off")
    rail.text(0.02, 0.985, "HOW TO READ", color=_INK, fontsize=9, fontweight="bold", va="top")
    # example ring stack: agreement vs disagreement
    for r, c in ((0.030, _LINEAGE[0]), (0.032, _LINEAGE[1])):
        rail.add_patch(plt.Circle((0.16, 0.87), r, fill=False, ec=c, lw=1.6, alpha=0.8))
    rail.text(0.30, 0.87, "rings agree:\nfuture is determined", color=_MUTED, fontsize=7.8,
              va="center")
    for r, c in ((0.018, _LINEAGE[0]), (0.036, _LINEAGE[1]), (0.052, _LINEAGE[2])):
        rail.add_patch(plt.Circle((0.16, 0.74), r, fill=False, ec=c, lw=1.6, alpha=0.8))
    rail.text(0.30, 0.74, "rings fan out:\nfutures disagree", color=_MUTED, fontsize=7.8,
              va="center")
    rail.text(0.02, 0.63, "each ring = one candidate future\nring area = mass it sends to the "
              "cell\ncolour = surviving hypothesis line", color=_MUTED, fontsize=7.8, va="top")
    # timeline + sparkline live in figure coordinates set by the caller each frame
    rail.text(0.02, 0.44, "PROPAGATED UNCERTAINTY\n(radius of the set of futures, nats)",
              color=_INK, fontsize=8, fontweight="bold", va="top")
    days = [s["day"] for s in states]
    etas = [s["eta"] for s in states]
    x = np.linspace(0.06, 0.94, len(days))
    y0, y1 = 0.16, 0.36
    ymax = max(max(etas), 1e-9)
    rail.plot(x, [y0] * len(x), color=_MUTED, lw=0.8)
    for xi, d in zip(x, days):
        rail.text(xi, y0 - 0.035, f"D{d:g}", color=_MUTED, fontsize=7, ha="center")
    line, = rail.plot([], [], color=_LINEAGE[0], lw=1.8, marker="o", ms=3.5)
    cursor, = rail.plot([], [], marker="v", color=_INK, ms=6, lw=0)
    return x, y0, y1, ymax, line, cursor


def main(record_path, run_dir, budget, seed, coords_path, hold, fade, intro, fps):
    rec = np.load(record_path, allow_pickle=True)
    coords = read_fle_coords(coords_path)
    bx, by, bfate = load_identity_base(run_dir)
    states = _states(rec, run_dir, budget, seed, coords)
    days = rec["days"].tolist()
    rhat = float(np.median([np.median([g["rhat_med"] for g in
                                       json.loads(str(rec[f"diag_{a:g}->{b:g}"]))])
                            for a, b in zip(days[:-1], days[1:])]))

    fig = plt.figure(figsize=(10.4, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.02,
                          left=0.01, right=0.99, top=0.92, bottom=0.04)
    ax = fig.add_subplot(gs[0])
    rail = fig.add_subplot(gs[1])
    draw_identity_base(ax, bx, by, bfate)
    tx, ty0, ty1, ymax, spark, cursor = _draw_rail(rail, states)
    etas = [s["eta"] for s in states]

    title = fig.suptitle("", color=_INK, fontsize=12, fontweight="bold", x=0.02, ha="left")
    caption = rail.text(0.02, 0.10, "", color=_INK, fontsize=8.6, va="top", fontweight="bold")
    fig.text(0.985, 0.005, "PLACEHOLDER quality: sampler R-hat_med "
             f"{rhat:.2f} | HFPD-OT particle filter | GSE122662, WOT layout",
             color=_MUTED, fontsize=7, ha="right")
    intro_box = ax.text(0.5, 0.55, _INTRO, transform=ax.transAxes, color=_INK, fontsize=11.5,
                        ha="center", va="center", fontweight="bold", linespacing=1.5,
                        bbox=dict(boxstyle="round,pad=0.9", fc="white", ec=_MUTED, alpha=0.94))

    def circles_of(state, alpha):
        arts = []
        xy = state["xy"]
        ok = np.isfinite(xy[:, 0])
        for p in np.argsort(-state["mass"]):
            w = state["W"][p][ok]
            sizes = 3200.0 * w
            live = sizes > 0.5
            arts.append(ax.scatter(xy[ok][live, 0], xy[ok][live, 1], s=sizes[live],
                                   facecolors="none", edgecolors=state["colour"][p],
                                   lw=1.4, alpha=alpha, zorder=3 + 0.01 * p))
        return arts

    frames = [("intro", k) for k in range(intro)]
    for si in range(len(states)):
        frames += [("hold", si, k) for k in range(hold)]
        if si + 1 < len(states):
            frames += [("fade", si, t / fade) for t in range(1, fade + 1)]

    live_arts = []

    def show_state(si):
        s = states[si]
        n = len(s["W"])
        title.set_text(f"Day {s['day']:g}  —  {n} candidate future{'s' if n > 1 else ''} "
                       "of the cell population")
        caption.set_text(s["caption"])
        spark.set_data(tx[:si + 1], [ty0 + (ty1 - ty0) * e / ymax for e in etas[:si + 1]])
        cursor.set_data([tx[si]], [ty1 + 0.03])

    def update(frame):
        nonlocal live_arts
        for a in live_arts:
            a.remove()
        live_arts = []
        if frame[0] == "intro":
            intro_box.set_visible(True)
            # dots pulse so intro frames stay distinct (identical frames collapse in the GIF
            # encoder, cutting the card's dwell time)
            title.set_text("HFPD-OT: lineage maps with uncertainty" +
                           " ." * (1 + frame[1] % 3))
            return []
        intro_box.set_visible(False)
        if frame[0] == "hold":
            si, k = frame[1], frame[2]
            show_state(si)
            pulse = 0.55 + 0.06 * np.sin(2 * np.pi * k / hold)
            live_arts = circles_of(states[si], alpha=pulse)
        else:
            _, si, t = frame
            show_state(si + 1 if t > 0.5 else si)
            title.set_text(f"Day {states[si]['day']:g} → {states[si + 1]['day']:g}  —  "
                           "propagating every candidate through sampled transport maps")
            live_arts = circles_of(states[si], alpha=0.55 * (1 - t)) + \
                        circles_of(states[si + 1], alpha=0.55 * t)
        return live_arts

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = Path(record_path).with_suffix("").as_posix() + "_propagation.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"video ({len(frames)} frames @ {fps} fps) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", help="particle_filter_*.npz from run_particle_filter.py")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--budget", type=int, default=30, help="must match the record's run")
    p.add_argument("--seed", type=int, default=0, help="must match the record's run")
    p.add_argument("--coords", default=str(ROOT / "data" / "fle_coords.txt"))
    p.add_argument("--hold", type=int, default=18, help="frames held on each day state")
    p.add_argument("--fade", type=int, default=10, help="cross-fade frames between days")
    p.add_argument("--intro", type=int, default=28, help="intro-card frames")
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args()
    main(a.record, a.run_dir or latest_run(), a.budget, a.seed, a.coords, a.hold, a.fade,
         a.intro, a.fps)
