"""Fate-flow branches on the FLE -- the posterior as spaghetti-of-draws (v1, fate level).

Each strand is ONE posterior draw's flow between two fate territories. Per day pair, the
raw per-draw fate tables (``tables_{tag}``, in-record) are row-normalised (per-source-fate
routes; scale-invariant, so the documented chart mass artifact cancels) and every retained
draw contributes one quadratic-Bezier strand from source to target territory centroid,
width proportional to that draw's flow and bow offset by draw index so the bundle stays
visible. Consistent flows render as tight uniform bundles; contested flows as ragged fans
of unequal widths -- the fuzziness IS the posterior spread, with no derived statistic in
between. Self-renewal flows are not drawn (they would be loops); territory centroids come
from the L1 overlay record (median position of cells whose MAP fate is that population --
display geometry only, every plotted width is a per-draw posterior quantity).

``--video`` renders the animated version for a reader meeting the method cold: an intro
card states the idea in one sentence, strands then ACCUMULATE draw by draw within each
day pair -- many candidate lineage maps visibly piling up -- before the window advances,
with an auto-caption naming the most confident and the most contested route of the step.
Geometry, colours and width mapping are fixed across frames and windows (nothing
rescales), so bundles are comparable throughout.

Colour: strands carry a darker mark-step of their SOURCE territory's hue (identity is
also encoded by origin position, so colour is never the sole channel); territory tints
stay as the recessive base.

v2 (cell -> fate branches from per-source-cell descendant profiles) needs a
(draws, II, K) record key -- a schema change, decided separately.

Run from the repo root:
    python scripts/fle_flow_branches.py <record> --run-dir <run_dir> [--video]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir, latest_artifact
from fle_plan_bands import _INK, _MUTED, coarse_of, draw_identity_base, load_identity_base

#: mark-strength strand colours per coarse territory (darker steps of the base tints;
#: IPS/Neural are sibling blues by palette design -- separated by lightness, and origin
#: position always duplicates the identity)
_STRANDS = {"Stromal": "#6b6560", "MET": "#2f8f68", "Epithelial": "#8d5a8d",
            "IPS": "#34518f", "Neural": "#4a9bc9", "Trophoblast": "#967d2e"}

_INTRO = ("Where do reprogramming cells go?\n\n"
          "One transport map is one hypothesis about every cell's route.\n"
          "HFPD-OT samples thousands of plausible maps from the data.\n\n"
          "Each strand below is ONE sampled map's flow between two fates.\n"
          "Tight, even bundles: the route is determined.\n"
          "Ragged fans: the data leave that route genuinely open.")


def _centroids(run_dir):
    """Fine-fate territory centroids from the L1 overlay record (display geometry)."""
    d = np.load(latest_artifact(run_dir, "fle_uncertainty") / "fle_uncertainty.npz",
                allow_pickle=True)
    pops = [str(p) for p in d["populations"]]
    x, y, mf = d["x"], d["y"], d["map_fate"]
    cent = np.full((len(pops), 2), np.nan)
    for f in range(len(pops)):
        sel = mf == f
        if sel.any():
            cent[f] = np.median(x[sel]), np.median(y[sel])
    return pops, cent


def _prepare(record_path, run_dir, thin, min_flow):
    """Everything both renderers share: geometry, colours, and per-pair thinned routes."""
    run_dir = Path(run_dir)
    rec = np.load(record_path, allow_pickle=True)
    days = rec["days"].tolist()
    pops = [str(p) for p in rec["populations"]]
    l1_pops, cent = _centroids(run_dir)
    assert l1_pops == pops, "L1 record and PF record disagree on populations"
    base = load_identity_base(run_dir)
    colour = [_STRANDS.get(coarse_of(p), "#555555") for p in pops]
    pairs = []
    for a, b in zip(days[:-1], days[1:]):
        T = np.asarray(rec[f"tables_{a:g}->{b:g}"], dtype=float)
        keep = np.linspace(0, len(T) - 1, min(thin, len(T))).astype(int)
        rows = T[keep].sum(axis=2, keepdims=True)
        R = np.divide(T[keep], rows, out=np.zeros_like(T[keep]), where=rows > 0)
        pairs.append((a, b, R))
    return run_dir, days, pops, cent, colour, base, pairs


def _routes(R, cent, min_flow):
    """Live off-diagonal routes of a pair: (r, c, flows) with finite endpoints."""
    K = R.shape[1]
    med = np.median(R, axis=0)
    out = []
    for r in range(K):
        for c in range(K):
            if c == r or not (np.isfinite(cent[r]).all() and np.isfinite(cent[c]).all()):
                continue
            if med[r, c] >= min_flow or R[:, r, c].max() >= 2 * min_flow:
                out.append((r, c, R[:, r, c]))
    return out


def _caption(routes, pops):
    """Most confident and most contested substantial route, in plain language."""
    scored = [(r, c, float(np.median(f)), float(np.std(f) / max(np.mean(f), 1e-12)))
              for r, c, f in routes if np.median(f) >= 0.03]
    if not scored:
        return "All mass self-renews at this step."
    sure = min((s for s in scored if s[2] >= 0.05), key=lambda s: s[3], default=None)
    fuzzy = max(scored, key=lambda s: s[3])
    parts = []
    if sure is not None:
        parts.append(f"Confident route: {pops[sure[0]]} → {pops[sure[1]]}")
    parts.append(f"Most contested: {pops[fuzzy[0]]} → {pops[fuzzy[1]]}")
    return "   |   ".join(parts)


def _strand(ax, p0, p1, bow, width, colour, alpha):
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
    ctrl = mid + n * bow
    t = np.linspace(0.0, 1.0, 24)[:, None]
    c = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p1
    return ax.plot(c[:, 0], c[:, 1], color=colour, lw=width, alpha=alpha,
                   solid_capstyle="round", zorder=4)[0]


def _draw_base(ax, base, cent, colour, pops, label):
    draw_identity_base(ax, *base, label_territories=label)
    for f in range(len(pops)):
        if np.isfinite(cent[f]).all():
            ax.scatter(*cent[f], s=26, c=colour[f], edgecolors="white", lw=0.8, zorder=5)


def _pair_strands(ax, R, routes, cent, colour, alpha, draw_sel=None):
    """Strand artists for the given draws (all if draw_sel is None), returned for removal."""
    artists = []
    n_draws = len(R)
    for r, c, flows in routes:
        dist = float(np.linalg.norm(cent[c] - cent[r]))
        for k, f in enumerate(flows):
            if draw_sel is not None and k not in draw_sel:
                continue
            if f < 1e-3:
                continue
            bow = (k / max(n_draws - 1, 1) - 0.5) * 0.22 * dist
            artists.append(_strand(ax, cent[r], cent[c], bow, 0.4 + 9.0 * f,
                                   colour[r], alpha))
    return artists


def figure(record_path, run_dir, thin, min_flow, alpha):
    run_dir, days, pops, cent, colour, base, pairs = _prepare(record_path, run_dir,
                                                              thin, min_flow)
    ncol = min(4, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, (a, b, R) in enumerate(pairs):
        routes = _routes(R, cent, min_flow)
        _draw_base(axes[i], base, cent, colour, pops, label=(i == 0))
        _pair_strands(axes[i], R, routes, cent, colour, alpha)
        axes[i].set_title(f"D{a:g} → D{b:g}", color=_INK, fontsize=10)
        print(f"  D{a:g}->D{b:g}: {_caption(routes, pops)}")
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.suptitle(f"Fate-flow branches, one strand per posterior draw "
                 f"(row-normalised routes; {thin} draws/pair; flows < {min_flow:g} and "
                 "self-renewal not drawn)", color=_MUTED, fontsize=10)
    out_dir = artifact_dir(run_dir, "fle_flow_branches",
                           config=dict(record=str(record_path), thin=thin,
                                       min_flow=min_flow, alpha=alpha, video=False))
    out = out_dir / f"fle_branches_D{days[0]:g}-D{days[-1]:g}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"branches -> {out}")


def video(record_path, run_dir, thin, min_flow, alpha, fps, intro_n, accum_n, hold_n,
          fade_n):
    run_dir, days, pops, cent, colour, base, pairs = _prepare(record_path, run_dir,
                                                              thin, min_flow)
    routes_of = [_routes(R, cent, min_flow) for _, _, R in pairs]
    captions = [_caption(rt, pops) for rt in routes_of]
    per_pair = accum_n + hold_n + fade_n
    total = intro_n + per_pair * len(pairs)

    fig = plt.figure(figsize=(9.6, 6.4))
    ax = fig.add_axes((0.02, 0.10, 0.70, 0.86))
    rail = fig.add_axes((0.735, 0.10, 0.255, 0.86))
    rail.set_xlim(0, 1); rail.set_ylim(0, 1); rail.axis("off")
    _draw_base(ax, base, cent, colour, pops, label=True)
    rail.text(0.02, 0.98, "HOW TO READ", color=_INK, fontsize=9, fontweight="bold",
              va="top")
    rail.text(0.02, 0.93, "One strand = one sampled\nlineage map's flow.\n\n"
                          "Tight even bundle:\nroute determined.\n\n"
                          "Ragged fan:\nroute genuinely open.\n\n"
                          "Strand colour = source\nterritory; width = flow.",
              color=_MUTED, fontsize=8, va="top")
    rail.text(0.02, 0.40, "Timecourse", color=_INK, fontsize=8.5, fontweight="bold")
    for i, (a, b, _) in enumerate(pairs):
        rail.text(0.06, 0.36 - 0.028 * i, f"D{a:g} → D{b:g}", color=_MUTED,
                  fontsize=7.5)
    marker = rail.text(0.0, 0.36, "▶", color=_INK, fontsize=8)
    day_txt = ax.text(0.015, 0.985, "", transform=ax.transAxes, color=_INK, fontsize=13,
                      fontweight="bold", va="top")
    cap_txt = fig.text(0.02, 0.035, "", color=_INK, fontsize=10)
    intro_txt = fig.text(0.5, 0.55, _INTRO, color=_INK, fontsize=12, ha="center",
                         va="center",
                         bbox=dict(boxstyle="round,pad=0.9", fc="white", ec=_MUTED))
    state = {"pair": -1, "artists": []}

    def update(frame):
        if frame < intro_n:
            return []
        intro_txt.set_visible(False)
        f = frame - intro_n
        p, ph = divmod(f, per_pair)
        a, b, R = pairs[p]
        n_draws = len(R)
        if p != state["pair"]:
            for art in state["artists"]:
                art.remove()
            state.update(pair=p, artists=[])
            day_txt.set_text(f"D{a:g} → D{b:g}")
            cap_txt.set_text(captions[p])
            marker.set_position((0.0, 0.36 - 0.028 * p))
        if ph < accum_n:                                   # strands pile up draw by draw
            per_frame = int(np.ceil(n_draws / accum_n))
            new = range(ph * per_frame, min((ph + 1) * per_frame, n_draws))
            state["artists"] += _pair_strands(ax, R, routes_of[p], cent, colour, alpha,
                                              draw_sel=set(new))
        elif ph >= accum_n + hold_n:                       # fade toward the next step
            k = ph - accum_n - hold_n + 1
            for art in state["artists"]:
                art.set_alpha(alpha * max(0.0, 1.0 - k / fade_n))
        return state["artists"]

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / fps)
    out_dir = artifact_dir(run_dir, "fle_flow_branches",
                           config=dict(record=str(record_path), thin=thin,
                                       min_flow=min_flow, alpha=alpha, video=True,
                                       fps=fps))
    out = out_dir / f"fle_branches_D{days[0]:g}-D{days[-1]:g}.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    for (a, b, _), cap in zip(pairs, captions):
        print(f"  D{a:g}->D{b:g}: {cap}")
    print(f"branches video -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", help="particle_filter_*.npz from run_particle_filter.py")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--video", action="store_true", help="render the animated version")
    p.add_argument("--thin", type=int, default=64, help="draws rendered per pair (even thinning)")
    p.add_argument("--min-flow", type=float, default=0.02,
                   help="row-normalised flow below which a route is not drawn")
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--intro", type=int, default=24, help="intro-card frames")
    p.add_argument("--accum", type=int, default=10, help="strand-accumulation frames per pair")
    p.add_argument("--hold", type=int, default=16, help="hold frames per pair")
    p.add_argument("--fade", type=int, default=5, help="fade-out frames per pair")
    a = p.parse_args()
    rd = a.run_dir or latest_run()
    if a.video:
        video(a.record, rd, a.thin, a.min_flow, a.alpha, a.fps, a.intro, a.accum,
              a.hold, a.fade)
    else:
        figure(a.record, rd, a.thin, a.min_flow, a.alpha)
