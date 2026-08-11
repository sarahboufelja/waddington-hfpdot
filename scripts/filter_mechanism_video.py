"""The particle filter as a methods animation -- the system propagating in time (PLACEHOLDER data).

The companion of the landscape video, aimed at the paper's methods narrative: no landscape, just
the algorithm's state. Marginals are drawn as composition glyphs (six bars, one per coarse fate,
in the territory tints), laid out in day columns, and the three moves of the filter cycle on
screen:

  1  SAMPLE     from each particle's own hyperprior ``S^o(pi | mu_p)``, n transport plans; each
                plan pushes the particle to a future composition. The prior target of the pair is
                drawn DASHED in the same column: futures visibly scatter around it rather than
                hitting it, because the marginals are KL-penalised, not constrained (the UOT
                point).
  2  TRIM       farthest-point selection keeps L maximally-spread futures; the pruned fade out and
                their probability mass reappears as the survivors' mass chips (absorption keeps
                the ensemble unbiased).
  3  ADVANCE    each survivor conditions its own hyperprior at the next pair -- exact filtering,
                no moment collapse -- and the cycle repeats.

Glyph frames carry the hypothesis-lineage colour (inherited through parent pointers, root in
neutral ink); bar heights are the fate composition, normalised per glyph to the tallest bar. All
data come from a ``run_particle_filter.py`` record that includes the pre-trim pools (``pool_*``
keys) and fate projections (``W_fates_*``) -- nothing is re-embedded or re-sampled here.

Run from the repo root:
    python scripts/filter_mechanism_video.py assets/gmvae_runs/<run>/particle_filter_<window>.npz
"""
import argparse
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
from matplotlib.patches import FancyBboxPatch, Rectangle

from fle_plan_bands import _COARSE_ORDER, _INK, _MUTED, _TINTS

_LINEAGE = ["#2b6cb0", "#b45309", "#0f766e", "#7c3aed", "#be185d", "#4d7c0f"]

_CAPTIONS = {
    "intro": "The marginal particle filter: how coupling uncertainty propagates through time.",
    "sample": ("1) SAMPLE  n transport plans  π ~ S°(· | μ_p)  from each "
               "particle's own hyperprior; every plan pushes the particle to a future "
               "composition.\nFutures scatter around the pair's prior target (dashed): marginals "
               "are KL-penalised, not constrained (unbalanced OT)."),
    "trim": ("2) TRIM  farthest-point selection keeps the most spread-out futures; every pruned "
             "future donates its probability mass to its nearest survivor\n(mass chips), so the "
             "ensemble stays an unbiased weighted particle set and the extremes survive."),
    "advance": ("3) ADVANCE  each survivor conditions its own hyperprior at the next pair -- "
                "exact filtering, no moment collapse -- and the cycle repeats."),
}


def _fate_profiles(weights, W_fates):
    """(N, m) cell marginals -> (N, 6) renormalised coarse-fate compositions."""
    F = np.atleast_2d(weights) @ W_fates
    tot = F.sum(axis=1, keepdims=True)
    return np.where(tot > 0, F / np.maximum(tot, 1e-12), 1.0 / F.shape[1])


def _glyph(ax, x, y, profile, frame, alpha=1.0, scale=1.0, dashed=False, lw=1.5):
    """One composition glyph: fate-tinted bars in a lineage-coloured frame. Returns artists."""
    w, h = 0.46 * scale, 0.115 * scale
    arts = [FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.008,rounding_size=0.012", fill=True,
                           fc="white", ec=frame, lw=lw, alpha=alpha,
                           ls=(0, (3, 2)) if dashed else "-", zorder=4)]
    pmax = max(float(np.max(profile)), 1e-12)
    bw = w / len(profile)
    for k, g in enumerate(_COARSE_ORDER):
        bh = 0.84 * h * float(profile[k]) / pmax
        arts.append(Rectangle((x - w / 2 + k * bw + 0.08 * bw, y - h / 2 + 0.08 * h),
                              0.84 * bw, bh, fc=_TINTS[g], ec="none", alpha=alpha, zorder=5))
    for a in arts:
        ax.add_patch(a)
    return arts


def _build_states(rec):
    """Per-pair display data: source slots, displayed pool futures, survivors, colours."""
    days = rec["days"].tolist()
    steps = []
    src_colours = [_INK]
    for a, b in zip(days[:-1], days[1:]):
        tag = f"{a:g}->{b:g}"
        Wf = rec[f"W_fates_{b:g}"]
        pool = _fate_profiles(rec[f"pool_{tag}"], Wf)
        pool_par = rec[f"pool_parents_{tag}"]
        ens = _fate_profiles(rec[f"ensemble_{tag}"], Wf)
        ens_mass = np.exp(rec[f"ensemble_mass_{tag}"])
        ens_par = rec[f"parents_{tag}"]
        nu0 = rec[f"nu0_fates_{tag}"]; nu0 = nu0 / nu0.sum()
        # locate the survivors inside the pool (trim copies weight vectors verbatim)
        pool_raw = rec[f"pool_{tag}"]; ens_raw = rec[f"ensemble_{tag}"]
        surv_idx = [int(np.argmin(np.abs(pool_raw - e).sum(axis=1))) for e in ens_raw]
        # display subset: survivors always, plus evenly-thinned others (cap keeps it readable)
        cap = 13
        others = [i for i in range(len(pool)) if i not in surv_idx]
        show = sorted(set(surv_idx) |
                      set(others[:: max(1, len(others) // max(cap - len(surv_idx), 1))]
                          [:cap - len(surv_idx)]))
        if len(src_colours) == 1:
            child_colours = {i: _LINEAGE[surv_idx.index(i) % len(_LINEAGE)]
                             if i in surv_idx else _INK for i in show}
        else:
            child_colours = {i: src_colours[int(pool_par[i])] for i in show}
        steps.append({"day_from": a, "day_to": b, "pool": pool, "pool_par": pool_par,
                      "show": show, "surv": surv_idx, "ens": ens, "ens_mass": ens_mass,
                      "ens_par": ens_par, "nu0": nu0, "src_colours": list(src_colours),
                      "child_colours": child_colours})
        src_colours = [child_colours[i] for i in surv_idx]
    return days, steps


def main(record_path, hold, fade, intro, fps):
    rec = np.load(record_path, allow_pickle=True)
    days, steps = _build_states(rec)
    n_draws = len(steps[0]["pool"])

    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(-0.55, len(days) - 0.25)
    ax.set_ylim(-0.02, 1.06)
    ax.axis("off")
    for i, d in enumerate(days):
        ax.text(i, 1.035, f"Day {d:g}", color=_INK, fontsize=11, fontweight="bold", ha="center")
        ax.axvline(i, color=_MUTED, lw=0.5, alpha=0.25, zorder=1)
    # fate legend, once
    for k, g in enumerate(_COARSE_ORDER):
        ax.add_patch(Rectangle((-0.52, 0.985 - 0.033 * k), 0.035, 0.02, fc=_TINTS[g], ec="none"))
        ax.text(-0.47, 0.995 - 0.033 * k, g, color=_MUTED, fontsize=7.2, va="center")
    caption = fig.text(0.02, 0.055, "", color=_INK, fontsize=9.3, va="top", linespacing=1.35)
    fig.text(0.99, 0.005, "PLACEHOLDER sampler quality -- mechanism illustration on real "
             "GSE122662 marginals", color=_MUTED, fontsize=7, ha="right")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.13)

    def slots(n, centre=0.52, span=0.82):
        if n == 1:
            return [centre]
        lo, hi = centre - span / 2, centre + span / 2
        return list(np.linspace(hi, lo, n))

    # source positions per column, filled as the animation advances
    src_pos = {0: slots(1)}
    for si, st in enumerate(steps):
        src_pos[si + 1] = slots(len(st["surv"]))

    live = []

    def draw_sources(col, profiles, colours, masses=None, alpha=1.0):
        arts = []
        for y, pr, c, i in zip(src_pos[col], profiles, colours, range(len(profiles))):
            arts += _glyph(ax, col, y, pr, c, alpha=alpha)
            if masses is not None:
                arts.append(ax.text(col, y - 0.085, f"{masses[i]:.0%}", color=c, fontsize=7.5,
                                    ha="center", fontweight="bold"))
        return arts

    def root_profile():
        tag = f"{days[0]:g}->{days[1]:g}"
        Wf = rec[f"W_fates_{days[1]:g}"]      # root is uniform on its own support; nu0 of the
        return steps[0]["nu0"]                 # first pair is the closest stored projection

    def frame_list():
        fr = [("intro", k) for k in range(intro)]
        for si in range(len(steps)):
            fr += [("sample", si, min(1.0, (k + 1) / fade), None) for k in range(fade)]
            fr += [("sample", si, 1.0, k) for k in range(hold)]
            fr += [("trim", si, min(1.0, (k + 1) / fade), None) for k in range(fade)]
            fr += [("trim", si, 1.0, k) for k in range(hold)]
        fr += [("trim", len(steps) - 1, 1.0, k) for k in range(hold)]
        return fr

    def update(frame):
        nonlocal live
        for a in live:
            try:
                a.remove()
            except ValueError:
                pass
        live = []
        kind = frame[0]
        if kind == "intro":
            caption.set_text(_CAPTIONS["intro"] + " ." * (1 + frame[1] % 3))
            live += draw_sources(0, [root_profile()], [_INK])
            return live
        si, t = frame[1], frame[2]
        st = steps[si]
        # everything settled so far: sources of every earlier column
        for c in range(si + 1):
            prev = steps[c - 1] if c > 0 else None
            profs = [root_profile()] if c == 0 else prev["ens"]
            cols = [_INK] if c == 0 else [prev["child_colours"][i] for i in prev["surv"]]
            mass = None if c == 0 else prev["ens_mass"] / prev["ens_mass"].sum()
            live += draw_sources(c, profs, cols, mass, alpha=0.35 if c < si else 1.0)
        col = si + 1
        ys = slots(len(st["show"]), span=0.95)
        # gentle pulse keeps hold frames distinct in the GIF (identical frames collapse and cut
        # the dwell time); it modulates only the fan-out connectors and the dashed prior
        k = frame[3] if len(frame) > 3 else None
        pulse = 0.85 + 0.15 * np.sin(2 * np.pi * k / hold) if k is not None else 1.0
        live += _glyph(ax, col + 0.33, 0.52, st["nu0"], _MUTED, alpha=0.55 + 0.35 * pulse,
                       dashed=True, scale=0.9)
        live.append(ax.text(col + 0.33, 0.595, "prior target", color=_MUTED, fontsize=7.2,
                            ha="center", va="bottom"))
        if kind == "sample":
            caption.set_text(_CAPTIONS["sample"].replace("n transport", f"n={n_draws} transport"))
            for y, i in zip(ys, st["show"]):
                c = st["child_colours"][i]
                sy = src_pos[si][int(st["pool_par"][i])]
                live.append(ax.plot([si + 0.24, col - 0.20], [sy, y], color=c, lw=0.7,
                                    alpha=0.30 * t * pulse, zorder=2)[0])
                live += _glyph(ax, col, y, st["pool"][i], c, alpha=0.75 * t, scale=0.5)
        else:                                   # trim
            caption.set_text(_CAPTIONS["trim"])
            mass = st["ens_mass"] / st["ens_mass"].sum()
            for y, i in zip(ys, st["show"]):
                c = st["child_colours"][i]
                if i in st["surv"]:
                    k = st["surv"].index(i)
                    yt = src_pos[si + 1][k]
                    yy = y + (yt - y) * t       # survivors glide to their settled slots
                    live += _glyph(ax, col, yy, st["pool"][i], c, alpha=0.75 + 0.25 * t,
                                   scale=0.5 + 0.5 * t)
                    if t >= 1.0:
                        live.append(ax.text(col, yy - 0.085, f"{mass[k]:.0%}", color=c,
                                            fontsize=7.5, ha="center", fontweight="bold"))
                else:
                    live += _glyph(ax, col, y, st["pool"][i], c, alpha=0.75 * (1 - t),
                                   scale=0.5 * (1 - 0.5 * t))
        return live

    frames = frame_list()
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = Path(record_path).with_suffix("").as_posix() + "_mechanism.gif"
    anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"video ({len(frames)} frames @ {fps} fps) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("record", help="particle_filter_*.npz WITH pool_* keys")
    p.add_argument("--hold", type=int, default=14)
    p.add_argument("--fade", type=int, default=10)
    p.add_argument("--intro", type=int, default=20)
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args()
    main(a.record, a.hold, a.fade, a.intro, a.fps)
