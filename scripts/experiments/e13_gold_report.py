"""E13 report: gold-anchor figure -- chart-family bias ladders against the gold standard.

Reads the record written by ``e13_gold_anchor.py``. Four panels:
  A  R_mu bias (chart - gold) vs gamma, per rank and phase, with gamma -> 0 extrapolation
     VARIANTS (linear and quadratic LS on the gate-passing rows only) drawn to gamma = 0 --
     the spread of intercepts displays the identification limit of the ladder: a nonzero
     gamma -> 0 residual is the volume-term hypothesis, but the ladder alone brackets it.
     NOTE: no limit is computed as a headline number here; the intercept marks are model
     variants, not measurements. The section chart (exact 1/2 log det G, hard gauge, no
     ridge) is the construction that decides without a limit.
  B  g_pi0 bias vs gamma (same layout): how far the chart family sits from the true S^o
     spread around pi^o.
  C  median table-entry width ratio sd_chart / sd_gold vs gamma.
  D  R-hat_max vs gamma: the small-gamma mixing degradation that clouds the smallest rungs.
     Panel A omits rows failing the gate entirely (their only content is D's story); open
     markers in B flag rows with R-hat_max > 1.25.

Reproduce:  python scripts/experiments/e13_gold_report.py --exp gold
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GATE_RHAT_MAX = 1.25
_LIVE_MEAN, _LIVE_BAND = 1e-4, 1e-6


def load(z, phase):
    K = len(z[f"{phase}_populations"])
    seeds = sorted({int(k.split("gold")[1].split("_")[0]) for k in z.files
                    if k.startswith(f"{phase}_gold")})
    gold = {s: {st: z[f"{phase}_gold{s}_{st}"] for st in ("mean", "sd", "band", "mcse")}
            for s in seeds}
    gmean = np.mean([gold[s]["mean"] for s in seeds], axis=0)
    gsd = np.mean([gold[s]["sd"] for s in seeds], axis=0)
    gband = np.mean([gold[s]["band"] for s in seeds], axis=0)
    live = (gmean[:K * K] > _LIVE_MEAN) & (gband[:K * K] > _LIVE_BAND)
    ranks = sorted({int(k.split("_chart_r")[1].split("_")[0]) for k in z.files
                    if "_chart_r" in k})
    gammas = sorted({float(k.split("_g")[-1].split("_")[0]) for k in z.files
                     if f"_chart_r{ranks[0]}_g" in k and k.endswith("_mean")})
    chart = {}
    for r in ranks:
        for g in gammas:
            p = f"{phase}_chart_r{r}_g{g:g}"
            chart[(r, g)] = dict(mean=z[f"{p}_mean"], sd=z[f"{p}_sd"],
                                 mcse=z[f"{p}_mcse"],
                                 gates=json.loads(str(z[f"{p}_gates"])))
    return dict(K=K, gmean=gmean, gsd=gsd, live=live, ranks=ranks,
                gammas=np.array(gammas), chart=chart)


def main(run_dir, budget, exp):
    run_dir = Path(run_dir)
    z = np.load(run_dir / f"e13_gold_anchor_b{budget}_{exp}.npz", allow_pickle=True)
    P = {ph: load(z, ph) for ph in ("dox", "serum")}

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("E13 gold anchor (N=8): chart-family bias ladders vs the full-rank gold",
                 fontsize=12.5)
    rank_cols = {2: "#1f6fb2", 4: "#8a5fbf", 8: "#c0504d"}
    phase_ls = {"dox": "-", "serum": "--"}
    gg = np.linspace(0, 0.55, 60)

    from matplotlib.lines import Line2D
    handles = ([Line2D([], [], color=c, marker="o", ls="", ms=5, label=f"r={r}")
                for r, c in rank_cols.items()] +
               [Line2D([], [], color="0.3", ls=ls, label=ph)
                for ph, ls in phase_ls.items()])

    def idx(K, name):
        return {"R_mu": K * K, "g_pi0": K * K + 2}[name]

    # --- A: R_mu bias ladders + gamma=0 intercept brackets + rank-trend inset ---
    a = ax[0, 0]
    for ph, d in P.items():
        j = idx(d["K"], "R_mu")
        for r in d["ranks"]:
            b = np.array([d["chart"][(r, g)]["mean"][j] - d["gmean"][j]
                          for g in d["gammas"]])
            ok = np.array([d["chart"][(r, g)]["gates"]["rhat_max"] <= GATE_RHAT_MAX
                           for g in d["gammas"]])
            a.plot(d["gammas"][ok], b[ok], "o", color=rank_cols[r], ms=5,
                   ls=phase_ls[ph], lw=1.2)
            if ok.sum() >= 3:
                ints = [np.polyval(np.polyfit(d["gammas"][ok], b[ok], deg), 0.0)
                        for deg in (1, 2)]
                a.plot([0, 0], [min(ints), max(ints)], color=rank_cols[r], lw=3,
                       alpha=0.6, solid_capstyle="butt")
    a.axhline(0, color="0.3", lw=1.0)
    a.text(0.53, 0.0, "gold", ha="right", va="bottom", fontsize=8, color="0.3")
    a.annotate("gamma = 0 intercept brackets\n(linear..quad fits -- plausible\nrange, not a point estimate)",
               xy=(0.005, -0.40), xytext=(0.10, -0.30), fontsize=7.5, color="0.35",
               arrowprops=dict(arrowstyle="->", lw=0.7, color="0.35"))
    a.set_xlabel("$\\gamma$")
    a.set_ylabel("$\\hat{E}_{chart} - \\hat{E}_{gold}$ (R_mu, nats)")
    a.set_title("A  R_mu bias vs $\\gamma$: contraction onto $\\pi^o$ (gold's mode) suppresses\n"
                "the spread term as $\\gamma$ grows; no fit variant closes onto gold at 0",
                fontsize=9.5)
    a.set_xlim(-0.02, 0.55)
    a.legend(handles=handles, fontsize=7, ncol=2, loc="upper left")
    ia = a.inset_axes([0.44, 0.62, 0.52, 0.34])
    for ph, d in P.items():
        j = idx(d["K"], "R_mu")
        rb = [d["chart"][(r, 0.2)]["mean"][j] - d["gmean"][j] for r in d["ranks"]]
        ia.plot(d["ranks"], rb, "o", ls=phase_ls[ph], color="0.25", ms=4, lw=1.0)
    ia.set_xticks(P["dox"]["ranks"])
    ia.tick_params(labelsize=7)
    ia.set_title("rank trend at $\\gamma$=0.2 (fixed ridge)", fontsize=7.5)
    ia.set_xlabel("r", fontsize=7)

    # --- B: ABSOLUTE g_pi0, log scale: gamma-response visible against the gold gap ---
    b = ax[0, 1]
    for ph, d in P.items():
        j = idx(d["K"], "g_pi0")
        for r in d["ranks"]:
            v = np.array([d["chart"][(r, g)]["mean"][j] for g in d["gammas"]])
            ok = np.array([d["chart"][(r, g)]["gates"]["rhat_max"] <= GATE_RHAT_MAX
                           for g in d["gammas"]])
            b.plot(d["gammas"][ok], v[ok], "o", color=rank_cols[r], ms=5,
                   ls=phase_ls[ph], lw=1.2)
            b.plot(d["gammas"][~ok], v[~ok], "o", color=rank_cols[r], ms=6,
                   mfc="none", mew=1.4)
        b.axhline(d["gmean"][j], color="0.25", lw=1.2, ls=phase_ls[ph])
        b.text(0.53, d["gmean"][j], f"gold ({ph})", ha="right", va="bottom",
               fontsize=8, color="0.25")
    b.set_yscale("log")
    b.set_xlabel("$\\gamma$")
    b.set_ylabel("$\\hat{E}[g_{\\pi^o}]$ (nats, log scale)")
    b.set_title("B  ABSOLUTE g_pi0: the ridge response (~2.4x, bowl regime) rides on a\n"
                "gamma-INDEPENDENT concentration gap to gold (~x10)", fontsize=9.5)
    b.set_xlim(-0.02, 0.55)

    # --- C: width ratio, legend included ---
    c = ax[1, 0]
    for ph, d in P.items():
        for r in d["ranks"]:
            wr = []
            for g in d["gammas"]:
                sdc = d["chart"][(r, g)]["sd"][:d["K"] ** 2][d["live"]]
                wr.append(np.median(sdc / np.maximum(d["gsd"][:d["K"] ** 2][d["live"]],
                                                     1e-12)))
            c.plot(d["gammas"], wr, "o", color=rank_cols[r], ms=5, ls=phase_ls[ph],
                   lw=1.2)
    c.axhline(1.0, color="0.3", lw=1.0)
    c.text(0.53, 1.0, "gold width", ha="right", va="bottom", fontsize=8, color="0.3")
    c.set_xlabel("$\\gamma$")
    c.set_ylabel("median sd$_{chart}$ / sd$_{gold}$")
    c.set_title("C  TABLE-ENTRY width ratio: slightly OVER gold at $\\gamma$=0.5, inflating\n"
                "as $\\gamma\\to 0$ (plan-divergence spread is simultaneously UNDER: anisotropy)",
                fontsize=9.5)
    c.set_xlim(-0.02, 0.55)
    c.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")

    # --- D: gates, legend + provenance ---
    dpan = ax[1, 1]
    for ph, d in P.items():
        for r in d["ranks"]:
            rh = [d["chart"][(r, g)]["gates"]["rhat_max"] for g in d["gammas"]]
            dpan.plot(d["gammas"], rh, "o", color=rank_cols[r], ms=5, ls=phase_ls[ph],
                      lw=1.2)
    dpan.axhline(GATE_RHAT_MAX, color="#2a7345", lw=0.9, ls="--")
    dpan.text(0.53, GATE_RHAT_MAX, "R1 gate 1.25 (pre-registered: loose max-tier over\n"
              "~2000 per-coordinate R-hats; the tight test is R-hat_med <= 1.05)",
              ha="right", va="bottom", fontsize=7, color="#2a7345")
    dpan.set_xlabel("$\\gamma$")
    dpan.set_ylabel("R-hat_max")
    dpan.set_title("D  GATES: the gauge unpins as $\\gamma\\to 0$", fontsize=9.5)
    dpan.set_xlim(-0.02, 0.55)
    dpan.legend(handles=handles, fontsize=7, ncol=2, loc="upper right")

    fig.text(0.5, 0.005, "solid = dox, dashed = serum; A omits rows failing the R1 gate "
             "(see D); open markers in B = R-hat_max > 1.25 (excluded from fits)",
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = run_dir / f"e13_gold_anchor_b{budget}_{exp}.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--exp", type=str, required=True)
    a = p.parse_args()
    if a.run_dir is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from gmvae_confusion import latest_run
        a.run_dir = latest_run()
    main(a.run_dir, a.budget, a.exp)
