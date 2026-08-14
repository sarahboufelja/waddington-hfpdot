"""E12 report: ridge-bias grid analysis and figure (ratification protocol R2, step 1).

Reads the record written by ``e12_ridge_bias_grid.py`` and produces (i) the certificate
tables on stdout and (ii) ``e12_bias_grid.png`` next to the record.

Notation (locked):
  E_gamma[O]    posterior mean of observable O under the gamma-tilted posterior
                p_gamma propto p_0 exp(-gamma T), T = ||theta||^2; estimated by the
                per-entry MC mean Ehat_gamma[O] (4 chains x 500 thinned draws).
  Ehat_0[O]     quadratic least-squares intercept of {Ehat_gamma[O]} at gamma = 0. An
                EXTRAPOLANT, not a sampled quantity: the gamma = 0 chart posterior is
                improper along the GL(r) orbits, so no chain exists there. Its
                model-independence rests on the near-linearity of the mean curves
                (finite-difference slopes constant to ~1%) and is cross-checked by the
                tilt identity d E_gamma[O]/d gamma = -Cov_gamma(O, T).
  bias(gamma)   Ehat_gamma[O] - Ehat_0[O]  (grid-extrapolation bias; certificate input).
  bias1(gamma)  -gamma Covhat_gamma(O, T): first-order tilt-identity estimate. Quoted
                only where |Cov| exceeds its MC error; noise-dominated for weakly
                coupled observables (table entries), where it overstates bias(gamma).
  sd(gamma)     posterior sd of O under gamma; band(gamma) = 2.5-97.5 interquantile
                width. "Live" table entries: Ehat_{gamma*} > 1e-4 and band > 1e-6.

Certificate at gamma* (strengthened form -- the raw 1/4-sd gate is threshold-sensitive
and is retained only as a summary statistic):
  1. COVERAGE BUDGET. A location shift of b sd erodes nominal 95% credible coverage at
     second order only: erosion ~ 1.96 phi(1.96) b^2 ~ 0.115 b^2. The gate is stated as
     a coverage budget (<= 0.7 pp of the 95% level, i.e. b <= 0.25 sd); any alternative
     threshold maps to a coverage number that must be defended on operational grounds.
  2. DUAL-COLUMN CONCLUSION STABILITY. Both Ehat_gamma* and Ehat_0 are reported (the
     latter with its total extrapolation error: statistical se of the weighted quad-LS
     intercept + model-form spread max(|quad - linear|, grid jackknife)); every
     qualitative claim (entry ranking, per-row dominant destination) must be invariant
     between the two columns. This makes the argument threshold-free: the correction is
     the same order as its own error at the current grid, so neither column is
     privileged, and conclusions must not depend on the choice.
  3. The end-to-end audit is empirical held-out coverage (results plan D2); this
     certificate is the mechanistic decomposition, not the sole defence.
  4. Width: sd(gamma) trend reported, but the gamma -> 0 width limit is NOT identified
     from the grid (power-law regime) -- the small-N gold anchor (R2 step 3) calibrates
     width and family-wide systematics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GAMMA_STAR = 0.5
SCALARS = ("R_mu", "R_nu", "g_pi0")


def record_grid(z, phase):
    """Gamma grid read from the record's keys (records differ across campaigns)."""
    gs = {float(k.split("_")[1]) for k in z.files
          if k.startswith(f"{phase}_") and k.endswith("_mean")}
    return np.array(sorted(gs))


def load_phase(z, phase, grid):
    K = len(z[f"{phase}_populations"])
    d = {k: np.stack([z[f"{phase}_{g:g}_{k}"] for g in grid])
         for k in ("mean", "sd", "band", "cov_ot", "mcse")}
    d["gates"] = [json.loads(str(z[f"{phase}_{g:g}_gates"])) for g in grid]
    d["t_mean"] = np.array([z[f"{phase}_{g:g}_t_mean"] for g in grid])
    d["pops"] = z[f"{phase}_populations"]
    d["K"] = K
    ref_key = f"{phase}_{grid[0]:g}_r_mu_pi0"
    d["r_mu_pi0"] = float(z[ref_key]) if ref_key in z.files else None
    return d


def quad_intercept(grid, means):
    """Ehat_0 per observable: quadratic LS fit over the grid, evaluated at gamma = 0."""
    return np.array([np.polyval(np.polyfit(grid, means[:, j], 2), 0.0)
                     for j in range(means.shape[1])])


def wls_intercept(grid, y, deg, mcse=None):
    """Weighted-LS polynomial intercept at gamma = 0 and its statistical se."""
    X = np.vander(grid, deg + 1, increasing=True)
    W = np.eye(len(grid)) if mcse is None else np.diag(1 / np.maximum(mcse, 1e-12) ** 2)
    A = np.linalg.inv(X.T @ W @ X)
    return (A @ X.T @ W @ y)[0], np.sqrt(A[0, 0])


def extrapolation_error(grid, y, mcse):
    """Total error of Ehat_0: statistical se (+) model-form spread.

    Model-form = max(|quad - linear| over the full grid, half-range of the quad
    intercept under leave-one-grid-point-out jackknife). A cubic bracket is NOT used:
    four parameters on five noisy points overfit MC noise, and the intercept's
    extrapolation leverage inflates the spread beyond the bias being estimated.
    """
    q, se = wls_intercept(grid, y, 2, mcse)
    lin, _ = wls_intercept(grid, y, 1, mcse)
    jk = [wls_intercept(np.delete(grid, i), np.delete(y, i), 2, np.delete(mcse, i))[0]
          for i in range(len(grid))]
    model = max(abs(q - lin), (max(jk) - min(jk)) / 2)
    return q, float(np.hypot(se, model))


def scalar_indices(K):
    return {"R_mu": K * K, "R_nu": K * K + 1, "g_pi0": K * K + 2}


def live_mask(d, gstar_row=0):
    K = d["K"]
    return (d["mean"][gstar_row, :K * K] > 1e-4) & (d["band"][gstar_row, :K * K] > 1e-6)


def report_phase(phase, d, grid, gstar):
    K, idx = d["K"], scalar_indices(d["K"])
    mean, sd, band, cov = d["mean"], d["sd"], d["band"], d["cov_ot"]
    e0 = quad_intercept(grid, mean)
    live = live_mask(d)
    g0 = int(np.argmin(np.abs(grid - gstar)))

    print(f"\n=== {phase.upper()} (K={K}) ===")
    print("gates:", "; ".join(
        f"g={g:g}: rhat {gt['rhat_med']:.2f}/{gt['rhat_max']:.2f} "
        f"ess {gt['ess_med']:.0f} ebfmi {gt['ebfmi_min']:.3f}"
        for g, gt in zip(grid, d["gates"])))
    print("E[T] = E[||theta||^2]:", np.array2string(d["t_mean"], precision=1))

    print(f"\n{'obs':>6} {'segment':>12} {'FD slope':>10} {'-Cov avg':>10} {'ratio':>7}")
    for name, j in idx.items():
        for a in range(len(grid) - 1):
            fd = (mean[a + 1, j] - mean[a, j]) / (grid[a + 1] - grid[a])
            ident = -(cov[a, j] + cov[a + 1, j]) / 2
            print(f"{name:>6} {grid[a]:>5.1f}->{grid[a + 1]:<5.1f} {fd:>10.3f} "
                  f"{ident:>10.3f} {fd / ident if ident else np.nan:>7.2f}")

    print(f"\nEhat_0 (quad-LS intercept) and bias at gamma*={gstar}:")
    for name, j in idx.items():
        b = mean[g0, j] - e0[j]
        print(f"  {name:>6}: Ehat_g* {mean[g0, j]:>9.3f} -> Ehat_0 {e0[j]:>9.3f}  "
              f"bias {b:>+9.3f} nats  |b|/sd {abs(b) / sd[g0, j]:>7.2f}  "
              f"(first-order {-gstar * cov[g0, j]:>+9.3f})")

    print(f"\nlive table entries: {int(live.sum())}/{K * K}")
    if not live.any():
        print("  table certificate SKIPPED: observable vacuous for this pair")
        return
    tb = np.abs((mean[g0, :K * K] - e0[:K * K])[live])
    t1 = np.abs(gstar * cov[g0, :K * K][live])
    s0, b0 = sd[g0, :K * K][live], band[g0, :K * K][live]
    wr = np.median(sd[:, :K * K][:, live] / np.maximum(s0, 1e-12), axis=1)
    print(f"  certificate (bias = Ehat_g* - Ehat_0):")
    print(f"    |b|/sd    med {np.median(tb / s0):.3f}  max {np.max(tb / s0):.3f}  "
          f"(gate 0.25; first-order med {np.median(t1 / s0):.3f} -- noise-dominated)")
    print(f"    |b|/band  med {np.median(tb / b0):.3f}  max {np.max(tb / b0):.3f}")
    print(f"    |b| abs   med {np.median(tb):.5f}  max {np.max(tb):.5f}  "
          f"(entry means med {np.median(mean[g0, :K * K][live]):.4f})")
    print(f"    mcse/sd   med {np.median(d['mcse'][g0, :K * K][live] / s0):.3f}")
    p = np.polyfit(np.log(grid), np.log(wr), 1)
    print(f"    width: med sd ratio vs gamma* {np.array2string(wr, precision=3)} "
          f"~ gamma^{p[0]:.2f}; gamma->0 limit UNIDENTIFIED (gold anchor calibrates)")

    # Strengthened certificate: coverage budget + dual-column conclusion stability.
    lj = np.where(live)[0]
    e0w, err = zip(*(extrapolation_error(grid, mean[:, j], d["mcse"][:, j]) for j in lj))
    e0w, err = np.array(e0w), np.array(err)
    bshift = np.abs(mean[g0, lj] - e0w) / s0
    erosion_pp = 100 * 0.115 * bshift ** 2
    print(f"  coverage budget (95% nominal; erosion ~ 0.115 b^2, b in sd units):")
    print(f"    erosion med {np.median(erosion_pp):.2f}pp  max {np.max(erosion_pp):.2f}pp  "
          f"(budget 0.7pp <-> gate 0.25 sd)")
    print(f"  dual-column stability (Ehat_gamma* vs Ehat_0 +- err):")
    print(f"    extrapolation err/sd  med {np.median(err / s0):.3f}  "
          f"max {np.max(err / s0):.3f}  (same order as the bias -- neither column "
          f"privileged)")
    rk_g = np.argsort(np.argsort(-mean[g0, lj]))
    rk_0 = np.argsort(np.argsort(-e0w))
    n_top = min(10, len(lj))
    stable_top = set(np.argsort(-mean[g0, lj])[:n_top]) == set(np.argsort(-e0w)[:n_top])
    dom_changes = 0
    for row in range(K):
        rj = [i for i, j in enumerate(lj) if j // K == row]
        if len(rj) > 1:
            dom_changes += int(rj[int(np.argmax(mean[g0, lj][rj]))]
                               != rj[int(np.argmax(e0w[rj]))])
    print(f"    max rank shift {int(np.max(np.abs(rk_g - rk_0)))}, "
          f"top-{n_top} set identical: {stable_top}, "
          f"rows changing dominant destination: {dom_changes}")


def make_figure(S, D, grid, gstar, out_path):
    K = S["K"]
    idx = scalar_indices(K)
    e0s = quad_intercept(grid, S["mean"])
    live = live_mask(S)
    g0 = int(np.argmin(np.abs(grid - gstar)))
    gg = np.linspace(0, grid[-1] * 1.02, 100)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    fig.suptitle("E12 ridge-bias grid (N=500): location bias vs width bias vs gates",
                 fontsize=12.5)

    # A: scalar mean curves, quad fits extended to 0, Ehat_0 intercepts.
    a = ax[0, 0]
    cols = {"R_mu": "#1f6fb2", "R_nu": "#7aa8d0", "g_pi0": "#c0504d"}
    for name, j in idx.items():
        a.plot(grid, S["mean"][:, j], "o-", color=cols[name], ms=4,
               label=f"{name} (serum)")
        a.plot(gg, np.polyval(np.polyfit(grid, S["mean"][:, j], 2), gg), "--",
               color=cols[name], lw=0.8)
        a.plot(0, e0s[j], "*", color=cols[name], ms=11, mec="k", mew=0.4)
        a.plot(grid, D["mean"][:, j], "s:", color=cols[name], alpha=0.35, ms=3)
    a.plot([0, gstar], [e0s[idx["R_mu"]], S["mean"][g0, idx["R_mu"]]],
           lw=2.2, color="#1f6fb2", alpha=0.25)
    a.annotate("bias$(\\gamma^*)$ = $\\hat{E}_{\\gamma^*}-\\hat{E}_0$",
               xy=(gstar, S["mean"][g0, idx["R_mu"]]), xytext=(0.62, 34), fontsize=8.5,
               arrowprops=dict(arrowstyle="-", lw=0.6))
    a.axvline(gstar, color="0.6", lw=0.7, ls=":")
    if S.get("r_mu_pi0") is not None:
        # design-point asymptote: R_mu decays toward R_mu(pi^o) as gamma -> inf; the offset
        # above it at gamma* is the posterior's modelled spread, not bias
        a.axhline(S["r_mu_pi0"], color="0.35", lw=0.8, ls="--")
        a.text(grid[-1] * 0.99, S["r_mu_pi0"], "$R_\\mu(\\pi^o)$ floor", ha="right",
               va="bottom", fontsize=7.5, color="0.35")
    a.set_xlabel("$\\gamma$")
    a.set_ylabel("$\\hat{E}_\\gamma[O]$ (nats)")
    a.set_title("A  scalars: near-linear in $\\gamma$; stars = $\\hat{E}_0$ "
                "(quad-LS intercept, extrapolated)", fontsize=9.5)
    a.legend(fontsize=7.5, loc="upper left")
    a.set_xlim(-0.07, grid[-1] * 1.02)

    # Per-phase table statistics for panels B and C: location bias in sd units and the
    # width ratio, each on the phase's own live set.
    phase_styles = (("serum", S, "#1f6fb2", "#12365a"),
                    ("dox", D, "#e08a4f", "#a03d13"))
    tables = {}
    for name, P, faint, bold in phase_styles:
        lv = live_mask(P)
        e0p = quad_intercept(grid, P["mean"])
        tb = (P["mean"][:, :K * K][:, lv] - e0p[:K * K][lv]) / \
            np.maximum(P["sd"][g0, :K * K][lv], 1e-12)
        wr = P["sd"][:, :K * K][:, lv] / np.maximum(P["sd"][g0, :K * K][lv], 1e-12)
        tables[name] = (lv, tb, wr)

    # B: table-cell location bias, both phases overlaid.
    b = ax[0, 1]
    for name, P, faint, bold in phase_styles:
        lv, tb, _ = tables[name]
        b.plot(grid, tb, color=faint, alpha=0.10, lw=0.8)
        b.plot(grid, np.median(tb, axis=1), "o-", color=bold, lw=2, ms=4,
               label=f"median ({name}, {int(lv.sum())} entries)")
        b.plot(grid, np.percentile(np.abs(tb), 95, axis=1), "^--", color=bold,
               lw=1.2, ms=4, alpha=0.8, label=f"p95 |bias| ({name})")
    b.axhspan(-0.25, 0.25, color="#3a9d5c", alpha=0.12)
    b.axhline(0, color="0.5", lw=0.6)
    b.axvline(gstar, color="0.6", lw=0.7, ls=":")
    b.text(grid[-1] * 0.99, 0.235, "certificate gate $\\pm$0.25 sd", ha="right",
           va="top", fontsize=8, color="#2a7345")
    b.set_xlabel("$\\gamma$")
    b.set_ylabel("$(\\hat{E}_\\gamma[O] - \\hat{E}_0[O])\\; /\\; $sd$(\\gamma^*)$")
    b.set_title("B  LOCATION: live table entries per phase "
                "($\\hat{E}_0$ extrapolated, not sampled)", fontsize=9.5)
    b.legend(fontsize=7, loc="lower left")
    b.set_xlim(-0.07, grid[-1] * 1.02)

    # C: table-cell width ratio, both phases; the gamma -> 0 plateau is outside the data.
    c = ax[1, 0]
    for name, P, faint, bold in phase_styles:
        _, _, wr = tables[name]
        c.plot(grid, wr, color=faint, alpha=0.08, lw=0.8)
        med = np.median(wr, axis=1)
        c.plot(grid, med, "o-", color=bold, lw=2, ms=4, label=f"median entry ({name})")
        p = np.polyfit(np.log(grid), np.log(med), 1)
        c.plot(gg[gg > 0.05], np.exp(np.polyval(p, np.log(gg[gg > 0.05]))), "--",
               color=bold, lw=1.2, alpha=0.8,
               label=f"$\\gamma^{{{p[0]:.2f}}}$ ({name})")
    c.axvspan(0, gstar, color="#b2551f", alpha=0.08)
    c.text(gstar / 2, 0.35, "plateau\nunidentified:\ngold anchor\n(step 3)",
           ha="center", fontsize=8, color="#7c3a12")
    c.axvline(gstar, color="0.6", lw=0.7, ls=":")
    c.axhline(1.0, color="0.5", lw=0.6)
    c.set_xlabel("$\\gamma$")
    c.set_ylabel("sd$(\\gamma)$ / sd$(\\gamma^*)$")
    c.set_title("C  WIDTH: ridge narrows table posteriors; dox steeper than serum",
                fontsize=9.5)
    c.legend(fontsize=7.5)
    c.set_xlim(-0.07, grid[-1] * 1.02)
    c.set_ylim(0, 1.9)

    # D: gates vs gamma; no chain exists at gamma = 0 (improper target).
    d_ = ax[1, 1]
    for ph, dd, lw, alp in (("serum", S, 1.8, 1.0), ("dox", D, 1.2, 0.45)):
        d_.plot(grid, [g["rhat_max"] for g in dd["gates"]], "o-", color="#12365a",
                lw=lw, alpha=alp, ms=4, label=f"R-hat_max ({ph})")
    d_.plot([0.3, 0.3], [1.50, 1.31], "x", color="#a01313", ms=9, mew=2.2)
    d_.annotate("R1: $\\gamma=0.30$ FAILS\n(both chain seeds)", xy=(0.3, 1.4),
                xytext=(0.60, 1.44), fontsize=8, color="#a01313",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#a01313"))
    d_.axhline(1.25, color="#2a7345", lw=0.8, ls="--")
    d_.text(1.15, 1.262, "R1 gate 1.25", fontsize=8, color="#2a7345")
    d_.axvspan(0, 0.28, color="#a01313", alpha=0.07)
    d_.text(0.15, 1.13, "$\\gamma\\to 0$:\nimproper target,\nno invariant law\n"
            "-- gates undefined", ha="center", fontsize=7.5, color="#6b1010")
    d_.axvline(gstar, color="0.6", lw=0.7, ls=":")
    d2 = d_.twinx()
    for ph, dd, alp in (("serum", S, 1.0), ("dox", D, 0.45)):
        d2.plot(grid, [g["ebfmi_min"] for g in dd["gates"]], "s--", color="#b2551f",
                alpha=alp, ms=3.5, lw=1.1)
    d2.set_ylabel("eBFMI_min (dashed)", color="#b2551f", fontsize=9)
    d2.tick_params(axis="y", colors="#b2551f")
    d_.set_xlabel("$\\gamma$")
    d_.set_ylabel("R-hat_max")
    d_.set_title("D  GATES: mixing degrades as $\\gamma\\downarrow$; no chain exists "
                 "at 0", fontsize=9.5)
    d_.legend(fontsize=7.5, loc="center right")
    d_.set_xlim(-0.07, grid[-1] * 1.02)
    d_.set_ylim(1.0, 1.58)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nfigure -> {out_path}")


def main(run_dir, budget, gstar, exp):
    run_dir = Path(run_dir)
    z = np.load(run_dir / f"e12_ridge_bias_b{budget}_{exp}.npz", allow_pickle=True)
    grid = record_grid(z, "serum")
    print(f"record grid: {grid}  gamma* = {gstar}")
    S, D = load_phase(z, "serum", grid), load_phase(z, "dox", grid)
    for phase, d in (("dox", D), ("serum", S)):
        report_phase(phase, d, grid, gstar)
    make_figure(S, D, grid, gstar, run_dir / f"e12_bias_grid_{budget}_{exp}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--gamma-star", type=float, default=GAMMA_STAR)
    p.add_argument("--exp", type=str, default=None)
    a = p.parse_args()
    if a.run_dir is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from gmvae_confusion import latest_run
        a.run_dir = latest_run()
    main(a.run_dir, a.budget, a.gamma_star, a.exp)
