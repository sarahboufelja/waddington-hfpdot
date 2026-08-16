"""Dobrushin contraction with credible bands -- the C1 ladder under the posterior.

Pure reader of the growth-aware window records; the C1 -> C2 bridge. Per window, each of
``--chains`` independent-draw chains composes row-normalised per-draw fate tables from the
window root outward (one stored draw per step, sampled proportionally to posterior mass)
and evaluates the deterministic ladder's own statistic -- ``row_spread``, the largest TV
distance between two source fates' destination rows -- at every horizon. The result is
spread-vs-horizon with a 95% credible band, and per-chain log-linear fits give a credible
interval on the effective per-step retention delta_eff.

Overlaid: the deterministic composed curves from the lineage baselines at the locked
epsilon grid. The windowed horizon cap (max 14 steps) is a property of the RECORDS
(windowed execution is operational, not a framework limit); the deterministic CK ladder
anchors longer horizons, and reportability of any long-horizon composed claim is governed
by the CK non-identifiability gate regardless of windowing.

Outputs figure + json to ``dobrushin_bands/<stamp>/``. Run from the repo root:
    python scripts/dobrushin_bands.py [run_dir]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_e2e_hfpdot as E2E
from diag_blur_composition import row_spread
from fle_plan_bands import _INK, _MUTED
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir

QS = [0.025, 0.5, 0.975]


def _w(log_mass):
    m = np.exp(log_mass - np.max(log_mass))
    return m / m.sum()


def window_chains(rec, days, n_chains, rng):
    """(n_chains, n_steps) row-spread trajectories of composed sampled tables."""
    steps = []
    for a, b in zip(days, days[1:]):
        tag = f"{a:g}->{b:g}"
        T = np.asarray(rec[f"tables_{tag}"], dtype=float)
        rows = T.sum(axis=2, keepdims=True)
        steps.append((np.divide(T, rows, out=np.zeros_like(T), where=rows > 0),
                      _w(np.asarray(rec[f"table_log_mass_{tag}"]))))
    out = np.full((n_chains, len(steps)), np.nan)
    for n in range(n_chains):
        C = None
        for s, (R, w) in enumerate(steps):
            Rd = R[rng.choice(len(w), p=w)]
            C = Rd if C is None else C @ Rd
            live = C.sum(axis=1, keepdims=True)
            C = np.divide(C, live, out=np.zeros_like(C), where=live > 0)
            out[n, s] = row_spread(C)
    return out


def deterministic_curve(lineage, days):
    """Composed row spread of the deterministic per-pair tables along the window."""
    if lineage is None:
        return None
    pairs, M = np.asarray(lineage["pairs"]), np.asarray(lineage["matrices"])
    C, curve = None, []
    for a, b in zip(days, days[1:]):
        hit = np.where((pairs[:, 0] == a) & (pairs[:, 1] == b))[0]
        if not len(hit):
            return None
        T = np.asarray(M[hit[0]])
        rows = T.sum(axis=1, keepdims=True)
        R = np.divide(T, rows, out=np.zeros_like(T), where=rows > 0)
        C = R if C is None else C @ R
        live = C.sum(axis=1, keepdims=True)
        C = np.divide(C, live, out=np.zeros_like(C), where=live > 0)
        curve.append(row_spread(C))
    return np.asarray(curve)


def delta_eff(traj):
    """Per-chain retention exp(slope) of log-spread vs horizon (chains with >= 3 finite)."""
    out = []
    for row in traj:
        ok = np.isfinite(row) & (row > 0)
        if ok.sum() >= 3:
            n = np.arange(1, len(row) + 1)[ok]
            out.append(float(np.exp(np.polyfit(n, np.log(row[ok]), 1)[0])))
    return np.asarray(out)


def main(run_dir, budget, n_chains, seed):
    run_dir = Path(run_dir)
    lineage = {}
    for eps in (0.1, 0.2):
        child = E2E._adopt(run_dir, "lineage", dict(budget=budget, reg=eps),
                           E2E._lineage_complete)
        lineage[eps] = (np.load(child / "transition_tables.npz", allow_pickle=True)
                       if child else None)
    fig, axes = plt.subplots(1, len(E2E.WINDOWS), figsize=(4.0 * len(E2E.WINDOWS), 3.9),
                             sharey=True)
    result = {}
    rng = np.random.default_rng(seed)
    for ax, days in zip(np.atleast_1d(axes), E2E.WINDOWS):
        name = f"D{days[0]:g}-D{days[-1]:g}"
        cfg = dict(E2E.PF_DEFAULTS, days=list(days), budget=budget)
        child = E2E._adopt(run_dir, "particle_filter", cfg, E2E._pf_complete(days))
        if child is None:
            raise SystemExit(f"no complete record for {name}")
        rec = np.load(E2E._record_of(child), allow_pickle=True)
        traj = window_chains(rec, days, n_chains, rng)
        n = np.arange(1, traj.shape[1] + 1)
        lo, med, hi = np.nanquantile(traj, QS, axis=0)
        d_eff = delta_eff(traj)
        d_cri = (np.quantile(d_eff, QS).tolist() if len(d_eff) else None)
        ax.fill_between(n, lo, hi, color="#7c96c9", alpha=0.35, lw=0,
                        label="95% credible band")
        ax.plot(n, med, color=_INK, lw=2.0, label="posterior median")
        for eps, ls in ((0.1, "--"), (0.2, ":")):
            det = deterministic_curve(lineage[eps], days)
            if det is not None:
                ax.plot(n, det, color="#b45309", lw=1.4, ls=ls,
                        label=f"deterministic eps={eps:g}")
        ax.set_yscale("log")
        ax.set_xlabel("composed steps from window root")
        ax.set_title(name + (f"   delta_eff {d_cri[1]:.3f} [{d_cri[0]:.3f}, {d_cri[2]:.3f}]"
                             if d_cri else ""), fontsize=9, color=_INK)
        ax.spines[["top", "right"]].set_visible(False)
        result[name] = dict(horizons=n.tolist(), spread_cri=np.stack([lo, med, hi]).tolist(),
                            delta_eff_cri=d_cri,
                            deterministic={str(e): (deterministic_curve(lineage[e], days).tolist()
                                                    if deterministic_curve(lineage[e], days) is not None
                                                    else None) for e in (0.1, 0.2)})
        print(f"{name}: spread {med[0]:.3f} -> {med[-1]:.3f} over {len(n)} steps | "
              f"delta_eff {d_cri}")
    np.atleast_1d(axes)[0].set_ylabel("source-fate row spread (max pairwise TV)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Dobrushin contraction of composed sampled tables, with credible bands "
                 "(within-window horizons; deterministic CK ladder anchors longer spans)",
                 fontsize=10, color=_MUTED)
    out_dir = artifact_dir(run_dir, "dobrushin_bands",
                           config=dict(budget=budget, chains=n_chains, seed=seed))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "dobrushin_bands.png", dpi=160)
    plt.close(fig)
    (out_dir / "dobrushin_bands.json").write_text(json.dumps(result, indent=2))
    print(f"dobrushin bands -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--chains", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260816)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.chains, a.seed)
