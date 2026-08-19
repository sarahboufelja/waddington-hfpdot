"""D2 predictive-coverage reader -- the calibration read-out of the hierarchical predictive.

Pure reader of d2_coverage triplet records (wadd_schema.D2_KEYS); needs no model, no
sampler and no data access -- records are self-describing. For every gated triplet,
each stored posterior composition p-hat_d receives ``LOCK["replicates"]`` multinomial
replicates at the record's real N; per-fate central equal-tailed intervals of the
pooled predictive samples are tested against the observed m-bar. Coverage at the
primary nominal level is the calibration claim (C2); the same statistic at the
secondary levels draws the calibration curve. The transport-blind baselines
(persistence, linear mixture) run under the IDENTICAL protocol from the record's
endpoint compositions, with matched predictive sample counts.

The protocol in ``LOCK`` is pre-registered in the claims ledger BEFORE the first run;
any post-lock change is a reported protocol violation.

Run from the repo root:
    python scripts/d2_coverage_reader.py [d2_dir]   # default: <run>/d2_coverage/latest
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

from fle_plan_bands import _INK, _MUTED
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_schema import validate_d2_record

#: predictor display order and colors (baselines in the house accent palette)
_PREDICTORS = (("hierwot", _INK), ("persistence", "#b45309"), ("mixture", "#2a7345"))

#: pre-registered protocol -- values mirror the ledger block; change = protocol violation
LOCK = dict(
    nominal=0.95,                      # primary claim level (the paper's CrI convention)
    levels=(0.50, 0.80, 0.90, 0.95),   # calibration-curve levels; secondary, same statistic
    replicates=20,                     # multinomial replicates per posterior draw:
    #                                    D*R = 10,000 predictive samples per triplet/fate,
    #                                    tail-quantile MC error negligible; baselines drawn
    #                                    at the SAME matched count from their single p
    noise_seed=20260818,               # reader-side rng, substreamed per triplet by
    #                                    (seed, round(100*t)) -- records replay independently
    median_floor=True,                 # AMENDED 2026-08-18: floored coverage (fates with
    #                                    predictive median > 1/N, per predictor) IS the
    #                                    coverage statistic; the all-fates rate is
    #                                    descriptive inventory only (see ledger amendment)
    gates=dict(rhat_med=1.05, rhat_max=1.25, ess_med=300),   # R1 (SIAM ledger); a failing
    #                                    triplet is EXCLUDED and reported, never silent
    readout=("coverage", "median_width"),   # per predictor per level: coverage AND
    #                                    sharpness -- coverage alone rewards wide
    #                                    intervals; no point-accuracy metrics (scope)
)

def tag_of(rec):
    t = np.asarray(rec["d2_days"], dtype=float)
    return f"D{t[0]:g}->D{t[1]:g}->D{t[2]:g}"


def load_records(d2_dir):
    """Schema-gated triplet records from a d2_coverage dir, sorted by t1.

    Every npz must pass validate_d2_record -- a half-written or stale-schema file is
    not a record and raises here rather than skewing coverage silently.
    """
    recs = []
    for npz in sorted(Path(d2_dir).glob("*.npz")):
        z = np.load(npz, allow_pickle=True)
        rec = {k: z[k] for k in z.files}
        validate_d2_record(rec)
        recs.append(rec)
    if not recs:
        raise SystemExit(f"no triplet records in {d2_dir}")
    recs.sort(key=lambda r: float(np.asarray(r["d2_days"], dtype=float)[0]))
    return recs


def gate_split(records):
    """(gated, excluded) by the locked R1 gates; exclusions carry their reasons.

    Excluded triplets are REPORTED, never silently dropped -- the exclusion list is
    part of the result (a gate failure voids the triplet, not the protocol).
    """
    g = LOCK["gates"]
    gated, excluded = [], []
    for rec in records:
        d = json.loads(np.asarray(rec["d2_diag"]).item())
        fails = [f"{k}={d[k]:.3g}>{g[k]:g}" for k in ("rhat_med", "rhat_max") if d[k] > g[k]]
        if d["ess_med"] < g["ess_med"]:
            fails.append(f"ess_med={d['ess_med']:.3g}<{g['ess_med']:g}")
        if fails:
            excluded.append((tag_of(rec), fails))
        else:
            gated.append(rec)
    return gated, excluded


def triplet_rng(rec):
    """The locked per-triplet substream: (noise_seed, round(100 * t) for the 3 stamps)."""
    t = np.asarray(rec["d2_days"], dtype=float)
    return np.random.default_rng([LOCK["noise_seed"], *(int(round(100 * x)) for x in t)])


def predictive_samples(rec):
    """{predictor: (S, K) pooled predictive fraction samples}, S = D * replicates.

    HierWOT: every stored posterior composition p-hat_d receives ``replicates``
    multinomial replicates at the record's real N (plan uncertainty + sampling noise).
    Baselines: the SAME S draws from their single point p (sampling noise only) --
    matched counts, identical noise layer, one shared substream per triplet.
    """
    comp = np.asarray(rec["d2_comp"], dtype=np.float64)
    comp = comp / comp.sum(axis=1, keepdims=True)
    t1 = np.asarray(rec["d2_comp_t1"], dtype=np.float64)
    t3 = np.asarray(rec["d2_comp_t3"], dtype=np.float64)
    alpha = float(rec["d2_alpha"])
    n = int(rec["d2_real_n"])
    R = LOCK["replicates"]
    S = len(comp) * R
    mix = (1.0 - alpha) * t1 + alpha * t3
    rng = triplet_rng(rec)
    return {"hierwot": rng.multinomial(n, np.repeat(comp, R, axis=0)) / n,
            "persistence": rng.multinomial(n, t1 / t1.sum(), size=S) / n,
            "mixture": rng.multinomial(n, mix / mix.sum(), size=S) / n}


def evaluate(rec):
    """Per predictor: resolvability mask + per-level covered flags and widths.

    Intervals are central equal-tailed quantiles of the pooled samples; covered is
    inclusive. The resolvability mask (predictive median > 1/N) is PER-PREDICTOR:
    each predictor's secondary coverage runs over the fates its own predictive deems
    resolvable (a shared mask would tie baseline coverage to our model; an
    observation-based mask would condition on the outcome). Denominators are
    reported with the rates.
    """
    m = np.asarray(rec["d2_real"], dtype=np.float64)
    n = int(rec["d2_real_n"])
    out = {}
    for name, S in predictive_samples(rec).items():
        ent = {"mask": np.median(S, axis=0) > 1.0 / n, "levels": {}}
        for lev in LOCK["levels"]:
            lo = np.quantile(S, (1.0 - lev) / 2.0, axis=0)
            hi = np.quantile(S, (1.0 + lev) / 2.0, axis=0)
            ent["levels"][lev] = {"covered": (lo <= m) & (m <= hi), "width": hi - lo}
        out[name] = ent
    return out


def aggregate(gated):
    """Pooled floored coverage and widths per predictor/level, with per-triplet detail.

    The AMENDED rule: coverage pools covered resolvable cells over all gated triplets
    (denominators carried); the all-fates count is kept as descriptive inventory only.
    """
    pool = {n: {lev: dict(cov=0, n=0, widths=[]) for lev in LOCK["levels"]}
            for n, _ in _PREDICTORS}
    inventory = {n: dict(cov=0, n=0) for n, _ in _PREDICTORS}
    detail = []
    for rec in gated:
        res = evaluate(rec)
        t = np.asarray(rec["d2_days"], dtype=float)
        row = dict(tag=tag_of(rec), t2=float(t[1]), alpha=float(rec["d2_alpha"]))
        for name, _ in _PREDICTORS:
            ent = res[name]
            k = ent["mask"]
            prim = ent["levels"][LOCK["nominal"]]
            inventory[name]["cov"] += int(prim["covered"].sum())
            inventory[name]["n"] += len(k)
            for lev in LOCK["levels"]:
                e = ent["levels"][lev]
                pool[name][lev]["cov"] += int(e["covered"][k].sum())
                pool[name][lev]["n"] += int(k.sum())
                pool[name][lev]["widths"] += list(e["width"][k])
            row[name] = dict(resolvable=int(k.sum()),
                             covered=int(prim["covered"][k].sum()),
                             med_width=float(np.median(prim["width"][k])) if k.any() else None)
        detail.append(row)
    return pool, inventory, detail


def figure(pool, detail, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    lev_grid = list(LOCK["levels"])
    for name, color in _PREDICTORS:
        emp = [pool[name][lev]["cov"] / max(pool[name][lev]["n"], 1) for lev in lev_grid]
        axes[0].plot(lev_grid, emp, "o-", color=color, lw=1.6, ms=4, label=name)
        t2 = [r["t2"] for r in detail]
        rate = [r[name]["covered"] / r[name]["resolvable"] if r[name]["resolvable"] else np.nan
                for r in detail]
        axes[1].plot(t2, rate, "o-", color=color, lw=1.2, ms=3.5, alpha=0.9)
        axes[2].plot(t2, [r[name]["med_width"] for r in detail], "o-", color=color,
                     lw=1.2, ms=3.5, alpha=0.9)
    axes[0].plot([0, 1], [0, 1], color=_MUTED, lw=0.8, ls="--", zorder=0)
    axes[0].set_xlim(0.45, 1.0); axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("nominal level"); axes[0].set_ylabel("empirical coverage (floored)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(LOCK["nominal"], color=_MUTED, lw=0.8, ls="--")
    axes[1].set_xlabel("held-out day t2"); axes[1].set_ylabel(f"coverage at {LOCK['nominal']:g}")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("held-out day t2"); axes[2].set_ylabel("median width (resolvable)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("D2 held-out predictive coverage -- floored statistic (ledger amendment "
                 "2026-08-18); baselines carry counting noise only", fontsize=9.5, color=_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "d2_coverage.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("d2_dir", nargs="?", default=None,
                   help="d2_coverage artifact dir (default: <latest run>/d2_coverage/latest)")
    a = p.parse_args()
    d2_dir = (Path(a.d2_dir) if a.d2_dir
              else Path(latest_run()) / "d2_coverage" / "latest").resolve()
    records = load_records(d2_dir)
    gated, excluded = gate_split(records)
    print(f"{len(records)} records | {len(gated)} gated, {len(excluded)} excluded")
    for tag, fails in excluded:
        print(f"  EXCLUDED {tag}: {', '.join(fails)}")
    pool, inventory, detail = aggregate(gated)
    print(f"\n{'predictor':>12} " + " ".join(f"{f'cov@{lev:g}':>12}" for lev in LOCK["levels"])
          + f" {'med width@' + format(LOCK['nominal'], 'g'):>14}")
    for name, _ in _PREDICTORS:
        cells = " ".join(f"{pool[name][lev]['cov']}/{pool[name][lev]['n']:<5}"
                         f"({pool[name][lev]['cov'] / max(pool[name][lev]['n'], 1):.2f})"
                         .rjust(12) for lev in LOCK["levels"])
        w = np.median(pool[name][LOCK["nominal"]]["widths"])
        print(f"{name:>12} {cells} {w:>14.4f}")
    inv = ", ".join(f"{n}: {v['cov']}/{v['n']}" for n, v in inventory.items())
    print(f"all-fates inventory (descriptive only, boundary artifact -- see ledger): {inv}")

    run_dir = d2_dir.parents[1]
    out_dir = artifact_dir(run_dir, "d2_coverage_reader",
                           config=dict(lock={k: (list(v) if isinstance(v, tuple) else v)
                                             for k, v in LOCK.items()},
                                       source=str(d2_dir)))
    figure(pool, detail, out_dir)
    result = dict(lock={k: (list(v) if isinstance(v, tuple) else v) for k, v in LOCK.items()},
                  amendment="2026-08-18 floored-primary (see claims ledger)",
                  source=str(d2_dir), n_records=len(records),
                  excluded=[dict(tag=t, reasons=r) for t, r in excluded],
                  pooled={n: {str(lev): dict(covered=pool[n][lev]["cov"], n=pool[n][lev]["n"],
                                             median_width=float(np.median(pool[n][lev]["widths"]))
                                             if pool[n][lev]["widths"] else None)
                              for lev in LOCK["levels"]} for n, _ in _PREDICTORS},
                  all_fates_inventory=inventory, triplets=detail)
    (out_dir / "d2_coverage.json").write_text(json.dumps(result, indent=2))
    print(f"d2 coverage reader -> {out_dir}")
