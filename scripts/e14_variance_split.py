"""E14 -- the law-of-total-variance split of transition-table uncertainty.

Pure reader of the growth-aware window records. Per pair and per table entry, the total
posterior variance decomposes EXACTLY (by conditioning, not by assuming independence):

    Var[T] = E_draws[ Var_m[T | draw] ]  +  Var_draws[ E_m[T | draw] ]
             identity term (stored          plan term (variance of the stored
             ``table_identity_var``)        ``table_identity_mean`` across draws)

Both expectations over draws are mass-weighted. The identity resamples carry the z -> W
channel of the single-source dependence exactly (Rao-Blackwellised: c integrated
analytically); the z -> cost -> plan channel is FROZEN by construction and is the
documented scope limit of this measurement.

Outputs: per-pair split summaries + the identity-share trajectory across developmental
time (figure + json) to ``e14_split/<stamp>/``.

Run from the repo root:
    python scripts/e14_variance_split.py [run_dir]
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
from fle_plan_bands import _INK, _MUTED
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir


def _w(log_mass):
    m = np.exp(log_mass - np.max(log_mass))
    return m / m.sum()


def split_pair(rec, tag):
    """(identity, plan) per-entry variance terms, mass-weighted over draws."""
    w = _w(np.asarray(rec[f"table_log_mass_{tag}"]))
    iv = np.asarray(rec[f"table_identity_var_{tag}"])
    im = np.asarray(rec[f"table_identity_mean_{tag}"])
    identity = np.tensordot(w, iv, axes=1)
    mbar = np.tensordot(w, im, axes=1)
    plan = np.tensordot(w, (im - mbar) ** 2, axes=1)
    return identity, plan


def main(run_dir, budget):
    run_dir = Path(run_dir)
    rows = []
    for days in E2E.WINDOWS:
        cfg = dict(E2E.PF_DEFAULTS, days=list(days), budget=budget)
        child = E2E._adopt(run_dir, "particle_filter", cfg, E2E._pf_complete(days))
        if child is None:
            raise SystemExit(f"no complete record for D{days[0]:g}-D{days[-1]:g}")
        rec = np.load(E2E._record_of(child), allow_pickle=True)
        pops = [str(p) for p in rec["populations"]]
        for a, b in zip(days, days[1:]):
            tag = f"{a:g}->{b:g}"
            identity, plan = split_pair(rec, tag)
            live = identity + plan > 0
            share = identity[live] / (identity + plan)[live]
            peak = np.unravel_index(np.argmax(np.where(live, identity, 0.0)), identity.shape)
            rows.append(dict(
                day=b, pair=tag, window=f"D{days[0]:g}-D{days[-1]:g}",
                share_median=float(np.median(share)),
                share_p90=float(np.quantile(share, 0.90)),
                share_of_total_variance=float(identity[live].sum()
                                              / (identity + plan)[live].sum()),
                peak_entry=f"{pops[peak[0]]}->{pops[peak[1]]}",
                peak_share=float((identity / np.maximum(identity + plan, 1e-300))[peak]),
                identity=identity, plan=plan))

    peak_row = max(rows, key=lambda r: r["share_p90"])
    fig, (ax, hx) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                 gridspec_kw=dict(width_ratios=(1.55, 1.0)))
    day = [r["day"] for r in rows]
    ax.plot(day, [r["share_p90"] for r in rows], color="#b45309", lw=2.0,
            label="p90 over live entries")
    ax.plot(day, [r["share_of_total_variance"] for r in rows], color=_INK, lw=2.0,
            label="share of total table variance")
    ax.plot(day, [r["share_median"] for r in rows], color=_MUTED, lw=1.5, ls="--",
            label="median over live entries")
    for days in E2E.WINDOWS[1:]:
        ax.axvline(days[0], color=_MUTED, lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("day"); ax.set_ylabel("identity share of Var[T]")
    ax.set_title("Identity term of the table-variance split (frozen-cost scope)",
                 fontsize=10, color=_INK)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    identity, plan = peak_row["identity"], peak_row["plan"]
    share_map = identity / np.maximum(identity + plan, 1e-300)
    im = hx.imshow(share_map, cmap="Blues", vmin=0, vmax=max(share_map.max(), 0.2))
    hx.set_xticks(range(len(pops))); hx.set_yticks(range(len(pops)))
    hx.set_xticklabels(pops, rotation=90, fontsize=6)
    hx.set_yticklabels(pops, fontsize=6)
    hx.set_title(f"per-entry identity share, {peak_row['pair']} (peak pair)",
                 fontsize=9, color=_INK)
    fig.colorbar(im, ax=hx, shrink=0.8)

    out_dir = artifact_dir(run_dir, "e14_split", config=dict(budget=budget))
    fig.tight_layout()
    fig.savefig(out_dir / "e14_variance_split.png", dpi=160)
    plt.close(fig)
    payload = [{k: v for k, v in r.items() if k not in ("identity", "plan")} for r in rows]
    (out_dir / "e14_split.json").write_text(json.dumps(
        dict(scope="frozen-cost: z->W channel exact, z->cost->plan channel deferred",
             pairs=payload), indent=2))

    print(f"{'pair':>12} {'share med':>10} {'p90':>7} {'of total var':>13}  peak entry")
    for r in rows:
        print(f"{r['pair']:>12} {r['share_median']:>10.4f} {r['share_p90']:>7.3f} "
              f"{r['share_of_total_variance']:>13.4f}  {r['peak_entry']} "
              f"({r['peak_share']:.2f})")
    print(f"\ne14 split -> {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget)
