"""Chapman-Kolmogorov ladder: is the long-range lineage answer identified?

Under a Markov process with exactly estimated kernels, Chapman-Kolmogorov guarantees that every
decomposition of the same interval agrees:

    K(t0 -> t1)  =  K1 K2 ... Kn      for ANY subdivision into n steps.

So computing one fixed span at several granularities is a self-consistency test that needs no ground
truth. Any systematic drift with the number of steps is a direct lower bound on per-step estimation
error -- the entropic blur compounding, plus finite-sample and cost misspecification.

D5 -> D18 is 28 half-day steps and 28 = 2^2 * 7, so the ladder n = 1, 2, 4, 7, 14, 28 lands exactly on
observed timepoints with no interpolation.

Two readings, and neither recovers truth:

  * If log(spread) falls linearly in n, the multiplicative per-step blur model holds and its slope is
    the per-step retention factor.
  * The answer's dependence on n IS the finding. There is no n at which the estimator is unbiased --
    just as there is no such epsilon (small epsilon drives the plan to a permutation, large epsilon to
    the independent coupling; both are artifacts). The ladder quantifies NON-IDENTIFIABILITY, it does
    not extrapolate to a true value.

A drift also cannot be attributed: Chapman-Kolmogorov failure is a joint test of Markovianity AND exact
estimation. It does not license the conclusion that the cell process is non-Markov.

Run from the repo root:
    python scripts/diag_ck_ladder.py --eps 0.1
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

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae_confusion import latest_run
from diag_blur_composition import stage, row_spread, _plan, _table_from_kernel
from wadd_lineage import compose_plans


def _divisor_ladder(n_steps):
    """Subdivisions of the span that land exactly on observed timepoints."""
    return [d for d in range(1, n_steps + 1) if n_steps % d == 0]


def main(run_dir, budget, eps, reg_m, day_from, day_to):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    probe, _, _ = R.assemble(days[:1])
    model = GMVAENet(x_dim=probe[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    staged, pops = stage(model, cfg, days, budget, device)

    day_of = [s["day"] for s in staged]
    i0, i1 = day_of.index(day_from), day_of.index(day_to)
    span = i1 - i0
    print(f"run: {run_dir.name} | span D{day_from:g}->D{day_to:g} = {span} half-day steps "
          f"| eps={eps} | budget={budget} | device={device}")

    rows, tables = [], {}
    print(f"\n{'n_steps':>8} {'gap(days)':>10} {'spread':>8} {'vs direct':>10}  conv")
    for n in _divisor_ladder(span):
        stride = span // n
        idx = [i0 + k * stride for k in range(n + 1)]
        plans, ok_all = [], True
        for a, b in zip(idx, idx[1:]):
            P, ok = _plan(staged[a], staged[b], eps, reg_m)
            plans.append(P); ok_all &= ok
        K = compose_plans(plans) if n > 1 else plans[0] / plans[0].sum(axis=1, keepdims=True)
        T = _table_from_kernel(K, staged[i0], staged[i1], pops)
        tables[n] = T
        s = row_spread(T)
        # mean row-wise TV against the single-plan (n=1) answer: the Chapman-Kolmogorov discrepancy
        if 1 in tables:
            live = tables[1].sum(axis=1) > 0.5
            dv = float(np.mean(0.5 * np.abs(T[live] - tables[1][live]).sum(axis=1)))
        else:
            dv = 0.0
        rows.append((n, stride * 0.5, s, dv))
        print(f"{n:>8} {stride * 0.5:>10.1f} {s:>8.3f} {dv:>10.3f}  "
              f"{'ok' if ok_all else 'NOT-CONVERGED'}")

    arr = np.array(rows)
    n_arr, sp = arr[:, 0], arr[:, 2]
    good = sp > 0
    if good.sum() >= 3:
        b, a = np.polyfit(n_arr[good], np.log(sp[good]), 1)
        pred = a + b * n_arr[good]
        ss_res = float(((np.log(sp[good]) - pred) ** 2).sum())
        ss_tot = float(((np.log(sp[good]) - np.log(sp[good]).mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"\nlog-linear fit  log(spread) = {a:.3f} + {b:.4f}*n     R^2 = {r2:.3f}")
        print(f"  implied per-step retention  exp(slope) = {np.exp(b):.3f}")
        print(f"  intercept at n=0            exp({a:.3f}) = {np.exp(a):.3f}   "
              f"(NOT a bias-free value -- see module docstring)")

    out = run_dir / "lineage" / "ck_ladder.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, n_steps=arr[:, 0], gap_days=arr[:, 1], spread=arr[:, 2], vs_direct=arr[:, 3],
             epsilon=eps, day_from=day_from, day_to=day_to)
    print(f"\nseries -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--reg-m", type=float, nargs=2, default=(1.0, 50.0))
    p.add_argument("--from-day", type=float, default=5.0)
    p.add_argument("--to-day", type=float, default=18.0)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.eps, tuple(a.reg_m), a.from_day, a.to_day)
