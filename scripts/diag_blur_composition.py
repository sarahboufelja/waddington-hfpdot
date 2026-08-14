"""Diagnostic: how much of the lineage memory loss is entropic blur rather than biology?

Composing transport plans multiplies their Dobrushin ergodicity coefficients, so source-fate
distinction decays geometrically along a trajectory. That decay mixes two causes which a single
composed plan cannot separate:

  * real developmental stochasticity -- a cell's descendants genuinely spread across fates;
  * entropic blur -- the regularisation ``epsilon`` smears each plan, and chaining n plans compounds
    that smearing n times.

Two contrasts separate them.

  DIRECT vs COMPOSED   one plan from t0 to t1 carries ONE blur; the chain of consecutive plans over
                       the same span carries one per step. Both describe the same biology, so the
                       difference between them is the compounding artifact.
  EPSILON SWEEP        if the composed separation keeps rising as epsilon falls, the loss is
                       epsilon-dominated; if it plateaus, what remains is biological.

Reports the source-fate row spread: the largest total-variation distance between the destination
distributions of two different source fates. 1.0 = fates fully distinguishable at the far end,
0.0 = the lineage has forgotten where it started.

Run from the repo root:
    python scripts/diag_blur_composition.py
    python scripts/diag_blur_composition.py --budget 300 --eps 0.02 0.05 0.1 0.2
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_dim_reduction import CoverageSubsampler
from wadd_ot import CellCloud, GaussianW2, uot_plan
from wadd_lineage import transition_table, compose_plans, population_distributions


def row_spread(T):
    """Largest total-variation distance between two source fates' destination rows.

    The quantity Dobrushin's coefficient bounds: how distinguishable the futures of two different
    starting fates still are. Rows with no mass are skipped (an absent fate has no future).
    """
    live = T[T.sum(axis=1) > 0.5]
    if len(live) < 2:
        return float("nan")
    return max(0.5 * np.abs(live[i] - live[j]).sum()
               for i in range(len(live)) for j in range(i + 1, len(live)))


def stage(model, cfg, days, budget, device):
    """Embed and subsample every day once; the plans below are recomputed per epsilon."""
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))
    matrices, memberships, pops = R.assemble(days)
    sub = CoverageSubsampler()
    staged = []
    for exp, memb in zip(matrices, memberships):
        post = embedder.embed(exp)
        idx = sub.indices(post, min(budget, len(post)))
        staged.append({"day": exp.day, "posterior": post.select(idx),
                       "membership": memb.align_populations(pops).matrix[idx]})
    return staged, pops


def _cloud(s):
    return CellCloud(means=s["posterior"].means, stds=s["posterior"].stds)


def _plan(a, b, eps, reg_m):
    """UOT plan plus a validity flag.

    Two checks, because the failure mode here is not slow convergence but numerical DIVERGENCE: POT's
    stabilised unbalanced Sinkhorn is only partially log-domain, and below about ``eps = 0.05`` on this
    cost it blows up -- plans come back with total mass of 1e4 to 1e25 instead of ~1, and raising the
    iteration cap makes it worse. A returned plan is trusted only if Sinkhorn did not warn AND its mass
    is finite and near 1; without the mass check a diverged plan yields confident nonsense downstream.
    (POT 0.9.6 offers no ``sinkhorn_log`` for the unbalanced problem, so low ``eps`` needs a different
    solver -- epsilon-scaling, majorisation-minimisation, or our own log-domain implementation.)
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        P = uot_plan(_cloud(a), _cloud(b), cost=GaussianW2(), reg=eps, reg_m=reg_m)
        quiet = not any("not converge" in str(x.message) for x in w)
    mass = float(P.sum())
    sane = bool(np.isfinite(P).all()) and 0.5 < mass < 2.0
    return P, (quiet and sane)


def _table_from_kernel(K, a, b, pops):
    """Population table from an already-composed cell-level kernel."""
    T = population_distributions(a["membership"]) @ K @ b["membership"]
    s = T.sum(axis=1, keepdims=True)
    return np.divide(T, s, out=np.zeros_like(T), where=s > 0)


def main(run_dir, budget, eps_list, reg_m, days_spec=None):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = R._parse_days(days_spec) if days_spec else cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run: {run_dir.name} | days: {len(days)} | budget: {budget} | device: {device}")

    probe, _, _ = R.assemble(days[:1])
    model = GMVAENet(x_dim=probe[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    staged, pops = stage(model, cfg, days, budget, device)

    day_of = [s["day"] for s in staged]
    start = next(i for i, s in enumerate(staged) if (s["membership"].sum(axis=1) > 0).sum() >= 10)
    spans = [(start, i) for i in (start + 4, start + 10, start + 18, len(staged) - 1)
             if i < len(staged)]
    print(f"lineage root: D{day_of[start]:g}\n")

    print(f"{'epsilon':>8} {'span':>16} {'steps':>6} {'direct':>8} {'composed':>9} {'ratio':>7}  conv")
    results = []
    for eps in eps_list:
        plans, conv_all = [], True
        for a, b in zip(staged, staged[1:]):
            P, ok = _plan(a, b, eps, reg_m)
            plans.append(P); conv_all &= ok
        for (i, j) in spans:
            a, b = staged[i], staged[j]
            n_steps = j - i
            Pd, ok_d = _plan(a, b, eps, reg_m)                       # ONE blur
            Td = transition_table(Pd, a["membership"], b["membership"], pops, pops).matrix
            Kc = compose_plans(plans[i:j])                            # n_steps blurs
            Tc = _table_from_kernel(Kc, a, b, pops)
            sd, sc = row_spread(Td), row_spread(Tc)
            ratio = sc / sd if sd and np.isfinite(sd) and sd > 0 else float("nan")
            flag = "ok" if (ok_d and conv_all) else "NOT-CONVERGED"
            print(f"{eps:>8.3f} D{day_of[i]:g}->D{day_of[j]:<7g} {n_steps:>6} "
                  f"{sd:>8.3f} {sc:>9.3f} {ratio:>7.3f}  {flag}")
            results.append((eps, day_of[i], day_of[j], n_steps, sd, sc))
        print()

    arr = np.array(results)
    out = artifact_dir(run_dir, "blur_diagnostic") / "blur_diagnostic.npz"
    np.savez(out, epsilon=arr[:, 0], day_from=arr[:, 1], day_to=arr[:, 2], steps=arr[:, 3],
             spread_direct=arr[:, 4], spread_composed=arr[:, 5])

    # Effective per-step Dobrushin coefficient implied by the composed decay, at each epsilon.
    print("implied effective Dobrushin coefficient per step (from the composed chain):")
    for eps in eps_list:
        m = arr[arr[:, 0] == eps]
        longest = m[np.argmax(m[:, 3])]
        n, sc, sd = longest[3], longest[5], longest[4]
        if sd > 0 and sc > 0:
            print(f"  epsilon={eps:<6.3f} over {int(n)} steps: delta_eff = "
                  f"{(sc / sd) ** (1.0 / n):.3f}   (composed/direct = {sc / sd:.4f})")
    print(f"\nseries -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--days", default=None)
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument("--reg-m", type=float, nargs=2, default=(1.0, 50.0))
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.eps, tuple(a.reg_m), a.days)
