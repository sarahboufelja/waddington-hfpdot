"""Transition tables from genuine HFPD-OT hyperprior samples, not a single deterministic plan.

Everything so far has read lineage off one point-estimate plan, so no result has carried a credible
interval. This draws plans from the optimal hyperprior itself,

    S^o(pi|K) ~ exp(-lambda_1 KL(mu||mu_0)) S~(pi|K) exp(-lambda_2 KL(nu||nu_0)),

and turns each draw into a population transition table, giving a distribution over tables rather than
a single one.

The specific question it answers: **how wide is the plan uncertainty when the marginals are tightly
pinned?** Replicate lanes put the marginal radius at about 0.009 nats, which is small -- but a small
radius constrains the MARGINALS, not the COUPLING. As eta -> 0 the hyperprior concentrates on the
transport polytope Pi(mu_0, nu_0) (paper Remark 1, Eq 26), which is a set of dimension ~m^2-2m+1, not
a point, and S^o remains spread over it with the temperature of the KL(pi||pi_I) term (coefficient 1
in the parametric form, Eq 32). So plan uncertainty survives a well-determined marginal, and its size
is an empirical question.

Rather than solve lambda(eta) -- which needs the sampler in the loop -- lambda is swept directly, as in
the paper's own descriptive analysis (Fig 6). Large lambda means tightly-held marginals.

Deliberately small: the plan has m^2 dimensions, and MALA on this support degrades badly with
dimension, so the budget is tens of cells per timepoint. This builds intuition; it is not the
production path.

Run from the repo root:
    python scripts/diag_plan_uncertainty.py --budget 40 --lambdas 1 10 100
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Pin JAX to CPU with one host device per chain, BEFORE anything imports jax. On a multi-GPU box JAX
# otherwise claims both devices and the sampler's per-chain vmap turns into a multi-device collective,
# which fails outright (NCCL ncclAlltoAll ... unhandled system error). The GMVAE embedding below still
# uses CUDA through torch; only the sampler is pinned.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_dim_reduction import CoverageSubsampler
from wadd_ot import CellCloud, GaussianW2
from wadd_lineage import transition_table
from diag_blur_composition import row_spread
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior


def main(run_dir, budget, lambdas, lam_I, eps, day_from, day_to, n_samples, chains, seed):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    matrices, memberships, pops2 = R.assemble([day_from, day_to])
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))
    sub = CoverageSubsampler()

    staged = []
    for exp, memb in zip(matrices, memberships):
        post = embedder.embed(exp)
        idx = sub.indices(post, min(budget, len(post)))
        staged.append({"post": post.select(idx),
                       "W": memb.align_populations(pops2).matrix[idx]})
    a, b = staged
    II, JJ = len(a["post"]), len(b["post"])
    cost = GaussianW2()(CellCloud(a["post"].means, a["post"].stds),
                        CellCloud(b["post"].means, b["post"].stds))
    mu_0 = np.full(II, 1.0 / II)
    nu_0 = np.full(JJ, 1.0 / JJ)
    print(f"D{day_from:g} -> D{day_to:g} | plan {II}x{JJ} = {II * JJ} dims | eps={eps} | "
          f"device={device}")

    print(f"\n{"lambda":>8} {"draws":>8} {'R-hat':>7} {'ESS':>7} {'spread mean':>12} "
          f"{'spread sd':>10} {'max entry sd':>13}")
    results = []
    for lam in lambdas:
        prior = HFPDOTHyperprior(mu_0=mu_0, nu_0=nu_0, lambda_1=lam, lambda_2=lam,
                                 lambda_I_1=lam_I, lambda_I_2=lam_I,
                                 cost_fn=np.asarray(cost).reshape(-1), epsilon=eps,
                                 support="simplex")
        sampler = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            shape=II * JJ, support="simplex", num_parallel_chains=chains,
            num_samples=n_samples, num_burnin=n_samples // 2, warm_up_steps=500,
            step_size=0.01, seed=seed, initial_plan=prior.sinkhorn_init())
        res = sampler.sample(with_diagnostics=True)
        plans = np.asarray(res.samples)                          # (chains, draws, II*JJ)
        plans = plans.reshape(-1, plans.shape[-1])               # pool chains
        diag = res.diagnostics
        rhat = float(np.asarray(diag.bulk_rhat_max)) if diag is not None else float("nan")
        ess = float(np.asarray(diag.ess_min)) if diag is not None else float("nan")

        tabs = []
        for p in plans[:: max(1, len(plans) // 300)]:            # thin for cost
            T = transition_table(np.asarray(p).reshape(II, JJ), a["W"], b["W"], pops, pops,
                                 day_from, day_to)
            tabs.append(T.matrix)
        tabs = np.stack(tabs)
        spreads = np.array([row_spread(t) for t in tabs])
        spreads = spreads[np.isfinite(spreads)]
        live = tabs[0].sum(axis=1) > 0.5
        entry_sd = float(tabs[:, live].std(axis=0).max()) if live.any() else float("nan")
        print(f"{lam:>8.2f} {len(plans):>8} {rhat:>7.2f} {ess:>7.0f} "
              f"{spreads.mean():>12.3f} {spreads.std():>10.3f} {entry_sd:>13.3f}")
        results.append((lam, len(plans), rhat, ess,
                        float(spreads.mean()), float(spreads.std()), entry_sd))

    arr = np.array(results)
    out = run_dir / "plan_uncertainty.npz"
    np.savez(out, lam=arr[:, 0], draws=arr[:, 1], rhat=arr[:, 2], ess=arr[:, 3],
             spread_mean=arr[:, 4], spread_sd=arr[:, 5], entry_sd=arr[:, 6],
             populations=np.array(pops))
    print(f"\nspread sd is the plan-driven uncertainty in the fate-separation statistic;")
    print(f"max entry sd is the largest standard deviation of any transition-table entry.")
    print(f"series -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p.add_argument("--lam-I", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--from-day", type=float, default=12.0)
    p.add_argument("--to-day", type=float, default=12.5)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.lambdas, a.lam_I, a.eps,
         a.from_day, a.to_day, a.samples, a.chains, a.seed)
