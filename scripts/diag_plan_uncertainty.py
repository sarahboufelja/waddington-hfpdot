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

Deliberately small: the budget is tens of cells per timepoint. Even under the ridge
parametrisation (``--strategy low_rank``, which samples r(II+JJ) free coordinates rather than all
II*JJ plan entries) this builds intuition; it is not the production path.

Run from the repo root:
    python scripts/diag_plan_uncertainty.py --budget 40 --lambdas 1 10 100
"""
import argparse
import json
import os
import sys
from pathlib import Path

# These MUST be set before anything imports jax, and that is easy to get wrong here: langevin_sampler
# sets them itself at its own line 6, but `import ot` (POT probes for a jax backend at import time)
# pulls jax in first, so by the time the sampler module is reached jax has already initialised and the
# flags are inert. The symptom is a hard failure inside the chain sharding
# (NCCL ncclAlltoAll ... unhandled system error), not a graceful fallback. Keeping them here preserves
# the sampler's design: chains are sharded across BOTH GPUs via NamedSharding on a ("chains",) mesh.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np
import torch

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_dim_reduction import RandomSubsampler
from wadd_ot import CellCloud, GaussianW2
from wadd_lineage import transition_table
from diag_blur_composition import row_spread
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior


def main(run_dir, budget, lambdas, lam_I, eps, day_from, day_to, n_samples, chains, seed,
         strategy, rank, burnin, warmup, step, warm_sigma=(0.1, 0.3), support="simplex"):
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
    # Random, not CoverageSubsampler: FPS is deliberately coverage-biased (tail-heavy), which
    # contradicts the uniform mu_0 below. A uniform subsample IS the day's empirical measure, so
    # uniform weights over it are unbiased for the composition and the atoms remain actual cells.
    # A composition-faithful landmark quantisation (FPS + Voronoi occupancy masses) is the
    # second-step upgrade if budgets shrink below the rare-fate scale.
    sub = RandomSubsampler(seed=seed)

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
    free = II * JJ if strategy == "full_rank" else rank * (II + JJ)
    print(f"D{day_from:g} -> D{day_to:g} | plan {II}x{JJ} = {II * JJ} entries | "
          f"{strategy} -> {free} sampled dims | eps={eps} | device={device}")

    print(f"\n{'lambda':>8} {'draws':>8} {'R-hat':>7} {'ESS':>7} {'liveR':>7} {'liveESS':>8} "
          f"{'acc':>5} {'eBFMI':>7} {'spread mean':>12} {'spread sd':>10} {'max entry sd':>13}")
    results, per_coord = [], {}
    for lam in lambdas:
        prior = HFPDOTHyperprior(mu_0=mu_0, nu_0=nu_0, lambda_1=lam, lambda_2=lam,
                                 lambda_I_1=lam_I, lambda_I_2=lam_I,
                                 cost_fn=np.asarray(cost).reshape(-1), epsilon=eps,
                                 support=support)
        # `full_rank` is the constructor DEFAULT and is the wrong parametrisation here: it samples
        # all II*JJ plan entries, where MALA degrades badly. `low_rank` (the ridge path) samples
        # r(II+JJ) free coordinates of a rank-r log-correction to the Gibbs kernel instead -- two
        # orders of magnitude fewer dimensions, and the only setting under which this sampler has
        # ever reported usable diagnostics (experiments.md E10: ESS 899 / R-hat 1.43).
        low_rank = {} if strategy == "full_rank" else {
            "II": II, "JJ": JJ, "rank": rank, "cost": jnp.asarray(cost), "epsilon": eps}
        sampler = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            shape=II * JJ, support=support, sampling_strategy=strategy,
            num_parallel_chains=chains,
            num_samples=n_samples, num_burnin=burnin, warm_up_steps=warmup,
            step_size=step, seed=seed, initial_plan=prior.sinkhorn_init(),
            warm_start_sigma=warm_sigma, **low_rank)
        res = sampler.sample(with_diagnostics=True)
        # device_get, not np.asarray: chains are sharded across GPUs on a ("chains",) mesh, and
        # pulling a sharded array over the host boundary implicitly is what breaks.
        plans = np.asarray(jax.device_get(res.samples))          # (chains, draws, II*JJ)
        plans = plans.reshape(-1, plans.shape[-1])               # pool chains
        d = res.diagnostics
        rhat, ess, ebfmi = (jax.device_get((d.bulk_rhat_max, d.ess_min, d.ebfmi_min))
                            if d is not None else (np.nan,) * 3)
        # Per-coordinate view: max/min summaries are dominated by numerically-dead coordinates
        # (pi entries ~ e^-C/eps never move at float precision -> zero within-chain variance), so
        # keep the whole profile plus each coordinate's mean mass to separate dead from live.
        acc = np.asarray(jax.device_get(res.stacked_mala_states.stacked_num_accepted_samples))
        acc_rate = acc.reshape(len(acc), -1)[:, -1] / (n_samples + burnin)
        per_coord[lam] = {"rhat": np.asarray(d.bulk_rhat), "ess": np.asarray(d.ess),
                          "mass": plans.mean(axis=0), "accept": acc_rate}
        live_mask = per_coord[lam]["mass"] > 1.0 / (10 * II * JJ)
        live_rhat = float(np.nanmax(per_coord[lam]["rhat"][live_mask]))
        live_ess = float(np.nanmin(per_coord[lam]["ess"][live_mask]))

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
        print(f"{lam:>8.2f} {len(plans):>8} {rhat:>7.2f} {ess:>7.0f} {live_rhat:>7.2f} "
              f"{live_ess:>8.0f} {acc_rate.mean():>5.2f} {ebfmi:>7.3f} "
              f"{spreads.mean():>12.3f} {spreads.std():>10.3f} {entry_sd:>13.3f}")
        results.append((lam, len(plans), rhat, ess, live_rhat, live_ess,
                        float(acc_rate.mean()), ebfmi,
                        float(spreads.mean()), float(spreads.std()), entry_sd))

    arr = np.array(results)
    out = run_dir / "plan_uncertainty.npz"
    coord_dump = {f"{k}_{lam:g}": v[k] for lam, v in per_coord.items()
                  for k in ("rhat", "ess", "mass", "accept")}
    np.savez(out, lam=arr[:, 0], draws=arr[:, 1], rhat=arr[:, 2], ess=arr[:, 3],
             live_rhat=arr[:, 4], live_ess=arr[:, 5], accept=arr[:, 6],
             ebfmi=arr[:, 7], spread_mean=arr[:, 8], spread_sd=arr[:, 9], entry_sd=arr[:, 10],
             strategy=np.array(strategy), rank=np.array(rank), populations=np.array(pops),
             **coord_dump)
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
    # sampler budget defaults = the README bridge config (experiments.md E3), under which the ridge
    # baselines (simplex R-hat 1.12/1.04, ESS 76/1596) were recorded -- keep comparisons faithful
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--burnin", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=600)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategy", default="low_rank",
                   choices=["low_rank", "low_rank_section", "full_rank"])
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--warm-sigma", type=float, nargs=2, default=(0.1, 0.3),
                   help="per-chain warm-start perturbation (lo hi); collapse to ~0 to test "
                        "whether chain disagreement is initialisation-basin sensitivity")
    p.add_argument("--support", default="simplex", choices=["simplex", "positive_orthant"],
                   help="balanced (simplex) vs unbalanced (orthant); the E8/E9 ridge baselines "
                        "were recorded on positive_orthant")
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.lambdas, a.lam_I, a.eps,
         a.from_day, a.to_day, a.samples, a.chains, a.seed, a.strategy, a.rank,
         a.burnin, a.warmup, a.step, tuple(a.warm_sigma), a.support)
