"""E13 -- the gold anchor: R2 step 3, the measurement outside the chart+ridge family.

At N = 8 the plan has II*JJ = 64 coordinates, so ``full_rank`` samples S^o directly on the
orthant: no low-rank chart (no chart-restriction bias), no GL(r) gauge, no ridge (gamma = 0
is proper -- the impropriety lived in the chart's gauge orbits). Both arms run the SAME MALA
schema (kernel family, windowed warm-up, acceptance target, gates), differing ONLY in the
parametrisation, so any gold-vs-chart difference is attributable to the chart+ridge and not
to sampler heterogeneity. No tempering: the fallback ladder is MALA-internal (more draws,
more chains, smaller N).

Deliverables per phase and observable (K x K table entries, R_mu, R_nu, g_pi0):
  1. chart-restriction bias, read at gamma -> 0 along a small-gamma ladder: the ridge
     scale does NOT transfer across N (gamma* = 0.5 was ratified against the concentrated
     N = 500 posterior; at N = 8 the same gamma crushes a far wider target), so the chart
     arm runs gammas {0.05..0.5} and chart bias = lim_{gamma->0} Ehat_chart - Ehat_gold.
     At r = min(II,JJ) = 8 the chart is exhaustive, so the gamma -> 0 residual must vanish
     -- the built-in closure check on the ladder extrapolation itself. The gamma* row is
     kept as the production-scheme reference, not a chart-bias estimate.
  2. width calibration        sd_chart / sd_gold (the gamma -> 0 width limit the grid
     cannot identify).
  3. the direct gold read of E_{S^o}[R_mu] etc. -- no extrapolation -- arbitrating the
     model-limited small-gamma quadratic intercept.
Gold reproducibility is reported across two independent chain seeds.

Reproduce:  python scripts/experiments/e13_gold_anchor.py --budget 8 --exp gold
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import jax
import jax.numpy as jnp
import numpy as np

from e12_ridge_bias_grid import stage, gkl, _SHARPNESS
from device_info import print_device_banner
from gmvae_confusion import latest_run
from mcmc_diagnostics import MCMCDiagnostics
from wadd_lineage import transition_table
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

THIN_PER_CHAIN = 2000
_LIVE_MEAN, _LIVE_BAND = 1e-4, 1e-6


def build_prior(C, n):
    eps = float(C.max() - C.min()) / _SHARPNESS
    mu0, nu0 = np.full(n, 1 / n), 1.3 * np.full(n, 1 / n)
    prior = HFPDOTHyperprior(mu_0=mu0, nu_0=nu0, lambda_1=10.0, lambda_2=10.0,
                             lambda_I_1=0.5, lambda_I_2=0.5, cost_fn=C.reshape(-1),
                             epsilon=eps, support="positive_orthant")
    return prior, eps, mu0, nu0


def run_arm(C, Ws, Wt, pops, day_from, day_to, *, strategy, rank, ridge, centered,
            cfg):
    """One arm (gold full_rank or chart low_rank) -> observable statistics dict."""
    n = C.shape[0]
    prior, eps, mu0, nu0 = build_prior(C, n)
    kwargs = dict(cost=jnp.asarray(C), epsilon=eps)
    if strategy == "low_rank":
        kwargs.update(II=n, JJ=n, rank=rank, ridge=ridge,
                      ridge_center="warm_start" if centered else None)
    smp = MetropolisAdjustedLangevinSampler(
        target_log_prob_fn=prior.hyperprior_log_prob_fun,
        target_score_fn=prior.hyperprior_score_fun,
        shape=n * n, support="positive_orthant", sampling_strategy=strategy,
        initial_plan=prior.sinkhorn_init(), **kwargs, **cfg)
    pi0 = np.asarray(prior.sinkhorn_init()).reshape(-1)

    t0 = time.time()
    res = smp.sample(with_diagnostics=True, max_pi_coords=2000)
    d = res.diagnostics
    gates = dict(rhat_med=float(np.nanmedian(d.bulk_rhat)),
                 rhat_max=float(np.nanmax(d.bulk_rhat)),
                 ess_med=float(np.nanmedian(d.ess)), ebfmi_min=float(np.nanmin(d.ebfmi)))

    lat = np.asarray(jax.device_get(res.stacked_mala_states.stacked_latent_states))
    lat = lat.reshape(lat.shape[0], lat.shape[1], -1)[:, -cfg["num_samples"]:, :]
    keep = np.linspace(0, cfg["num_samples"] - 1, THIN_PER_CHAIN).astype(int)
    lat = lat[:, keep, :]
    Cn, D, m = lat.shape

    K = len(pops)
    obs = np.empty((Cn, D, K * K + 3))
    pis = np.asarray(jax.device_get(
        smp.support.to_constrained(jnp.asarray(lat.reshape(Cn * D, m))))).reshape(Cn, D, -1)
    for c in range(Cn):
        for i in range(D):
            P = pis[c, i].reshape(n, n)
            T = transition_table(P, Ws, Wt, pops, pops, day_from, day_to).matrix
            obs[c, i, :K * K] = T.reshape(-1)
            obs[c, i, K * K] = gkl(P.sum(1), mu0)
            obs[c, i, K * K + 1] = gkl(P.sum(0), nu0)
            obs[c, i, K * K + 2] = gkl(pis[c, i], pi0)
    mins = (time.time() - t0) / 60

    od = MCMCDiagnostics(constrained_chains=jnp.asarray(obs))
    ess_o = np.asarray(od.ess(jnp.asarray(obs)))
    flat = obs.reshape(-1, obs.shape[-1])
    mean, sd = flat.mean(axis=0), flat.std(axis=0, ddof=1)
    mcse = sd / np.sqrt(np.maximum(ess_o, 1.0))
    qs = np.quantile(flat[:, :K * K], [0.025, 0.975], axis=0)
    return dict(gates=gates, mean=mean, sd=sd, mcse=mcse, ess=ess_o,
                band=qs[1] - qs[0], minutes=mins)


def main(run_dir, budget, exp, gold_seeds, ranks, gammas, gold_cfg, chart_cfg, seed):
    print_device_banner()
    out = {}
    for day_from, day_to, phase in ((2.0, 2.5, "dox"), (12.0, 12.5, "serum")):
        C, Ws, Wt, pops = stage(run_dir, budget, day_from, day_to, seed)
        n, K = C.shape[0], len(pops)
        iR, iN, iG = K * K, K * K + 1, K * K + 2
        print(f"\n[{phase}] D{day_from:g}->D{day_to:g}  N={n}  plan dim {n * n}")

        golds = []
        for cs in gold_seeds:
            g = run_arm(C, Ws, Wt, pops, day_from, day_to, strategy="full_rank",
                        rank=None, ridge=0.0, centered=False,
                        cfg=dict(gold_cfg, seed=cs))
            golds.append(g)
            gt = g["gates"]
            print(f"  gold cs{cs}: rhat {gt['rhat_med']:.2f}/{gt['rhat_max']:.2f} "
                  f"ess {gt['ess_med']:.0f} ebfmi {gt['ebfmi_min']:.3f} | "
                  f"R_mu {g['mean'][iR]:.4f}+-{g['mcse'][iR]:.4f}  "
                  f"g_pi0 {g['mean'][iG]:.3f}  ({g['minutes']:.1f} min)", flush=True)
            for k, v in g.items():
                out[f"{phase}_gold{cs}_{k}"] = json.dumps(v) if k == "gates" else v
        # gold reproducibility + pooled reference
        dmu = abs(golds[0]["mean"][iR] - golds[1]["mean"][iR]) if len(golds) > 1 else 0.0
        se = np.hypot(golds[0]["mcse"][iR], golds[-1]["mcse"][iR])
        print(f"  gold seed agreement: |dR_mu| = {dmu:.4f} ({dmu / max(se, 1e-12):.1f} MCSE)")
        gmean = np.mean([g["mean"] for g in golds], axis=0)
        gsd = np.mean([g["sd"] for g in golds], axis=0)
        live = (gmean[:K * K] > _LIVE_MEAN) & \
               (np.mean([g["band"] for g in golds], axis=0) > _LIVE_BAND)
        print(f"  live table entries (gold): {int(live.sum())}/{K * K}")

        print(f"  {'rank':>4} {'gamma':>5} {'gates':>11} | {'R_mu bias':>9} {'/sd_g':>6} | "
              f"{'g_pi0 bias':>10} | {'tbl med|b|/sd_g':>15} {'tbl width med':>13}")
        for r in ranks:
            for gam in gammas:
                ch = run_arm(C, Ws, Wt, pops, day_from, day_to, strategy="low_rank",
                             rank=r, ridge=gam, centered=True,
                             cfg=dict(chart_cfg, seed=seed))
                for k, v in ch.items():
                    out[f"{phase}_chart_r{r}_g{gam:g}_{k}"] = \
                        json.dumps(v) if k == "gates" else v
                bias = ch["mean"] - gmean
                wr = ch["sd"][:K * K][live] / np.maximum(gsd[:K * K][live], 1e-12)
                tb = np.abs(bias[:K * K][live]) / np.maximum(gsd[:K * K][live], 1e-12)
                gt = ch["gates"]
                print(f"  {r:>4} {gam:>5.2f} {gt['rhat_med']:.2f}/{gt['rhat_max']:.2f} "
                      f"{gt['ess_med']:>5.0f} | {bias[iR]:>+9.4f} "
                      f"{abs(bias[iR]) / max(gsd[iR], 1e-12):>6.2f} | {bias[iG]:>+10.3f} | "
                      f"{np.median(tb):>15.3f} {np.median(wr):>13.3f}", flush=True)
        out[f"{phase}_populations"] = np.array(pops)
        np.savez(Path(run_dir) / f"e13_gold_anchor_b{budget}_{exp}.npz", **out)
    path = Path(run_dir) / f"e13_gold_anchor_b{budget}_{exp}.npz"
    np.savez(path, **out)
    print(f"\nrecord -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=8)
    p.add_argument("--exp", type=str, required=True)
    p.add_argument("--gold-seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5],
                   help="chart-arm ridge ladder: small gammas isolate chart bias at gamma->0;\
                         the production gamma* row is the scheme reference")
    p.add_argument("--gold-samples", type=int, default=300_000)
    p.add_argument("--chart-samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    gold_cfg = dict(num_samples=a.gold_samples, num_burnin=20_000, warm_up_steps=1_200,
                    step_size=0.02, num_parallel_chains=4)
    chart_cfg = dict(num_samples=a.chart_samples, num_burnin=5_000, warm_up_steps=1_200,
                     step_size=0.02, num_parallel_chains=4)
    main(a.run_dir or latest_run(), a.budget, a.exp, a.gold_seeds, a.ranks, a.gammas,
         gold_cfg, chart_cfg, a.seed)
