"""E11 -- the gauge ridge at production scale: real cost, strong-lambda regime, N per CLI.

E10/E11 background: the ridge low-rank sampler had been run with ``ridge = 0.0`` in every recorded
experiment, leaving the GL(r) gauge of the factor parametrisation unpenalised -- exact flat
directions, a saddle at the warm start (46/120 non-positive Hessian directions at N=30), and the
never-strictly-converged ceiling across the whole E-series. At N=30 on the real target
(GaussianW2 cost, lambda=10, lambda_I=0.5, sharpness (C_max-C_min)/eps = 33, positive orthant),
switching the ridge on collapsed the warm-start condition number 12,324 -> 8.9 and produced the
first approximately-converged run on record (R-hat_med 1.01 / max 1.10, ESS_med 810, eBFMI 0.21).

This script is the scale gate: the same three-cell comparison (ridge 0.5 first -- the answer that
matters -- then 0.1, then the 0.0 control) at production N. Reports the warm-start spectrum and
the full nan-aware diagnostic set per cell.

Reproduce:  python scripts/experiments/e11_ridge_scale.py --budget 500
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

import jax.numpy as jnp
import numpy as np

from device_info import print_device_banner
from diag_sampler_factorial import BRIDGE, hessian_spectrum, real_cost
from gmvae_confusion import latest_run
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior


def main(run_dir, budget, ridges, day_from, day_to, seed, sharpness, lam, lam_I, skip_hessian,
         chain_seed=None, centered=False, warmup="windowed", warm_up_steps=None):
    print_device_banner()
    C = real_cost(run_dir, budget, day_from, day_to, seed)
    eps = float(C.max() - C.min()) / sharpness
    n = C.shape[0]
    # chain_seed varies the MCMC randomness at a FIXED subsample/cost (mixing replication);
    # `seed` varies the subsample itself (a different target cell)
    cfg = dict(BRIDGE, seed=seed if chain_seed is None else chain_seed)
    if warm_up_steps is not None:
        cfg["warm_up_steps"] = warm_up_steps
    print(f"E11 | N={n} (m=2r(II+JJ)={2 * 2 * n * 2 // 2}) | D{day_from:g}->D{day_to:g} | "
          f"lambda={lam} lam_I={lam_I} s={sharpness} (eps={eps:.4f}) | orthant, bridge config | "
          f"chain seed {cfg['seed']} | ridge {'CENTERED at pi^o' if centered else 'uncentered'} | warmup {warmup}")
    print(f"{'ridge':>7} {'kappa':>10} {'n<=0':>5} {'acc':>5} {'Rhat_med':>9} {'Rhat_max':>9} "
          f"{'ESS_med':>8} {'ESS_min':>8} {'eBFMI':>7} {'frozen':>7} {'mins':>6}")
    rows = []
    for ridge in ridges:
        prior = HFPDOTHyperprior(mu_0=np.full(n, 1 / n), nu_0=1.3 * np.full(n, 1 / n),
                                 lambda_1=lam, lambda_2=lam, lambda_I_1=lam_I, lambda_I_2=lam_I,
                                 cost_fn=C.reshape(-1), epsilon=eps, support="positive_orthant")
        smp = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            shape=n * n, support="positive_orthant", sampling_strategy="low_rank",
            II=n, JJ=n, rank=2, cost=jnp.asarray(C), epsilon=eps, ridge=ridge,
            ridge_center="warm_start" if centered else None,
            warmup=warmup,
            initial_plan=prior.sinkhorn_init(), **cfg)
        spec = ({"kappa": float("nan"), "n_nonpos": -1} if skip_hessian
                else hessian_spectrum(smp, prior))
        t0 = time.time()
        res = smp.sample(with_diagnostics=True, max_pi_coords=2000)
        mins = (time.time() - t0) / 60
        d = res.diagnostics
        acc = np.asarray(res.stacked_mala_states.stacked_num_accepted_samples)
        acc = float(np.mean(acc.reshape(len(acc), -1)[:, -1] /
                            (cfg["num_samples"] + cfg["num_burnin"])))
        ebfmi_chain = np.asarray(d.ebfmi).reshape(-1)
        frozen = int(np.sum(ebfmi_chain < 0.01))          # E4: dead chains hide behind medians
        row = dict(ridge=ridge, kappa=spec["kappa"], n_nonpos=spec["n_nonpos"], acc=acc,
                   rhat_med=float(np.nanmedian(d.bulk_rhat)),
                   rhat_max=float(np.nanmax(d.bulk_rhat)),
                   ess_med=float(np.nanmedian(d.ess)), ess_min=float(np.nanmin(d.ess)),
                   ebfmi_min=float(np.nanmin(d.ebfmi)),
                   ebfmi_chain=ebfmi_chain.tolist(), frozen=frozen,
                   chain_seed=cfg["seed"], minutes=mins)
        rows.append(row)
        print(f"{ridge:>7.2f} {row['kappa']:>10.1f} {row['n_nonpos']:>5d} {acc:>5.2f} "
              f"{row['rhat_med']:>9.2f} {row['rhat_max']:>9.2f} {row['ess_med']:>8.0f} "
              f"{row['ess_min']:>8.0f} {row['ebfmi_min']:>7.3f} {frozen:>7d} {mins:>6.1f}",
              flush=True)
    tag = ("cen_" if centered else "") + ("wwin_" if warmup == "windowed" else "")
    out = Path(run_dir) / f"e11_ridge_scale_{tag}b{budget}_D{day_from:g}_cs{cfg['seed']}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--ridges", type=float, nargs="+", default=[0.5, 0.1, 0.0])
    p.add_argument("--from-day", type=float, default=12.0)
    p.add_argument("--to-day", type=float, default=12.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sharpness", type=float, default=33.0)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--lam-I", type=float, default=0.5)
    p.add_argument("--skip-hessian", action="store_true",
                   help="skip the warm-start spectrum (m^2 memory at large N)")
    p.add_argument("--chain-seed", type=int, default=None,
                   help="MCMC seed at a FIXED subsample (mixing replication); default = --seed")
    p.add_argument("--centered", action="store_true",
                   help="centered ridge gamma||theta - theta_0||^2, theta_0 = warm start (pi^o)")
    p.add_argument("--warmup", choices=["windowed", "legacy"], default="windowed",
                   help="warm-up scheme (legacy = historical unadapted/pooled flow)")
    p.add_argument("--warm-up-steps", type=int, default=None,
                   help="override BRIDGE warm_up_steps (windowed: more steps = better mass)")
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.ridges, a.from_day, a.to_day, a.seed,
         a.sharpness, a.lam, a.lam_I, a.skip_hessian, a.chain_seed, a.centered, a.warmup,
         a.warm_up_steps)
