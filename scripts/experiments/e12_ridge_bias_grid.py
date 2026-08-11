"""E12 step 1 -- the instrumented ridge grid: R2 bias quantification at production scale.

For each ridge value r on the grid, at each phase pair (Dox D2->2.5, Serum D12->12.5), this runs
the ratified production cell (N=500, orthant, low-rank r=2, lambda=10, lambda_I=0.5, s=33; double
bridge draws) and instruments the chains for the R2 protocol:

  per retained (thinned) draw i:  theta_i -> pi_i (chunked chart pushforward), then
    - the K x K transition table T_i,
    - the marginal-KL statistics R_mu = gKL(mu_pi||mu_0), R_nu = gKL(nu_pi||nu_0)
      (the moments lambda(eta) matches -- the observable that decides whether the ratified ridge
      shifts the lambda calibration),
    - the Gibbs divergence g = gKL(pi||pi_I) (the direct shrinkage readout),
    - t_i = ||theta_i||^2 (the sufficient statistic of the ridge tilt).

  per observable O: mean, sd, per-observable ESS/R-hat (chain structure preserved), MCSE,
  the slope Cov(O, t) -- so the first-order bias  bias_O(r) ~= -r Cov_r(O, t)  and the width
  sensitivity  dVar/dr = -Cov(O^2, t) + 2 E[O] Cov(O, t)  come from the same chains -- and the
  2.5/50/97.5% band per table entry (the denominator of the bias/band certificate).

Everything is saved per (pair, ridge) to one npz; the printed summary shows the mixing gates and
the two certificate ratios for the headline observables.

Reproduce:  python scripts/experiments/e12_ridge_bias_grid.py --budget 500
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

import jax
import jax.numpy as jnp
import numpy as np
import torch

import run_gmvae_train as R
from device_info import print_device_banner
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_dim_reduction import RandomSubsampler
from wadd_ot import CellCloud, GaussianW2
from wadd_lineage import transition_table
from mcmc_diagnostics import MCMCDiagnostics
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

# Gibbs sharpness s = (C_max - C_min)/epsilon held at the R1 operating point (factorial: epsilon
# acts on conditioning only through s). Per-pair epsilon = range/s keeps Dox and Serum cells at
# the same operating point, so gamma is the only factor varying across the grid.
_SHARPNESS = 33.0

GRID = (0.4, 0.6, 0.8, 1.2, 2.0)
CFG = dict(num_samples=10_000, num_burnin=2_000, warm_up_steps=600, step_size=0.02,
           num_parallel_chains=4, seed=0)
THIN_PER_CHAIN = 500
CHUNK = 100
_EPS = 1e-30


def stage(run_dir, budget, day_from, day_to, seed):
    """Embed the pair once: cost + q(c|z) membership rows of the sampled support.

    W is the GMVAE posterior responsibilities ``prob_cat`` (single-source doctrine), NOT
    the Schiebinger cell-set annotations: annotation coverage is 0% before D6 (all-zero
    tables), while q(c|z) is defined for every cell at every day. Component k carries the
    label of its seed population (population-seeded prior), so ``pops`` still names the
    table axes. Torch memory is freed after staging.
    """
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    matrices, _, pops2 = R.assemble([day_from, day_to])
    assert pops2 == pops
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"])
    # Embed on CPU: staging runs once per pair and must not compete with the jax sampling
    # pool for device memory (jax grows its pool across cells and never shrinks it; on a
    # shared or single-GPU box the embedder then finds no headroom).
    model.to("cpu")
    emb = VaDEEmbedder(model, device="cpu", batch_size=cfg.get("batch_size", 512))
    sub = RandomSubsampler(seed=seed)
    staged = []
    for exp in matrices:
        post = emb.embed(exp)
        sel = post.select(sub.indices(post, budget))
        staged.append({"post": sel,
                       "W": np.asarray(sel.prob_cat, dtype=np.float64)})
    a, b = staged
    C = np.asarray(GaussianW2()(CellCloud(a["post"].means, a["post"].stds),
                                CellCloud(b["post"].means, b["post"].stds)))
    del model, emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return C, a["W"], b["W"], pops


def gkl(p, q):
    """Generalized KL (unnormalised marginals): sum p log p/q - sum p + sum q."""
    p = np.maximum(p, _EPS)
    return float(np.sum(p * (np.log(p) - np.log(np.maximum(q, _EPS)))) - p.sum() + q.sum())


def run_cell(C, Ws, Wt, pops, ridge, day_from, day_to):
    n = C.shape[0]
    eps = float(C.max() - C.min()) / _SHARPNESS
    mu0, nu0 = np.full(n, 1 / n), 1.3 * np.full(n, 1 / n)
    prior = HFPDOTHyperprior(mu_0=mu0, nu_0=nu0, lambda_1=10.0, lambda_2=10.0,
                             lambda_I_1=0.5, lambda_I_2=0.5, cost_fn=C.reshape(-1),
                             epsilon=eps, support="positive_orthant")
    smp = MetropolisAdjustedLangevinSampler(
        target_log_prob_fn=prior.hyperprior_log_prob_fun,
        target_score_fn=prior.hyperprior_score_fun,
        shape=n * n, support="positive_orthant", sampling_strategy="low_rank",
        II=n, JJ=n, rank=2, cost=jnp.asarray(C), epsilon=eps, ridge=ridge,
        initial_plan=prior.sinkhorn_init(), **CFG)
    pi_I = np.asarray(prior.sinkhorn_init()).reshape(-1)

    t0 = time.time()
    res = smp.sample(with_diagnostics=True, max_pi_coords=2000)
    d = res.diagnostics
    gates = dict(rhat_med=float(np.nanmedian(d.bulk_rhat)),
                 rhat_max=float(np.nanmax(d.bulk_rhat)),
                 ess_med=float(np.nanmedian(d.ess)), ebfmi_min=float(np.nanmin(d.ebfmi)))

    # thinned latent draws, retained segment only: (C, THIN, m)
    lat = np.asarray(jax.device_get(res.stacked_mala_states.stacked_latent_states))
    lat = lat.reshape(lat.shape[0], lat.shape[1], -1)[:, -CFG["num_samples"]:, :]
    keep = np.linspace(0, CFG["num_samples"] - 1, THIN_PER_CHAIN).astype(int)
    lat = lat[:, keep, :]
    Cn, D, m = lat.shape
    t_stat = (lat ** 2).sum(axis=2)                                 # (C, D)

    K = len(pops)
    obs = np.empty((Cn, D, K * K + 3))
    for c in range(Cn):
        for s in range(0, D, CHUNK):
            th = jnp.asarray(lat[c, s:s + CHUNK])
            pis = np.asarray(smp.support.to_constrained(th))        # (chunk, n*n)
            for j, pi in enumerate(pis):
                P = pi.reshape(n, n)
                T = transition_table(P, Ws, Wt, pops, pops, day_from, day_to).matrix
                obs[c, s + j, :K * K] = T.reshape(-1)
                obs[c, s + j, K * K] = gkl(P.sum(1), mu0)           # R_mu
                obs[c, s + j, K * K + 1] = gkl(P.sum(0), nu0)       # R_nu
                obs[c, s + j, K * K + 2] = gkl(pi, pi_I)            # Gibbs divergence
    mins = (time.time() - t0) / 60

    # per-observable statistics with chain structure preserved
    od = MCMCDiagnostics(constrained_chains=jnp.asarray(obs))
    ess_o = np.asarray(od.ess(jnp.asarray(obs)))
    flat = obs.reshape(-1, obs.shape[-1])
    tflat = t_stat.reshape(-1)
    mean = flat.mean(axis=0)
    sd = flat.std(axis=0, ddof=1)
    tc = (tflat - tflat.mean())[:, None]
    cov_ot = ((flat - mean) * tc).mean(axis=0)
    # width sensitivity in the centered form: dVar/dr = -Cov((O - E[O])^2, T), algebraically
    # equal to -Cov(O^2,T) + 2 E[O] Cov(O,T) but without large-magnitude cancellation
    dev2 = (flat - mean) ** 2
    dvar = -((dev2 - dev2.mean(axis=0)) * tc).mean(axis=0)
    mcse = sd / np.sqrt(np.maximum(ess_o, 1.0))
    qs = np.quantile(flat[:, :K * K], [0.025, 0.5, 0.975], axis=0)
    return dict(gates=gates, mean=mean, sd=sd, cov_ot=cov_ot, dvar=dvar, mcse=mcse,
                ess=ess_o, band=qs[2] - qs[0], t_mean=float(tflat.mean()),
                minutes=mins, n_obs=obs.shape[-1])


def main(run_dir, budget, ridges, seed):
    print_device_banner()
    out = {}
    for day_from, day_to, phase in ((2.0, 2.5, "dox"), (12.0, 12.5, "serum")):
        C, Ws, Wt, pops = stage(run_dir, budget, day_from, day_to, seed)
        K = len(pops)
        iR, iN, iG = K * K, K * K + 1, K * K + 2
        print(f"\n[{phase}] D{day_from:g}->D{day_to:g}  N={C.shape[0]}  "
              f"contrast {(C.max() - C.min()) / np.median(C):.2f}")
        print(f"{'ridge':>6} {'rhat_med':>8} {'max':>5} | {'R_mu':>8} {'+-mcse':>7} "
              f"{'bias1':>8} {'|b|/sd':>7} {'dVar/Var':>9} | {'g_piI':>8} {'|b|/sd':>7} | "
              f"{'tbl med|b|/sd':>13} {'max|b|/band':>11} {'mins':>5}")
        for ridge in ridges:
            r = run_cell(C, Ws, Wt, pops, ridge, day_from, day_to)
            bias1 = -ridge * r["cov_ot"]                            # first-order location bias
            live = (r["mean"][:K * K] > 1e-4) & (r["band"] > 1e-6)
            if live.any():
                tbl_bsd = np.abs(bias1[:K * K][live]) / np.maximum(r["sd"][:K * K][live], 1e-12)
                tbl_bband = np.abs(bias1[:K * K][live]) / r["band"][live]
            else:
                tbl_bsd = tbl_bband = np.array([np.nan])
            g = r["gates"]
            print(f"{ridge:>6.2f} {g['rhat_med']:>8.2f} {g['rhat_max']:>5.2f} | "
                  f"{r['mean'][iR]:>8.4f} {r['mcse'][iR]:>7.4f} {bias1[iR]:>8.4f} "
                  f"{abs(bias1[iR]) / max(r['sd'][iR], 1e-12):>7.3f} "
                  f"{r['dvar'][iR] * ridge / max(r['sd'][iR] ** 2, 1e-12):>9.3f} | "
                  f"{r['mean'][iG]:>8.2f} {abs(bias1[iG]) / max(r['sd'][iG], 1e-12):>7.3f} | "
                  f"{np.median(tbl_bsd):>13.3f} {np.max(tbl_bband):>11.3f} {r['minutes']:>5.1f}",
                  flush=True)
            for k, v in r.items():
                if k != "gates":
                    out[f"{phase}_{ridge:g}_{k}"] = v
            out[f"{phase}_{ridge:g}_gates"] = json.dumps(r["gates"])
            out[f"{phase}_populations"] = np.array(pops)
            # incremental checkpoint: a killed run keeps every completed cell
            np.savez(Path(run_dir) / f"e12_ridge_bias_b{budget}.npz", **out)
    path = Path(run_dir) / f"e12_ridge_bias_b{budget}.npz"
    np.savez(path, **out)
    print(f"\nrecord -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--ridges", type=float, nargs="+", default=list(GRID))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.ridges, a.seed)
