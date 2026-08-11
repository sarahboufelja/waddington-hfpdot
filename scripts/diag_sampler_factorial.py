"""Factorial attribution of the ridge sampler's real-target mixing failure.

Every validated run (E-series) and every failing run (real hyperprior) differs in BOTH of the two
untested factors at once, so the existing evidence cannot attribute the failure:

  cost    E-series: squared Euclidean between iid 100-d Gaussians. High-dimensional distance
          concentration makes that matrix nearly flat (low contrast, no structure) -- an
          intrinsically easy, near-isotropic target. Real: GaussianW2 on 10-d clustered
          posteriors -- contrast ~3.4x around the median, block structure from the fate clusters.
  regime  E-series: lambda_1 = lambda_2 = 1, lambda_I = 0.01, eps = 0.01. Real: lambda = 10,
          lambda_I = 0.5, eps = 0.1 -- and the strong-lambda regime is what lambda(eta) will
          produce for tight-marginal days, so it cannot be retreated from.

This runs the 2x2 (cost x regime), plus two eps-isolation cells on the real cost, all else fixed:
N = 30 per side, positive orthant, ridge low-rank r = 2, bridge config (5000/1000/600, step .02,
4 chains). Each cell reports the mixing triplet (median R-hat / median ESS / min eBFMI, nan-aware),
acceptance, and the measured target statistics: cost contrast (max-min)/median and effective rank
(SVD entropy), and the exact eigenspectrum of the latent-target Hessian at the warm start
(m = 120 -> exact and cheap): condition number and quartiles. Whatever moves the mixing must move
with its factor.

Run from the repo root:
    python scripts/diag_sampler_factorial.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

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
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

BRIDGE = dict(num_samples=5000, num_burnin=1000, warm_up_steps=600, step_size=0.02,
              num_parallel_chains=4, seed=0)
REGIMES = {"e9": dict(lam=1.0, lam_I=0.01, eps=0.01),
           "real": dict(lam=10.0, lam_I=0.5, eps=0.1)}


def real_cost(run_dir, budget, day_from, day_to, seed):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    matrices, _, _ = R.assemble([day_from, day_to])
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to("cuda" if torch.cuda.is_available() else "cpu")
    emb = VaDEEmbedder(model, device=model.parameters().__next__().device,
                      batch_size=cfg.get("batch_size", 512))
    sub = RandomSubsampler(seed=seed)
    posts = []
    for exp in matrices:
        post = emb.embed(exp)
        posts.append(post.select(sub.indices(post, budget)))
    a, b = posts
    C = np.asarray(GaussianW2()(CellCloud(a.means, a.stds), CellCloud(b.means, b.stds)))
    # Release the embedder's GPU memory: torch's caching allocator otherwise keeps holding it,
    # and at production N the jax sampler OOMs mid-autotune while the cache sits idle.
    del model, emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return C


def synthetic_cost(n, latent=100, seed=0):
    """The E-series construction verbatim: sq-Euclidean between iid Gaussians, median-normalised."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    src = jax.random.normal(k1, (n, latent))
    tgt = jax.random.normal(k2, (n, latent)) + 0.5
    C = jnp.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=-1)
    return np.asarray(C / jnp.median(C))


def cost_stats(C):
    s = np.linalg.svd(C - C.mean(), compute_uv=False)
    p = s / s.sum()
    erank = float(np.exp(-np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0.0))))
    return {"contrast": float((C.max() - C.min()) / np.median(C)), "erank": erank}


def hessian_spectrum(smp, prior):
    theta0 = smp.support.to_unconstrained(jnp.asarray(prior.sinkhorn_init()).reshape(1, -1))
    f = lambda th: -jnp.sum(smp.latent_log_prob_function(th.reshape(1, -1)))
    H = np.asarray(jax.hessian(f)(theta0.reshape(-1)))
    if not np.all(np.isfinite(H)):
        return {"kappa": float("nan"), "ev_min": float("nan"), "ev_q25": float("nan"),
                "ev_med": float("nan"), "ev_max": float("nan"), "n_nonpos": -1}
    ev = np.linalg.eigvalsh((H + H.T) / 2)
    pos = ev[ev > 1e-10]
    kappa = float(pos.max() / pos.min()) if len(pos) else float("inf")
    return {"kappa": kappa, "ev_min": float(ev.min()), "ev_q25": float(np.quantile(ev, .25)),
            "ev_med": float(np.median(ev)), "ev_max": float(ev.max()),
            "n_nonpos": int((ev <= 1e-10).sum())}


def run_cell(name, C, reg, n):
    prior = HFPDOTHyperprior(mu_0=np.full(n, 1 / n), nu_0=1.3 * np.full(n, 1 / n),
                             lambda_1=reg["lam"], lambda_2=reg["lam"],
                             lambda_I_1=reg["lam_I"], lambda_I_2=reg["lam_I"],
                             cost_fn=C.reshape(-1), epsilon=reg["eps"],
                             support="positive_orthant")
    smp = MetropolisAdjustedLangevinSampler(
        target_log_prob_fn=prior.hyperprior_log_prob_fun,
        target_score_fn=prior.hyperprior_score_fun,
        shape=n * n, support="positive_orthant", sampling_strategy="low_rank",
        II=n, JJ=n, rank=2, cost=jnp.asarray(C), epsilon=reg["eps"],
        initial_plan=prior.sinkhorn_init(), **BRIDGE)
    spec = hessian_spectrum(smp, prior)
    res = smp.sample(with_diagnostics=True)
    d = res.diagnostics
    acc = np.asarray(res.stacked_mala_states.stacked_num_accepted_samples)
    acc = float(np.mean(acc.reshape(len(acc), -1)[:, -1] /
                        (BRIDGE["num_samples"] + BRIDGE["num_burnin"])))
    cs = cost_stats(C)
    row = dict(cell=name, contrast=cs["contrast"], erank=cs["erank"], acc=acc,
               rhat_med=float(np.nanmedian(d.bulk_rhat)), ess_med=float(np.nanmedian(d.ess)),
               ebfmi_min=float(np.nanmin(d.ebfmi)), **spec)
    print(f"{name:>22} | contrast {row['contrast']:6.2f} erank {row['erank']:5.1f} | "
          f"kappa {row['kappa']:9.1f} ev[min/med/max] {row['ev_min']:8.2f}/"
          f"{row['ev_med']:7.2f}/{row['ev_max']:8.2f} n<=0 {row['n_nonpos']:2d} | "
          f"acc {acc:4.2f} Rhat_med {row['rhat_med']:5.2f} ESS_med {row['ess_med']:6.0f} "
          f"eBFMI {row['ebfmi_min']:6.3f}", flush=True)
    return row


def main(run_dir, budget, day_from, day_to, seed):
    """2x2x2: cost structure x potential strength x Gibbs sharpness.

    ``eps`` is NOT a factor on its own -- it only acts through the sharpness ``s =
    (C_max - C_min)/eps``, so each cell derives eps from its cost's contrast (running the E9
    eps=0.01 verbatim on the real cost puts 337 nats in the exponent and overflows: the E9 regime
    is only well-defined because distance concentration flattens its synthetic cost). One extra
    continuity cell reproduces E9's literal eps on the synthetic cost.
    """
    Creal = real_cost(run_dir, budget, day_from, day_to, seed)
    Csyn = synthetic_cost(budget, seed=seed)
    print(f"N={budget} orthant ridge r=2 bridge config | real pair D{day_from:g}->D{day_to:g}")
    for name, C in (("syn", Csyn), ("real", Creal)):
        print(f"  {name} cost: contrast {(C.max() - C.min()) / np.median(C):.2f} | "
              f"E9-literal sharpness (eps=.01) would be {(C.max() - C.min()) / 0.01:.0f} nats")
    print()
    rows = []
    for cname, C in (("syn", Csyn), ("real", Creal)):
        span = float(C.max() - C.min())
        for lname, (lam, lam_I) in (("weak-lam", (1.0, 0.01)), ("strong-lam", (10.0, 0.5))):
            for s in (3.3, 33.0):
                reg = dict(lam=lam, lam_I=lam_I, eps=span / s)
                rows.append(run_cell(f"{cname}|{lname}|s={s:g}", C, reg, budget))
    rows.append(run_cell("syn|weak-lam|eps=.01(E9)", Csyn,
                         dict(lam=1.0, lam_I=0.01, eps=0.01), budget))
    out = Path(run_dir) / f"sampler_factorial_b{budget}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nrows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--from-day", type=float, default=12.0)
    p.add_argument("--to-day", type=float, default=12.5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.from_day, a.to_day, a.seed)
