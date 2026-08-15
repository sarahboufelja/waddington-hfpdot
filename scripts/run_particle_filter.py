"""Particle-filtered HFPD-OT propagation over a day window -- credible bands under the fixed kernel.

Runs the module-5 marginal particle filter over consecutive day pairs with the CERTIFIED fixed
kernel injected: centered ridge ``gamma ||theta - theta_0||^2`` at the ratified ``gamma = 0.5``
(``theta_0`` = warm start = the chart coordinates of pi^o), windowed warm-up, and per-pair
``epsilon = (C_max - C_min) / 33`` (the ratified Gibbs sharpness) unless ``--eps`` overrides it.
The full diagnostic triplet is stored next to every pair and gates it: a pair that fails the R1
gates is not a result, whatever the bands look like. Fate membership ``W`` is the GMVAE posterior
responsibilities ``q(c|z)`` (single-source doctrine), not curated annotations.

Per pair, for every particle: the particle's marginal conditions its own hyperprior
(``mu_0 :=`` particle weights, floored and renormalised; target prior uniform), plans are drawn,
each retained draw becomes a transition table and a future marginal; the pooled futures are
coverage-trimmed back to the particle budget with mass absorption (``wadd_propagation``). Outputs
per pair: mass-weighted credible bands over the K x K transition table and the ensemble tube width
``ensemble_spread`` -- the propagated-uncertainty curve to set against the identity radius
``eta(day)``.

Space level (per the locked design): random subsample of ``--budget`` cells per day, uniform
``nu_0`` on the target support.

Run from the repo root (newest run, defaults are a smoke; see --help):
    python scripts/run_particle_filter.py --days 12 12.5 13 --budget 20 --particles 2 \
        --samples 600 --burnin 200 --warmup 200
Production (certified kernel, single-GPU):
    CUDA_VISIBLE_DEVICES=1 python scripts/run_particle_filter.py \
        --days 12 12.5 13 13.5 14 --budget 500 --particles 8 \
        --samples 10000 --burnin 2000 --warmup 1200 --support positive_orthant
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

import jax.numpy as jnp
import numpy as np
import torch

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_dim_reduction import RandomSubsampler
from wadd_ot import CellCloud, GaussianW2
from wadd_lineage import transition_table
from wadd_propagation import (MarginalParticle, initial_ensemble, kl_ball, pairwise_tv,
                              project_particles, propagate_pair, weighted_quantiles)
from langevin_sampler import MetropolisAdjustedLangevinSampler, HFPDOTHyperprior

_MU_FLOOR = 1e-6      # conditioning floor: a particle weight of exactly 0 would put an infinite
                      # KL wall at that cell; the floor keeps the hyperprior proper and is far
                      # below any mass the bands could resolve at these draw counts
_SHARPNESS = 33.0     # ratified Gibbs sharpness: eps = (C_max - C_min) / 33 per pair (E-series)


def make_sampler(cost, II, JJ, lam, lam_I, eps, support, rank, gamma, cfg, seed):
    """Close over the pair's fixed data; the returned callable is the injected ``PlanSampler``.

    ``gamma`` is the centered-ridge scale (the certified kernel); ``gamma = 0`` reproduces the
    historical ridge-free runs and is a diagnostic setting only.
    """
    def sample(mu_0, nu_0):
        mu = np.maximum(mu_0, _MU_FLOOR); mu = mu / mu.sum()
        prior = HFPDOTHyperprior(mu_0=mu, nu_0=nu_0, lambda_1=lam, lambda_2=lam,
                                 lambda_I_1=lam_I, lambda_I_2=lam_I,
                                 cost_fn=np.asarray(cost).reshape(-1), epsilon=eps,
                                 support=support)
        smp = MetropolisAdjustedLangevinSampler(
            target_log_prob_fn=prior.hyperprior_log_prob_fun,
            target_score_fn=prior.hyperprior_score_fun,
            shape=II * JJ, support=support, sampling_strategy="low_rank",
            II=II, JJ=JJ, rank=rank, cost=jnp.asarray(cost), epsilon=eps,
            ridge=gamma, ridge_center="warm_start" if gamma > 0 else None,
            num_parallel_chains=cfg["chains"], num_samples=cfg["samples"],
            num_burnin=cfg["burnin"], warm_up_steps=cfg["warmup"], step_size=cfg["step"],
            seed=seed, initial_plan=prior.sinkhorn_init())
        res = smp.sample(with_diagnostics=True, max_pi_coords=2000)
        d = res.diagnostics
        # per_chain=125 halves the peak transient (500 plans ~ 1 GB): on a shared host the
        # OOM killer targets the largest single process, so peak RSS is survival-relevant.
        plans = smp.thinned_plans(res, per_chain=125, chunk=50)
        acc = np.asarray(res.stacked_mala_states.stacked_num_accepted_samples)
        acc_rate = float(np.mean(acc.reshape(len(acc), -1)[:, -1] /
                                 (cfg["samples"] + cfg["burnin"])))
        diag = {"rhat_med": float(np.nanmedian(d.bulk_rhat)),
                "rhat_max": float(np.nanmax(d.bulk_rhat)),
                "ess_med": float(np.nanmedian(d.ess)),
                "ebfmi_min": float(np.nanmin(d.ebfmi)),
                "accept": acc_rate}
        return plans, diag
    return sample


def _rebuild_ensemble(z, tag):
    """The exact ensemble that left pair ``tag``: MarginalParticle is (weights, log_mass, parent)."""
    E, M, par = z[f"ensemble_{tag}"], z[f"ensemble_mass_{tag}"], z[f"parents_{tag}"]
    return [MarginalParticle(weights=E[i], log_mass=float(M[i]),
                             parent=None if par[i] < 0 else int(par[i]))
            for i in range(len(M))]


def _find_resume(run_dir, config, tags):
    """Partial record with the longest completed-pair prefix for this exact config.

    Completed pairs are always a prefix (the loop checkpoints sequentially), so progress is
    the prefix length. Config equality is checked against the manifest after a json round
    trip; the `latest` symlink is skipped to avoid double-visiting its target.
    """
    config = json.loads(json.dumps(config, default=str))
    best, best_n = None, 0
    for man in sorted((Path(run_dir) / "particle_filter").glob("*/manifest.json")):
        if man.parent.name == "latest":
            continue
        try:
            stored = json.loads(man.read_text()).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        stored.pop("resumed_from", None)               # resumed records stay resumable
        if stored != config:
            continue
        for npz in man.parent.glob("*.npz"):
            try:
                z = np.load(npz, allow_pickle=True)
            except (OSError, ValueError):
                continue
            n = 0
            while n < len(tags) and f"ensemble_{tags[n]}" in z.files:
                n += 1
            if n > best_n:
                best, best_n = npz, n
    return best, best_n


def main(run_dir, days, budget, particles, draws_kept, lam, lam_I, eps, support, rank,
         gamma, seed, sampler_cfg, resume=False):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run: {run_dir.name} | window: {days} | budget {budget} cells/day | "
          f"P={particles} D={draws_kept} | lambda={lam} lam_I={lam_I} "
          f"eps={'range/33' if eps <= 0 else eps} {support} | gamma={gamma} (centered) | "
          f"sampler {sampler_cfg} | device {device}")
    print("NOTE: per-pair diagnostics gate the bands -- a pair failing the R1 gates is not a "
          "result, whatever its bands look like.")

    # Artifact dir created at LAUNCH so the record can checkpoint incrementally: a run
    # killed mid-window keeps every completed pair (the manifest timestamp marks launch).
    config = dict(days=list(days), budget=budget, particles=particles,
                  draws_kept=draws_kept, lam=lam, lam_I=lam_I, eps=eps,
                  support=support, rank=rank, gamma=gamma, seed=seed,
                  sampler=sampler_cfg)
    tags = [f"{a:g}->{b:g}" for a, b in zip(days, days[1:])]
    resume_src, done = (None, 0)
    if resume:
        resume_src, done = _find_resume(run_dir, config, tags)
        print(f"resume: {resume_src} carries {done}/{len(tags)} pairs"
              if resume_src else "resume: no matching partial record, starting fresh")
        if done == len(tags):
            print("window already complete; nothing to do")
            return
    adir = artifact_dir(run_dir, "particle_filter",
                        config=dict(config, resumed_from=str(resume_src) if resume_src else ""))
    path = adir / (f"particle_filter_D{days[0]:g}-D{days[-1]:g}_P{particles}"
                   f"_b{budget}_lam{lam:g}_g{gamma:g}.npz")

    matrices, memberships, pops2 = R.assemble(days)
    assert pops2 == pops
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))
    sub = RandomSubsampler(seed=seed)

    staged = []
    for exp in matrices:
        post = embedder.embed(exp)
        idx = sub.indices(post, min(budget, len(post)))
        sel = post.select(idx)
        # W = q(c|z) responsibilities (single-source doctrine): soft fate membership from the
        # same posterior that drives every other uncertainty; defined at every day.
        staged.append({"day": exp.day, "post": sel,
                       "W": np.asarray(sel.prob_cat, dtype=np.float64)})
    # Staging is the only consumer of the expression matrices, the embedder and the model;
    # holding them for the whole window (~1 GB per day of matrices alone) is what pushed
    # wide windows past host memory. The subsampled posteriors in `staged` are all the
    # pair loop needs.
    del matrices, memberships, embedder, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ensemble = initial_ensemble(len(staged[0]["post"]))
    out = {"days": np.array(days), "populations": np.array(pops),
           "lam": lam, "lam_I": lam_I, "eps": eps, "support": support, "gamma": gamma,
           "budget": budget, "seed": seed}
    # the tube: KL ball (mean = the propagated radius eta_prop, hyperprior moment condition) and
    # pairwise-TV diameter (Dobrushin-contraction readout), on fates and on cells
    spreads = [(days[0], 0.0, 0.0, 0.0, 0.0)]          # tube starts closed at the window root

    if resume_src is not None and done:
        z = np.load(resume_src, allow_pickle=True)
        for key in z.files:                            # completed pairs carried forward wholesale
            out[key] = z[key]
        ensemble = _rebuild_ensemble(z, tags[done - 1])
        # The tube statistics for completed pairs are recomputed from the stored ensembles
        # (cheap; partial records checkpoint per-pair keys only, the tube arrays are written
        # at completion).
        for j in range(done):
            ens_j = _rebuild_ensemble(z, tags[j])
            b = staged[j + 1]
            fates = project_particles(ens_j, b["W"])
            e_f, _ = kl_ball(fates)
            _, tv_f = pairwise_tv(fates)
            e_c, _ = kl_ball(ens_j)
            _, tv_c = pairwise_tv(ens_j)
            spreads.append((b["day"], e_f, tv_f, e_c, tv_c))
        np.savez(path, **out)                          # the new record is whole from step one
    print(f"\n{'pair':>14} {'draws':>6} {'rhat_med':>9} {'ess_med':>8} {'eBFMI':>7} {'acc':>5} "
          f"{'eta_prop':>9} {'TVdiam':>7} {'band width med':>15} {'max':>6}")
    for i, (a, b) in enumerate(zip(staged[:-1], staged[1:])):
        if i < done:                                   # completed in the resumed record
            continue
        II, JJ = len(a["post"]), len(b["post"])
        cost = GaussianW2()(CellCloud(a["post"].means, a["post"].stds),
                            CellCloud(b["post"].means, b["post"].stds))
        # eps <= 0 selects the ratified Gibbs sharpness per pair (the E-series regime).
        eps_pair = eps if eps > 0 else float((np.asarray(cost).max() -
                                              np.asarray(cost).min()) / _SHARPNESS)
        sampler = make_sampler(cost, II, JJ, lam, lam_I, eps_pair, support, rank,
                               gamma, sampler_cfg, seed)
        out[f"eps_{a['day']:g}->{b['day']:g}"] = eps_pair
        table_of = lambda d, a=a, b=b, II=II, JJ=JJ: transition_table(
            np.asarray(d).reshape(II, JJ), a["W"], b["W"], pops, pops,
            a["day"], b["day"]).matrix
        rec = propagate_pair(ensemble, sampler, np.full(JJ, 1.0 / JJ), II, JJ,
                             budget=particles, table_of_plan=table_of,
                             day_from=a["day"], day_to=b["day"],
                             max_futures_per_particle=draws_kept)
        ensemble = rec.futures
        qs = weighted_quantiles(rec.tables, rec.table_log_mass, [0.025, 0.5, 0.975])
        width = qs[2] - qs[0]
        live = qs[2] > 0                     # occupied entries only: structural zeros are not bands
        fates = project_particles(ensemble, b["W"])
        eta_prop, _ = kl_ball(fates)
        _, tv_diam = pairwise_tv(fates)
        eta_cells, _ = kl_ball(ensemble)
        _, tv_cells = pairwise_tv(ensemble)
        spreads.append((b["day"], eta_prop, tv_diam, eta_cells, tv_cells))
        dmed = {k: float(np.median([g[k] for g in rec.diagnostics]))
                for k in rec.diagnostics[0]}
        tag = f"{a['day']:g}->{b['day']:g}"
        print(f"{tag:>14} {len(rec.tables):>6} {dmed['rhat_med']:>9.2f} {dmed['ess_med']:>8.0f} "
              f"{dmed['ebfmi_min']:>7.3f} {dmed['accept']:>5.2f} {eta_prop:>9.4f} "
              f"{tv_diam:>7.3f} {np.nanmedian(width[live]):>15.4f} {np.nanmax(width[live]):>6.3f}")
        out[f"tables_q_{tag}"] = qs
        out[f"table_log_mass_{tag}"] = rec.table_log_mass
        out[f"diag_{tag}"] = json.dumps(rec.diagnostics)
        out[f"ensemble_{tag}"] = np.stack([p.weights for p in ensemble])
        out[f"ensemble_mass_{tag}"] = np.array([p.log_mass for p in ensemble])
        out[f"parents_{tag}"] = np.array([-1 if p.parent is None else p.parent
                                          for p in ensemble])
        # the full pre-trim pool + fate projections: what the trim selected from, and the
        # coarse-axis profiles the mechanism video draws without re-embedding anything
        out[f"pool_{tag}"] = np.stack([p.weights for p in rec.pool])
        out[f"pool_mass_{tag}"] = np.array([p.log_mass for p in rec.pool])
        out[f"pool_parents_{tag}"] = np.array([p.parent for p in rec.pool])
        out[f"W_fates_{b['day']:g}"] = b["W"]
        out[f"nu0_fates_{tag}"] = np.full(JJ, 1.0 / JJ) @ b["W"]
        np.savez(path, **out)               # incremental checkpoint: pair-level durability

    out["tube_day"] = np.array([s[0] for s in spreads])
    out["tube_eta_prop"] = np.array([s[1] for s in spreads])       # KL ball mean, fate simplex
    out["tube_tv_diam"] = np.array([s[2] for s in spreads])        # TV diameter, fate simplex
    out["tube_eta_cells"] = np.array([s[3] for s in spreads])
    out["tube_tv_cells"] = np.array([s[4] for s in spreads])
    np.savez(path, **out)
    print(f"\neta_prop (propagated KL-ball radius, fate simplex, nats): "
          f"{'  '.join(f'D{s[0]:g}:{s[1]:.4f}' for s in spreads)}")
    print(f"TV diameter (fates, Dobrushin-comparable):                 "
          f"{'  '.join(f'D{s[0]:g}:{s[2]:.3f}' for s in spreads)}")
    print(f"record -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--days", type=float, nargs="+", default=[12.0, 12.5, 13.0])
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--particles", type=int, default=2)
    p.add_argument("--draws-kept", type=int, default=50)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--lam-I", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=0.0,
                   help="entropic regularisation; <= 0 selects the ratified per-pair "
                        "(C_max - C_min)/33, a positive value is an absolute override")
    p.add_argument("--support", default="simplex", choices=["simplex", "positive_orthant"])
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="centered-ridge scale (certified gamma* = 0.5 at budget 500); "
                        "0 reproduces the historical ridge-free kernel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue from the config-matching partial record with the most "
                        "completed pairs (per-pair checkpoints make kills resumable)")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--burnin", type=int, default=200)
    p.add_argument("--warmup", type=int, default=1200)
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--chains", type=int, default=4)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.days, a.budget, a.particles, a.draws_kept, a.lam,
         a.lam_I, a.eps, a.support, a.rank, a.gamma, a.seed,
         {"samples": a.samples, "burnin": a.burnin, "warmup": a.warmup, "step": a.step,
          "chains": a.chains}, resume=a.resume)
