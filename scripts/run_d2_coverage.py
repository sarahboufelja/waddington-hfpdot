"""D2 held-out predictive coverage -- the runner.

For each consecutive triplet (t1, t2, t3) of the certified snapshot grid, the DIRECT
(t1, t3) coupling is sampled with the certified kernel (one root at t1, no history;
never composed -- no CK gate applies) and each posterior plan draw is collapsed to a
mid-point composition p-hat at t2 through the staged G-tensor of responsibilities at
interpolated latents. The record stores the clean (D, K) predictions next to the
observed composition m-bar of the real t2 subset; multinomial noise, intervals and
coverage are the READER's job (see wadd_schema.D2_KEYS) and are never stored here.

One npz per triplet; a triplet is complete only if its file passes the schema gate,
so interrupted runs resume by re-running exactly the missing or broken triplets.

The sampler configuration is pinned to the certified campaign values (E2E defaults)
and is deliberately NOT a CLI knob: the D2 kernel identity must be the campaign's.

Run from the repo root (single-GPU convention):
    CUDA_VISIBLE_DEVICES=0 python scripts/run_d2_coverage.py [run_dir] --list
    CUDA_VISIBLE_DEVICES=0 python scripts/run_d2_coverage.py [run_dir] [--resume] [--max N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

import run_e2e_hfpdot as E2E
import run_gmvae_train as R
import run_particle_filter as PF
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_dim_reduction import RandomSubsampler
from wadd_ot import CellCloud, GaussianW2
from wadd_propagation import initial_ensemble, propagate_pair
from wadd_schema import validate_d2_record

#: D2 stores no transition tables; propagate_pair's mandatory table channel is stubbed
#: with an empty array and the stacked result is dropped at record assembly
_NO_TABLE = np.zeros((0, 0))


def triplet_grid():
    """The consecutive (t1, t2, t3) triplets of the certified snapshot grid.

    The grid is the concatenation of the E2E window day tuples with the shared joint
    days deduplicated -- the single source of the stamps, never a hand-typed list.
    """
    days = []
    for w in E2E.WINDOWS:
        days += [d for d in w if not days or d > days[-1]]
    return [(days[i], days[i + 1], days[i + 2]) for i in range(len(days) - 2)]


def alpha_of(t):
    """Mid-point weight from the real stamps; 0.5 only where the grid is regular."""
    return (t[1] - t[0]) / (t[2] - t[0])


class Stager:
    """Embeds and subsamples triplet days; the model loads once, matrices per triplet.

    Unlike the particle filter (which stages a whole window once, then frees the model),
    every triplet stages through the model, so it persists for the run; the expression
    matrices -- the heavy part, ~1 GB per day -- are assembled and freed per triplet.
    Subsampling is stateless per (seed, day), so a day's subset is identical in every
    triplet role AND identical to the campaign's staged subset at the same seed/budget.
    """

    def __init__(self, run_dir, seed):
        run_dir = Path(run_dir)
        self.cfg = json.loads((run_dir / "diagnostics.json").read_text())
        self.ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
        self.pops = self.ckpt["population_names"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sub = RandomSubsampler(seed=seed)
        self.model = None

    def _ensure_model(self, n_genes):
        if self.model is None:
            self.model = GMVAENet(x_dim=n_genes, num_clusters=len(self.pops),
                                  latent_dim=self.cfg["latent_dim"],
                                  hidden_dim=self.cfg["hidden_dim"])
            self.model.load_state_dict(self.ckpt["model"])
            self.model.to(self.device)
            self.embedder = VaDEEmbedder(self.model, device=self.device,
                                         batch_size=self.cfg.get("batch_size", 512))

    def stage(self, t, budget):
        """Role-keyed staging {a: t1, mid: t2, b: t3}; t2 is HELD OUT.

        The mid subset never reaches the sampler -- it exists only to yield the observed
        composition m-bar and its cell count N. W = q(c|z) responsibilities, the
        single-source membership used everywhere else in the pipeline.
        """
        matrices, _, pops2 = R.assemble(list(t))
        assert pops2 == self.pops
        self._ensure_model(matrices[0].n_genes)
        staged = {}
        for role, exp in zip(("a", "mid", "b"), matrices):
            post = self.embedder.embed(exp)
            sel = post.select(self.sub.indices(post, min(budget, len(post))))
            staged[role] = {"day": exp.day, "post": sel,
                            "W": np.asarray(sel.prob_cat, dtype=np.float64)}
        del matrices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return staged

    def g_tensor(self, a, b, alpha, batch=8192):
        """(II, JJ, K) responsibilities at the McCann interpolation points.

        The displacement interpolant of a COUPLING is supported on the pair points
        z_ij = (1 - alpha) z_i + alpha z_j -- one hypothetical t2 cell per (ancestor,
        descendant) pair, because mass splitting is the branching structure (a Monge
        map would collapse this to II atoms; an entropic plan never does). Collapsing
        to barycentric destinations first would bias compositions through the
        nonlinearity of gamma (Jensen gap), so gamma is evaluated at ALL II*JJ points,
        batched through the same inference net as every other membership. Latents are
        the posterior means (the deterministic membership contract); identity is NOT
        resampled here -- frozen-G scope, justified by E14 (identity <= 2% of variance
        mass). Built once per triplet; only the plan contraction varies per draw.
        """
        za = np.asarray(a["post"].means, dtype=np.float64)
        zb = np.asarray(b["post"].means, dtype=np.float64)
        pts = ((1.0 - alpha) * za[:, None, :]
               + alpha * zb[None, :, :]).reshape(-1, za.shape[1])
        out = []
        with torch.no_grad():
            for k in range(0, len(pts), batch):
                lg = self.model.inference_net.responsibilities(
                    torch.as_tensor(pts[k:k + batch], dtype=torch.float32,
                                    device=self.device))
                out.append(np.exp(lg.cpu().numpy()))
        return np.concatenate(out).astype(np.float64).reshape(len(za), len(zb), -1)


def comp_of_plan(G):
    """Per-draw McCann composition: ``comp(plan) = p-hat``, the extras closure.

    The draw's own total mass normalises the plan INSIDE the closure, so extras rows
    arrive as ready compositions -- scale-free by construction, the mass-scale artifact
    cannot leak in. The raw mass is untouched here; it reaches the record as the sum of
    the growth channel's row sums (table_log_mass is the genealogy share, constant for
    the single root, NOT the plan mass). Row sums inherit gamma's normalisation:
    sum_c p-hat[c] = sum_ij pi-hat_ij = 1.
    """
    II, JJ, _ = G.shape

    def comp(d):
        pi = np.asarray(d, dtype=np.float64).reshape(II, JJ)
        return np.einsum("ij,ijc->c", pi / pi.sum(), G)
    return comp


def pair_sampler(a, b, lam, lam_I, lam_pi, eps, support, rank, gamma, sampler_cfg, seed):
    """Certified-kernel sampler for the DIRECT (t1, t3) pair -- PF machinery bit for bit.

    ``make_sampler`` is imported from the particle filter, so the D2 kernel identity is
    the campaign's by construction, never a copy; ``eps <= 0`` selects the ratified
    per-pair Gibbs sharpness (C-range / 33).
    """
    II, JJ = len(a["post"]), len(b["post"])
    cost = GaussianW2()(CellCloud(a["post"].means, a["post"].stds),
                        CellCloud(b["post"].means, b["post"].stds))
    eps_pair = eps if eps > 0 else float((np.asarray(cost).max()
                                          - np.asarray(cost).min()) / PF._SHARPNESS)
    sampler = PF.make_sampler(cost, II, JJ, lam, lam_I, lam_pi, eps_pair, support, rank,
                              gamma, sampler_cfg, seed)
    return sampler, eps_pair, II, JJ


def triplet_path(adir, t):
    return Path(adir) / f"d2_D{t[0]:g}-D{t[1]:g}-D{t[2]:g}.npz"


def triplet_complete(path):
    """A triplet counts only if its file loads AND passes the schema gate."""
    try:
        z = np.load(path, allow_pickle=True)
        validate_d2_record({k: z[k] for k in z.files})
        return True
    except Exception:
        return False


def _find_run(run_dir, config):
    """Latest existing d2_coverage dir whose manifest config matches after a json round trip."""
    config = json.loads(json.dumps(config, default=str))
    hit = None
    for man in sorted((Path(run_dir) / "d2_coverage").glob("*/manifest.json")):
        if man.parent.name == "latest":
            continue
        try:
            stored = json.loads(man.read_text()).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        if stored == config:
            hit = man.parent
    return hit


def main(run_dir, budget, draws, lam, lam_I, lam_pi, eps, support, rank, gamma, seed,
         max_triplets, resume):
    run_dir = Path(run_dir)
    sampler_cfg = dict(E2E.PF_DEFAULTS["sampler"])
    config = dict(budget=budget, draws=draws, lam=lam, lam_I=lam_I, lam_pi=lam_pi,
                  eps=eps, support=support, rank=rank, gamma=gamma, seed=seed,
                  sampler=sampler_cfg)
    triplets = triplet_grid()
    adir = _find_run(run_dir, config) if resume else None
    print(f"resume: continuing into {adir}" if adir else
          ("resume: no matching run, starting fresh" if resume else "fresh run"))
    if adir is None:
        adir = artifact_dir(run_dir, "d2_coverage", config=config)
    pending = [t for t in triplets if not triplet_complete(triplet_path(adir, t))]
    print(f"d2 coverage: {len(pending)}/{len(triplets)} triplets pending -> {adir}")
    print(f"knobs: budget {budget} | D<={draws} | lam {lam} lam_I {lam_I} lam_pi {lam_pi:g} "
          f"| eps {'range/33' if eps <= 0 else eps} {support} | gamma {gamma} | seed {seed}")
    print("NOTE: per-triplet diagnostics gate coverage -- a triplet failing the R1 gates "
          "is excluded by the reader, whatever its intervals look like.")
    if lam_pi != 1:
        print(f"DIAGNOSTIC KERNEL: lam_pi={lam_pi:g} -- NOT the HFPD-OT hyperprior (Def 2); "
              "the kernel certificate does not apply.")
    if not pending:
        print("all triplets complete; nothing to do")
        return
    if max_triplets:
        pending = pending[:max_triplets]

    stager = Stager(run_dir, seed)
    print(f"\n{'triplet':>22} {'alpha':>6} {'D':>4} {'rhat_med':>9} {'ess_med':>8} "
          f"{'eBFMI':>7} {'acc':>5} {'TV(med,real)':>13} {'mass':>6} {'sec':>5}")
    for t in pending:
        t0 = time.time()
        alpha = alpha_of(t)
        staged = stager.stage(t, budget)
        G = stager.g_tensor(staged["a"], staged["b"], alpha)
        sampler, eps_pair, II, JJ = pair_sampler(
            staged["a"], staged["b"], lam, lam_I, lam_pi, eps, support, rank, gamma,
            sampler_cfg, seed)
        rec = propagate_pair(initial_ensemble(II), sampler, np.full(JJ, 1.0 / JJ), II, JJ,
                             budget=1, table_of_plan=lambda d: _NO_TABLE,
                             extra_of_plan=comp_of_plan(G),
                             day_from=t[0], day_to=t[2], max_futures_per_particle=draws)
        comp = np.asarray(rec.extras, dtype=np.float64)
        out = {"days": np.array([t[0], t[2]], dtype=float),
               "populations": np.array(stager.pops),
               "lam": lam, "lam_I": lam_I, "lam_pi": lam_pi, "eps": eps,
               "support": support, "gamma": gamma, "budget": budget, "seed": seed,
               "d2_days": np.array(t, dtype=float), "d2_alpha": alpha,
               "d2_eps": eps_pair,
               "d2_comp": comp,
               "d2_total_mass": np.asarray(rec.growth, dtype=np.float64).sum(axis=1),
               "d2_real": staged["mid"]["W"].mean(axis=0),
               "d2_real_n": len(staged["mid"]["W"]),
               "d2_comp_t1": staged["a"]["W"].mean(axis=0),
               "d2_comp_t3": staged["b"]["W"].mean(axis=0),
               "d2_diag": json.dumps(rec.diagnostics[0])}
        validate_d2_record(out)
        np.savez(triplet_path(adir, t), **out)
        d = rec.diagnostics[0]
        tv = 0.5 * np.abs(np.median(comp, axis=0) - out["d2_real"]).sum()
        tag = f"D{t[0]:g}->D{t[1]:g}->D{t[2]:g}"
        print(f"{tag:>22} {alpha:>6.3g} {len(comp):>4} {d['rhat_med']:>9.2f} "
              f"{d['ess_med']:>8.0f} {d['ebfmi_min']:>7.3f} {d['accept']:>5.2f} "
              f"{tv:>13.3f} {out['d2_total_mass'].mean():>6.2f} {time.time() - t0:>5.0f}")
    print(f"\nd2 records -> {adir}")


if __name__ == "__main__":
    PD = E2E.PF_DEFAULTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None, help="GMVAE run dir (default: latest)")
    p.add_argument("--list", action="store_true", help="print the triplet grid and exit")
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--draws", type=int, default=500, help="max posterior draws kept per triplet")
    p.add_argument("--lam", type=float, default=PD["lam"])
    p.add_argument("--lam-I", type=float, default=PD["lam_I"])
    p.add_argument("--lam-pi", type=float, default=PD["lam_pi"])
    p.add_argument("--eps", type=float, default=PD["eps"])
    p.add_argument("--support", default=PD["support"])
    p.add_argument("--rank", type=int, default=PD["rank"])
    p.add_argument("--gamma", type=float, default=PD["gamma"])
    p.add_argument("--seed", type=int, default=PD["seed"])
    p.add_argument("--max", type=int, default=0, help="run at most N pending triplets (smoke)")
    p.add_argument("--resume", action="store_true",
                   help="continue into the latest config-matching d2_coverage dir")
    a = p.parse_args()
    if a.list:
        for t in triplet_grid():
            print(f"D{t[0]:g} -> D{t[1]:g} -> D{t[2]:g}   alpha={alpha_of(t):g}")
        sys.exit(0)
    main(a.run_dir or latest_run(), a.budget, a.draws, a.lam, a.lam_I, a.lam_pi, a.eps,
         a.support, a.rank, a.gamma, a.seed, a.max, a.resume)
