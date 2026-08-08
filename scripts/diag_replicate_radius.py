"""Estimate the marginal-uncertainty radius eta from replicate lanes.

The HFPD-OT hyperprior is conditioned on eta bounding E[KL(mu || mu_0)] -- how wrong the nominal
marginal can be. Two of the three contributions are computable from a single snapshot and both turn
out to be negligible:

  * finite sampling of cells, and
  * ambiguity in each cell's fate assignment,

which together are exactly the multinomial variance of the fate proportions, giving
E[KL] ~ (K-1)/(2n) ~ 0.002 nats at n ~ 3000. A radius that small pins the marginals and collapses
HFPD-OT onto deterministic EOT.

The term that actually matters is the one that does NOT vanish with n: capture efficiency, batch, and
dish-to-dish biological variation. It is not identifiable from one sample -- but this archive runs TWO
10x lanes (C1, C2) at every timepoint, which are independent captures of the same underlying
population. The fate composition estimated from each lane therefore differs by sampling noise PLUS
exactly the non-vanishing term we cannot otherwise see, so the excess of the observed between-lane
divergence over its sampling-noise expectation is a direct, data-derived estimate of eta.

Fate composition per lane is the mean GMVAE responsibility, w_c = mean_k gamma_kc, so every cell
contributes (labelled or not) and the estimate stays single-source.

Sampling baseline: for two independent multinomial estimates of the same p,
E[KL(p1 || p2)] ~ (K-1)/2 * (1/n1 + 1/n2). Excess above that is not explicable by sampling.

Run from the repo root:
    python scripts/diag_replicate_radius.py
"""
import argparse
import json
import re
import sys
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

_LANE = re.compile(r"_C(\d+)_")


def _lane_of(cell_id):
    m = _LANE.search(cell_id)
    return m.group(1) if m else "?"


def _kl(p, q, floor=1e-12):
    p = np.maximum(p, floor); q = np.maximum(q, floor)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def main(run_dir, days_spec=None):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = R._parse_days(days_spec) if days_spec else cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = len(pops)

    probe, _, _ = R.assemble(days[:1])
    model = GMVAENet(x_dim=probe[0].n_genes, num_clusters=K,
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))

    matrices, _, _ = R.assemble(days)
    print(f"run: {run_dir.name} | {len(days)} days | K={K} fates | device={device}")
    print(f"\n{'day':>6} {'n(C1)':>7} {'n(C2)':>7} {'KL_sym':>9} {'sampling':>9} {'excess':>9} {'ratio':>7}")
    rows = []
    for exp in matrices:
        post = embedder.embed(exp)
        lanes = np.array([_lane_of(c) for c in post.cells.ids])
        uniq = sorted(set(lanes) - {"?"})
        if len(uniq) < 2:
            continue
        a, b = uniq[0], uniq[1]
        wa = post.prob_cat[lanes == a].mean(axis=0)
        wb = post.prob_cat[lanes == b].mean(axis=0)
        na, nb = int((lanes == a).sum()), int((lanes == b).sum())
        kl_sym = 0.5 * (_kl(wa, wb) + _kl(wb, wa))
        sampling = 0.5 * (K - 1) * (1.0 / na + 1.0 / nb)      # expected under pure sampling noise
        excess = max(kl_sym - sampling, 0.0)
        rows.append((exp.day, na, nb, kl_sym, sampling, excess))
        print(f"{exp.day:>6} {na:>7} {nb:>7} {kl_sym:>9.4f} {sampling:>9.4f} {excess:>9.4f} "
              f"{kl_sym / sampling:>7.1f}")

    arr = np.array(rows)
    kl, samp, exc = arr[:, 3], arr[:, 4], arr[:, 5]
    print(f"\nbetween-lane KL   median {np.median(kl):.4f}  [{kl.min():.4f}, {kl.max():.4f}]")
    print(f"sampling baseline median {np.median(samp):.4f}")
    print(f"EXCESS (= eta)    median {np.median(exc):.4f}  [{exc.min():.4f}, {exc.max():.4f}]")
    print(f"ratio observed/sampling: median {np.median(kl / samp):.1f}x")
    print(f"\nfor scale: log K = {np.log(K):.3f} nats is the largest possible KL on this simplex")

    out = run_dir / "replicate_radius.npz"
    np.savez(out, day=arr[:, 0], n1=arr[:, 1], n2=arr[:, 2], kl_sym=kl, sampling=samp, excess=exc,
             populations=np.array(pops))
    print(f"series -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--days", default=None)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.days)
