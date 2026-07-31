"""Per-fate confusion diagnostic for a trained GMVAE run.

Loads a checkpoint, re-embeds the cells, and compares the predicted fate (argmax responsibility)
against the labelled true fate (argmax target) on the LABELLED cells only. Writes a row-normalised
confusion heatmap and prints per-fate precision/recall plus the macro (unweighted) average -- the
view the aggregate micro accuracy hides when one or two fates dominate the cell count.

Run from the repo root (defaults to the newest run):
    python scripts/gmvae_confusion.py
    python scripts/gmvae_confusion.py assets/gmvae_runs/<tag>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_gmvae_train as R
from gmvae.networks import GMVAENet
from gmvae.embedder import VaDEEmbedder
from gmvae.training import build_targets


def latest_run():
    runs = sorted(p.parent for p in (ROOT / "assets" / "gmvae_runs").glob("*/model.pt"))
    if not runs:
        raise SystemExit("no runs under assets/gmvae_runs/*/model.pt")
    return runs[-1]


def confusion(run_dir, days_spec=None):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = R._parse_days(days_spec) if days_spec else cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run: {run_dir.name} | days: {len(days)} | device: {device}")

    matrices, memberships, pops2 = R.assemble(days)
    assert pops2 == pops, "population axis mismatch between checkpoint and assembled data"

    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"])
    model.to(device)

    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))
    P = np.vstack([embedder.embed(m).prob_cat for m in matrices])      # (N, K) responsibilities
    T = build_targets(matrices, memberships, pops)                    # (N, K) soft targets

    lab = T.sum(1) > 0
    y_true = T[lab].argmax(1)
    y_pred = P[lab].argmax(1)
    K = len(pops)
    C = np.zeros((K, K), dtype=int)
    np.add.at(C, (y_true, y_pred), 1)                                 # rows = true, cols = predicted

    support = C.sum(1)
    recall = np.divide(C.diagonal(), support, out=np.zeros(K), where=support > 0)
    pred_tot = C.sum(0)
    precision = np.divide(C.diagonal(), pred_tot, out=np.zeros(K), where=pred_tot > 0)

    # where each fate's recall leaks to (biggest off-diagonal in its row)
    leak_to, leak_frac = [], []
    for i in range(K):
        row = C[i].copy(); row[i] = -1
        j = int(row.argmax())
        leak_to.append(pops[j])
        leak_frac.append(C[i, j] / support[i] if support[i] else 0.0)

    print(f"\nlabelled cells: {int(lab.sum())} / {len(lab)}")
    print(f"micro accuracy (aggregate): {C.diagonal().sum() / support.sum():.3f}")
    print(f"macro recall  (unweighted): {recall.mean():.3f}   <- the imbalance-honest number")
    print(f"\n{'fate':>24} {'support':>8} {'recall':>7} {'precis':>7}   biggest leak")
    order = np.argsort(-support)                                       # commonest fates first
    for i in order:
        print(f"{pops[i]:>24} {support[i]:>8} {recall[i]:>7.3f} {precision[i]:>7.3f}"
              f"   -> {leak_to[i]} ({leak_frac[i]:.2f})")

    # row-normalised heatmap (recall view)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cn = C / support[:, None]
    Cn = np.nan_to_num(Cn)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(Cn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(pops, rotation=90, fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(pops, fontsize=8)
    ax.set_xlabel("predicted fate"); ax.set_ylabel("true (labelled) fate")
    ax.set_title(f"Row-normalised confusion (recall) -- {run_dir.name}")
    for i in range(K):
        for j in range(K):
            if Cn[i, j] >= 0.01:
                ax.text(j, i, f"{Cn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if Cn[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = run_dir / "confusion.png"
    fig.savefig(out, dpi=140)
    np.save(run_dir / "confusion_counts.npy", C)
    print(f"\nheatmap -> {out}")
    return C, recall, precision, support


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None, help="run dir (default: newest)")
    p.add_argument("--days", default=None, help="override day set (default: the run's own days)")
    a = p.parse_args()
    confusion(a.run_dir or latest_run(), a.days)
