"""Two-stage VaDE GMVAE training on the GSE122662 reprogramming timecourse.

Assembles per-day ``(ExpressionMatrix, Membership)`` from the 10x HDF5 archive and the
``cell_sets.gmt`` fate labels, then runs Stage A -> SEED -> Stage B via ``gmvae.training.train``.
Defaults to a small plumbing smoke -- three days, three epochs per stage -- to prove the pipeline
end to end before committing to a full run.

Run from the repo root:
    python scripts/run_gmvae_train.py                              # smoke: D6/D12/D18, 3+3 epochs
    python scripts/run_gmvae_train.py --days all \\
        --pretrain-epochs 50 --joint-epochs 100                    # full run (all in-scope days)

The gene axis is Ensembl (unique key); library-size normalisation and highly-variable-gene
selection are deliberately NOT applied yet -- both are quality levers to add once this foundation
runs. See the model docs for what they entail.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from wadd_data_ingest import H5Reader, read_cell_sets_gmt, membership_from_cell_sets
from gmvae.networks import GMVAENet
from gmvae.training import train, build_targets
from gmvae.embedder import VaDEEmbedder

DATA_DIR = ROOT / "data" / "GSE122662_RAW"
GMT_PATH = ROOT / "data" / "cell_sets.gmt"
SMOKE_DAYS = [6.0, 12.0, 18.0]      # labelled + multi-fate + spans the trajectory; each has unlabelled cells too


def assemble(days):
    """Per-day parallel ``(matrices, memberships)`` lists and the canonical fate axis.

    The fate axis is the GMT key order (13 fates); each day's membership is built over that day's
    ExpressionMatrix cell axis, so it is row-aligned to the counts by construction.
    """
    cell_sets = read_cell_sets_gmt(GMT_PATH)
    reader = H5Reader(matrix_dir=DATA_DIR)
    matrices, memberships = [], []
    for d in days:
        exp = reader.read(d)
        memberships.append(membership_from_cell_sets(cell_sets, exp.cells))
        matrices.append(exp)
        print(f"  day {d:>5}: {len(exp):6d} cells x {exp.n_genes} genes")
    return matrices, memberships, list(cell_sets)


def evaluate(model, matrices, memberships, population_names, device, batch_size):
    """Post-hoc smoke checks on the trained model: does supervision take, and does the latent use
    more than one component?

    - true_fate_mass: mean responsibility placed on the true fate over labelled cells (argmax of the
      soft target). Above the ~1/K chance level means the anchor took hold.
    - argmax_accuracy: fraction of labelled cells whose top responsibility is the true fate.
    - occupied_clusters / occupancy: how many of the K components are some cell's argmax -- the
      anti-collapse check, given the IPS-heavy imbalance.
    """
    embedder = VaDEEmbedder(model, device=device, batch_size=batch_size)
    probs = [embedder.embed(exp).prob_cat for exp in matrices]      # each row-aligned to exp.cells
    P = np.vstack(probs)                                            # (N, K) responsibilities
    T = build_targets(matrices, memberships, population_names)      # (N, K) soft targets, same order

    pred = P.argmax(1)
    occ = np.bincount(pred, minlength=len(population_names))
    labelled = T.sum(1) > 0
    true = T.argmax(1)
    true_mass = float(P[labelled, true[labelled]].mean()) if labelled.any() else float("nan")
    accuracy = float((pred[labelled] == true[labelled]).mean()) if labelled.any() else float("nan")
    return {
        "n_cells": int(P.shape[0]),
        "n_labelled": int(labelled.sum()),
        "chance_level": 1.0 / len(population_names),
        "true_fate_mass": true_mass,
        "argmax_accuracy": accuracy,
        "occupied_clusters": int((occ > 0).sum()),
        "n_clusters": len(population_names),
        "occupancy": {name: int(c) for name, c in zip(population_names, occ)},
    }


def main(days, pretrain_epochs, joint_epochs, batch_size, latent_dim, hidden_dim, lr, lambda_sup,
         class_weight_scheme, seed, save):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | days: {days}")

    print("assembling days...")
    matrices, memberships, pops = assemble(days)
    n_total = sum(len(m) for m in matrices)
    print(f"  pooled: {n_total} cells, {len(pops)} fates, {matrices[0].n_genes} genes")

    torch.manual_seed(seed)
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=latent_dim, hidden_dim=hidden_dim)

    print(f"training: Stage A ({pretrain_epochs}) -> SEED -> Stage B ({joint_epochs})")
    pre_hist, joint_hist = train(
        model, matrices, memberships, pops,
        pretrain_epochs=pretrain_epochs, joint_epochs=joint_epochs, batch_size=batch_size,
        lambda_sup=lambda_sup, class_weight_scheme=class_weight_scheme, temperature=0.5,
        lr=lr, device=device, rng=np.random.default_rng(seed), verbose=True)

    print("evaluating...")
    diag = evaluate(model, matrices, memberships, pops, device, batch_size)
    diag.update(finite_losses=bool(np.all(np.isfinite(pre_hist)) and np.all(np.isfinite(joint_hist))))
    print(json.dumps(diag, indent=2))

    if save:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = ROOT / "assets" / "gmvae_runs" / tag
        outdir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "population_names": pops}, outdir / "model.pt")
        np.savez(outdir / "histories.npz", pretrain=pre_hist, joint=joint_hist)
        (outdir / "diagnostics.json").write_text(json.dumps(
            {"days": days, "pretrain_epochs": pretrain_epochs, "joint_epochs": joint_epochs,
             "batch_size": batch_size, "latent_dim": latent_dim, "hidden_dim": hidden_dim,
             "lr": lr, "lambda_sup": lambda_sup, "class_weight_scheme": class_weight_scheme,
             "seed": seed, **diag}, indent=2))
        print(f"  artifacts written to {outdir}")
    return diag


def _parse_days(spec):
    if spec == "smoke":
        return SMOKE_DAYS
    if spec == "all":
        return H5Reader(matrix_dir=DATA_DIR).available_days()
    return [float(x) for x in spec.split(",")]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", default="smoke",
                   help="'smoke' (D6/D12/D18), 'all' (every in-scope day), or a comma list e.g. 6,12,18")
    p.add_argument("--pretrain-epochs", type=int, default=3)
    p.add_argument("--joint-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--latent-dim", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-sup", type=float, default=1.0)
    p.add_argument("--class-weight-scheme", default="uniform", choices=["uniform", "inverse", "inverse_sqrt"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-save", action="store_true", help="skip writing the checkpoint/diagnostics")
    a = p.parse_args()
    main(_parse_days(a.days), a.pretrain_epochs, a.joint_epochs, a.batch_size, a.latent_dim,
         a.hidden_dim, a.lr, a.lambda_sup, a.class_weight_scheme, a.seed, save=not a.no_save)
