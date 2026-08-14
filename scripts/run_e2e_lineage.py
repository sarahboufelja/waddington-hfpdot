"""End-to-end lineage run: trained GMVAE -> transport plans -> transition tables + a fate landscape.

Chains the modules on the real GSE122662 timecourse:

    embed (module 3)  ->  radius per day (module 3b)  ->  subsample to the transport budget
                      ->  cost + plan per consecutive day pair (module 2)
                      ->  population transition tables (module 4)  ->  composed lineage + layout

Two result sets are written under the run directory: the per-pair and composed **transition tables**
(the quantitative lineage answer) and a **force-directed layout** of the latent landscape coloured by
day and by fate (the qualitative one).

Fate membership is the GMVAE posterior responsibility matrix ``q(c|z)`` throughout (single-source
doctrine): the soft memberships carry the identity uncertainty into the tables, are defined for
every cell at every day (curated annotations have 0% coverage before D6), and let the composed
lineage root at the first timepoint.

PROVISIONAL: the per-day radius reported here is the multimodality gap ``KL(psi_bar || psi_0)``, which
measures how far the latent population departs from a single Gaussian. It is a spread statistic, NOT
an uncertainty about the marginal, so it is carried as a placeholder to unblock the pipeline and is
not yet fed to the hyperprior. Deriving the real radius is the open item; see the module docstring of
``wadd_potential`` and section 4 of docs/mathematical_derivation.md.

Run from the repo root:
    python scripts/run_e2e_lineage.py                    # newest run, its own days
    python scripts/run_e2e_lineage.py --budget 400 --layout-per-day 80
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
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_dim_reduction import CoverageSubsampler, gaussianity_gap
from wadd_ot import CellCloud, GaussianW2, uot_plan
from wadd_lineage import (transition_table, compose_plans, population_distributions,
                          TransitionTable)

_INK, _MUTED, _GRID = "#1e293b", "#64748b", "#e2e8f0"
# Below this max-responsibility the MAP fate is unreliable; such cells are greyed in the layout.
_MIN_FATE_CONF = 0.5


def _stage(model, cfg, days, budget, device):
    """Embed every day, take its placeholder radius, and subsample to the transport budget.

    Returns per-day dicts carrying the subsampled latent posterior and membership -- everything the
    transport and lineage steps need, already row-aligned. Membership is the GMVAE posterior
    responsibility matrix ``q(c|z)`` (single-source doctrine): the same posterior that drives every
    other uncertainty also carries each cell's fate membership, soft -- so the tables inherit the
    membership uncertainty instead of thresholding it away. Curated annotations (0% coverage before
    D6) are not used. Component ``k`` carries the label of its seed population, so ``pops`` still
    names the table axes.
    """
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))
    matrices, _, pops = R.assemble(days)
    sub = CoverageSubsampler()
    staged = []
    print(f"\n{'day':>6} {'cells':>7} {'kept':>6} {'radius(gap)':>12} {'med max q(c|z)':>15}")
    for exp in matrices:
        post = embedder.embed(exp)
        radius = gaussianity_gap(post, n_samples=1024)          # PROVISIONAL placeholder (see docstring)
        idx = sub.indices(post, min(budget, len(post)))
        sel = post.select(idx)
        W = np.asarray(sel.prob_cat, dtype=np.float64)
        staged.append({"day": exp.day, "n": len(post), "radius": radius,
                       "posterior": sel, "membership": W})
        print(f"{exp.day:>6} {len(post):>7} {len(idx):>6} {radius:>12.3f} "
              f"{np.median(W.max(axis=1)):>15.3f}")
    return staged, pops


def _cloud(post):
    return CellCloud(means=post.means, stds=post.stds)


def _pairwise_tables(staged, pops, reg, reg_m):
    """One transport plan and one transition table per consecutive day pair.

    The plans are returned alongside the tables because spanning many timepoints is done by composing
    the PLANS (cell level) and coarse-graining once, not by chaining the tables.
    """
    tables, plans = [], []
    print(f"\n{'pair':>14} {'plan mass':>10} {'med max q':>10} {'top flow'}")
    for a, b in zip(staged, staged[1:]):
        plan = uot_plan(_cloud(a["posterior"]), _cloud(b["posterior"]),
                        cost=GaussianW2(), reg=reg, reg_m=reg_m)
        T = transition_table(plan, a["membership"], b["membership"], pops, pops,
                             source_day=a["day"], target_day=b["day"])
        tables.append(T); plans.append(plan)
        conf = float(np.median(a["membership"].max(axis=1)))
        live = [(c, d, v) for c, row in zip(pops, T.matrix) for d, v in zip(pops, row) if v > 0]
        top = max(live, key=lambda t: t[2]) if live else ("-", "-", 0.0)
        print(f"D{a['day']:<5g}->D{b['day']:<5g} {plan.sum():>10.4f} {conf:>10.3f} "
              f"{top[0]} -> {top[1]} ({top[2]:.2f})")
    return tables, plans


def _span_table(staged, plans, pops, start_idx):
    """Lineage from ``staged[start_idx]`` to the final timepoint, composed at the CELL level.

    Composing plans assumes the cell process is Markov in the latent state (Chapman-Kolmogorov over
    the intermediate timepoints) -- an assumption, but the weaker one available: chaining population
    tables instead would need that same property plus lumpability of the fate partition.
    """
    K = compose_plans(plans[start_idx:])
    a, b = staged[start_idx], staged[-1]
    P = population_distributions(a["membership"])                  # (n_pops, n_cells_start)
    T = P @ K @ b["membership"]
    return TransitionTable(matrix=_row_normalize_safe(T), source_populations=list(pops),
                           target_populations=list(pops),
                           source_day=a["day"], target_day=b["day"])


def _row_normalize_safe(M):
    s = M.sum(axis=1, keepdims=True)
    return np.divide(M, s, out=np.zeros_like(M), where=s > 0)


def _plot_table(T, out, title):
    """Row-normalised transition heatmap: rows are source fates, columns their destinations."""
    K = len(T.source_populations)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * K, 0.9 + 0.5 * K))
    im = ax.imshow(T.matrix, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(T.target_populations)))
    ax.set_xticklabels(T.target_populations, rotation=90, fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(T.source_populations, fontsize=7)
    ax.set_xlabel(f"destination fate  (D{T.target_day:g})", color=_INK, fontsize=9)
    ax.set_ylabel(f"source fate  (D{T.source_day:g})", color=_INK, fontsize=9)
    ax.set_title(title, color=_INK, fontsize=10, fontweight="bold", loc="left", pad=10)
    for i in range(K):
        for j in range(len(T.target_populations)):
            v = T.matrix[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def _layout(staged, pops, per_day, seed=0):
    """Force-directed layout of the latent landscape (a genuine FLE: kNN graph + spring embedding).

    Built on OUR latent space rather than the published Waddington-OT coordinates, so the picture is
    of the model we actually trained. A per-day subsample keeps the graph small enough for the
    force simulation; edges are mutual-kNN in latent space.
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    xs, days, fates = [], [], []
    for s in staged:
        post = s["posterior"]
        take = rng.choice(len(post), size=min(per_day, len(post)), replace=False)
        xs.append(post.means[take])
        days.append(np.full(len(take), s["day"]))
        fates.append(np.argmax(s["membership"][take], axis=1))
        fates[-1][s["membership"][take].max(axis=1) < _MIN_FATE_CONF] = -1   # low-confidence q(c|z)
    X = np.vstack(xs); day = np.concatenate(days); fate = np.concatenate(fates)

    k = 8
    tree = cKDTree(X)
    _, nbr = tree.query(X, k=k + 1)
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    G.add_edges_from((i, int(j)) for i, row in enumerate(nbr) for j in row[1:])
    print(f"\nlayout graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> spring layout")
    pos = nx.spring_layout(G, seed=seed, iterations=60)
    P = np.array([pos[i] for i in range(len(X))])
    return P, day, fate, G


def _plot_layout(P, day, fate, pops, out):
    """Two panels of the same layout: coloured by timepoint, and by MAP fate."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(_GRID)

    sc = axes[0].scatter(P[:, 0], P[:, 1], c=day, cmap="viridis", s=4, linewidths=0)
    axes[0].set_title("coloured by timepoint", color=_INK, fontsize=11, fontweight="bold", loc="left")
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04); cb.set_label("day", fontsize=9)

    present = [c for c in np.unique(fate) if c >= 0]
    cmap = plt.get_cmap("tab20")
    axes[1].scatter(P[fate < 0, 0], P[fate < 0, 1], c="#d4d4d8", s=4, linewidths=0,
                    label=f"uncertain (max q < {_MIN_FATE_CONF:g})")
    for i, c in enumerate(present):
        m = fate == c
        axes[1].scatter(P[m, 0], P[m, 1], color=cmap(i % 20), s=5, linewidths=0, label=pops[c])
    axes[1].set_title("coloured by fate", color=_INK, fontsize=11, fontweight="bold", loc="left")
    axes[1].legend(fontsize=6.5, markerscale=2.0, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=False)

    fig.suptitle("Latent fate landscape (force-directed layout) — GSE122662 serum/Dox",
                 color=_INK, fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150); plt.close(fig)


def main(run_dir, budget, layout_per_day, reg, reg_m, days_spec=None):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = R._parse_days(days_spec) if days_spec else cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run: {run_dir.name} | days: {len(days)} | budget: {budget} | device: {device}")

    matrices_probe, _, pops2 = R.assemble(days[:1])
    model = GMVAENet(x_dim=matrices_probe[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)

    staged, pops = _stage(model, cfg, days, budget, device)
    tables, plans = _pairwise_tables(staged, pops, reg, reg_m)

    # q(c|z) membership is defined for every cell at every day, so the composed lineage roots at the
    # first timepoint. (Curated annotations forced a later root: coverage is 0% before D6.)
    start = 0
    overall = _span_table(staged, plans, pops, start)
    print(f"\nlineage root: D{staged[start]['day']:g}")

    out = artifact_dir(run_dir, "lineage",
                       config=dict(budget=budget, layout_per_day=layout_per_day,
                                   reg=reg, reg_m=list(reg_m), days=list(days)))
    np.savez(out / "transition_tables.npz",
             pairs=np.array([[t.source_day, t.target_day] for t in tables]),
             matrices=np.stack([t.matrix for t in tables]),
             composed=overall.matrix, composed_days=np.array([overall.source_day, overall.target_day]),
             populations=np.array(pops), radius=np.array([s["radius"] for s in staged]),
             day=np.array([s["day"] for s in staged]))
    _plot_table(overall, out / "transition_composed.png",
                f"Lineage  D{overall.source_day:g} -> D{overall.target_day:g}  (cell-level composition)")
    _plot_table(tables[-1], out / "transition_last_pair.png",
                f"Transition  D{tables[-1].source_day:g} -> D{tables[-1].target_day:g}")

    print(f"\nlineage D{overall.source_day:g} -> D{overall.target_day:g}, top destinations:")
    for src, dests in overall.top_destinations(3).items():
        if dests:
            print(f"  {src:>24} -> " + ", ".join(f"{d} {v:.2f}" for d, v in dests))

    P, day, fate, _ = _layout(staged, pops, layout_per_day)
    np.savez(out / "layout.npz", coords=P, day=day, fate=fate, populations=np.array(pops))
    _plot_layout(P, day, fate, pops, out / "fate_landscape.png")
    print(f"\nartifacts -> {out}")
    return overall


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--days", default=None)
    p.add_argument("--budget", type=int, default=400, help="cells per timepoint for the transport")
    p.add_argument("--layout-per-day", type=int, default=80, help="cells per timepoint in the layout")
    p.add_argument("--reg", type=float, default=1e-1,
                   help="entropic regularisation (cost is median-normalised to 1, so this is a "
                        "fraction of the typical transport cost; 0.05 can fail to converge on "
                        "widely separated timepoints)")
    p.add_argument("--reg-m", type=float, nargs=2, default=(1.0, 50.0),
                   help="UOT marginal relaxations (source, target)")
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget, a.layout_per_day, a.reg, tuple(a.reg_m), a.days)
