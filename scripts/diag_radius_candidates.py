"""The adopted latent radius ``eta = H(pi_bar)`` and its exact decomposition, on the real embedding.

The radius is the entropy of the day's fate composition ``pi_bar = (1/n) sum_i q(c|z_i)``. It gives
the complete account of the latent identity uncertainty, because it is exactly the sum of the two
quantities either of which alone tells half the story:

    H(pi_bar) = I(C;Z) + E_i H(q(c|z_i))
              = resolved identity spread + residual per-cell ambiguity,

with ``I(C;Z) = (1/n) sum_i KL(q(c|z_i) || pi_bar)`` the fate information the latent transmits (the
direction forced by I = KL(joint||product), section 4.2) and ``E_i H`` the ambiguity left after
seeing the latent. Both microstates that produce the same radius -- many confidently-occupied fates
versus uniformly ambiguous cells -- grant the marginals the same freedom; the decomposition is
reported so they stay distinguishable.

Against the section-4.4 checklist: bounded by ``log K`` (a fate alphabet, not the ``log n`` that made
I(cell;z) degenerate); exact in closed form (a mixture of categoricals is categorical, so no
Gaussianity gap exists on this channel); intensive (``pi_bar`` is a mean over cells, so the radius
depends on n only through O(n^-1/2) plug-in error); endpoint-correct (homogeneous single-fate day
=> eta -> 0 => marginal KL-balls shrink to the priors; fully ambiguous day => eta -> log K).

On the eta -> 0 endpoint, precisely: pinning the marginals does NOT pin the plan. As eta -> 0 the
hyperprior degenerates onto the transport polytope Pi(mu_0, nu_0) -- a set of dimension ~m^2-2m+1,
not a point (paper Remark 1, Eq 26) -- over which S^o remains spread at the temperature of the
KL(pi||pi_I) term. Coupling uncertainty therefore survives exactly-known marginals; deterministic
EOT is *not* recovered in the limit.

The retired candidate ``eta_boot = E_b KL(mu_b || mu_0)`` (resample cell fates, measure the drawn
composition against the point estimate) is kept as the disqualifying contrast: it is the *standard
error* of the composition -- the law of large numbers pins ``mu_b`` and eta_boot = O(1/n), the
intensivity failure -- and it inverts monotonicity (confident cells give zero, ambiguous cells give
the sampling-noise null). It remains useful as the model's own finite-sample noise floor, the
in-model analog of the replicate-lane measurement.

The map to the data-space marginal simplex is fixed by the section-4.4 endpoints with the *constant*
normaliser (the day's own value cannot normalise itself):  rho = log(m) . eta / log(K),  so
rho(0) = 0 and rho(log K) = log m with no overshoot.

Run from the repo root (newest run, its own days):
    python scripts/diag_radius_candidates.py
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

_INK, _MUTED, _GRID, _LINE = "#1e293b", "#64748b", "#e2e8f0", "#2b6cb0"
_ACCENT, _WARN = "#b45309", "#9f1239"
_EPS = 1e-12


# -- the two candidates ---------------------------------------------------------------------------

def _entropy(p, axis=-1):
    """Shannon entropy in nats, safe at p=0 (0 log 0 := 0)."""
    return -np.sum(np.where(p > _EPS, p * np.log(np.maximum(p, _EPS)), 0.0), axis=axis)


def identity_radius(prob_cat):
    """``eta = H(pi_bar)`` and its exact split into resolved identity and residual ambiguity.

    Exact, O(n K), no Monte Carlo: ``pi_bar = mean_i q(c|z_i)`` is itself categorical, and the chain
    rule gives ``H(pi_bar) = I(C;Z) + E_i H(q(c|z_i))`` with no approximation -- ``I`` is the
    generalized Jensen-Shannon divergence of the responsibility rows, ``E_i H`` the mean per-cell
    fate entropy. Returns ``(eta, mi, ambiguity, k_eff)`` where ``k_eff = exp(eta)`` is the
    effective number of occupied fates; ``mi`` is clamped at 0 against float cancellation only
    (Jensen guarantees ``H(pi_bar) >= E_i H`` in exact arithmetic).
    """
    pi_bar = prob_cat.mean(axis=0)
    eta = float(_entropy(pi_bar))
    ambiguity = float(np.mean(_entropy(prob_cat, axis=1)))
    return eta, max(eta - ambiguity, 0.0), ambiguity, float(np.exp(eta))


def bootstrap_radius(prob_cat, n_draws=512, seed=0):
    """``eta_boot = E_b KL(mu_b || mu_0)`` on the K-fate simplex, by resampling cell identities.

    Each draw assigns every cell a fate ``c_i ~ q(c|z_i)`` and forms the induced composition. Returns
    ``(mc, second_order)``: the Monte-Carlo value and the closed form of its second-order expansion
    ``1/2 sum_a Var(mu_a)/mu_0a = (1/2n^2) sum_a [sum_i p_ia(1-p_ia)] / pi_bar_a``, whose explicit
    ``1/n^2 . n = 1/n`` is the intensivity failure in algebra rather than in a plot.

    z is held at the encoder mean, so this is the identity arm alone; adding the z-resampling can
    only add variance and leaves the ``n``-scaling untouched, which is the property under test.
    """
    rng = np.random.default_rng(seed)
    n, K = prob_cat.shape
    pi_bar = prob_cat.mean(axis=0)
    cdf = np.cumsum(prob_cat, axis=1)

    draws = np.empty(n_draws)
    for b in range(n_draws):
        c = (rng.random((n, 1)) < cdf).argmax(axis=1)         # inverse-CDF categorical, vectorised
        mu = np.bincount(c, minlength=K) / n
        live = mu > _EPS                                      # 0 log 0 := 0 on empty fates
        draws[b] = np.sum(mu[live] * np.log(mu[live] / np.maximum(pi_bar[live], _EPS)))

    var = (prob_cat * (1.0 - prob_cat)).sum(axis=0) / n ** 2  # Var_b(mu_a), cells independent
    second = 0.5 * float(np.sum(var / np.maximum(pi_bar, _EPS)))
    return float(draws.mean()), second


def to_marginal(eta, K, m):
    """Map the latent radius onto the ``m``-cell marginal simplex: ``rho = log m . eta / log K``.

    Section 4.4 fixes the endpoints, not the shape: ``eta -> 0`` must give ``rho -> 0`` and ``eta``
    at its ceiling must give ``rho_max = log m`` **without overshoot**. The normaliser is the
    *constant* ``log K`` -- with ``eta = H(pi_bar)`` the day's own value cannot normalise itself --
    and rescaling by ``log m`` is what makes the two sides dimensionally comparable: a latent radius
    capped at ``log 13`` cannot be handed to a simplex whose diameter is ``log 500`` on its raw
    scale.
    """
    return float(np.log(m)) * eta / float(np.log(K))


# -- figures --------------------------------------------------------------------------------------

def _frame(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=_GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)


def _plot_identity(days, eta, mi, out, K):
    """``eta(day) = H(pi_bar)`` with its exact decomposition stacked under the curve.

    The two fills sum to the radius by the chain rule -- resolved identity ``I(C;z)`` from the axis
    up, residual ambiguity ``H(C|Z) = eta - I`` on top -- so the figure carries both the adopted
    definition and the two microstates it deliberately does not distinguish. The cap is the constant
    ``log K``: it does not move with the day's cell count, which is the intensivity statement made
    visually.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    _frame(ax)
    ax.axvline(8.25, color=_MUTED, lw=1.0, ls=(0, (4, 3)), alpha=0.6, zorder=1)
    ax.text(8.05, 0.97, "Dox → serum", color=_MUTED, fontsize=8, va="top", ha="right",
            transform=ax.get_xaxis_transform())

    cap = float(np.log(K))
    ax.axhline(cap, color=_MUTED, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.text(days[-1] + 0.3, cap + 0.03, f"cap  log K = {cap:.2f}", color=_MUTED, fontsize=8.5,
            va="bottom", ha="right")

    ax.fill_between(days, 0, mi, color=_LINE, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(days, mi, eta, color=_ACCENT, alpha=0.22, lw=0, zorder=1)
    ax.plot(days, eta, color=_LINE, lw=2.0, marker="o", ms=4.5, mfc=_LINE, mec="white",
            mew=0.6, zorder=3)
    ax.text(days[-1], eta[-1], r"  $\eta = H(\bar\pi)$", color=_LINE, fontsize=8.5, va="center",
            fontweight="bold")
    imid = int(np.searchsorted(days, 14.0))
    ax.text(days[imid], mi[imid] * 0.5, "resolved identity  $I(C;z)$", color=_LINE, fontsize=8.5,
            ha="center", fontweight="bold")
    ax.text(days[imid], (mi[imid] + eta[imid]) * 0.5 + 0.06, "ambiguity  $H(C|Z)$", color=_ACCENT,
            fontsize=8.5, ha="center", fontweight="bold")

    ipk = int(np.argmax(eta))
    ax.annotate(f"peak  D{days[ipk]:g}", (days[ipk], eta[ipk]), textcoords="offset points",
                xytext=(6, 8), fontsize=8.5, color=_INK, fontweight="bold")

    ax.set_xlim(days[0] - 0.4, days[-1] + 2.9)
    ax.set_ylim(0, cap * 1.10)
    ax.set_xlabel("day  (reprogramming timecourse)", color=_INK, fontsize=10)
    ax.set_ylabel("identity radius  (nats)", color=_INK, fontsize=10)
    ax.set_title(r"Latent identity radius  $\eta(t)=H(\bar\pi)=I(C;z)+\mathbb{E}_i H(q(c|z_i))$"
                 "  —  GSE122662", color=_INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_intensivity(sizes, curves, out):
    """The checklist test: radius vs subsample size, each series normalised to its full-population value.

    An intensive radius is a flat line at 1. ``eta_boot`` is plotted on the same axes precisely
    because it is not flat -- the 1/n decay is the disqualifying evidence, so it belongs in the same
    frame as the candidate that passes.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _frame(ax)
    ax.set_xscale("log")
    ax.axhline(1.0, color=_MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.text(sizes[0], 1.03, "intensive: flat at 1", color=_MUTED, fontsize=8.5, va="bottom")

    for (label, vals, colour) in curves:
        ax.plot(sizes, vals, color=colour, lw=2.0, marker="o", ms=4.5, mfc=colour, mec="white",
                mew=0.6, zorder=3)
        ax.text(sizes[-1], vals[-1], f"  {label}", color=colour, fontsize=8.5, va="center",
                fontweight="bold")

    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 3.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("cells retained  (random subsample, log scale)", color=_INK, fontsize=10)
    ax.set_ylabel("radius / full-population radius", color=_INK, fontsize=10)
    ax.set_title("Intensivity — the §4.4 box that decides between the candidates", color=_INK,
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -- driver ---------------------------------------------------------------------------------------

def main(run_dir, days_spec, n_draws, m_marginal, intensivity_day, seed):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "diagnostics.json").read_text())
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
    pops = ckpt["population_names"]
    days = R._parse_days(days_spec) if days_spec else cfg["days"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"run: {run_dir.name} | days: {len(days)} | K={len(pops)} | device: {device}")

    matrices, _, pops2 = R.assemble(days)
    assert pops2 == pops
    model = GMVAENet(x_dim=matrices[0].n_genes, num_clusters=len(pops),
                     latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"])
    model.load_state_dict(ckpt["model"]); model.to(device)
    embedder = VaDEEmbedder(model, device=device, batch_size=cfg.get("batch_size", 512))

    K = len(pops)
    print(f"\n{'day':>6} {'n':>7} {'eta':>8} {'I(C;z)':>8} {'H(C|Z)':>8} {'K_eff':>6} "
          f"{'rho':>6} {'eta_boot':>9} {'2nd-order':>10} {'boot.n':>8}")
    rows, keep = [], {}
    for exp in matrices:
        post = embedder.embed(exp)
        p = np.asarray(post.prob_cat, dtype=np.float64)
        eta, mi, amb, k_eff = identity_radius(p)
        boot, second = bootstrap_radius(p, n_draws=n_draws, seed=seed)
        rho = to_marginal(eta, K, m_marginal)
        n = len(post)
        keep[exp.day] = p
        rows.append((exp.day, n, eta, mi, amb, k_eff, rho, boot, second))
        print(f"{exp.day:>6} {n:>7} {eta:>8.4f} {mi:>8.4f} {amb:>8.4f} {k_eff:>6.2f} "
              f"{rho:>6.3f} {boot:>9.5f} {second:>10.5f} {boot * n:>8.3f}")

    order = np.argsort([r[0] for r in rows])
    g = lambda j: np.array([rows[i][j] for i in order])
    day, n_arr, eta_arr, mi_arr = g(0), g(1), g(2), g(3)
    amb_arr, k_eff, rho_arr, boot_arr = g(4), g(5), g(6), g(7)

    print(f"\neta = H(pi_bar): median {np.median(eta_arr):.4f} nats  "
          f"[{eta_arr.min():.4f}, {eta_arr.max():.4f}]  cap log K = {np.log(K):.4f}; "
          f"K_eff median {np.median(k_eff):.2f}")
    print(f"decomposition:   I(C;z) median {np.median(mi_arr):.4f} + H(C|Z) median "
          f"{np.median(amb_arr):.4f}  (ambiguity share median "
          f"{np.median(amb_arr / np.maximum(eta_arr, _EPS)):.3f})")
    print(f"rho = log(m).eta/log(K): median {np.median(rho_arr):.3f} nats on the m={m_marginal} "
          f"simplex (rho_max = log m = {np.log(m_marginal):.3f})")
    print(f"eta_boot (retired): median {np.median(boot_arr):.5f} nats; n.eta_boot median "
          f"{np.median(boot_arr * n_arr):.3f}  <- roughly constant means eta_boot = O(1/n)")

    # the checklist test, on one day, at increasing subsample size
    p = keep.get(intensivity_day, keep[sorted(keep)[len(keep) // 2]])
    rng = np.random.default_rng(seed)
    sizes = np.array([s for s in (50, 100, 250, 500, 1000, 2000) if s < len(p)])
    e_full = identity_radius(p)[0]
    b_full, _ = bootstrap_radius(p, n_draws=n_draws, seed=seed)
    e_rel, b_rel = [], []
    print(f"\nintensivity on D{intensivity_day:g} (n={len(p)}): ratio to the full-population value")
    print(f"{'m':>7} {'eta':>10} {'eta_boot':>10}")
    for s in sizes:
        idx = rng.choice(len(p), size=int(s), replace=False)
        e_s = identity_radius(p[idx])[0]
        b_s, _ = bootstrap_radius(p[idx], n_draws=max(64, n_draws // 4), seed=seed)
        e_rel.append(e_s / e_full); b_rel.append(b_s / b_full)
        print(f"{s:>7} {e_s / e_full:>10.3f} {b_s / b_full:>10.3f}")

    np.savez(run_dir / "radius_candidates.npz", day=day, n=n_arr, eta=eta_arr,
             mi=mi_arr, ambiguity=amb_arr, k_eff=k_eff, rho=rho_arr, eta_boot=boot_arr,
             sizes=sizes, eta_rel=np.array(e_rel), eta_boot_rel=np.array(b_rel),
             m_marginal=np.array(m_marginal), populations=np.array(pops))
    _plot_identity(day, eta_arr, mi_arr, run_dir / "identity_radius_by_day.png", K)
    _plot_intensivity(sizes, [(r"$\eta=H(\bar\pi)$", e_rel, _LINE),
                              (r"$\eta_{boot}$", b_rel, _WARN)],
                      run_dir / "radius_intensivity.png")
    print(f"\nseries -> {run_dir/'radius_candidates.npz'}")
    print(f"figures -> {run_dir/'identity_radius_by_day.png'}, "
          f"{run_dir/'radius_intensivity.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--days", default=None)
    p.add_argument("--draws", type=int, default=512)
    p.add_argument("--m", type=int, default=500, help="cells in the marginal simplex (rho_max=log m)")
    p.add_argument("--intensivity-day", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.days, a.draws, a.m, a.intensivity_day, a.seed)
