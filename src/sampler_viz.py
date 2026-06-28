"""Plotting and visual-diagnostic helpers for the Langevin sampler.

This module is intentionally separate from ``langevin_sampler`` so that the core
sampler imports without a hard dependency on matplotlib. Import these helpers
only where plotting is actually needed.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

from transport_summary import transport_plan_summary


def plot_transport_plan(
    samples,
    II,
    JJ,
    mu_0=None,
    nu_0=None,
    *,
    log_scale=False,
    row_normalize=False,
    row_order=None,
    col_order=None,
    save_path=None,
):
    """Visualize a posterior over HFPD-OT / OT plans.

    Layout: the posterior-mean plan as a heatmap, flanked by its source (rows, mu)
    and target (cols, nu) marginals with posterior credible bands and the nominal
    marginals mu_0/nu_0 overlaid; a separate per-entry posterior-std heatmap shows
    where the plan is uncertain.

    Args:
        samples: ``(S, II*JJ)`` or ``(S, II, JJ)`` posterior plan samples.
        II, JJ: source / target sizes.
        mu_0, nu_0: optional nominal marginals to overlay as reference.
        log_scale: log color scale for the mean plan (reveals small flows).
        row_normalize: show the mean plan row-normalized (where each source's mass goes).
        row_order, col_order: optional permutations (e.g. from ``dendrogram_order``)
            applied consistently to the plan, std, and marginals.
        save_path: if given, write the figure there.
    """
    plans = np.asarray(samples, dtype=float)
    if plans.ndim == 2:
        plans = plans.reshape(plans.shape[0], II, JJ)

    # Apply the semantic ordering consistently before summarizing.
    if row_order is not None:
        plans = plans[:, np.asarray(row_order), :]
    if col_order is not None:
        plans = plans[:, :, np.asarray(col_order)]
    mu_0 = None if mu_0 is None else np.asarray(mu_0)[row_order] if row_order is not None else np.asarray(mu_0)
    nu_0 = None if nu_0 is None else np.asarray(nu_0)[col_order] if col_order is not None else np.asarray(nu_0)

    s = transport_plan_summary(plans)
    mean_plan, std_plan = np.asarray(s.mean_plan), np.asarray(s.std_plan)

    display_plan = mean_plan
    if row_normalize:
        display_plan = mean_plan / np.maximum(mean_plan.sum(axis=1, keepdims=True), 1e-12)
    norm = LogNorm() if log_scale else None

    rows = np.arange(II)
    cols = np.arange(JJ)

    fig = plt.figure(figsize=(14, 6))
    # Column 2 is a thin spacer that gives the mean colorbar room without its
    # label bleeding into the std heatmap.
    gs = GridSpec(
        2, 4, width_ratios=[1, 4, 0.6, 4], height_ratios=[1, 4],
        wspace=0.05, hspace=0.05, figure=fig,
    )
    ax_nu = fig.add_subplot(gs[0, 1])     # target marginal, above the mean heatmap
    ax_mu = fig.add_subplot(gs[1, 0])     # source marginal, left of the mean heatmap
    ax_mean = fig.add_subplot(gs[1, 1], sharex=ax_nu, sharey=ax_mu)
    ax_std = fig.add_subplot(gs[1, 3])

    # --- mean plan heatmap ---
    im = ax_mean.imshow(display_plan, aspect="auto", origin="upper", cmap="viridis", norm=norm)
    ax_mean.set_xlabel("target (nu)")
    ax_mean.set_yticks([])
    fig.colorbar(im, ax=ax_mean, fraction=0.046, pad=0.04,
                 label="row-normalized mass" if row_normalize else "mass")

    # --- target marginal (top): nu_mean +/- band, nu_0 overlaid ---
    ax_nu.plot(cols, s.mean_second_marginal, color="C0", lw=1.5, label="posterior mean")
    ax_nu.fill_between(cols, s.lo_second_marginal, s.hi_second_marginal, color="C0", alpha=0.3, label="95% CI")
    if nu_0 is not None:
        ax_nu.plot(cols, nu_0, color="k", ls="--", lw=1, label=r"$\nu_0$")
    ax_nu.set_ylabel(r"$\nu$")
    ax_nu.tick_params(labelbottom=False)
    ax_nu.legend(fontsize=7, loc="upper right")

    # --- source marginal (left): mu_mean +/- band, mu_0 overlaid (horizontal) ---
    ax_mu.plot(s.mean_first_marginal, rows, color="C1", lw=1.5)
    ax_mu.fill_betweenx(rows, s.lo_first_marginal, s.hi_first_marginal, color="C1", alpha=0.3)
    if mu_0 is not None:
        ax_mu.plot(mu_0, rows, color="k", ls="--", lw=1)
    ax_mu.set_ylabel("source (mu)")
    ax_mu.set_xlabel(r"$\mu$")
    ax_mu.invert_xaxis()      # grow leftward, toward the heatmap
    # row order (source 0 at top) comes from imshow's origin="upper" via sharey;
    # do NOT invert the y-axis here or it double-flips against the heatmap.

    # --- per-entry posterior std heatmap ---
    im_std = ax_std.imshow(std_plan, aspect="auto", origin="upper", cmap="magma")
    ax_std.set_title("per-entry posterior std")
    ax_std.set_xlabel("target (nu)")
    ax_std.set_yticks([])
    fig.colorbar(im_std, ax=ax_std, fraction=0.046, pad=0.04, label="std")

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def validate_sampler(samples, mu, sigma_diag, reference_samples, base_alpha):
    """Validate the Langevin sampler by overlaying samples on the target mean/std bands.

    Args:
        samples: array of shape (num_samples, D) drawn by the sampler.
        mu: target mean, shape (D,).
        sigma_diag: target marginal variances, shape (D,).
        reference_samples: optional reference draws to overlay, shape (num_ref, D).
        base_alpha: base opacity, scaled down by sqrt(num_samples) for legibility.
    """
    D = mu.shape[0]
    num_samples = samples.shape[0]
    # Adjust the plot opacity to decrease with the sqrt of the number of samples.
    adjusted_alpha = base_alpha / (np.sqrt(num_samples))

    x = np.arange(D)

    # Plot the target mean and std bands.
    plt.plot(x, mu, label="Target Mean", color="red", linestyle="--")
    plt.fill_between(x, mu - 1.96 * np.sqrt(sigma_diag), mu + 1.96 * np.sqrt(sigma_diag), alpha=0.4, color="red")

    for sample in samples:
        plt.plot(x, sample, color="blue", alpha=adjusted_alpha)

    if reference_samples is not None:
        for ref_sample in reference_samples:
            plt.plot(x, ref_sample, color="green", alpha=0.3, linestyle=":")

    sample_proxy = Line2D([0], [0], color="blue", alpha=1.0, label="samples")
    ref_proxy = Line2D([0], [0], color="green", alpha=1.0, linestyle="--", label="reference samples")
    handles = [
        Line2D([], [], color="red", linestyle="--", label="Ideal mean"),
        Line2D([], [], color="red", alpha=0.2, label="1.96 Std bands"),
        sample_proxy,
    ]
    if reference_samples is not None:
        handles.append(ref_proxy)

    plt.legend(handles=handles)

    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


def validate_hyperprior_sampler(samples, first_marginal, second_marginal, first_uncertainty_radius, second_uncertainty_radius, base_alpha):
    """Overlay the sampled transport-plan marginals on the target marginals."""
    D1 = first_marginal.shape[0]
    D2 = second_marginal.shape[0]
    x1 = np.arange(D1)
    x2 = np.arange(D2)

    # Adjust the plot opacity to decrease with the sqrt of the number of samples.
    adjusted_alpha = base_alpha / np.sqrt(samples.shape[0])

    for sample in samples:
        pi_mat = jnp.reshape(sample, (first_marginal.shape[0], second_marginal.shape[0]))
        mu = jnp.sum(pi_mat, axis=1)
        nu = jnp.sum(pi_mat, axis=0)

        # Plot the target first marginal.
        plt.plot(x1, first_marginal, label="Target First Marginal", color="darkblue", linestyle="--")
        plt.plot(x1, mu, color="red", alpha=adjusted_alpha)
        plt.xlabel("Dimension of First Marginal")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()

        # Plot the target second marginal.
        plt.plot(x2, second_marginal, label="Target Second Marginal", color="darkblue", linestyle="--")
        plt.plot(x2, nu, color="red", alpha=adjusted_alpha)
        plt.xlabel("Dimension of Second Marginal")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.show()


def visualise_rhat(rhat_vec, dim):
    """Render a per-entry R-hat convergence heatmap for a (dim x dim) transport plan."""
    rhat_mat = rhat_vec.reshape((dim, dim))
    plt.figure(figsize=(8, 10))
    plt.imshow(rhat_mat, cmap="YlOrRd", vmin=1, vmax=1.1)
    plt.colorbar()
    plt.title("Convergence heatmap (R-hat)")
    plt.show()


def plot_transport_plans_traces(samples, num_chains, num_samples):
    """Plot per-entry MCMC traces for a 10x10 transport plan across chains."""
    grid_samples = samples.reshape(num_chains, num_samples, 10, 10)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True)
    for i in range(10):
        for j in range(10):
            for chain in range(num_chains):
                axes[i, j].plot(grid_samples[chain, :, i, j], alpha=0.5)
            axes[i, j].set_title(f"Entry ({i}, {j})")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.tight_layout()
    plt.show()
