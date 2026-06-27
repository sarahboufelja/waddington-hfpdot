"""Plotting and visual-diagnostic helpers for the Langevin sampler.

This module is intentionally separate from ``langevin_sampler`` so that the core
sampler imports without a hard dependency on matplotlib. Import these helpers
only where plotting is actually needed.
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
