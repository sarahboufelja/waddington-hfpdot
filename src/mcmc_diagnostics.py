"""MCMC convergence diagnostics for the Langevin sampler.

Every metric operates on chains shaped ``(num_chains, num_draws, dim)``: one row
per chain, one column per retained draw, trailing parameter axis. Keeping the
chain structure explicit (rather than a flattened ``(N, dim)`` array) is what
makes split-R-hat and eBFMI correct -- the previous implementation flattened the
draws first, which let eBFMI's differences cross chain boundaries.

References:
  - Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021), "Rank-normalization,
    folding, and localization: An improved R-hat...", Bayesian Analysis 16(2).
  - Betancourt (2016), "Diagnosing Suboptimal Cotangent Disintegrations" (eBFMI).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import ndtri


class MCMCDiagnostics:
    """Convergence diagnostics over ``(num_chains, num_draws, dim)`` draws.

    Args:
        log_prob_fn: maps a single latent state ``(dim,)`` to a scalar log-density.
            Required only for :meth:`ebfmi`.
    """

    def __init__(self, log_prob_fn: Callable[[Array], Array] | None = None):
        self.log_prob_fn = log_prob_fn

    # --------------------------------------------------------------------- #
    # Rank-normalized split-R-hat (Vehtari et al. 2021).
    # --------------------------------------------------------------------- #
    @staticmethod
    def _split(chains: Array) -> Array:
        """(C, S, D) -> (2C, S//2, D): split each chain into two half-chains."""
        n_half = chains.shape[1] // 2
        first = chains[:, :n_half, :]
        second = chains[:, n_half: 2 * n_half, :]
        return jnp.concatenate([first, second], axis=0)

    @staticmethod
    def _rank_normalize(chains: Array) -> Array:
        """Rank-normalize per parameter across all (chains x draws). (C, S, D) -> (C, S, D)."""
        C, S, D = chains.shape
        pooled = chains.reshape(C * S, D)
        # Rank within each column (parameter) independently; ties broken by position.
        ranks = jnp.argsort(jnp.argsort(pooled, axis=0), axis=0)  # 0-based ranks
        # Blom transform to z-scores: r is 1-based, hence ranks + 1.
        normalized = (ranks + 1 - 0.375) / (C * S + 0.25)
        z = ndtri(normalized)
        return z.reshape(C, S, D)

    @staticmethod
    def _rhat(chains: Array) -> Array:
        """Classic split-R-hat ANOVA on (C, S, D) -> (D,)."""
        C, S, D = chains.shape
        chain_means = jnp.mean(chains, axis=1)               # (C, D)
        chain_vars = jnp.var(chains, axis=1, ddof=1)         # (C, D)
        W = jnp.mean(chain_vars, axis=0)                     # within  (D,)
        B = S * jnp.var(chain_means, axis=0, ddof=1)         # between (D,)
        var_plus = (S - 1) / S * W + B / S
        return jnp.sqrt(var_plus / jnp.maximum(W, 1e-12))

    @classmethod
    def bulk_rhat(cls, chains: Array) -> Array:
        """Rank-normalized split-R-hat per parameter (the 'bulk' R-hat). -> (D,)."""
        return cls._rhat(cls._rank_normalize(cls._split(chains)))
        
    @classmethod
    def tail_rhat(cls, chains: Array) -> Array:
        """Folded rank-normalized split-R-hat per parameter (sensitive to the tails). -> (D,)."""
        median = jnp.median(chains, axis=(0, 1), keepdims=True)
        folded = jnp.abs(chains - median)
        return cls._rhat(cls._rank_normalize(cls._split(folded)))

    # --------------------------------------------------------------------- #
    # Effective sample size (autocorrelation-based, multi-chain).
    # --------------------------------------------------------------------- #
    @staticmethod
    def ess(chains: Array) -> Array:
        """Effective sample size per parameter: (C, S, D) -> (D,)."""
        C, N, D = chains.shape
        # First, divide the chains into two halves to detect the within-chain variance
        n_half = N // 2
        split = jnp.concatenate([chains[:, :n_half, :], chains[:, n_half: 2 * n_half, :]], axis=0)
        M, Nh, _ = split.shape

        # Biased within-chain auto-covariance via FFT
        centered = split - jnp.mean(split, axis=1, keepdims=True)
        flat_centered = centered.transpose(0, 2, 1).reshape(-1, Nh)
        
        # Next power of two >= 2*Nh - 1: enough zero-padding so the linear autocovariance
        # (lags 0..Nh-1) does not wrap around circularly; power of two is for FFT speed.
        nfft = 1 << (2 * Nh - 1).bit_length()

        freq = jnp.fft.rfft(flat_centered, n=nfft, axis=1)
        psd = freq.real ** 2 + freq.imag ** 2

        flat_acov = jnp.fft.irfft(psd, n=nfft, axis=-1)[:, :Nh]
        acov = flat_acov.reshape(M, D, Nh).transpose(0, 2, 1) / Nh
        
        # Combine the within-chain variance and the autocorrelation

        # Unbiased estimate of the total variance
        W = jnp.mean(acov[:, 0, :] * Nh / (Nh - 1), axis=0) 
        # Unbiased estimate of the between-chain variance
        B = Nh * jnp.var(jnp.mean(split, axis=1), axis=0, ddof=1)
        #  Estimate of the true marginal variance of the target distribution
        var_plus = (Nh - 1) / Nh  * W + B / Nh

        rho = 1.0 - (W[None, :] - jnp.mean(acov, axis=0)) / var_plus[None, :]

        # Use Geyer monotone sequebce pver pair of sums
        K = Nh // 2
        gamma = rho[: 2 * K, :].reshape(K, 2, D).sum(axis=1)
        positive = jnp.cumprod((gamma > 0).astype(chains.dtype), axis=0)
        gamma_monotone = jax.lax.cummin(gamma, axis=0)
        tau = jnp.maximum(-1.0 + 2.0 * jnp.sum(gamma_monotone * positive, axis=0), 1.0)
        return (M * Nh) / tau

    # --------------------------------------------------------------------- #
    # Energy-based fraction of missing information (per chain).
    # --------------------------------------------------------------------- #
    def ebfmi(self, chains: Array) -> Array:
        """eBFMI per chain on the energy E = -log_prob. (C, N, D) -> (C,).

        Differences are taken *within* each chain (the previous implementation
        differenced a flattened array, mixing chains).
        """
        if self.log_prob_fn is None:
            raise ValueError("ebfmi requires a log_prob_fn at construction.")
        C, N = chains.shape[0], chains.shape[1]
        energies = -jax.vmap(jax.vmap(self.log_prob_fn))(chains)
        energies = energies.reshape(C, N)                        # robust to scalar/(1,) returns
        diffs = jnp.diff(energies, axis=1)
        numerator = jnp.mean(diffs ** 2, axis=1)
        denominator = jnp.var(energies, axis=1)
        return numerator / jnp.maximum(denominator, 1e-12)

    # --------------------------------------------------------------------- #
    @staticmethod
    def acceptance_rate(num_accepted: Array, num_steps: int) -> Array:
        """Acceptance fraction per chain from the cumulative accepted count."""
        return jnp.asarray(num_accepted) / num_steps

    # --------------------------------------------------------------------- #
    def summarize(self, chains: Array) -> dict:
        """Compute the full diagnostic suite and conservative scalar summaries."""
        bulk = self.bulk_rhat(chains)
        tail = self.tail_rhat(chains)
        ess = self.ess(chains)
        out = {
            "bulk_rhat": bulk,
            "bulk_rhat_max": jnp.max(bulk),
            "tail_rhat": tail,
            "tail_rhat_max": jnp.max(tail),
            "ess": ess,
            "ess_min": jnp.min(ess),
        }
        if self.log_prob_fn is not None:
            ebfmi = self.ebfmi(chains)
            out |= {"ebfmi": ebfmi, "ebfmi_min": jnp.min(ebfmi)}
        return out
