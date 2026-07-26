"""Training utilities for the VaDE GMVAE.

The GMVAE is trained on the WHOLE cell population pooled across all timepoints -- it is not
time-conditional (day feeds the downstream transport layer, not the encoder). So training data is the
per-day ``ExpressionMatrix`` objects concatenated into one big sparse matrix, iterated as
global-shuffled minibatches that mix timepoints (per-day batches would bias BatchNorm and let the
encoder learn a day shortcut).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse


def pool_counts(matrices):
    """Concatenate per-day sparse count matrices into one ``(n_total, n_genes)`` CSR.

    The pooled object is deliberately NOT an ``ExpressionMatrix`` -- that is single-day by contract
    (its ``CellAxis`` carries one day). Cell identity and day are irrelevant to Stage A training, so
    the pool is just the counts; label alignment for Stage B is done separately, once, at pool time.

    Pooling requires a shared gene axis (same genes, same order); a differing order is a hard stop,
    because vstack would silently misalign features. Reordering to a canonical axis is a separate
    step (the gene-order-mismatch task) that must land before training on files whose axes differ.
    """
    if not matrices:
        raise ValueError("pool_counts requires at least one ExpressionMatrix")
    genes = list(matrices[0].gene_names)
    for m in matrices[1:]:
        if list(m.gene_names) != genes:
            raise ValueError("gene axes differ across days; a canonical gene reorder is required "
                             "before pooling (vstack would misalign features)")
    return sparse.vstack([m.counts for m in matrices]).tocsr()


def iter_minibatches(counts, batch_size, rng=None, shuffle=True, drop_last=True,
                     device=None, dtype=torch.float32):
    """Yield dense count minibatches over a pooled CSR, on ``device``.

    One densify per batch -> peak memory is ``batch x genes``. ``drop_last`` guards a real BatchNorm
    crash: BatchNorm1d in train mode raises on a batch of size 1, so a final singleton batch must be
    dropped. Pass a caller-owned ``rng`` (a numpy Generator) so successive epochs get different
    permutations while the whole run stays reproducible; creating the rng inside would reshuffle
    identically every epoch.
    """
    n = counts.shape[0]
    order = rng.permutation(n) if (shuffle and rng is not None) else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:
            break
        dense = np.asarray(counts[idx].todense(), dtype=np.float32)
        yield torch.as_tensor(dense, device=device, dtype=dtype)


def population_gmm_params(posterior, membership, temperature=0.5):
    """The SEED computation: summarise each labelled population as one latent Gaussian, for
    ClusterPrior.initialise. Pure numpy.

    ``posterior`` is a CellPosterior over the labelled cells (their latent means/variances, from the
    Stage-A-trained encoder); ``membership`` is a Membership over the SAME cells (soft weights
    ``w[i,c]``, rows sum <= 1, so overlapping labels are already fractional upstream). Returns
    ``(means[K,d], variances[K,d], weights[K])``.

    Membership supplies the *which* (weights); the posterior supplies the *where* (latent
    coordinates) -- neither alone is enough, since population centroids are points in the encoder's
    latent space. Per population c:

        mu_c   = weighted mean of the cell means            (centroid)
        var_c  = within + between   (law of total variance):
                   within  = weighted mean of cell variances     (measurement uncertainty)
                   between = weighted Var of cell means           (population heterogeneity)
        pi_c   propto n_c ** temperature   (n_c = effective count; temperature<1 tempers the
                                            90:1 fate imbalance so rare fates survive)
    """
    W = membership.for_cells(posterior.cells)                # (n, K), alignment checked, not assumed
    mu, var = posterior.means, posterior.variances           # (n, d)
    n_c = W.sum(0)                                            # (K,) effective per-population counts
    safe = np.maximum(n_c, 1e-12)[:, None]                   # avoid 0/0 for absent populations

    mu_c = (W.T @ mu) / safe                                 # (K, d) weighted centroid
    within = (W.T @ var) / safe                              # (K, d) mean within-cell variance
    between = (W.T @ (mu ** 2)) / safe - mu_c ** 2           # (K, d) Var(mu_i) = E[mu^2] - E[mu]^2
    var_c = within + np.maximum(between, 0.0)                # (K, d) total variance

    empty = n_c == 0                                         # a population with no labelled cells
    if empty.any():                                          # fall back to global stats, weight 0
        mu_c[empty] = mu.mean(0)
        var_c[empty] = var.mean(0) + mu.var(0)
    weights = n_c ** temperature                            # tempered; initialise() softmaxes log
    return mu_c, var_c, weights


def pretrain(model, counts, epochs, batch_size, lr=1e-3, beta=1.0,
             device=None, rng=None, verbose=False):
    """Stage A: train the encoder + decoder (+ per-gene dispersion) as a plain VAE on the pooled
    counts, so the latent has structure before the GMM prior is seeded.

    Optimises ONLY ``model.pretrain_parameters()`` -- the GMM prior (mu_c, logvar_c, pi_logits) is
    left untouched for the SEED phase. Single-device (cuda if available, else cpu); multi-GPU is a
    later step. Returns a ``(epochs, 3)`` array of per-epoch mean ``(loss, recon, kl)`` -- watching
    recon against KL is the pretraining collapse diagnostic.
    """
    device = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    opt = torch.optim.Adam(model.pretrain_parameters(), lr=lr)   # built after .to(device)
    rng = rng if rng is not None else np.random.default_rng(0)

    history = []
    for epoch in range(epochs):
        parts = []
        for x in iter_minibatches(counts, batch_size, rng=rng, device=device):
            opt.zero_grad()
            loss, recon, kl = model.pretrain_loss(x, beta=beta)
            loss.backward()
            opt.step()
            parts.append((float(loss.detach()), float(recon.detach()), float(kl.detach())))
        row = np.mean(parts, axis=0)
        history.append(row)
        if verbose:
            print(f"[pretrain] epoch {epoch + 1}/{epochs}  "
                  f"loss={row[0]:.3f}  recon={row[1]:.3f}  kl={row[2]:.3f}")
    return np.asarray(history)
