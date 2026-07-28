"""Training utilities for the VaDE GMVAE.

The GMVAE is trained on the WHOLE cell population pooled across all timepoints -- it is not
time-conditional (day feeds the downstream transport layer, not the encoder). So training data is the
per-day ``ExpressionMatrix`` objects concatenated into one big sparse matrix, iterated as
global-shuffled minibatches that mix timepoints (per-day batches would bias BatchNorm and let the
encoder learn a day shortcut).

Two layers live here:

* **setup** (numpy, run once): ``pool_counts``, ``build_targets``, ``population_gmm_params``,
  ``class_weights`` -- prepare the big arrays from numpy inputs. Not differentiable, not in any hot
  loop, so numpy/CPU, and deliberately kept CPU so the per-batch GPU transfer happens in one place.
* **runtime** (torch, per-batch): ``iter_minibatches`` / ``iter_labelled_minibatches`` (the GPU
  boundary) and the training loops. These slice the setup arrays one batch at a time and move that
  batch to the device.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import sparse

from gmvae.embedder import VaDEEmbedder
from gmvae.networks import GMVAENet
from wadd_data_ingest import ExpressionMatrix, Membership
from wadd_dim_reduction import CellPosterior
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

Array = np.ndarray
CSR = sparse.csr_matrix


# --------------------------------------------------------------------------------------------------
# Setup layer (numpy, one-time)
# --------------------------------------------------------------------------------------------------

def pool_counts(matrices: Sequence[ExpressionMatrix]) -> CSR:
    """Concatenate per-day sparse count matrices into one ``(n_total, n_genes)`` CSR.

    The pooled object is deliberately NOT an ``ExpressionMatrix`` -- that is single-day by contract
    (its ``CellAxis`` carries one day). Cell identity and day are irrelevant to Stage A training, so
    the pool is just the counts; label alignment for Stage B is done separately, once (see
    ``build_targets``).

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


def build_targets(matrices: Sequence[ExpressionMatrix], memberships: Sequence[Membership],
                  population_names: Sequence[str]) -> Array:
    """Row-aligned soft label targets ``(n_total, K)`` for the pooled counts (Stage B anchor).

    ``matrices`` and ``memberships`` are parallel per-day lists in the SAME order as ``pool_counts``,
    so the target rows line up with the pooled count rows. Each day's membership is aligned to that
    day's cells (``for_cells`` runs ``require_same`` on day + cell order), then renormalised to a
    distribution over the ``K`` populations for labelled cells (row sums to 1) and left all-zero for
    unlabelled cells -- which therefore contribute nothing to the anchor.

    ``population_names`` fixes the column order; it must match the order used to seed the GMM prior,
    so a target column indexes the same component as the corresponding responsibility.
    """
    if len(matrices) != len(memberships):
        raise ValueError(f"{len(matrices)} matrices but {len(memberships)} memberships")
    names = list(population_names)
    rows = []
    for exp, memb in zip(matrices, memberships):
        if list(memb.population_names) != names:
            logger.info("population axes differ across days; proceeding to columns alignment before "
                             "building targets")                       # K population alignment
        ord_memb = memb.align_populations(names)
        M = ord_memb.for_cells(exp.cells)                              # (n_day, K), alignment checked
        s = M.sum(1, keepdims=True)
        rows.append(np.divide(M, s, out=np.zeros_like(M), where=s > 0))   # labelled -> sum 1; else 0
    return np.vstack(rows)


def class_weights(counts_per_pop: Array, scheme: str = "uniform") -> Array:
    """Per-population balance weights ``w_c`` for the anchor, normalised to mean 1.

    ``uniform`` (all ones) is the starting choice; ``inverse`` / ``inverse_sqrt`` upweight rare fates
    to counter the ~90:1 imbalance, to be tuned once the system runs. Normalising to mean 1 keeps the
    anchor's overall scale (and thus ``lambda_sup``) comparable across schemes.
    """
    n = np.maximum(np.asarray(counts_per_pop, dtype=float), 1e-12)
    if scheme == "uniform":
        w = np.ones_like(n)
    elif scheme == "inverse":
        w = 1.0 / n
    elif scheme == "inverse_sqrt":
        w = 1.0 / np.sqrt(n)
    else:
        raise ValueError(f"unknown class-weight scheme {scheme!r}")
    return w * (len(w) / w.sum())                               # mean(w) == 1


def population_gmm_params(posterior: CellPosterior, membership: Membership,
                          temperature: float = 0.5) -> Tuple[Array, Array, Array]:
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
    return _gmm_from_arrays(W, posterior.means, posterior.variances, temperature)


def _gmm_from_arrays(W: Array, mu: Array, var: Array, temperature: float) -> Tuple[Array, Array,
                                                                                   Array]:
    """The weighted GMM reduction shared by the single- and multi-day seed paths. ``W`` (n, K) is the
    membership weights, ``mu``/``var`` (n, d) the cells' latent moments; rows may be pooled across
    days (unlabelled cells carry a zero W row and so contribute nothing)."""
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


# --------------------------------------------------------------------------------------------------
# Runtime layer (torch, per-batch): the sparse->dense->device boundary
# --------------------------------------------------------------------------------------------------

def _batch_slices(n: int, batch_size: int, rng: Optional[np.random.Generator],
                  shuffle: bool, drop_last: bool) -> Iterator[Array]:
    """Yield row-index slices for one epoch. Shared by the labelled/unlabelled feeders so the
    shuffle, batching, and drop_last logic lives in exactly one place. A caller-owned ``rng`` gives
    reproducible per-epoch permutations; creating one inside would reshuffle identically every epoch.
    """
    order = rng.permutation(n) if (shuffle and rng is not None) else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if drop_last and len(idx) < batch_size:                # only the final slice can be short
            return
        yield idx


def _dense_batch(counts: CSR, idx: Array, device, dtype: torch.dtype) -> torch.Tensor:
    """Densify the given rows and move them to the device. The one sparse->dense->GPU boundary."""
    dense = np.asarray(counts[idx].todense(), dtype=np.float32)
    return torch.as_tensor(dense, device=device, dtype=dtype)


def iter_minibatches(counts: CSR, batch_size: int, targets: Optional[Array] = None,
                     rng: Optional[np.random.Generator] = None, shuffle: bool = True,
                     drop_last: bool = True, device=None, dtype: torch.dtype = torch.float32
                     ) -> Iterator[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Yield ``(counts_batch, targets_batch)`` minibatches over a pooled CSR, on ``device``.

    ``targets`` is the row-aligned label array from ``build_targets`` (Stage B); pass ``None`` for
    Stage A, in which case the second element of each yield is ``None``. Counts and targets are always
    sliced by the SAME shuffle, so their rows stay aligned.

    One densify per batch -> peak memory is ``batch x genes``. ``drop_last`` guards a real BatchNorm
    crash: BatchNorm1d in train mode raises on a batch of size 1, so a final singleton batch is
    dropped.
    """
    if targets is not None and targets.shape[0] != counts.shape[0]:
        raise ValueError(f"targets has {targets.shape[0]} rows for {counts.shape[0]} count rows")
    for idx in _batch_slices(counts.shape[0], batch_size, rng, shuffle, drop_last):
        x = _dense_batch(counts, idx, device, dtype)
        t = None if targets is None else torch.as_tensor(targets[idx], device=device, dtype=dtype)
        yield x, t


def pretrain(model: GMVAENet, counts: CSR, epochs: int, batch_size: int, lr: float = 1e-3,
             beta: float = 1.0, device=None, rng: Optional[np.random.Generator] = None,
             verbose: bool = False) -> Array:
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
        for x, _ in iter_minibatches(counts, batch_size, rng=rng, device=device):
            opt.zero_grad()
            loss, recon, kl = model.pretrain_loss(x, beta=beta)
            loss.backward()
            opt.step()
            # Keep the scalars on-device and sync once per epoch: reading a GPU scalar (float()/
            # .item()/.cpu()) forces a host synchronisation, so doing it per batch stalls the loop.
            parts.append(torch.stack((loss.detach(), recon.detach(), kl.detach())))
        row = torch.stack(parts).mean(0).cpu().numpy()
        history.append(row)
        if verbose:
            print(f"[pretrain] epoch {epoch + 1}/{epochs}  "
                  f"loss={row[0]:.3f}  recon={row[1]:.3f}  kl={row[2]:.3f}")
    return np.asarray(history)


def joint_train(model: GMVAENet, counts: CSR, targets: Array, class_weights: Array, epochs: int,
                batch_size: int, lambda_sup: float = 1.0, lr: float = 1e-3, device=None,
                rng: Optional[np.random.Generator] = None, verbose: bool = False) -> Array:
    """Stage B: joint training on the seeded model. Per batch,

        L = ELBO(all cells)  +  lambda_sup * anchor(labelled cells),

    optimised over ALL parameters -- crucially the GMM prior trains now (unlike Stage A), and the
    anchor is what keeps it pinned to the populations while it moves. The ELBO's KL weight is the
    model's own ``beta``; ``lambda_sup`` is the fixed anchor weight (both tuned later). ``targets``
    is from ``build_targets`` (row-aligned to ``counts``); ``class_weights`` is per-population.
    Returns a ``(epochs, 3)`` array of per-epoch mean ``(total, elbo, anchor)``.
    """
    device = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)           # ALL params, incl. the GMM prior
    cw = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    rng = rng if rng is not None else np.random.default_rng(0)

    history = []
    for epoch in range(epochs):
        parts = []
        for x, t in iter_minibatches(counts, batch_size, targets=targets, rng=rng, device=device):
            opt.zero_grad()
            out = model(x)                                      # VaDE forward -> ELBO in out.total_loss
            anchor = model.losses.anchor_loss(out.log_gamma_c, t, cw)
            loss = out.total_loss + lambda_sup * anchor
            loss.backward()
            opt.step()
            # On-device accumulation, one host sync per epoch (see pretrain for the rationale).
            parts.append(torch.stack((loss.detach(), out.total_loss.detach(), anchor.detach())))
        row = torch.stack(parts).mean(0).cpu().numpy()
        history.append(row)
        if verbose:
            print(f"[joint] epoch {epoch + 1}/{epochs}  "
                  f"loss={row[0]:.3f}  elbo={row[1]:.3f}  anchor={row[2]:.3f}")
    return np.asarray(history)


# --------------------------------------------------------------------------------------------------
# SEED phase orchestration + the full two-stage driver
# --------------------------------------------------------------------------------------------------

def seed_prior(model: GMVAENet, matrices: Sequence[ExpressionMatrix],
               memberships: Sequence[Membership], population_names: Sequence[str],
               temperature: float = 0.5, batch_size: int = 4096, device=None) -> None:
    """SEED: population-seed the GMM prior from the Stage-A-trained encoder, in place.

    Embeds each day's cells (``VaDEEmbedder``, eval + no_grad) and aggregates the labelled cells
    ACROSS days into one weighted GMM reduction, then writes it into ``model.cluster_prior``. The
    whole day is embedded; unlabelled cells carry a zero membership row and contribute nothing, so no
    subsetting is needed. ``matrices`` and ``memberships`` are parallel per-day lists.
    """
    if len(matrices) != len(memberships):
        raise ValueError(f"{len(matrices)} matrices but {len(memberships)} memberships")
    names = list(population_names)
    embedder = VaDEEmbedder(model, device=device, batch_size=batch_size)

    mus, variances, weights = [], [], []
    for exp, memb in zip(matrices, memberships):
        if list(memb.population_names) != names:
            logger.info("population axes differ across days; align columns before seeding")
        ord_memb = memb.align_populations(names)                  # population alignment checked
        post = embedder.embed(exp)                                # CellPosterior over this day's cells
        weights.append(ord_memb.for_cells(post.cells))            # (n_day, K), alignment checked
        mus.append(post.means)
        variances.append(post.variances)

    W = np.vstack(weights)                                    # all cells across days
    means, vars_c, w_c = _gmm_from_arrays(W, np.vstack(mus), np.vstack(variances), temperature)
    model.cluster_prior.initialise(means, vars_c, w_c)


def train(model: GMVAENet, matrices: Sequence[ExpressionMatrix],
          memberships: Sequence[Membership], population_names: Sequence[str],
          pretrain_epochs: int, joint_epochs: int, batch_size: int, lambda_sup: float = 1.0,
          class_weight_scheme: str = "uniform", temperature: float = 0.5, lr: float = 1e-3,
          device=None, rng: Optional[np.random.Generator] = None,
          verbose: bool = False) -> Tuple[Array, Array]:
    """The full two-stage driver: Stage A (pretrain) -> SEED (population-seed the prior) -> Stage B
    (joint ELBO + anchor). Single-device; multi-GPU is a later step. Returns
    ``(pretrain_history, joint_history)``.

    ``matrices``/``memberships`` are parallel per-day lists; ``population_names`` fixes the K-column
    order used for both the seed and the anchor targets, so a component always indexes the same fate.
    """
    device = torch.device(device) if device is not None else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = rng if rng is not None else np.random.default_rng(0)

    counts = pool_counts(matrices)                            # all days, one CSR
    pre_hist = pretrain(model, counts, pretrain_epochs, batch_size, lr=lr, device=device, rng=rng,
                        verbose=verbose)                     # Stage A (GMM prior untouched)
    seed_prior(model, matrices, memberships, population_names, temperature=temperature,
               batch_size=batch_size, device=device)         # SEED

    targets = build_targets(matrices, memberships, population_names)
    counts_per_pop = sum(np.asarray(memb.matrix).sum(0) for memb in memberships)   # (K,)
    cw = class_weights(counts_per_pop, class_weight_scheme)
    joint_hist = joint_train(model, counts, targets, cw, joint_epochs, batch_size,
                             lambda_sup=lambda_sup, lr=lr, device=device, rng=rng, verbose=verbose)  # Stage B (GMM trained)
    return pre_hist, joint_hist
