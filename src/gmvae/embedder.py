"""VaDEEmbedder -- the concrete torch Embedder that maps a trained GMVAE onto a CellPosterior.

This sits on the torch side of the ``Embedder`` seam: ``wadd_dim_reduction`` defines the Protocol
(``embed(expr) -> CellPosterior``) and stays framework-free, while this module is the one place that
depends on the GMVAE.

The embedding is deterministic by construction, which the downstream radii and membership rely on:

* ``model.eval()`` so the encoder's BatchNorm uses its running statistics -- a cell's latent must not
  depend on which other cells share its minibatch.
* responsibilities are evaluated at the posterior mean ``mu~(x)``, not at a reparametrised sample.
  ``Gaussian.reparametrization`` draws ``randn_like`` regardless of ``eval()``, so a sampled ``z``
  would make ``prob_cat`` random across calls; the mean is the deterministic, standard VaDE choice.
* ``no_grad`` and per-batch densification keep peak memory at ``batch x genes``.

Outputs are cast to float64: the radius/KL math downstream is float64 and cancellation-sensitive, and
the posterior arrays are tiny (``n_cells x latent_dim`` / ``n_cells x K``) next to the dense counts.
"""
from __future__ import annotations

import numpy as np
import torch

from wadd_data_ingest import ExpressionMatrix
from wadd_dim_reduction import CellPosterior
from gmvae.networks import GMVAENet


class VaDEEmbedder:
    """Wrap a trained ``GMVAENet`` as an ``Embedder``: raw counts -> per-cell latent posterior."""

    def __init__(self, model: GMVAENet, device=None, batch_size: int = 4096,
                 dtype: np.dtype | str = np.float64):
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.batch_size = int(batch_size)
        self.dtype = dtype

    @torch.no_grad()
    def embed(self, expr: ExpressionMatrix) -> CellPosterior:
        inf = self.model.inference_net
        was_training = self.model.training
        self.model.eval()                       # BatchNorm -> running stats (batch-independent)
        try:
            means, variances, probs = [], [], []
            n = len(expr.cells)
            for start in range(0, n, self.batch_size):
                idx = np.arange(start, min(start + self.batch_size, n))
                dense = torch.as_tensor(expr.densify(idx, dtype=np.float32), device=self.device)
                gauss = inf.q_zx(dense)                          # deterministic mu, logvar
                log_gamma = inf.responsibilities(gauss.mu)       # gamma at the MEAN, not a sample
                means.append(gauss.mu.cpu().numpy())
                variances.append(torch.exp(gauss.logvar).cpu().numpy())
                probs.append(torch.exp(log_gamma).cpu().numpy())
        finally:
            self.model.train(was_training)                      # leave the model as we found it

        return CellPosterior(
            means=np.concatenate(means).astype(self.dtype),
            variances=np.concatenate(variances).astype(self.dtype),
            prob_cat=np.concatenate(probs).astype(self.dtype),
            cells=expr.cells,
        )
