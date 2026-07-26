"""Contract tests for VaDEEmbedder: a trained GMVAE -> CellPosterior, deterministically.

The two load-bearing tests are batch-invariance and call-to-call determinism. Together they pin the
two design decisions the embedder exists to enforce: eval() (so BatchNorm does not leak batch-mates
into a cell's latent) and responsibilities-at-the-mean (so prob_cat does not inherit the reparam
noise). If either regressed, the radii and the soft membership would quietly become nondeterministic.
"""
import os
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.networks import GMVAENet          # noqa: E402
from gmvae.embedder import VaDEEmbedder       # noqa: E402
from wadd_data_ingest import CellAxis, ExpressionMatrix  # noqa: E402
from wadd_dim_reduction import CellPosterior, MomentMatchedGaussian  # noqa: E402

N_CELLS, N_GENES, LATENT, HIDDEN, K = 20, 30, 8, 16, 4


def _expr(n=N_CELLS, seed=0):
    """A small sparse count matrix in the shape single-cell data arrives."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.3, size=(n, N_GENES)).astype(np.float32)   # ~90% zeros
    cells = CellAxis(ids=[f"c{i}" for i in range(n)], day=9.0)
    return ExpressionMatrix(counts=sparse.csr_matrix(dense), cells=cells,
                            gene_names=[f"g{j}" for j in range(N_GENES)])


def _model(seed=0):
    torch.manual_seed(seed)
    model = GMVAENet(x_dim=N_GENES, num_clusters=K, latent_dim=LATENT, hidden_dim=HIDDEN)
    # Populate BatchNorm running stats with a couple of train-mode passes, so eval() uses something
    # other than the (0, 1) defaults -- makes the batch-invariance test meaningful.
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = torch.as_tensor(_expr(seed=1).densify().astype(np.float32))
    for _ in range(3):
        opt.zero_grad(); model(counts).total_loss.backward(); opt.step()
    return model


# ---- the contract --------------------------------------------------------------------------------

def test_embed_returns_a_valid_cellposterior():
    post = VaDEEmbedder(_model()).embed(_expr())
    assert isinstance(post, CellPosterior)          # its __post_init__ validates shapes/normalisation
    assert post.means.shape == (N_CELLS, LATENT)
    assert post.variances.shape == (N_CELLS, LATENT)
    assert post.prob_cat.shape == (N_CELLS, K)
    assert post.cells.ids == _expr().cells.ids       # cell axis carried through, in order


def test_outputs_are_float64_and_well_formed():
    post = VaDEEmbedder(_model(), dtype=np.float64).embed(_expr())
    assert post.means.dtype == np.float64
    assert np.all(post.variances > 0), "exp(logvar) must be strictly positive"
    assert np.allclose(post.prob_cat.sum(axis=1), 1.0, atol=1e-5)


# ---- decision 1: eval() -> batch-invariance ------------------------------------------------------

def test_embedding_is_invariant_to_batch_size():
    """A cell's posterior must not depend on how the cells were chunked. Only true if BatchNorm runs
    on running stats (eval), not batch stats -- the whole reason eval() is mandatory."""
    model, expr = _model(), _expr()
    one_batch = VaDEEmbedder(model, batch_size=1000).embed(expr)     # all cells at once
    many = VaDEEmbedder(model, batch_size=3).embed(expr)             # 7 ragged batches
    assert np.allclose(one_batch.means, many.means, atol=1e-6)
    assert np.allclose(one_batch.variances, many.variances, atol=1e-6)
    assert np.allclose(one_batch.prob_cat, many.prob_cat, atol=1e-6)


# ---- decision 2: responsibilities at the mean -> determinism -------------------------------------

def test_prob_cat_is_deterministic_across_calls():
    """No seeding between calls. If prob_cat were computed from a reparametrised z it would vary; at
    the posterior mean it is fixed. This is the guard on the load-bearing decision."""
    model, expr = _model(), _expr()
    emb = VaDEEmbedder(model)
    a, b = emb.embed(expr), emb.embed(expr)
    assert np.array_equal(a.prob_cat, b.prob_cat)
    assert np.array_equal(a.means, b.means)


# ---- it leaves the model as it found it ----------------------------------------------------------

def test_model_training_mode_is_restored():
    model = _model()
    model.train()
    VaDEEmbedder(model).embed(_expr())
    assert model.training is True, "embed must restore the model's train/eval mode"
    model.eval()
    VaDEEmbedder(model).embed(_expr())
    assert model.training is False


# ---- it plugs into the radius estimator (the reason CellPosterior exists) -------------------------

def test_posterior_feeds_the_radius_estimator():
    post = VaDEEmbedder(_model()).embed(_expr())
    eta = MomentMatchedGaussian()(post)
    assert np.isfinite(eta) and eta >= 0.0
