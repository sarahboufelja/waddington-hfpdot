"""End-to-end tests for the two-stage driver: Stage A -> SEED -> Stage B.

Small synthetic data with two clearly separable populations spread across two timepoints (mixed
within each day, as in real data). The load-bearing test is that the whole pipeline concentrates
labelled cells' responsibilities onto their true fate; the seed test checks the cross-day aggregation.
"""
import os
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.networks import GMVAENet                                        # noqa: E402
from gmvae.embedder import VaDEEmbedder                                    # noqa: E402
from gmvae.training import seed_prior, train, _gmm_from_arrays             # noqa: E402
from wadd_data_ingest import CellAxis, ExpressionMatrix, Membership       # noqa: E402

N_GENES = 12
POPS = ["A", "B"]


def _multiday(days=(9.0, 10.0), per=15, seed=0):
    """Per-day (ExpressionMatrix, Membership) lists: two populations distinguishable by which half
    of the genes they express, both present at every day."""
    rng = np.random.default_rng(seed)
    half = N_GENES // 2
    hi_lo = np.r_[np.full(half, 3.0), np.full(N_GENES - half, 0.2)]
    matrices, memberships = [], []
    for d in days:
        g0 = rng.poisson(hi_lo, size=(per, N_GENES))
        g1 = rng.poisson(hi_lo[::-1], size=(per, N_GENES))
        dense = np.vstack([g0, g1]).astype(np.float32)
        n = 2 * per
        cells = CellAxis(ids=[f"D{d}_c{i}" for i in range(n)], day=d)
        matrices.append(ExpressionMatrix(sparse.csr_matrix(dense), cells,
                                         [f"g{j}" for j in range(N_GENES)]))
        M = np.zeros((n, 2)); M[:per, 0] = 1.0; M[per:, 1] = 1.0
        memberships.append(Membership(M, cells, POPS))
    return matrices, memberships


def _model(seed=0):
    torch.manual_seed(seed)
    return GMVAENet(x_dim=N_GENES, num_clusters=2, latent_dim=4, hidden_dim=16)


# ---- SEED aggregates across days -----------------------------------------------------------------

def test_seed_prior_aggregates_the_embedded_labelled_cells_across_days():
    """seed_prior must set the prior to the GMM reduction of ALL days' embeddings concatenated --
    reproduce that independently and compare."""
    model = _model()
    matrices, memberships = _multiday()
    seed_prior(model, matrices, memberships, POPS, temperature=0.5, device="cpu")

    embedder = VaDEEmbedder(model, device="cpu")               # embed with the (now seed-run) encoder
    mus, vars_, Ws = [], [], []
    for m, memb in zip(matrices, memberships):
        post = embedder.embed(m)
        mus.append(post.means); vars_.append(post.variances); Ws.append(memb.for_cells(post.cells))
    means, vars_c, w = _gmm_from_arrays(np.vstack(Ws), np.vstack(mus), np.vstack(vars_), 0.5)

    assert np.allclose(model.cluster_prior.mu_c.detach().numpy(), means, atol=1e-5)
    assert np.allclose(torch.exp(model.cluster_prior.logvar_c).detach().numpy(), vars_c, atol=1e-5)
    assert np.allclose(torch.softmax(model.cluster_prior.pi_logits, -1).detach().numpy(),
                       w / w.sum(), atol=1e-5)


def test_seed_prior_rejects_mismatched_list_lengths():
    matrices, memberships = _multiday()
    with pytest.raises(ValueError):
        seed_prior(_model(), matrices, memberships[:1], POPS, device="cpu")


# ---- the full driver -----------------------------------------------------------------------------

def test_train_returns_both_stage_histories():
    matrices, memberships = _multiday()
    pre, joint = train(_model(), matrices, memberships, POPS, pretrain_epochs=2, joint_epochs=2,
                       batch_size=10, device="cpu", rng=np.random.default_rng(0))
    assert pre.shape == (2, 3) and joint.shape == (2, 3)
    assert np.all(np.isfinite(pre)) and np.all(np.isfinite(joint))


def test_full_pipeline_concentrates_labels_on_true_fates():
    """Stage A -> SEED -> Stage B end to end: labelled cells' responsibilities must land on their
    true population far better than the untrained model."""
    model = _model()
    matrices, memberships = _multiday()
    x = torch.as_tensor(np.vstack([np.asarray(m.counts.todense()) for m in matrices]),
                        dtype=torch.float32)
    labels = torch.as_tensor(np.concatenate(
        [np.r_[np.zeros(len(m) // 2, int), np.ones(len(m) // 2, int)] for m in matrices]))

    def true_mass(m):
        m.eval()
        with torch.no_grad():
            gamma = m(x).prob_cat
        m.train()
        return float(gamma[torch.arange(len(labels)), labels].mean())

    before = true_mass(model)
    train(model, matrices, memberships, POPS, pretrain_epochs=15, joint_epochs=25, batch_size=10,
          lr=5e-3, lambda_sup=2.0, device="cpu", rng=np.random.default_rng(0))
    after = true_mass(model)
    assert after > 0.75, f"pipeline did not concentrate labels (mean true-fate mass {after:.2f})"
    assert after > before
