"""Feeder tests: pooling per-day matrices and iterating global-shuffled minibatches.

Pure CPU -- no training, no GPU. The pool guards the gene-axis assumption (misaligned features would
be a silent correctness bug); the iterator guards full coverage, BatchNorm-safe batch sizes, and
per-epoch shuffling.
"""
import os
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.training import pool_counts, iter_minibatches, pretrain  # noqa: E402
from gmvae.networks import GMVAENet                                  # noqa: E402
from wadd_data_ingest import CellAxis, ExpressionMatrix              # noqa: E402

N_GENES = 12


def _day(n, day, seed):
    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.3, size=(n, N_GENES)).astype(np.float32)
    cells = CellAxis(ids=[f"D{day}_c{i}" for i in range(n)], day=float(day))
    return ExpressionMatrix(counts=sparse.csr_matrix(dense), cells=cells,
                            gene_names=[f"g{j}" for j in range(N_GENES)])


# ---- pool_counts --------------------------------------------------------------------------------

def test_pool_concatenates_days_in_order():
    mats = [_day(5, 9, 0), _day(7, 10, 1), _day(3, 11, 2)]
    pooled = pool_counts(mats)
    assert pooled.shape == (15, N_GENES)
    assert sparse.issparse(pooled)
    # rows are the days stacked in the given order
    assert np.array_equal(np.asarray(pooled[:5].todense()), np.asarray(mats[0].counts.todense()))
    assert np.array_equal(np.asarray(pooled[5:12].todense()), np.asarray(mats[1].counts.todense()))


def test_pool_rejects_a_differing_gene_axis():
    a = _day(4, 9, 0)
    b = _day(4, 10, 1)
    scrambled = ExpressionMatrix(counts=b.counts, cells=b.cells,
                                 gene_names=list(reversed(b.gene_names)))   # same set, wrong order
    with pytest.raises(ValueError, match="gene axes differ"):
        pool_counts([a, scrambled])


def test_pool_requires_at_least_one_matrix():
    with pytest.raises(ValueError):
        pool_counts([])


# ---- iter_minibatches ---------------------------------------------------------------------------

def test_batches_cover_every_row_exactly_once_when_divisible():
    counts = pool_counts([_day(10, 9, 0), _day(10, 10, 1)])       # 20 rows
    rng = np.random.default_rng(0)
    total = 0
    for batch in iter_minibatches(counts, batch_size=5, rng=rng):
        assert batch.shape == (5, N_GENES)
        total += batch.shape[0]
    assert total == 20                                            # 4 full batches, all rows


def test_drop_last_drops_the_partial_final_batch():
    counts = pool_counts([_day(11, 9, 0)])                        # 11 rows, batch 5 -> 2 full + 1
    rng = np.random.default_rng(0)
    sizes = [b.shape[0] for b in iter_minibatches(counts, batch_size=5, rng=rng, drop_last=True)]
    assert sizes == [5, 5]                                        # the size-1 tail is dropped
    sizes_keep = [b.shape[0] for b in iter_minibatches(counts, batch_size=5, rng=rng,
                                                       drop_last=False)]
    assert sizes_keep == [5, 5, 1]


def test_no_batch_of_size_one_reaches_batchnorm():
    """The reason drop_last exists: BatchNorm1d(train) raises on a singleton batch."""
    counts = pool_counts([_day(13, 9, 0)])                        # 13 rows
    rng = np.random.default_rng(0)
    assert all(b.shape[0] > 1 for b in iter_minibatches(counts, batch_size=4, rng=rng))


def test_successive_epochs_shuffle_differently_under_a_shared_rng():
    counts = pool_counts([_day(20, 9, 0)])
    rng = np.random.default_rng(42)
    epoch1 = np.concatenate([b.sum(1).numpy() for b in iter_minibatches(counts, 4, rng=rng)])
    epoch2 = np.concatenate([b.sum(1).numpy() for b in iter_minibatches(counts, 4, rng=rng)])
    assert not np.array_equal(epoch1, epoch2), "a shared rng must give distinct per-epoch order"


def test_shuffle_is_a_permutation_not_a_resample():
    """Every row appears exactly once per epoch -- shuffling reorders, never duplicates/drops.
    Row sums are near-unique here, so the multiset of batch row-sums must equal the full set's."""
    counts = pool_counts([_day(16, 9, 0)])                       # 16 rows, batch 4 -> exact cover
    full = np.asarray(counts.todense()).sum(1)
    got = np.concatenate([b.sum(1).numpy()
                          for b in iter_minibatches(counts, 4, rng=np.random.default_rng(1))])
    assert sorted(got.tolist()) == sorted(full.tolist())


def test_device_and_dtype_are_honoured():
    counts = pool_counts([_day(8, 9, 0)])
    batch = next(iter_minibatches(counts, 4, rng=np.random.default_rng(0), dtype=torch.float64))
    assert batch.dtype == torch.float64
    assert batch.device.type == "cpu"


# ---- pretrain loop (Stage A) --------------------------------------------------------------------

def _model(seed=0):
    torch.manual_seed(seed)
    return GMVAENet(x_dim=N_GENES, num_clusters=3, latent_dim=4, hidden_dim=16)


def _gmm_snapshot(model):
    cp = model.cluster_prior
    return {k: v.detach().clone() for k, v in
            (("mu_c", cp.mu_c), ("logvar_c", cp.logvar_c), ("pi_logits", cp.pi_logits))}


def test_pretrain_returns_per_epoch_history():
    counts = pool_counts([_day(40, 9, 0), _day(40, 10, 1)])
    hist = pretrain(_model(), counts, epochs=3, batch_size=16, device="cpu",
                    rng=np.random.default_rng(0))
    assert hist.shape == (3, 3)                                   # (epochs, [loss, recon, kl])
    assert np.all(np.isfinite(hist))


def test_pretrain_reduces_the_loss():
    counts = pool_counts([_day(60, 9, 0), _day(60, 10, 1)])
    hist = pretrain(_model(), counts, epochs=8, batch_size=16, lr=1e-3, device="cpu",
                    rng=np.random.default_rng(0))
    assert hist[-1, 0] < hist[0, 0]                              # mean epoch loss trends down


def test_pretrain_leaves_the_gmm_prior_untouched_end_to_end():
    """The Stage A isolation guarantee, through the real loop: after a full pretrain run the GMM
    prior is byte-identical -- SEED must fit to an unmoved prior."""
    model = _model()
    before = _gmm_snapshot(model)
    counts = pool_counts([_day(50, 9, 0)])
    pretrain(model, counts, epochs=4, batch_size=16, device="cpu", rng=np.random.default_rng(0))
    for name, tensor in _gmm_snapshot(model).items():
        assert torch.equal(tensor, before[name]), f"{name} moved during Stage A"
