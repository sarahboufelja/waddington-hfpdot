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
    for batch, tgt in iter_minibatches(counts, batch_size=5, rng=rng):
        assert batch.shape == (5, N_GENES)
        assert tgt is None                                       # Stage A: no targets
        total += batch.shape[0]
    assert total == 20                                            # 4 full batches, all rows


def test_drop_last_drops_the_partial_final_batch():
    counts = pool_counts([_day(11, 9, 0)])                        # 11 rows, batch 5 -> 2 full + 1
    rng = np.random.default_rng(0)
    sizes = [b.shape[0] for b, _ in iter_minibatches(counts, batch_size=5, rng=rng, drop_last=True)]
    assert sizes == [5, 5]                                        # the size-1 tail is dropped
    sizes_keep = [b.shape[0] for b, _ in iter_minibatches(counts, batch_size=5, rng=rng,
                                                          drop_last=False)]
    assert sizes_keep == [5, 5, 1]


def test_no_batch_of_size_one_reaches_batchnorm():
    """The reason drop_last exists: BatchNorm1d(train) raises on a singleton batch."""
    counts = pool_counts([_day(13, 9, 0)])                        # 13 rows
    rng = np.random.default_rng(0)
    assert all(b.shape[0] > 1 for b, _ in iter_minibatches(counts, batch_size=4, rng=rng))


def test_successive_epochs_shuffle_differently_under_a_shared_rng():
    counts = pool_counts([_day(20, 9, 0)])
    rng = np.random.default_rng(42)
    epoch1 = np.concatenate([b.sum(1).numpy() for b, _ in iter_minibatches(counts, 4, rng=rng)])
    epoch2 = np.concatenate([b.sum(1).numpy() for b, _ in iter_minibatches(counts, 4, rng=rng)])
    assert not np.array_equal(epoch1, epoch2), "a shared rng must give distinct per-epoch order"


def test_shuffle_is_a_permutation_not_a_resample():
    """Every row appears exactly once per epoch -- shuffling reorders, never duplicates/drops.
    Row sums are near-unique here, so the multiset of batch row-sums must equal the full set's."""
    counts = pool_counts([_day(16, 9, 0)])                       # 16 rows, batch 4 -> exact cover
    full = np.asarray(counts.todense()).sum(1)
    got = np.concatenate([b.sum(1).numpy()
                          for b, _ in iter_minibatches(counts, 4, rng=np.random.default_rng(1))])
    assert sorted(got.tolist()) == sorted(full.tolist())


def test_device_and_dtype_are_honoured():
    counts = pool_counts([_day(8, 9, 0)])
    batch, _ = next(iter_minibatches(counts, 4, rng=np.random.default_rng(0), dtype=torch.float64))
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


# ---- build_targets / class_weights / labelled feeder (Stage B plumbing) --------------------------

from gmvae.training import build_targets, class_weights                              # noqa: E402
from wadd_data_ingest import Membership                                              # noqa: E402

POPS = ["A", "B"]


def _day_membership(matrix, day):
    n = len(matrix)
    cells = CellAxis(ids=[f"D{day}_c{i}" for i in range(n)], day=float(day))
    return Membership(matrix=np.asarray(matrix, float), cells=cells, population_names=POPS)


def test_build_targets_renormalises_and_aligns_to_pool_order():
    m0 = _day(2, 9, 0)
    m1 = _day(2, 10, 1)
    memb0 = _day_membership([[1.0, 0.0], [0.0, 0.0]], 9)      # cell0 labelled A; cell1 unlabelled
    memb1 = _day_membership([[0.5, 0.5], [0.3, 0.0]], 10)     # split; partial (0.3) -> renorm to 1
    T = build_targets([m0, m1], [memb0, memb1], POPS)
    assert T.shape == (4, 2)
    assert np.allclose(T[0], [1.0, 0.0])                      # one-hot
    assert np.allclose(T[1], [0.0, 0.0])                      # unlabelled stays zero
    assert np.allclose(T[2], [0.5, 0.5])                      # already a distribution
    assert np.allclose(T[3], [1.0, 0.0])                      # 0.3 on A renormalised to 1.0


def test_build_targets_rejects_a_differing_population_axis():
    m = _day(2, 9, 0)
    memb = _day_membership([[1.0, 0.0], [0.0, 1.0]], 9)
    with pytest.raises(ValueError, match="population axes differ"):
        build_targets([m], [memb], ["B", "A"])               # wrong column order


def test_class_weights_schemes():
    counts = np.array([100.0, 4.0])                          # imbalanced
    assert np.allclose(class_weights(counts, "uniform"), [1.0, 1.0])
    inv = class_weights(counts, "inverse")
    assert inv[1] > inv[0] and np.isclose(inv.mean(), 1.0)   # rare upweighted, mean 1
    assert np.isclose(class_weights(counts, "inverse").tolist()[1]
                      / class_weights(counts, "inverse").tolist()[0], 25.0)   # 100/4
    with pytest.raises(ValueError):
        class_weights(counts, "nonsense")


def test_labelled_feeder_keeps_counts_and_targets_row_aligned():
    """counts_batch[i] and targets_batch[i] must be the same cell -- verified via a row fingerprint
    planted identically in both."""
    n = 12
    counts = pool_counts([_day(n, 9, 0)])
    fingerprint = np.asarray(counts.todense()).sum(1)         # per-row count sum
    targets = np.zeros((n, 2))
    targets[:, 0] = fingerprint                               # plant the same fingerprint in targets
    targets[:, 1] = 1.0                                       # keep rows non-zero (labelled)
    rng = np.random.default_rng(0)
    for x, t in iter_minibatches(counts, 4, targets=targets, rng=rng):
        assert x.shape == (4, N_GENES) and t.shape == (4, 2)
        assert np.allclose(x.sum(1).numpy(), t[:, 0].numpy())  # same permutation applied to both


def test_labelled_feeder_rejects_row_count_mismatch():
    counts = pool_counts([_day(6, 9, 0)])
    with pytest.raises(ValueError, match="rows"):
        next(iter_minibatches(counts, 3, targets=np.zeros((5, 2))))


# ---- joint_train (Stage B) ----------------------------------------------------------------------

from gmvae.training import joint_train                          # noqa: E402


def _two_cluster_counts(per=25, seed=0):
    """Two clearly separable groups: group 0 expresses the first half of genes, group 1 the second.
    Returns pooled counts, one-hot targets, and the true labels."""
    rng = np.random.default_rng(seed)
    half = N_GENES // 2
    g0 = rng.poisson(np.r_[np.full(half, 3.0), np.full(N_GENES - half, 0.2)], size=(per, N_GENES))
    g1 = rng.poisson(np.r_[np.full(half, 0.2), np.full(N_GENES - half, 3.0)], size=(per, N_GENES))
    dense = np.vstack([g0, g1]).astype(np.float32)
    counts = sparse.csr_matrix(dense)
    targets = np.zeros((2 * per, 2)); targets[:per, 0] = 1.0; targets[per:, 1] = 1.0
    labels = np.r_[np.zeros(per, int), np.ones(per, int)]
    return counts, targets, labels


def _model2(seed=0):
    torch.manual_seed(seed)
    return GMVAENet(x_dim=N_GENES, num_clusters=2, latent_dim=4, hidden_dim=16)


def test_joint_train_returns_history():
    counts, targets, _ = _two_cluster_counts()
    hist = joint_train(_model2(), counts, targets, np.ones(2), epochs=3, batch_size=10,
                       device="cpu", rng=np.random.default_rng(0))
    assert hist.shape == (3, 3)                                  # (epochs, [total, elbo, anchor])
    assert np.all(np.isfinite(hist))


def test_joint_train_reduces_total_loss():
    counts, targets, _ = _two_cluster_counts()
    hist = joint_train(_model2(), counts, targets, np.ones(2), epochs=12, batch_size=10, lr=5e-3,
                       device="cpu", rng=np.random.default_rng(0))
    assert hist[-1, 0] < hist[0, 0]


def test_joint_train_updates_the_gmm_prior():
    """The Stage A contrast: in Stage B the prior TRAINS -- mu_c/logvar_c/pi_logits must move."""
    model = _model2()
    before = _gmm_snapshot(model)
    counts, targets, _ = _two_cluster_counts()
    joint_train(model, counts, targets, np.ones(2), epochs=5, batch_size=10, device="cpu",
                rng=np.random.default_rng(0))
    moved = [k for k, v in _gmm_snapshot(model).items() if not torch.equal(v, before[k])]
    assert set(moved) == {"mu_c", "logvar_c", "pi_logits"}, f"prior did not train: only {moved} moved"


def test_anchor_pulls_responsibilities_toward_labels():
    """The anchor doing its job: after joint training, labelled cells' responsibilities concentrate
    on their true population more than at initialisation."""
    model = _model2()
    counts, targets, labels = _two_cluster_counts()
    x = torch.as_tensor(np.asarray(counts.todense()), dtype=torch.float32)
    lab = torch.as_tensor(labels)

    def true_class_mass(m):
        m.eval()
        with torch.no_grad():
            gamma = m(x).prob_cat
        m.train()
        return float(gamma[torch.arange(len(lab)), lab].mean())

    before = true_class_mass(model)
    joint_train(model, counts, targets, np.ones(2), epochs=25, batch_size=10, lr=5e-3,
                lambda_sup=2.0, device="cpu", rng=np.random.default_rng(0))
    after = true_class_mass(model)
    assert after > before + 0.05, f"anchor did not concentrate responsibilities ({before:.2f} -> {after:.2f})"
