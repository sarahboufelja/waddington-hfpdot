"""Tests for the pure transport-plan reductions (matplotlib-free)."""

import numpy as np

from transport_summary import dendrogram_order, transport_plan_summary


def test_dendrogram_order_groups_clusters_contiguously():
    """Three well-separated clusters, shuffled -> the leaf order makes each cluster
    contiguous (so the number of label changes along the order is num_clusters - 1)."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [20.0, 20.0], [40.0, 0.0]])
    pts, labels = [], []
    for c, center in enumerate(centers):
        pts.append(center + rng.normal(scale=0.3, size=(6, 2)))
        labels += [c] * 6
    features = np.concatenate(pts, axis=0)
    labels = np.array(labels)

    perm = rng.permutation(len(features))
    features, labels = features[perm], labels[perm]

    order = dendrogram_order(features)
    assert sorted(order.tolist()) == list(range(len(features)))  # a valid permutation

    reordered = labels[order]
    changes = int(np.sum(reordered[1:] != reordered[:-1]))
    assert changes == len(centers) - 1, (changes, reordered)


def test_dendrogram_order_accepts_1d_coordinate():
    """A 1-D coordinate (e.g. pseudotime) yields a monotonic order (up to reflection)
    with optimal ordering -- close values become adjacent."""
    coord = np.array([3.0, 1.0, 2.0, 0.0, 4.0, 1.5])
    reordered = coord[dendrogram_order(coord, method="average", optimal_ordering=True)]
    ascending = np.all(np.diff(reordered) >= 0)
    descending = np.all(np.diff(reordered) <= 0)
    assert ascending or descending, reordered


def test_dendrogram_order_trivial_sizes():
    assert dendrogram_order(np.empty((0, 2))).tolist() == []
    assert dendrogram_order(np.array([[1.0, 2.0]])).tolist() == [0]


# --------------------------------------------------------------------------- #
# transport_plan_summary
# --------------------------------------------------------------------------- #
def _random_plans(S=400, II=5, JJ=3, seed=0):
    return np.abs(np.random.default_rng(seed).normal(size=(S, II, JJ)))


def test_transport_plan_summary_shapes():
    II, JJ = 5, 3
    s = transport_plan_summary(_random_plans(II=II, JJ=JJ))
    assert s.mean_plan.shape == (II, JJ)
    assert s.std_plan.shape == (II, JJ)
    for arr in (s.mean_first_marginal, s.lo_first_marginal, s.hi_first_marginal):
        assert arr.shape == (II,)
    for arr in (s.mean_second_marginal, s.lo_second_marginal, s.hi_second_marginal):
        assert arr.shape == (JJ,)


def test_transport_plan_summary_mean_and_marginal_consistency():
    plans = _random_plans()
    s = transport_plan_summary(plans)
    assert np.allclose(s.mean_plan, plans.mean(axis=0))
    # Mean of row/col sums equals row/col sums of the mean plan (linearity).
    assert np.allclose(s.mean_first_marginal, s.mean_plan.sum(axis=1))
    assert np.allclose(s.mean_second_marginal, s.mean_plan.sum(axis=0))


def test_transport_plan_summary_band_brackets_mean():
    s = transport_plan_summary(_random_plans())
    assert np.all(s.lo_first_marginal <= s.mean_first_marginal)
    assert np.all(s.mean_first_marginal <= s.hi_first_marginal)
    assert np.all(s.lo_second_marginal <= s.mean_second_marginal)
    assert np.all(s.mean_second_marginal <= s.hi_second_marginal)
