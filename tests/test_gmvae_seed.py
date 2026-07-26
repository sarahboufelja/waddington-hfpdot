"""SEED phase tests: writing a fit into the GMM prior (ClusterPrior.initialise).

The load-bearing test is the round-trip through the model: after seeding well-separated components,
the responsibilities evaluated at a component's own mean must be near-one-hot on that component --
i.e. the seed actually lands where the encoder will read it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from gmvae.networks import ClusterPrior, GMVAENet  # noqa: E402

K, D = 4, 6


def _prior(seed=0):
    torch.manual_seed(seed)
    return ClusterPrior(K, D)


# ---- the parameters are written exactly (up to the log-space representation) ---------------------

def test_initialise_sets_the_prior_from_the_fit():
    prior = _prior()
    means = torch.randn(K, D)
    variances = torch.rand(K, D) + 0.1
    weights = torch.tensor([0.4, 0.3, 0.2, 0.1])
    prior.initialise(means, variances, weights)
    assert torch.allclose(prior.mu_c, means)
    assert torch.allclose(torch.exp(prior.logvar_c), variances, atol=1e-6)      # logvar = log(var)
    assert torch.allclose(torch.softmax(prior.pi_logits, -1), weights, atol=1e-6)  # softmax recovers pi


def test_weights_need_not_presum_to_one():
    """pi_logits = log(weights); softmax normalises, so any non-negative weights are accepted."""
    prior = _prior()
    prior.initialise(torch.randn(K, D), torch.rand(K, D) + 0.1, torch.tensor([4.0, 3.0, 2.0, 1.0]))
    assert torch.allclose(torch.softmax(prior.pi_logits, -1),
                          torch.tensor([0.4, 0.3, 0.2, 0.1]), atol=1e-6)


def test_accepts_numpy_arrays():
    prior = _prior()
    prior.initialise(np.zeros((K, D)), np.ones((K, D)), np.full(K, 1.0 / K))
    assert torch.allclose(prior.mu_c, torch.zeros(K, D))


# ---- the round-trip that matters: seeded components are recoverable by responsibilities ----------

def test_responsibilities_are_near_one_hot_at_seeded_means():
    """Seed K well-separated components, then ask the model for q(c|z) at each component's own mean.
    It must return near-one-hot on that component -- the seed lands where inference reads it."""
    model = GMVAENet(x_dim=20, num_clusters=K, latent_dim=D, hidden_dim=16)
    means = torch.eye(K, D) * 10.0                          # far-apart component means
    model.cluster_prior.initialise(means, torch.full((K, D), 0.1), torch.full((K,), 1.0 / K))
    log_gamma = model.inference_net.responsibilities(means)  # evaluate q(c|z) at the means
    assert torch.allclose(torch.exp(log_gamma), torch.eye(K), atol=1e-3)


# ---- floors and validation ----------------------------------------------------------------------

def test_zero_variance_or_empty_population_stays_finite():
    """A one-cell population (zero within-population variance) or an absent one (zero weight) must not
    produce log(0) = -inf."""
    prior = _prior()
    variances = torch.rand(K, D) + 0.1
    variances[0] = 0.0                                      # a single-cell population
    weights = torch.tensor([0.5, 0.5, 0.0, 0.0])           # two absent populations
    prior.initialise(torch.randn(K, D), variances, weights)
    assert torch.isfinite(prior.logvar_c).all()
    assert torch.isfinite(prior.pi_logits).all()


@pytest.mark.parametrize("bad", [
    (torch.zeros(K + 1, D), torch.ones(K + 1, D), torch.ones(K + 1)),   # wrong K
    (torch.zeros(K, D + 1), torch.ones(K, D + 1), torch.ones(K)),       # wrong d
])
def test_shape_mismatch_raises(bad):
    with pytest.raises(ValueError):
        _prior().initialise(*bad)


def test_negative_variance_or_weight_raises():
    prior = _prior()
    with pytest.raises(ValueError):
        prior.initialise(torch.zeros(K, D), -torch.ones(K, D), torch.ones(K))
    with pytest.raises(ValueError):
        prior.initialise(torch.zeros(K, D), torch.ones(K, D), -torch.ones(K))


def test_initialise_does_not_leave_gradients_and_params_stay_learnable():
    prior = _prior()
    prior.initialise(torch.randn(K, D), torch.rand(K, D) + 0.1, torch.full((K,), 1.0 / K))
    for p in (prior.mu_c, prior.logvar_c, prior.pi_logits):
        assert p.requires_grad and p.grad is None            # still trainable, no stray grad


# ---- population_gmm_params: the SEED computation -------------------------------------------------

from gmvae.training import population_gmm_params            # noqa: E402
from wadd_dim_reduction import CellPosterior                 # noqa: E402
from wadd_data_ingest import CellAxis, Membership            # noqa: E402


def _posterior(means, variances):
    means = np.asarray(means, float)
    n, d = means.shape
    cells = CellAxis(ids=[f"c{i}" for i in range(n)], day=9.0)
    return CellPosterior(means=means, variances=np.asarray(variances, float),
                         prob_cat=np.full((n, 2), 0.5), cells=cells)


def _membership(matrix, cells, names):
    return Membership(matrix=np.asarray(matrix, float), cells=cells, population_names=names)


def test_centroid_and_total_variance_on_a_hand_case():
    """Two one-hot populations; check mu_c, within, between, var_c against a hand computation."""
    means = [[0.0, 0.0], [2.0, 0.0],      # population 0: cell means 0 and 2 -> centroid 1
             [10.0, 10.0]]                 # population 1: single cell
    variances = [[0.5, 0.5], [0.5, 0.5], [0.3, 0.3]]
    post = _posterior(means, variances)
    memb = _membership([[1, 0], [1, 0], [0, 1]], post.cells, ["A", "B"])
    mu_c, var_c, w = population_gmm_params(post, memb, temperature=1.0)

    assert np.allclose(mu_c[0], [1.0, 0.0])                  # centroid of 0 and 2
    assert np.allclose(mu_c[1], [10.0, 10.0])               # the single cell
    # pop 0: within = 0.5, between (dim 0) = mean([1,1]) = 1 -> var = 1.5; dim 1 within only = 0.5
    assert np.allclose(var_c[0], [1.5, 0.5])
    assert np.allclose(var_c[1], [0.3, 0.3])                # one cell -> between 0, within only
    assert np.allclose(w, [2.0, 1.0])                       # counts at temperature=1


def test_overlap_splits_a_cell_fractionally():
    """A cell half in each population contributes half its mean to each centroid."""
    post = _posterior([[0.0, 0.0], [4.0, 0.0]], [[0.1, 0.1], [0.1, 0.1]])
    # cell 0 fully in A; cell 1 split 0.5/0.5 between A and B
    memb = _membership([[1.0, 0.0], [0.5, 0.5]], post.cells, ["A", "B"])
    mu_c, _, w = population_gmm_params(post, memb, temperature=1.0)
    assert np.allclose(mu_c[0], [(0.0 * 1 + 4.0 * 0.5) / 1.5, 0.0])   # weighted centroid of A
    assert np.allclose(mu_c[1], [4.0, 0.0])                            # B sees only cell 1
    assert np.allclose(w, [1.5, 0.5])                                  # effective counts


def test_temperature_flattens_the_imbalance():
    """tau<1 shrinks the ratio between a big and a rare population's prior weight."""
    means = np.zeros((110, 2))
    post = _posterior(means, np.ones((110, 2)) * 0.1)
    M = np.zeros((110, 2)); M[:100, 0] = 1; M[100:, 1] = 1     # 100 vs 10 -> 10:1
    memb = _membership(M, post.cells, ["big", "rare"])
    _, _, w_raw = population_gmm_params(post, memb, temperature=1.0)
    _, _, w_temp = population_gmm_params(post, memb, temperature=0.5)
    assert np.isclose(w_raw[0] / w_raw[1], 10.0)
    assert w_temp[0] / w_temp[1] < w_raw[0] / w_raw[1]        # tempered ratio is smaller (sqrt10)
    assert np.isclose(w_temp[0] / w_temp[1], np.sqrt(10.0))


def test_empty_population_falls_back_to_global_stats_with_zero_weight():
    post = _posterior([[0.0, 0.0], [2.0, 2.0]], [[0.1, 0.1], [0.1, 0.1]])
    memb = _membership([[1, 0, 0], [1, 0, 0]], post.cells, ["A", "B", "C"])  # B, C empty
    mu_c, var_c, w = population_gmm_params(post, memb)
    assert w[1] == 0.0 and w[2] == 0.0                        # absent -> zero (floored later)
    assert np.allclose(mu_c[1], post.means.mean(0))           # global fallback centroid
    assert np.isfinite(var_c[1]).all() and np.isfinite(var_c[2]).all()


def test_seed_end_to_end_recovers_populations():
    """The whole SEED unit: params -> initialise -> responsibilities near-one-hot at each centroid."""
    from gmvae.networks import GMVAENet
    means = np.concatenate([np.zeros((5, 6)), np.ones((5, 6)) * 8.0])   # two far-apart pops
    post = _posterior(means, np.full((10, 6), 0.1))
    M = np.zeros((10, 2)); M[:5, 0] = 1; M[5:, 1] = 1
    memb = _membership(M, post.cells, ["A", "B"])
    model = GMVAENet(x_dim=20, num_clusters=2, latent_dim=6, hidden_dim=16)
    model.cluster_prior.initialise(*population_gmm_params(post, memb))
    log_gamma = model.inference_net.responsibilities(torch.as_tensor(model.cluster_prior.mu_c))
    assert torch.allclose(torch.exp(log_gamma), torch.eye(2), atol=1e-2)


def test_alignment_is_checked_not_assumed():
    post = _posterior([[0.0, 0.0], [1.0, 1.0]], [[0.1, 0.1], [0.1, 0.1]])
    other = CellAxis(ids=["x", "y"], day=9.0)                 # different cell ids
    memb = _membership([[1, 0], [0, 1]], other, ["A", "B"])
    with pytest.raises(ValueError):
        population_gmm_params(post, memb)
