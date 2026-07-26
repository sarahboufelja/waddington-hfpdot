"""Unit tests for wadd_dim_reduction. Pure numpy -- no modelling framework required."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wadd_data_ingest import CellAxis  # noqa: E402
from wadd_dim_reduction import (  # noqa: E402
    CellPosterior, MomentMatchedGaussian, MonteCarloMixture, uncertainty_radius,
    RandomSubsampler, CoverageSubsampler,
)


def _axis(n, day=9.0):
    return CellAxis(ids=[f"cell_{i}" for i in range(n)], day=day)


def _posterior(means, variances, prob_cat=None, day=9.0):
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    n = means.shape[0]
    if prob_cat is None:
        prob_cat = np.full((n, 2), 0.5)
    return CellPosterior(means=means, variances=variances, prob_cat=np.asarray(prob_cat, float),
                         cells=_axis(n, day))


def _identical_population(n=40, q=3, mu=0.0, var=1.0):
    return _posterior(np.full((n, q), mu), np.full((n, q), var))


# ---- CellPosterior ------------------------------------------------------------------------------

def test_posterior_validates_shapes_and_values():
    with pytest.raises(ValueError):                                   # variances vs means
        _posterior(np.zeros((3, 2)), np.zeros((3, 5)))
    with pytest.raises(ValueError):                                   # negative variance
        _posterior(np.zeros((2, 2)), -np.ones((2, 2)))
    with pytest.raises(ValueError):                                   # prob_cat rows must sum to 1
        _posterior(np.zeros((2, 2)), np.ones((2, 2)), prob_cat=[[0.5, 0.2], [0.5, 0.5]])
    with pytest.raises(ValueError):                                   # cells vs rows
        CellPosterior(np.zeros((3, 2)), np.ones((3, 2)), np.full((3, 2), 0.5), _axis(2))


def test_stds_are_the_square_root_of_variances():
    """One conversion point: the cost layer consumes stds, the model emits variances."""
    p = _posterior(np.zeros((2, 3)), np.full((2, 3), 4.0))
    assert np.allclose(p.stds, 2.0)


def test_posterior_exposes_shape_and_axis_metadata():
    p = _posterior(np.zeros((5, 3)), np.ones((5, 3)), prob_cat=np.full((5, 4), 0.25), day=8.25)
    assert len(p) == 5 and p.latent_dim == 3 and p.n_components == 4
    assert p.day == 8.25 and p.cell_ids[0] == "cell_0"


def test_select_keeps_every_array_and_the_axis_aligned():
    rng = np.random.default_rng(0)
    p = _posterior(rng.normal(size=(6, 3)), rng.uniform(0.5, 1.5, size=(6, 3)),
                   prob_cat=np.full((6, 2), 0.5))
    sub = p.select([4, 1])
    assert np.allclose(sub.means, p.means[[4, 1]])
    assert np.allclose(sub.variances, p.variances[[4, 1]])
    assert np.allclose(sub.prob_cat, p.prob_cat[[4, 1]])
    assert sub.cell_ids == ["cell_4", "cell_1"] and sub.day == p.day


# ---- radius: the cases with a known answer ------------------------------------------------------

@pytest.mark.parametrize("estimator", [MomentMatchedGaussian(), MonteCarloMixture(n_samples=4000)])
def test_identical_cells_have_zero_radius(estimator):
    """No diversity, no uncertainty: the mixture IS each component, so every KL term vanishes."""
    assert estimator(_identical_population()) == pytest.approx(0.0, abs=1e-6)


def test_moment_matched_radius_is_exact_for_identical_cells():
    """Not merely close: with identical cells the moment-matched mixture is the component itself."""
    assert MomentMatchedGaussian()(_identical_population(mu=3.0, var=0.25)) == pytest.approx(0.0,
                                                                                            abs=1e-12)


def test_radius_grows_with_population_spread():
    """The radius measures diversity, so separating the cells must increase it."""
    q, n = 2, 60
    rng = np.random.default_rng(1)
    base = rng.normal(size=(n, q))
    radii = [MomentMatchedGaussian()(_posterior(base * s, np.ones((n, q)))) for s in (0.1, 1.0, 5.0)]
    assert radii[0] < radii[1] < radii[2]


def test_radius_is_intensive_in_the_number_of_cells():
    """eta describes the population, not the sample size: drawing more cells from the same
    distribution must not move it. This is what makes a full-population radius a coherent
    constraint on a subsampled marginal."""
    rng = np.random.default_rng(2)
    q = 4
    radii = []
    for n in (200, 2000):
        p = _posterior(rng.normal(size=(n, q)), rng.uniform(0.5, 1.5, size=(n, q)))
        radii.append(MomentMatchedGaussian()(p))
    assert radii[1] == pytest.approx(radii[0], rel=0.15)


def test_radius_is_invariant_to_cell_ordering():
    rng = np.random.default_rng(3)
    n, q = 50, 3
    mu, var = rng.normal(size=(n, q)), rng.uniform(0.5, 1.5, size=(n, q))
    perm = rng.permutation(n)
    assert MomentMatchedGaussian()(_posterior(mu, var)) == pytest.approx(
        MomentMatchedGaussian()(_posterior(mu[perm], var[perm])))


# ---- moment-matched vs the Monte-Carlo reference -------------------------------------------------

def test_moment_match_agrees_with_monte_carlo_on_a_unimodal_population():
    """Where the Gaussian assumption holds, the O(n) estimator must track the reference."""
    rng = np.random.default_rng(4)
    n, q = 300, 3
    p = _posterior(rng.normal(size=(n, q)), rng.uniform(0.8, 1.2, size=(n, q)))
    analytic = MomentMatchedGaussian()(p)
    reference = MonteCarloMixture(n_samples=6000)(p)
    assert analytic == pytest.approx(reference, rel=0.25)


def test_moment_match_understates_a_strongly_bimodal_population():
    """The documented failure mode: one Gaussian cannot represent two separated modes, so the
    approximation is optimistic. Pinned so the limitation stays visible rather than surprising."""
    rng = np.random.default_rng(5)
    n, q = 200, 2
    far = np.concatenate([rng.normal(-20, 0.1, size=(n // 2, q)),
                          rng.normal(+20, 0.1, size=(n // 2, q))])
    p = _posterior(far, np.full((n, q), 0.05))
    analytic = MomentMatchedGaussian()(p)
    reference = MonteCarloMixture(n_samples=4000)(p)
    assert analytic < reference


def test_monte_carlo_estimator_is_reproducible():
    rng = np.random.default_rng(6)
    p = _posterior(rng.normal(size=(80, 3)), rng.uniform(0.5, 1.5, size=(80, 3)))
    assert MonteCarloMixture(seed=7)(p) == MonteCarloMixture(seed=7)(p)


# ---- the module-level entry point ---------------------------------------------------------------

def test_uncertainty_radius_defaults_to_the_analytic_estimator():
    rng = np.random.default_rng(8)
    p = _posterior(rng.normal(size=(40, 3)), rng.uniform(0.5, 1.5, size=(40, 3)))
    assert uncertainty_radius(p) == pytest.approx(MomentMatchedGaussian()(p))


def test_uncertainty_radius_accepts_an_alternative_estimator():
    p = _identical_population()
    assert uncertainty_radius(p, MonteCarloMixture(n_samples=500)) == pytest.approx(0.0, abs=1e-6)


# ---- Subsamplers --------------------------------------------------------------------------------

def _clustered(sizes, q=4, sep=8.0, seed=0):
    """An imbalanced, well-separated population: sizes[c] cells around a distinct centre per stratum,
    with prob_cat peaked at the true stratum. Mirrors the real data's 90:1 fate imbalance."""
    rng = np.random.default_rng(seed)
    K = len(sizes)
    centres = np.zeros((K, q))
    for c in range(1, K):
        centres[c, (c - 1) % q] = sep * (1 + (c - 1) // q)
    means, true = [], []
    for c, s in enumerate(sizes):
        means.append(centres[c] + rng.normal(scale=0.4, size=(s, q)))
        true += [c] * s
    means = np.vstack(means)
    n = means.shape[0]
    variances = np.abs(rng.normal(scale=0.2, size=(n, q))) + 0.1
    d2 = ((means[:, None, :] - centres[None]) ** 2).sum(-1)          # (n, K)
    prob_cat = np.exp(-d2 - (-d2).max(1, keepdims=True))
    prob_cat /= prob_cat.sum(1, keepdims=True)
    return _posterior(means, variances, prob_cat), np.array(true)


@pytest.mark.parametrize("sampler", [RandomSubsampler(), CoverageSubsampler()])
def test_subsampler_returns_valid_index_set(sampler):
    post, _ = _clustered([30, 10, 5])
    n, k = len(post), 12
    idx = sampler.indices(post, k)
    assert idx.shape == (k,)
    assert len(set(idx.tolist())) == k                              # unique
    assert list(idx) == sorted(idx)                                 # sorted -> stable cell order
    assert idx.min() >= 0 and idx.max() < n
    assert len(post.select(idx)) == k                               # composes with select


@pytest.mark.parametrize("sampler", [RandomSubsampler(), CoverageSubsampler()])
def test_budget_edges(sampler):
    post, _ = _clustered([20, 8, 4])
    assert np.array_equal(sampler.indices(post, 999), np.arange(len(post)))   # k>=n keeps all
    with pytest.raises(ValueError):
        sampler.indices(post, 0)                                    # k<=0


def test_random_is_reproducible_and_seed_sensitive():
    post, _ = _clustered([30, 10, 5])
    assert np.array_equal(RandomSubsampler(seed=1).indices(post, 12),
                          RandomSubsampler(seed=1).indices(post, 12))
    assert not np.array_equal(RandomSubsampler(seed=1).indices(post, 12),
                              RandomSubsampler(seed=2).indices(post, 12))


def test_coverage_represents_every_population():
    """The load-bearing guarantee: stratified seeding puts >=1 cell of each present population in the
    budget, even the rare one that uniform sampling routinely drops."""
    post, true = _clustered([60, 12, 3])                            # pop 2 is 3/75 cells
    idx = CoverageSubsampler().indices(post, 15)
    covered = set(true[idx].tolist())
    assert covered == {0, 1, 2}, f"missing populations: {{0,1,2}} - {covered}"


def test_coverage_keeps_the_rare_population_where_random_usually_loses_it():
    post, true = _clustered([200, 3])                               # ~1.5% rare fate
    assert 1 in true[CoverageSubsampler().indices(post, 10)]        # guaranteed
    # random, by contrast, misses it most of the time at this budget
    misses = sum(1 not in true[RandomSubsampler(seed=s).indices(post, 10)] for s in range(20))
    assert misses > 10, "the scenario should be one where random frequently drops the rare fate"


def test_coverage_is_deterministic():
    post, _ = _clustered([40, 15, 8])
    assert np.array_equal(CoverageSubsampler().indices(post, 20),
                          CoverageSubsampler().indices(post, 20))


def test_coverage_biases_the_radius_upward_relative_to_random():
    """Coverage deliberately favours extremes, so the retained population's uncertainty radius is
    larger than a uniform draw's -- the upward bias the module docstring calls out (median over
    random seeds, not a single draw)."""
    post, _ = _clustered([60, 12, 3])
    eta = MomentMatchedGaussian()
    e_cov = eta(post.select(CoverageSubsampler().indices(post, 15)))
    e_rnd = np.median([eta(post.select(RandomSubsampler(seed=s).indices(post, 15)))
                       for s in range(25)])
    assert e_cov > e_rnd


def test_coverage_warns_when_budget_is_below_population_count():
    post, true = _clustered([50, 20, 6, 2])                         # 4 populations
    with pytest.warns(UserWarning, match="per-population guarantee"):
        idx = CoverageSubsampler().indices(post, 2)                 # can only seat 2
    assert idx.shape == (2,)
    assert set(true[idx].tolist()) <= {0, 1}                        # the two most populous
