"""Unit tests for wadd_ot. numpy/POT only -- runnable in any env without torch/gmvae."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wadd_ot import (  # noqa: E402
    CellCloud, SqEuclideanLatent, GaussianW2, GaussianKL, Mahalanobis,
    as_marginal, eot_plan, uot_plan, wasserstein_distance, WOTIteration,
)


def _cloud(rng, n, d=5, with_std=True):
    means = rng.normal(size=(n, d))
    stds = np.abs(rng.normal(size=(n, d))) + 0.1 if with_std else None
    return CellCloud(means=means, stds=stds)


# ---- CellCloud / representation -------------------------------------------------------------------

def test_cellcloud_validates_shapes():
    with pytest.raises(ValueError):
        CellCloud(means=np.zeros((3,)))                       # not 2-D
    with pytest.raises(ValueError):
        CellCloud(means=np.zeros((3, 4)), stds=np.zeros((3, 2)))  # mismatched stds


def test_gaussian_metric_requires_stds():
    src = CellCloud(means=np.zeros((3, 4)))                   # no stds
    with pytest.raises(ValueError):
        GaussianW2()(src, src)


# ---- Cost metrics ---------------------------------------------------------------------------------

def test_gaussianw2_equals_augmented_sqeuclidean():
    rng = np.random.default_rng(0)
    src, tgt = _cloud(rng, 6), _cloud(rng, 7)
    C = GaussianW2(normalize=False)(src, tgt)
    # explicit: ||mu_k-mu_l||^2 + ||s_k-s_l||^2
    exp = np.zeros((6, 7))
    for k in range(6):
        for l in range(7):
            exp[k, l] = np.sum((src.means[k] - tgt.means[l]) ** 2) + np.sum((src.stds[k] - tgt.stds[l]) ** 2)
    assert np.allclose(C, exp)


def test_gaussianw2_reduces_to_sqeuclidean_when_stds_constant():
    rng = np.random.default_rng(1)
    means_s, means_t = rng.normal(size=(5, 4)), rng.normal(size=(6, 4))
    const = np.full((1, 4), 0.3)
    src = CellCloud(means=means_s, stds=np.broadcast_to(const, means_s.shape).copy())
    tgt = CellCloud(means=means_t, stds=np.broadcast_to(const, means_t.shape).copy())
    w2 = GaussianW2(normalize=False)(src, tgt)
    sq = SqEuclideanLatent(normalize=False)(src, tgt)
    assert np.allclose(w2, sq)                                # equal stds -> std term vanishes


def test_gaussiankl_zero_on_identical_and_symmetric_nonneg():
    rng = np.random.default_rng(2)
    src = _cloud(rng, 5)
    C = GaussianKL(normalize=False)(src, src)
    assert np.allclose(np.diag(C), 0.0, atol=1e-8)            # KL to self = 0
    assert np.all(C >= -1e-9)                                 # symmetric KL >= 0
    assert np.allclose(C, C.T)                                # symmetric matrix (src vs src)


def test_median_normalization_sets_median_to_one():
    rng = np.random.default_rng(3)
    src, tgt = _cloud(rng, 8), _cloud(rng, 9)
    C = GaussianW2(normalize=True)(src, tgt)
    assert np.isclose(np.median(C), 1.0)


def test_cost_does_not_mutate_inputs():
    rng = np.random.default_rng(4)
    src, tgt = _cloud(rng, 5), _cloud(rng, 5)
    m0, s0 = src.means.copy(), src.stds.copy()
    for metric in (SqEuclideanLatent(), GaussianW2(), GaussianKL(), Mahalanobis()):
        metric(src, tgt)
    assert np.array_equal(src.means, m0) and np.array_equal(src.stds, s0)


# ---- Marginals ------------------------------------------------------------------------------------

def test_as_marginal_uniform_and_normalized():
    assert np.allclose(as_marginal(None, 4), 0.25)
    assert np.isclose(as_marginal(np.array([1.0, 3.0]), 2).sum(), 1.0)
    with pytest.raises(ValueError):
        as_marginal(np.array([1.0, 2.0, 3.0]), 2)            # wrong length


# ---- Plans ----------------------------------------------------------------------------------------

def test_eot_plan_respects_marginals():
    rng = np.random.default_rng(5)
    src, tgt = _cloud(rng, 10), _cloud(rng, 12)
    a = as_marginal(rng.uniform(size=10), 10)
    b = as_marginal(rng.uniform(size=12), 12)
    P = eot_plan(src, tgt, a=a, b=b, reg=5e-2)
    assert P.shape == (10, 12) and np.all(P >= 0)
    assert np.allclose(P.sum(axis=1), a, atol=1e-2)          # balanced -> marginals recovered
    assert np.allclose(P.sum(axis=0), b, atol=1e-2)


def test_uot_plan_actually_uses_marginals():
    # Regression guard: the target marginal must actually influence the plan.
    rng = np.random.default_rng(6)
    src, tgt = _cloud(rng, 8), _cloud(rng, 8)
    b1 = as_marginal(np.ones(8), 8)
    b2 = as_marginal(np.arange(1, 9, dtype=float), 8)        # skewed target marginal
    P1 = uot_plan(src, tgt, b=b1, reg=5e-2, reg_m=(5.0, 50.0))
    P2 = uot_plan(src, tgt, b=b2, reg=5e-2, reg_m=(5.0, 50.0))
    assert P1.shape == (8, 8) and np.all(P1 >= 0)
    assert not np.allclose(P1, P2), "UOT ignored the target marginal"


def test_wasserstein_distance_zero_to_self_and_positive():
    rng = np.random.default_rng(7)
    src = _cloud(rng, 10)
    tgt = CellCloud(means=src.means + 3.0, stds=src.stds)    # shifted far away
    d_self = wasserstein_distance(src, src, reg=5e-2)
    d_far = wasserstein_distance(src, tgt, reg=5e-2)
    assert d_far > d_self
    assert d_self < d_far * 0.2                              # near-zero to self relative to far


# ---- Growth model ---------------------------------------------------------------------------------

def test_wot_growth_positive_and_converges():
    rng = np.random.default_rng(8)
    src, tgt = _cloud(rng, 12), _cloud(rng, 12)
    plan, growth = WOTIteration(reg=5e-2, reg_m=(1.0, 50.0), n_iters=5)(src, tgt, GaussianW2())
    assert plan.shape == (12, 12) and np.all(growth > 0)
    # running more iterations should not move growth much (fixed point)
    plan2, growth2 = WOTIteration(reg=5e-2, reg_m=(1.0, 50.0), n_iters=8)(src, tgt, GaussianW2())
    assert np.linalg.norm(growth - growth2) / np.linalg.norm(growth) < 0.25
