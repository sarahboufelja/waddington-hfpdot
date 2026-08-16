"""Tests for the marginal particle filter (wadd_propagation), sampler mocked throughout."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wadd_propagation import (MarginalParticle, PairRecord, ancestor_pr, ancestor_overlap,
                              dominant_destination_hist, extras_slices, fps_trim,
                              identity_readouts, initial_ensemble, kl_ball, pairwise_tv,
                              project_particles, propagate_pair, push_forward_marginal,
                              w_series_extras, weighted_quantiles, _logsumexp, _sym_kl)


def _particle(w, log_mass=0.0, parent=None):
    w = np.asarray(w, dtype=float)
    return MarginalParticle(weights=w / w.sum(), log_mass=log_mass, parent=parent)


class TestPushForward:
    def test_column_sums(self):
        plan = np.array([[0.2, 0.1], [0.3, 0.4]])
        nu = push_forward_marginal(plan.reshape(-1), 2, 2)
        np.testing.assert_allclose(nu, [0.5, 0.5])

    def test_unbalanced_plan_carries_mass(self):
        plan = np.array([[2.0, 0.0], [0.0, 6.0]])          # total mass 8 (growth)
        nu = push_forward_marginal(plan.reshape(-1), 2, 2)
        np.testing.assert_allclose(nu, [2.0, 6.0])         # masses, not a distribution:
        assert nu.sum() == 8.0                             # total mass IS the growth signal

    def test_zero_mass_rejected(self):
        with pytest.raises(ValueError, match="non-positive total mass"):
            push_forward_marginal(np.zeros(4), 2, 2)


class TestFpsTrim:
    def test_under_budget_untouched(self):
        parts = [_particle([1, 0, 0]), _particle([0, 1, 0])]
        assert fps_trim(parts, 5) is not parts             # copies the list
        assert [p.weights.tolist() for p in fps_trim(parts, 5)] == \
               [p.weights.tolist() for p in parts]

    def test_mass_conserved(self):
        rng = np.random.default_rng(0)
        parts = [_particle(rng.dirichlet(np.ones(6)), log_mass=float(np.log(1 / 40)))
                 for _ in range(40)]
        trimmed = fps_trim(parts, 7)
        assert len(trimmed) == 7
        total = _logsumexp(np.array([p.log_mass for p in trimmed]))
        np.testing.assert_allclose(np.exp(total), 1.0, atol=1e-12)

    def test_principal_mode_survives(self):
        rng = np.random.default_rng(1)
        parts = [_particle(rng.dirichlet(np.ones(4)), log_mass=-5.0) for _ in range(10)]
        heavy = _particle([1, 1, 1, 1], log_mass=0.0)
        trimmed = fps_trim(parts + [heavy], 3)
        assert any(np.allclose(p.weights, heavy.weights) for p in trimmed)

    def test_extremes_survive(self):
        # a tight cluster plus one far outlier: FPS must keep the outlier, resampling would not
        cluster = [_particle([1 + 0.01 * i, 1, 1], log_mass=np.log(0.24)) for i in range(4)]
        outlier = _particle([0.001, 0.001, 1.0], log_mass=np.log(0.04))
        trimmed = fps_trim(cluster + [outlier], 2)
        assert any(np.allclose(p.weights, outlier.weights) for p in trimmed)

    def test_deterministic(self):
        rng = np.random.default_rng(2)
        parts = [_particle(rng.dirichlet(np.ones(5)), log_mass=float(-np.log(20)))
                 for _ in range(20)]
        a = fps_trim(parts, 4); b = fps_trim(parts, 4)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.weights, y.weights)
            assert x.log_mass == y.log_mass


class TestPropagatePair:
    @staticmethod
    def _mock_sampler(draws):
        """Deterministic mock: perturbations of the outer product mu nu^T."""
        def sample(mu_0, nu_0):
            rng = np.random.default_rng(int(1e6 * mu_0[0]) % 2**31)
            base = np.outer(mu_0, nu_0)
            out = np.stack([(base * rng.uniform(0.5, 1.5, base.shape)).reshape(-1)
                            for _ in range(draws)])
            return out, {"mock": True}
        return sample

    def test_shapes_parents_and_mass(self):
        II = JJ = 5
        parts = [_particle(np.arange(1, II + 1), log_mass=np.log(0.5), parent=None),
                 _particle(np.ones(II), log_mass=np.log(0.5), parent=None)]
        rec = propagate_pair(parts, self._mock_sampler(8), np.full(JJ, 1 / JJ), II, JJ,
                             budget=3, table_of_plan=lambda d: d.reshape(II, JJ)[:2, :2],
                             day_from=1.0, day_to=2.0)
        assert isinstance(rec, PairRecord)
        assert rec.tables.shape == (16, 2, 2)
        assert len(rec.pool) == 16 and len(rec.futures) == 3
        assert {p.parent for p in rec.pool} == {0, 1}
        assert {p.parent for p in rec.futures} <= {0, 1}
        total = _logsumexp(np.array([p.log_mass for p in rec.futures]))
        np.testing.assert_allclose(np.exp(total), 1.0, atol=1e-12)
        assert len(rec.diagnostics) == 2 and rec.diagnostics[0]["mock"]

    def test_growth_aligns_with_draws(self):
        II = JJ = 4
        rec = propagate_pair(initial_ensemble(II), self._mock_sampler(6), np.full(JJ, 1 / JJ),
                             II, JJ, budget=2, table_of_plan=lambda d: d.reshape(II, JJ))
        assert rec.growth.shape == (6, II)
        # growth = transported source mass = row sums of each draw, in draw order
        np.testing.assert_allclose(rec.growth.sum(axis=1),
                                   [t.sum() for t in rec.tables], rtol=1e-12)

    def test_extra_channel_aligns_and_defaults_none(self):
        II = JJ = 4
        rec0 = propagate_pair(initial_ensemble(II), self._mock_sampler(4), np.full(JJ, 1 / JJ),
                              II, JJ, budget=2, table_of_plan=lambda d: d.reshape(II, JJ))
        assert rec0.extras is None
        rec = propagate_pair(initial_ensemble(II), self._mock_sampler(6), np.full(JJ, 1 / JJ),
                             II, JJ, budget=2, table_of_plan=lambda d: d.reshape(II, JJ),
                             extra_of_plan=lambda d: np.asarray(d).sum())
        assert rec.extras.shape == (6,)
        # same draws, same order as the table channel
        np.testing.assert_allclose(rec.extras, [t.sum() for t in rec.tables], rtol=1e-12)

    def test_futures_thinning(self):
        II = JJ = 4
        rec = propagate_pair(initial_ensemble(II), self._mock_sampler(50), np.full(JJ, 1 / JJ),
                             II, JJ, budget=2, table_of_plan=lambda d: d.reshape(II, JJ),
                             max_futures_per_particle=10)
        assert rec.tables.shape[0] == 10

    def test_bad_sampler_shape_rejected(self):
        II = JJ = 3
        bad = lambda mu, nu: (np.ones((4, 5)), {})
        with pytest.raises(ValueError, match="expected"):
            propagate_pair(initial_ensemble(II), bad, np.full(JJ, 1 / JJ), II, JJ,
                           budget=2, table_of_plan=lambda d: d)


class TestAncestorPr:
    def test_uniform_mass_counts_all_ancestors(self):
        II, JJ = 6, 5
        pi = np.full((II, JJ), 1.0 / (II * JJ))
        np.testing.assert_allclose(ancestor_pr(pi, np.ones((JJ, 1))), [II])

    def test_single_ancestor_reads_one(self):
        pi = np.zeros((5, 4)); pi[2] = 1.0
        np.testing.assert_allclose(ancestor_pr(pi, np.ones((4, 1))), [1.0])

    def test_scale_invariant_and_bounded(self):
        rng = np.random.default_rng(0)
        pi = rng.uniform(size=(6, 5))
        W = rng.dirichlet(np.ones(3), size=5)          # (JJ, K) soft memberships
        pr = ancestor_pr(pi, W)
        np.testing.assert_allclose(ancestor_pr(3.7 * pi, W), pr, rtol=1e-12)
        assert np.all(pr >= 1.0) and np.all(pr <= 6.0)

    def test_empty_fate_reads_zero(self):
        pi = np.ones((4, 3))
        W = np.zeros((3, 2)); W[:, 0] = 1.0            # second fate receives nothing
        pr = ancestor_pr(pi, W)
        assert pr[0] == 4.0 and pr[1] == 0.0


class TestChannelFunctionals:
    def test_dominant_destination_mass_weighted(self):
        # cell 0 (mass 3) dominant to fate 1; cell 1 (mass 1) dominant to fate 0
        A = np.array([[1.0, 2.0], [0.75, 0.25]])
        np.testing.assert_allclose(dominant_destination_hist(A), [0.25, 0.75])

    def test_overlap_diagonal_disjoint_and_empty(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])   # disjoint profiles, fate 2 empty
        O = ancestor_overlap(A)
        np.testing.assert_allclose(np.diag(O), [1.0, 1.0, 0.0])
        assert O[0, 1] == 0.0 and np.allclose(O, O.T)

    def test_overlap_identical_profiles(self):
        a = np.array([0.2, 0.5, 0.3])
        O = ancestor_overlap(np.stack([a, 4.0 * a], axis=1))
        np.testing.assert_allclose(O, 1.0)                 # scale-free: same shape -> BC = 1

    def test_identity_readouts_single_resample_zero_var(self):
        rng = np.random.default_rng(0)
        pi = rng.uniform(size=(5, 4))
        W = rng.dirichlet(np.ones(3), size=(1, 4))         # M = 1
        mean, var = identity_readouts(pi, rng.dirichlet(np.ones(3), size=(1, 5)), W)
        np.testing.assert_allclose(var, 0.0, atol=1e-15)

    def test_identity_readouts_match_loop(self):
        rng = np.random.default_rng(1)
        pi = rng.uniform(size=(6, 5))
        Ws_a = rng.dirichlet(np.ones(3), size=(4, 6))
        Ws_b = rng.dirichlet(np.ones(3), size=(4, 5))
        mean, var = identity_readouts(pi, Ws_a, Ws_b)
        T = np.stack([Ws_a[m].T @ pi @ Ws_b[m] for m in range(4)])
        np.testing.assert_allclose(mean, T.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(var, T.var(axis=0), rtol=1e-10, atol=1e-15)

    def test_w_series_extras_layout_roundtrip(self):
        rng = np.random.default_rng(2)
        K = 3
        pi = rng.uniform(size=(6, 5))
        W_b = rng.dirichlet(np.ones(K), size=5)
        Ws_a = rng.dirichlet(np.ones(K), size=(4, 6))
        Ws_b = rng.dirichlet(np.ones(K), size=(4, 5))
        v = w_series_extras(pi, W_b, Ws_a, Ws_b)
        sl = extras_slices(K)
        assert len(v) == 2 * K + 3 * K * K
        A = pi @ W_b
        np.testing.assert_allclose(v[sl["ancestor_pr"]], ancestor_pr(pi, W_b))
        np.testing.assert_allclose(v[sl["dom_dest"]], dominant_destination_hist(A))
        np.testing.assert_allclose(v[sl["ancestor_overlap"]].reshape(K, K),
                                   ancestor_overlap(A))
        m, va = identity_readouts(pi, Ws_a, Ws_b)
        np.testing.assert_allclose(v[sl["table_identity_mean"]].reshape(K, K), m)
        np.testing.assert_allclose(v[sl["table_identity_var"]].reshape(K, K), va)


class TestWeightedQuantiles:
    def test_uniform_weights_match_median(self):
        vals = np.arange(101, dtype=float).reshape(101, 1)
        q = weighted_quantiles(vals, np.zeros(101), [0.5])
        assert q[0, 0] == 50.0

    def test_mass_moves_the_quantile(self):
        vals = np.array([0.0, 1.0]).reshape(2, 1)
        q = weighted_quantiles(vals, np.log([0.99, 0.01]), [0.5])
        assert q[0, 0] == 0.0
        q = weighted_quantiles(vals, np.log([0.01, 0.99]), [0.5])
        assert q[0, 0] == 1.0

    def test_trailing_shape(self):
        rng = np.random.default_rng(3)
        vals = rng.normal(size=(200, 3, 4))
        q = weighted_quantiles(vals, np.zeros(200), [0.025, 0.5, 0.975])
        assert q.shape == (3, 3, 4)
        assert np.all(q[0] <= q[1]) and np.all(q[1] <= q[2])


class TestKlBall:
    def test_identical_particles_zero(self):
        parts = [_particle([1, 2, 3], log_mass=np.log(0.5)) for _ in range(2)]
        mean, mx = kl_ball(parts)
        assert mean == pytest.approx(0.0, abs=1e-12) and mx == pytest.approx(0.0, abs=1e-12)

    def test_disjoint_supports_bounded(self):
        # forward KL to the barycentre stays finite on disjoint particles: KL(p || (p+q)/2) = log 2
        parts = [_particle([1, 0, 0], log_mass=np.log(0.5)),
                 _particle([0, 0, 1], log_mass=np.log(0.5))]
        mean, mx = kl_ball(parts)
        assert mean == pytest.approx(np.log(2)) and mx == pytest.approx(np.log(2))

    def test_mean_is_moment_condition(self):
        rng = np.random.default_rng(0)
        parts = [_particle(rng.dirichlet(np.ones(5)), log_mass=np.log(w))
                 for w in (0.7, 0.2, 0.1)]
        W = np.stack([p.weights for p in parts]); m = np.array([0.7, 0.2, 0.1])
        bary = (m[:, None] * W).sum(axis=0)
        expect = sum(mi * np.sum(w * np.log(w / bary)) for mi, w in zip(m, W))
        assert kl_ball(parts)[0] == pytest.approx(expect)

    def test_negligible_mass_negligible_mean(self):
        anchor = _particle([1, 1, 1], log_mass=0.0)
        ghost = _particle([5, 0.1, 0.1], log_mass=-30.0)
        assert kl_ball([anchor, ghost])[0] == pytest.approx(0.0, abs=1e-8)


class TestPairwiseTv:
    def test_single_particle_zero(self):
        assert pairwise_tv([_particle([1, 2, 3])]) == (0.0, 0.0)

    def test_disjoint_is_one(self):
        parts = [_particle([1, 0]), _particle([0, 1])]
        mean, diam = pairwise_tv(parts)
        assert mean == pytest.approx(1.0) and diam == pytest.approx(1.0)

    def test_diameter_dominates_mean(self):
        rng = np.random.default_rng(1)
        parts = [_particle(rng.dirichlet(np.ones(4)), log_mass=-np.log(6)) for _ in range(6)]
        mean, diam = pairwise_tv(parts)
        assert 0.0 < mean <= diam <= 1.0

    def test_hand_value(self):
        parts = [_particle([0.5, 0.5], log_mass=np.log(0.5)),
                 _particle([0.8, 0.2], log_mass=np.log(0.5))]
        mean, diam = pairwise_tv(parts)
        assert mean == pytest.approx(0.3) and diam == pytest.approx(0.3)


class TestProjectParticles:
    def test_projection_aggregates_mass(self):
        W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]])   # third atom unlabeled
        p = _particle([0.5, 0.25, 0.25], log_mass=np.log(0.4), parent=3)
        out = project_particles([p], W)[0]
        # Aggregation, no renormalisation: the unlabeled atom's mass is dropped by its
        # zero row (mass-preserving exactly when rows sum to 1, e.g. q(c|z)).
        np.testing.assert_allclose(out.weights, [0.625, 0.125])
        assert out.log_mass == p.log_mass and out.parent == 3

    def test_contraction_under_lumping(self):
        # coarse-graining can only shrink both tube metrics (data-processing)
        rng = np.random.default_rng(2)
        W = np.zeros((6, 2)); W[:3, 0] = 1.0; W[3:, 1] = 1.0
        parts = [_particle(rng.dirichlet(np.ones(6)), log_mass=-np.log(5)) for _ in range(5)]
        proj = project_particles(parts, W)
        assert kl_ball(proj)[0] <= kl_ball(parts)[0] + 1e-12
        assert pairwise_tv(proj)[1] <= pairwise_tv(parts)[1] + 1e-12


class TestSymKl:
    def test_symmetric_and_zero_on_equal(self):
        p = np.array([0.2, 0.3, 0.5]); q = np.array([0.5, 0.25, 0.25])
        assert _sym_kl(p, q) == pytest.approx(_sym_kl(q, p))
        assert _sym_kl(p, p) == 0.0
