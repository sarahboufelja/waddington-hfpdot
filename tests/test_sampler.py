"""Smoke tests establishing a green baseline for the sampler overhaul.

These intentionally only assert that the module imports and that its core
public surface is present. The substantive math/engine/diagnostics tests are
added as those pieces are reworked (see the 3-day overhaul plan).
"""


def test_core_module_imports_without_plotting_deps():
    """The core sampler must import without matplotlib/blackjax."""
    import langevin_sampler as ls

    assert hasattr(ls, "MetropolisAdjustedLangevinSampler")
    assert hasattr(ls, "HFPDOTHyperprior")


def test_viz_helpers_live_in_separate_module():
    """Plotting helpers were moved out of the core module into sampler_viz."""
    import pytest

    import langevin_sampler as ls

    for name in (
        "validate_sampler",
        "validate_hyperprior_sampler",
        "visualise_rhat",
        "plot_transport_plans_traces",
    ):
        assert not hasattr(ls, name), f"{name} should no longer live in langevin_sampler"

    # The viz module itself requires matplotlib; skip its checks if absent.
    viz = pytest.importorskip("sampler_viz")
    for name in (
        "validate_sampler",
        "validate_hyperprior_sampler",
        "visualise_rhat",
        "plot_transport_plans_traces",
    ):
        assert hasattr(viz, name), f"sampler_viz missing {name}"


# =========================================================================== #
# Engine integration tests (issue #2): does the sampler actually target the
# distribution? We use a correlated 2-D Gaussian where the answer is known in
# closed form, exercising warm-up -> mass-matrix -> adapted main stage.
# =========================================================================== #
import jax
import jax.numpy as jnp


def _correlated_gaussian(mu, sigma):
    """Return (log_prob, score) for N(mu, sigma) acting on states shaped (1, d)."""
    precision = jnp.linalg.inv(sigma)

    def log_prob(x):
        d = x - mu
        return -0.5 * jnp.sum((d @ precision) * d)

    def score(x):
        return -(x - mu) @ precision

    return log_prob, score


def _build_sampler(**overrides):
    from langevin_sampler import MetropolisAdjustedLangevinSampler

    mu = jnp.array([1.0, -1.0])
    sigma = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    log_prob, score = _correlated_gaussian(mu, sigma)
    kwargs = dict(
        target_log_prob_fn=log_prob,
        target_score_fn=score,
        shape=2,
        support="unconstrained",
        num_parallel_chains=4,
        num_samples=2500,
        num_burnin=1200,
        warm_up_steps=600,
        step_size=0.3,
    )
    kwargs.update(overrides)
    sampler = MetropolisAdjustedLangevinSampler(**kwargs)
    sampler.init_key = jax.random.key(0)  # pin randomness for the test
    return sampler, mu, sigma


def test_sampler_recovers_correlated_gaussian_moments():
    sampler, mu, sigma = _build_sampler()
    state = sampler.sample(with_diagnostics=False)
    num_accepted = state.num_accepted_samples
    samples = state.samples.reshape(-1, state.samples.shape[-1])  # (C, N, d) -> (C*N, d)

    est_mean = jnp.mean(samples, axis=0)
    est_cov = jnp.cov(samples.T)
    assert jnp.allclose(est_mean, mu, atol=0.1), est_mean
    assert jnp.allclose(est_cov, sigma, atol=0.15), est_cov

    # Robbins-Monro adaptation should drive the final acceptance near TARGET_ACCEPT=0.574.
    final_accept = num_accepted[:, -1, 0] / sampler.tot_num_samples
    assert jnp.all(final_accept > 0.4) and jnp.all(final_accept < 0.75), final_accept


def test_sampler_is_reproducible_under_fixed_key():
    s1, _, _ = _build_sampler(num_samples=300, num_burnin=100, warm_up_steps=100, num_parallel_chains=2)
    s2, _, _ = _build_sampler(num_samples=300, num_burnin=100, warm_up_steps=100, num_parallel_chains=2)
    out1 = s1.sample(with_diagnostics=False).samples
    out2 = s2.sample(with_diagnostics=False).samples
    assert jnp.array_equal(out1, out2)
