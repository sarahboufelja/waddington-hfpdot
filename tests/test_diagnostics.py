"""Tests for the MCMCDiagnostics class (issue #3 / 2d).

Each metric is checked on synthetic data where the answer is qualitatively
known: converged vs non-converged chains for R-hat, i.i.d. vs autocorrelated
chains for ESS, well-mixed vs sticky chains for eBFMI.

float64 for stable autocorrelation/FFT arithmetic.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from mcmc_diagnostics import MCMCDiagnostics


def _iid_chains(key, C=4, N=3000, D=3):
    return jax.random.normal(key, (C, N, D))


def _ar1_chains(key, C=4, N=3000, D=3, phi=0.95):
    """Stationary AR(1) chains with autocorrelation phi (variance ~1)."""
    eps = jax.random.normal(key, (N, C, D)) * jnp.sqrt(1.0 - phi ** 2)

    def step(x_prev, e):
        x = phi * x_prev + e
        return x, x

    _, xs = jax.lax.scan(step, jnp.zeros((C, D)), eps)
    return jnp.transpose(xs, (1, 0, 2))


# --------------------------------------------------------------------------- #
# R-hat
# --------------------------------------------------------------------------- #
def test_bulk_rhat_near_one_for_converged_chains():
    chains = _iid_chains(jax.random.key(0))
    rhat = MCMCDiagnostics.bulk_rhat(chains)
    assert rhat.shape == (3,)
    assert jnp.max(rhat) < 1.05, float(jnp.max(rhat))


def test_bulk_rhat_large_for_nonconverged_chains():
    # Each chain centred at a very different location -> fails to mix.
    base = _iid_chains(jax.random.key(1))
    offsets = jnp.arange(base.shape[0]).reshape(-1, 1, 1) * 10.0
    chains = base + offsets
    rhat = MCMCDiagnostics.bulk_rhat(chains)
    assert jnp.max(rhat) > 1.5, float(jnp.max(rhat))


def test_tail_rhat_near_one_for_converged_chains():
    chains = _iid_chains(jax.random.key(2))
    rhat = MCMCDiagnostics.tail_rhat(chains)
    assert jnp.max(rhat) < 1.05, float(jnp.max(rhat))


# --------------------------------------------------------------------------- #
# ESS
# --------------------------------------------------------------------------- #
def test_ess_near_total_for_iid_chains():
    chains = _iid_chains(jax.random.key(3))
    C, N, _ = chains.shape
    ess = MCMCDiagnostics.ess(chains)
    assert ess.shape == (3,)
    # i.i.d. draws: ESS close to the total number of draws.
    assert jnp.all(ess > 0.7 * C * N)
    assert jnp.all(ess < 1.1 * C * N)


def test_ess_much_smaller_for_autocorrelated_chains():
    chains = _ar1_chains(jax.random.key(4), phi=0.95)
    C, N, _ = chains.shape
    ess = MCMCDiagnostics.ess(chains)
    # phi=0.95 -> strong autocorrelation -> ESS far below the raw count.
    assert jnp.all(ess < 0.3 * C * N), float(jnp.max(ess))


# --------------------------------------------------------------------------- #
# eBFMI
# --------------------------------------------------------------------------- #
def test_ebfmi_well_mixed_exceeds_sticky():
    log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    diag = MCMCDiagnostics(log_prob_fn=log_prob)

    well_mixed = _iid_chains(jax.random.key(5))
    sticky = _ar1_chains(jax.random.key(6), phi=0.99)

    bfmi_mixed = diag.ebfmi(well_mixed)
    bfmi_sticky = diag.ebfmi(sticky)

    assert bfmi_mixed.shape == (4,)
    assert jnp.all(bfmi_mixed > 0) and jnp.all(jnp.isfinite(bfmi_mixed))
    assert jnp.min(bfmi_mixed) > jnp.max(bfmi_sticky), (float(jnp.min(bfmi_mixed)), float(jnp.max(bfmi_sticky)))


def test_ebfmi_requires_log_prob_fn():
    with pytest.raises(ValueError):
        MCMCDiagnostics().ebfmi(_iid_chains(jax.random.key(7)))


# --------------------------------------------------------------------------- #
# acceptance + summarize
# --------------------------------------------------------------------------- #
def test_acceptance_rate():
    rate = MCMCDiagnostics.acceptance_rate(jnp.array([574, 600]), 1000)
    assert jnp.allclose(rate, jnp.array([0.574, 0.6]))


def test_summarize_keys_and_scalars():
    log_prob = lambda x: -0.5 * jnp.sum(x ** 2)
    diag = MCMCDiagnostics(log_prob_fn=log_prob)
    out = diag.summarize(_iid_chains(jax.random.key(8)))
    for key in ("bulk_rhat_max", "tail_rhat_max", "ess_min", "ebfmi_min"):
        assert key in out
        assert jnp.ndim(out[key]) == 0


# --------------------------------------------------------------------------- #
# Quantitative soundness cross-check against arviz (the reference). arviz is a
# dev-only dependency, so these are skipped if it is not installed.
# --------------------------------------------------------------------------- #
def test_ess_matches_arviz_across_regimes():
    az = pytest.importorskip("arviz")
    import numpy as np

    cases = {
        "iid": _iid_chains(jax.random.key(0)),
        "ar0.7": _ar1_chains(jax.random.key(1), phi=0.7),
        "ar0.95": _ar1_chains(jax.random.key(2), phi=0.95),
    }
    for name, chains in cases.items():
        ours = np.asarray(MCMCDiagnostics.ess(chains))
        # az 'mean' is ESS of the raw draws (our quantity); 'bulk' would rank-normalize first.
        ref = np.asarray(az.ess(np.asarray(chains), method="mean")).ravel()
        rel = np.max(np.abs(ours - ref) / ref)
        assert rel < 0.02, (name, rel, ours, ref)


def test_bulk_rhat_matches_arviz():
    az = pytest.importorskip("arviz")
    import numpy as np

    chains = _ar1_chains(jax.random.key(3), phi=0.9)
    ours = np.asarray(MCMCDiagnostics.bulk_rhat(chains))
    ref = np.asarray(az.rhat(np.asarray(chains), method="rank")).ravel()
    assert np.max(np.abs(ours - ref)) < 5e-3, (ours, ref)
