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
