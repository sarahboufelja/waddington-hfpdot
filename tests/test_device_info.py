"""Tests for device detection and GPU utilisation monitoring.

The point of this module is to make the hardware impossible to mistake, so the tests focus on the
failure it exists to catch: accelerators present while JAX quietly runs on CPU.
"""
import numpy as np
import pytest

from device_info import (
    DeviceReport, GpuDevice, GpuMonitor, GpuUtilisation, device_banner, device_report,
)


def _report(**kw) -> DeviceReport:
    base = dict(jax_available=True, jax_version="0.9.0", jax_platform="gpu", jax_device_count=2,
                jax_platforms_env=None, forced_device_count=None, gpus=[])
    base.update(kw)
    return DeviceReport(**base)


TWO_GPUS = [GpuDevice(0, "RTX 4070 Ti SUPER", 16376), GpuDevice(1, "RTX 4070 Ti SUPER", 16376)]


# ---- the trap: GPUs present, JAX on CPU --------------------------------------------------------

def test_detects_gpus_present_but_unused():
    r = _report(jax_platform="cpu", jax_platforms_env="cpu", gpus=TWO_GPUS)
    assert r.gpus_present_but_unused is True
    assert r.using_gpu is False


def test_banner_warns_loudly_when_gpus_are_idle_unintentionally():
    r = _report(jax_platform="cpu", jax_platforms_env="cpu", forced_device_count="4", gpus=TWO_GPUS)
    banner = device_banner(r)
    assert "WARNING" in banner
    assert "2 GPU(s) present but JAX is running on CPU" in banner
    assert "CPU numbers" in banner                      # the consequence is spelled out
    assert "JAX_PLATFORMS=cpu" in banner                # and the cause is named


def test_banner_suppresses_the_alarm_when_cpu_is_intentional():
    """The test suite pins CPU on purpose; the alarm must stay meaningful elsewhere."""
    r = _report(jax_platform="cpu", jax_platforms_env="cpu", gpus=TWO_GPUS)
    banner = device_banner(r, cpu_is_intentional=True)
    assert "WARNING" not in banner
    assert "idle by design" in banner


def test_no_warning_when_actually_using_gpu():
    r = _report(jax_platform="gpu", gpus=TWO_GPUS)
    assert r.gpus_present_but_unused is False
    assert r.using_gpu is True
    assert "WARNING" not in device_banner(r)


def test_no_warning_on_a_machine_without_gpus():
    r = _report(jax_platform="cpu", jax_platforms_env="cpu", gpus=[])
    assert r.gpus_present_but_unused is False
    banner = device_banner(r)
    assert "WARNING" not in banner
    assert "no GPUs visible" in banner


# ---- banner contents ---------------------------------------------------------------------------

def test_banner_reports_forced_host_device_count():
    """Simulated CPU devices must be visible -- 4 'devices' on CPU is not 4 accelerators."""
    r = _report(jax_platform="cpu", jax_device_count=4, forced_device_count="4", gpus=[])
    assert "host devices forced to 4" in device_banner(r)


def test_banner_lists_each_gpu_with_memory():
    banner = device_banner(_report(gpus=TWO_GPUS))
    assert "GPU 0: RTX 4070 Ti SUPER (16376 MiB)" in banner
    assert "GPU 1:" in banner


def test_banner_handles_missing_jax():
    banner = device_banner(_report(jax_available=False, jax_version=None, jax_platform=None,
                                   jax_device_count=0))
    assert "jax not importable" in banner


def test_device_report_runs_on_this_machine():
    r = device_report()
    assert isinstance(r, DeviceReport)
    assert r.jax_device_count >= 0
    device_banner(r)                                    # must not raise


# ---- utilisation monitoring --------------------------------------------------------------------

def test_utilisation_statistics():
    u = GpuUtilisation(0, samples=[10.0, 90.0, 50.0], memory_used_mib=[100.0, 800.0])
    assert np.isclose(u.mean, 50.0)
    assert np.isclose(u.peak, 90.0)
    assert np.isclose(u.peak_memory_mib, 800.0)


def test_utilisation_statistics_with_no_samples():
    u = GpuUtilisation(0)
    assert u.mean == 0.0 and u.peak == 0.0 and u.peak_memory_mib == 0.0


def test_monitor_flags_an_idle_device():
    """A multi-GPU run that only works one card is a sharding bug -- it must be called out."""
    mon = GpuMonitor()
    mon._available = True
    mon.stats = {0: GpuUtilisation(0, samples=[95.0, 99.0]),
                 1: GpuUtilisation(1, samples=[0.0, 1.0])}     # second card idle
    summary = mon.summary()
    assert "WARNING" in summary and "[1]" in summary
    assert "not spread across all devices" in summary


def test_monitor_is_quiet_when_all_devices_are_busy():
    mon = GpuMonitor()
    mon._available = True
    mon.stats = {0: GpuUtilisation(0, samples=[95.0]), 1: GpuUtilisation(1, samples=[93.0])}
    assert "WARNING" not in mon.summary()


def test_monitor_degrades_gracefully_without_nvidia_smi():
    mon = GpuMonitor()
    mon._available = False
    assert "unavailable" in mon.summary()


def test_monitor_context_manager_is_safe_to_use_anywhere():
    """Entering/exiting must never raise, GPUs or not -- it wraps real experiment runs."""
    with GpuMonitor(interval_s=0.01) as mon:
        _ = sum(range(1000))
    assert isinstance(mon.summary(), str)
