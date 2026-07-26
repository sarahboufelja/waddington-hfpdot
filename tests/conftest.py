"""Shared pytest configuration for the Langevin sampler test suite.

Puts ``src`` on the import path and forces JAX onto CPU with a small fixed
device count so tests are deterministic and don't require a GPU.
"""

import os
import sys
from pathlib import Path

# Run the test suite on CPU with a couple of fake devices so the multi-chain /
# sharding paths are exercised without needing real accelerators.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_report_header(config):
    """Announce the hardware every run, so nobody has to guess what a timing means.

    CPU is pinned above on purpose here, so the banner's "GPUs idle" alarm is suppressed -- it stays
    meaningful for scripts and experiments, where an unnoticed CPU fallback silently turns every
    timing into a CPU timing.
    """
    from device_info import device_banner

    return device_banner(cpu_is_intentional=True)
