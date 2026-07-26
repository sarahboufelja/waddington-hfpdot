"""Device detection and GPU utilisation monitoring.

Every script and test should announce what hardware it is actually running on. The motivating
failure: a whole series of scaling experiments ran with ``JAX_PLATFORMS=cpu`` on a machine with two
idle GPUs, and nothing in the output said so -- the timings were quietly CPU timings, which made the
compute budget look far smaller than it was.

So the loud case here is the **mismatch**: accelerators physically present but JAX not using them.

Dependency-free (stdlib + an optional lazy ``jax`` import), so it is safe to import from the
numpy-only modules as well as the sampler.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------------------------------
# What hardware exists, and what is JAX actually using?
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    memory_total_mib: int


def physical_gpus() -> List[GpuDevice]:
    """GPUs visible to the driver, via nvidia-smi. Independent of JAX. Empty if none/unavailable."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(GpuDevice(int(parts[0]), parts[1], int(float(parts[2]))))
            except ValueError:
                continue
    return gpus


@dataclass(frozen=True)
class DeviceReport:
    jax_available: bool
    jax_version: Optional[str]
    jax_platform: Optional[str]           # 'cpu' | 'gpu' | 'tpu'
    jax_device_count: int
    jax_platforms_env: Optional[str]      # the JAX_PLATFORMS override, if set
    forced_device_count: Optional[str]    # xla_force_host_platform_device_count, if set
    gpus: List[GpuDevice] = field(default_factory=list)

    @property
    def gpus_present_but_unused(self) -> bool:
        """The silent trap: accelerators exist, JAX is on CPU anyway."""
        return bool(self.gpus) and self.jax_available and self.jax_platform == "cpu"

    @property
    def using_gpu(self) -> bool:
        return self.jax_available and self.jax_platform == "gpu"


def device_report() -> DeviceReport:
    """Inspect the machine and the JAX runtime without importing JAX unless it is already usable."""
    jax_version = jax_platform = None
    jax_available, device_count = False, 0
    try:
        import jax  # noqa: PLC0415  (lazy on purpose: numpy-only modules must not require jax)
        jax_available = True
        jax_version = jax.__version__
        devices = jax.devices()
        device_count = len(devices)
        jax_platform = devices[0].platform if devices else None
    except Exception:
        pass

    xla_flags = os.environ.get("XLA_FLAGS", "")
    forced = None
    if "xla_force_host_platform_device_count" in xla_flags:
        for token in xla_flags.split():
            if token.startswith("--xla_force_host_platform_device_count"):
                forced = token.split("=")[-1]

    return DeviceReport(
        jax_available=jax_available, jax_version=jax_version, jax_platform=jax_platform,
        jax_device_count=device_count, jax_platforms_env=os.environ.get("JAX_PLATFORMS"),
        forced_device_count=forced, gpus=physical_gpus(),
    )


def device_banner(report: DeviceReport | None = None, *, cpu_is_intentional: bool = False) -> str:
    """One-block human-readable summary. Shouts if GPUs are present but unused.

    ``cpu_is_intentional=True`` for contexts that pin CPU on purpose (the test suite pins it for
    determinism), so the warning keeps its meaning instead of crying wolf on every run.
    """
    r = report or device_report()
    lines = ["-" * 78]
    if not r.jax_available:
        lines.append("DEVICE: jax not importable (numpy-only context)")
    else:
        detail = f"jax {r.jax_version} | platform={r.jax_platform} | devices={r.jax_device_count}"
        if r.forced_device_count:
            detail += f" (host devices forced to {r.forced_device_count})"
        lines.append(f"DEVICE: {detail}")
    if r.gpus:
        for g in r.gpus:
            lines.append(f"        GPU {g.index}: {g.name} ({g.memory_total_mib} MiB)")
    else:
        lines.append("        no GPUs visible to the driver")

    if r.gpus_present_but_unused:
        env = f" (JAX_PLATFORMS={r.jax_platforms_env})" if r.jax_platforms_env else ""
        if cpu_is_intentional:
            lines.append(f"        CPU pinned deliberately{env}; {len(r.gpus)} GPU(s) idle by design")
        else:
            lines += [
                "",
                f"  *** WARNING: {len(r.gpus)} GPU(s) present but JAX is running on CPU{env}. ***",
                "  *** Timings and any feasibility/compute conclusions are CPU numbers.        ***",
            ]
    return "\n".join(lines + ["-" * 78])


def print_device_banner(*, cpu_is_intentional: bool = False) -> DeviceReport:
    r = device_report()
    print(device_banner(r, cpu_is_intentional=cpu_is_intentional), flush=True)
    return r


# --------------------------------------------------------------------------------------------------
# Are BOTH GPUs actually being worked?
# --------------------------------------------------------------------------------------------------

@dataclass
class GpuUtilisation:
    index: int
    samples: List[float] = field(default_factory=list)
    memory_used_mib: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def peak(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def peak_memory_mib(self) -> float:
        return max(self.memory_used_mib) if self.memory_used_mib else 0.0


class GpuMonitor:
    """Context manager sampling per-GPU utilisation while a job runs.

    Use it to confirm that a multi-device run genuinely spreads across every GPU rather than
    hammering one and leaving the rest idle:

        with GpuMonitor() as mon:
            ...run the sampler...
        print(mon.summary())
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.stats: dict[int, GpuUtilisation] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._available = shutil.which("nvidia-smi") is not None

    def _sample_once(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True).stdout
        except (subprocess.SubprocessError, OSError):
            return
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                idx, util, mem = int(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            self.stats.setdefault(idx, GpuUtilisation(idx)).samples.append(util)
            self.stats[idx].memory_used_mib.append(mem)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuMonitor":
        if self._available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval_s + 5)

    def summary(self) -> str:
        if not self._available:
            return "GPU utilisation: nvidia-smi unavailable"
        if not self.stats:
            return "GPU utilisation: no samples collected"
        rows = ["GPU utilisation (mean / peak over the run):"]
        for idx in sorted(self.stats):
            s = self.stats[idx]
            rows.append(f"  GPU {idx}: {s.mean:5.1f}% / {s.peak:5.1f}%   "
                        f"peak mem {s.peak_memory_mib:.0f} MiB   ({len(s.samples)} samples)")
        idle = [i for i, s in self.stats.items() if s.peak < 5.0]
        if idle and len(self.stats) > 1:
            rows.append(f"  *** WARNING: GPU(s) {idle} stayed idle -- work is not spread across "
                        f"all devices. ***")
        return "\n".join(rows)


if __name__ == "__main__":
    print_device_banner()
