"""Timestamped artifact directories -- experiment hygiene for the pipeline scripts.

Every invocation of an artifact-producing script writes into its own timestamped directory

    <run_dir>/<script>/<YYYYmmdd_HHMMSS>/...

so nothing is ever overwritten and two runs are compared by diffing two directories. A
``latest`` symlink in ``<run_dir>/<script>/`` is refreshed per invocation, so downstream
consumers that just want "the current version" resolve it with ``latest_artifact`` and
need no flag-threading; pinning an older version is passing its explicit path instead.

Each directory carries a ``manifest.json`` (argv, resolved config, git commit, timestamp):
the manifest diff between two runs states exactly what changed between two artifact sets.

The E-series experiment records key their filenames by config (``--exp`` tags, budget,
gamma, day pair) and keep that scheme; this module is for the pipeline scripts whose fixed
output names previously clobbered earlier runs (``fle_uncertainty``, ``lineage``, ...).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _git_state(cwd: Path) -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def artifact_dir(run_dir, script: str, config: dict | None = None) -> Path:
    """Create ``<run_dir>/<script>/<stamp>/``, refresh ``latest``, write the manifest."""
    root = Path(run_dir) / script
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / stamp
    n = 1
    while out.exists():                       # same-second rerun
        n += 1
        out = root / f"{stamp}_{n}"
    out.mkdir(parents=True)
    link = root / "latest"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(out.name)                 # relative: survives moving the run dir
    manifest = {"created": stamp, "argv": sys.argv,
                "git": _git_state(Path(__file__).resolve().parent),
                "config": config or {}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


def latest_artifact(run_dir, script: str) -> Path:
    """Resolve ``<run_dir>/<script>/latest`` -> the newest artifact directory."""
    link = Path(run_dir) / script / "latest"
    if link.exists():
        return link.resolve()
    flat = Path(run_dir) / script            # pre-hygiene layout: fixed name at top level
    if flat.exists():
        return flat
    raise FileNotFoundError(
        f"no '{script}' artifacts under {run_dir}: run the producing script first")
