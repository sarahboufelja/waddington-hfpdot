"""End-to-end HFPD-OT results pipeline -- the single entry point.

One invocation produces (or adopts) one complete result set for the five result
families: (1) transition tables with plan+identity bands, (2) FLE identity and plan
overlays, (3) the propagation video, (4) Dobrushin/composition analysis inputs,
(5) the W-series channel. Downstream reader scripts enrich or summarise these
records; nothing downstream recomputes them.

    stage 1  CE backbone      run_e2e_lineage at each epsilon of --eps-grid
                              (deterministic tables: CE layer, W-series REG-check
                              baseline, composition analysis)
    stage 2  posterior store  run_particle_filter over the four certified windows
                              (bands, growth, raw draw tables, W-series + E14
                              channels), retry passes with per-pair resume
    stage 3  readers          fle_uncertainty -> fle_plan_bands + fle_plan_video per
                              window -> growth/tube summary (summary.json)

Children run as subprocesses of their existing CLIs: each stays standalone-
reproducible, and per-window process isolation bounds peak memory (the OOM-survival
property of the band campaign). Idempotence is adoption: a child whose manifest
config matches and whose record passes the declared schema (wadd_schema) is adopted,
not recomputed -- interrupting and relaunching the orchestrator is always safe, and
records produced outside the orchestrator are adopted the same way.

The master manifest e2e/<stamp>/manifest.json maps each stage to the child artifact
directories it produced or adopted.

Run from the repo root:
    python scripts/run_e2e_hfpdot.py --budget 500
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_schema import validate_record

#: the four certified windows of the band campaign (full GSE122662 timecourse)
WINDOWS = (
    (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6),
    (6, 6.5, 7, 7.5, 8, 8.25, 8.5, 8.75, 9, 9.5, 10, 10.5, 11, 11.5, 12),
    (12, 12.5, 13, 13.5, 14),
    (14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18),
)

#: certified fixed-kernel configuration (E11-E13 ratification; experiments.md)
PF_DEFAULTS = dict(particles=8, draws_kept=50, lam=10.0, lam_I=0.5, lam_pi=1.0, eps=0.0,
                   support="positive_orthant", rank=2, gamma=0.5, seed=0,
                   sampler=dict(samples=10000, burnin=2000, warmup=1200, step=0.02, chains=4))


def _tags(days):
    return [f"{a:g}->{b:g}" for a, b in zip(days, days[1:])]


def _norm(cfg):
    """The manifest stores config after a json round trip; compare in that space."""
    return json.loads(json.dumps(cfg, default=str))


def _adopt(run_dir, script, want, complete_fn):
    """Newest child dir of ``script`` whose manifest config matches ``want`` (subset,
    json-normalised) and whose payload ``complete_fn(dir)`` accepts; None otherwise."""
    want = _norm(want)
    for man in sorted((run_dir / script).glob("*/manifest.json"), reverse=True):
        if man.parent.name == "latest":
            continue
        try:
            stored = json.loads(man.read_text()).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        stored.pop("resumed_from", None)
        if all(stored.get(k) == v for k, v in want.items()) and complete_fn(man.parent):
            return man.parent
    return None


def _pf_complete(days):
    """Completeness check for a particle-filter window: schema-valid complete record."""
    def check(child):
        recs = sorted(child.glob("particle_filter_*.npz"))
        if not recs:
            return False
        try:
            z = np.load(recs[0], allow_pickle=True)
            validate_record(z, _tags(days), complete=True)
            return True
        except (OSError, ValueError, KeyError):
            return False
    return check


def _lineage_complete(child):
    return (child / "transition_tables.npz").exists() and (child / "layout.npz").exists()


def _record_of(child):
    return sorted(child.glob("particle_filter_*.npz"))[0]


class Runner:
    """Subprocess driver: one log file per child under the master dir, adoption first."""

    def __init__(self, run_dir, master):
        self.run_dir = run_dir
        self.logs = master / "logs"
        self.logs.mkdir(exist_ok=True)
        self.manifest = master / "manifest.json"
        self.children = {}

    def note(self, stage, name, child_dir):
        self.children.setdefault(stage, {})[name] = str(child_dir)
        man = json.loads(self.manifest.read_text())
        man["children"] = self.children
        self.manifest.write_text(json.dumps(man, indent=2))

    def run(self, name, cmd, gpu_pinned=False):
        env = dict(os.environ)
        if gpu_pinned:
            env["CUDA_VISIBLE_DEVICES"] = "0"
        log = self.logs / f"{name}.log"
        t0 = time.monotonic()
        with open(log, "a") as fh:
            fh.write(f"\n=== {' '.join(cmd)}\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                cwd=ROOT).returncode
        print(f"  {name}: exit {rc} in {time.monotonic() - t0:.0f}s  (log: {log})",
              flush=True)
        return rc


def stage1(run, budget, eps_grid):
    print("stage 1 -- CE backbone (deterministic lineage tables)", flush=True)
    for eps in eps_grid:
        want = dict(budget=budget, reg=eps)
        child = _adopt(run.run_dir, "lineage", want, _lineage_complete)
        if child is None:
            run.run(f"lineage_eps{eps:g}",
                    [sys.executable, "scripts/run_e2e_lineage.py", str(run.run_dir),
                     "--budget", str(budget), "--reg", str(eps)])
            child = _adopt(run.run_dir, "lineage", want, _lineage_complete)
        state = "MISSING (child failed)" if child is None else child.name
        print(f"  eps={eps:g}: {state}", flush=True)
        if child is not None:
            run.note("stage1", f"lineage_eps{eps:g}", child)


def stage2(run, budget, passes):
    print("stage 2 -- posterior store (certified particle-filter windows)", flush=True)
    pending = {days: dict(PF_DEFAULTS, days=list(days), budget=budget) for days in WINDOWS}
    for n in range(1, passes + 1):
        for days, cfg in list(pending.items()):
            child = _adopt(run.run_dir, "particle_filter", cfg, _pf_complete(days))
            if child is not None:
                print(f"  D{days[0]:g}-D{days[-1]:g}: complete ({child.name})", flush=True)
                run.note("stage2", f"D{days[0]:g}-D{days[-1]:g}", child)
                del pending[days]
                continue
            s = cfg["sampler"]
            run.run(f"pf_D{days[0]:g}-D{days[-1]:g}",
                    [sys.executable, "scripts/run_particle_filter.py", str(run.run_dir),
                     "--days", *[f"{d:g}" for d in days],
                     "--budget", str(budget), "--particles", str(cfg["particles"]),
                     "--samples", str(s["samples"]), "--burnin", str(s["burnin"]),
                     "--warmup", str(s["warmup"]), "--support", cfg["support"],
                     "--resume"], gpu_pinned=True)
        if not pending:
            break
        print(f"  pass {n}: {len(pending)} window(s) still incomplete", flush=True)
    return [days for days in WINDOWS if days not in pending]


def stage3(run, budget, complete_windows):
    print("stage 3 -- readers (overlays, video, summary)", flush=True)
    child = _adopt(run.run_dir, "fle_uncertainty", {}, lambda c: any(c.glob("*.npz")))
    if child is None:
        run.run("fle_uncertainty",
                [sys.executable, "scripts/fle_uncertainty.py", str(run.run_dir)])
        child = _adopt(run.run_dir, "fle_uncertainty", {}, lambda c: any(c.glob("*.npz")))
    if child is not None:
        run.note("stage3", "fle_uncertainty", child)

    summary = {}
    for days in complete_windows:
        cfg = dict(PF_DEFAULTS, days=list(days), budget=budget)
        child = _adopt(run.run_dir, "particle_filter", cfg, _pf_complete(days))
        rec = _record_of(child)
        name = f"D{days[0]:g}-D{days[-1]:g}"
        for reader, script in (("bands", "fle_plan_bands.py"), ("video", "fle_plan_video.py")):
            run.run(f"{reader}_{name}",
                    [sys.executable, f"scripts/{script}", str(rec),
                     "--run-dir", str(run.run_dir), "--budget", str(budget),
                     "--seed", str(PF_DEFAULTS["seed"])])
        run.run(f"branches_{name}",
                [sys.executable, "scripts/fle_flow_branches.py", str(rec),
                 "--run-dir", str(run.run_dir)])
        z = np.load(rec, allow_pickle=True)
        summary[name] = {
            "record": str(rec),
            "tube_eta_prop": np.asarray(z["tube_eta_prop"]).tolist(),
            "mass_ratio_median": {t: float(np.asarray(z[f"mass_ratio_q_{t}"]).ravel()[1])
                                  for t in _tags(days)},
            "growth_fate_rel_median_range": {
                t: [float(np.asarray(z[f"growth_fate_rel_q_{t}"])[1].min()),
                    float(np.asarray(z[f"growth_fate_rel_q_{t}"])[1].max())]
                for t in _tags(days)},
        }
    out = run.logs.parent / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  summary -> {out}", flush=True)


def main(run_dir, budget, eps_grid, stages, passes):
    run_dir = Path(run_dir)
    master = artifact_dir(run_dir, "e2e",
                          config=dict(budget=budget, eps_grid=list(eps_grid),
                                      stages=stages, passes=passes))
    print(f"e2e master: {master}", flush=True)
    run = Runner(run_dir, master)
    if 1 in stages:
        stage1(run, budget, eps_grid)
    complete = list(WINDOWS)
    if 2 in stages:
        complete = stage2(run, budget, passes)
    if 3 in stages:
        stage3(run, budget, complete)
    missing = [d for d in WINDOWS if d not in complete]
    if missing:
        print(f"INCOMPLETE windows after {passes} passes: "
              f"{['D%g-D%g' % (d[0], d[-1]) for d in missing]}", flush=True)
        return 1
    print("e2e complete", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    p.add_argument("--eps-grid", type=float, nargs="+", default=[0.1, 0.2])
    p.add_argument("--stages", default="1,2,3",
                   help="comma-separated subset of 1 (CE backbone), 2 (posterior store), "
                        "3 (readers)")
    p.add_argument("--passes", type=int, default=6,
                   help="stage-2 retry passes (each pass resumes incomplete windows)")
    a = p.parse_args()
    sys.exit(main(a.run_dir or latest_run(), a.budget, a.eps_grid,
                  tuple(int(s) for s in a.stages.split(",")), a.passes))
