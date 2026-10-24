"""Run one deterministic work unit (seed x scenario) of a profile inside a pod.

    python deploy/run_unit.py --config configs/paper.yaml --index 0 --tag nrp-cpu12-v1

Unit index -> (seed, scenario) is the row-major order of the config's ``seeds`` x ``scenarios``
(paper.yaml: 3 x 4 = 12 units, index 0 = seed 42 / iid). Inside a unit the existing runner runs
every method and ablation serially, so ``local`` is available to the KD round-0 determinism
diagnostic exactly as in the local run. Seeds, partitions, models and metrics are untouched;
only the output locations and (optionally) ``federation.client_num_cpus`` are overridden.

Layout (all on the persistent volume, unique per unit, so pods never write the same file):
    $RESULTS_ROOT/<tag>/unit_<idx>_seed<S>_<scenario>/{seed_S/..., hardware.json, runner.log, ledger.jsonl}
The data cache (processed splits, partitions) is rebuilt per pod in $SCRATCH from the read-only raw CSV.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heterofl.utils.config import load_config  # noqa: E402


def units(cfg: dict) -> list[tuple[int, str]]:
    return list(itertools.product(cfg["seeds"], cfg["scenarios"]))


def cgroup_cpu_limit() -> float | None:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        return None


def hardware_record() -> dict:
    model = None
    try:
        model = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                     if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NODE_NAME", "POD_NAME",
            "JOB_COMPLETION_INDEX", "CLIENT_NUM_CPUS")
    return {"cpu_model": model, "os_cpu_count": os.cpu_count(), "cgroup_cpu_limit": cgroup_cpu_limit(),
            "machine": platform.machine(), "platform": platform.platform(),
            "env": {k: os.environ.get(k) for k in keys},
            "image": os.environ.get("IMAGE_REF")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--index", type=int, default=None, help="default: $JOB_COMPLETION_INDEX")
    p.add_argument("--tag", default=os.environ.get("RUN_TAG"), help="run label, e.g. nrp-cpu12-v1")
    p.add_argument("--client-cpus", type=int, default=int(os.environ.get("CLIENT_NUM_CPUS", "4")),
                   help="federation.client_num_cpus (Ray reservation AND n_jobs per bank); 4 = original")
    p.add_argument("--raw", default=os.environ.get("RAW_CSV", "/data/raw/creditcard.csv"))
    p.add_argument("--results-root", default=os.environ.get("RESULTS_ROOT", "/results"))
    p.add_argument("--scratch", default=os.environ.get("SCRATCH", "/scratch"))
    p.add_argument("--list", action="store_true", help="print the unit table and exit")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="extra runner args after --, e.g. -- --no-ablations")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    table = units(cfg)
    if a.list:
        for i, (s, sc) in enumerate(table):
            print(f"{i:3d}  seed={s}  scenario={sc}")
        return 0
    idx = a.index if a.index is not None else int(os.environ["JOB_COMPLETION_INDEX"])
    if not a.tag:
        p.error("--tag (or $RUN_TAG) is required so results from different settings never mix")
    seed, scenario = table[idx]

    unit_dir = Path(a.results_root) / a.tag / f"unit_{idx:02d}_seed{seed}_{scenario}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "hardware.json").write_text(json.dumps(
        {**hardware_record(), "unit": idx, "seed": seed, "scenario": scenario,
         "client_num_cpus": a.client_cpus}, indent=2))

    # SIGTERM (preemption / `kubectl delete`) -> exception, so launch_subprocess kills Ray's process
    # group. The run's status.json stays "running" and the runner archives and redoes it on resume.
    def _term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _term)

    from heterofl.experiments.runner import main as runner_main

    extra = [x for x in a.extra if x != "--"]
    argv = ["--config", a.config, "--seeds", str(seed), "--scenarios", scenario, "--no-report",
            "--set", f"results_dir={unit_dir}",
            "--set", f"data.path={a.raw}",
            "--set", f"data.processed_dir={a.scratch}/processed",
            "--set", f"data.partitions_dir={a.scratch}/partitions",
            "--set", f"federation.client_num_cpus={a.client_cpus}", *extra]
    print(f"[unit {idx}] seed={seed} scenario={scenario} -> {unit_dir}", flush=True)
    try:
        return runner_main(argv)
    except KeyboardInterrupt:
        print("[unit] interrupted; rerun the same index to resume", flush=True)
        return 143


if __name__ == "__main__":
    sys.exit(main())
