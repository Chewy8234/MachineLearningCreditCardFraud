"""Parallel unit scheduler for the validation campaign.

A *unit* is one runner invocation for one (group, seed, scenario): ``runner --seeds S --scenarios SC``
with its own results directory (the runner's ``.heterofl.lock`` allows one process per results root).
Units are independent, so running many at once changes wall-clock time only -- every run keeps its own
seeds, partitions, configuration and identity hash (``results_dir`` is excluded from the hash).

    python scripts/validation/run_units.py --tree /path/to/frozen/worktree --campaign v1 \
        --groups paper,tuned,split_1 -P 8                    # run
    python scripts/validation/run_units.py ... --dry-run       # list the queue and exit
    python scripts/validation/run_units.py ... --units split_1:42:iid   # restrict (equivalence gate)

Design notes
  * Runs from a *frozen* git worktree (``--tree``): cwd + PYTHONPATH point there, so every run records one
    source commit with ``source_dirty=false`` (``--require-clean-source`` is always passed).
  * Data cache is pre-warmed serially (``prepare_bundle`` / ``prepare_scenario`` are idempotent and keyed by
    a spec hash), then units only read it -- no concurrent-write races.
  * Longest-job-first ordering, per-unit logs, one retry on failure, ``.unit_done`` marker for fast resume.
  * Child env pins BLAS/OpenMP to 1 thread (explicit ``n_jobs`` governs XGBoost / Random Forest).
  * Ctrl-C / SIGTERM sends SIGINT to every runner first so it can kill its Ray process tree cleanly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PERMS = ["MRX", "MXR", "RMX", "RXM", "XMR"]

# group -> (config, extra --set overrides). Config paths are relative to the tree.
GROUPS: dict[str, tuple[str, list[str]]] = {
    "paper": ("configs/paper.yaml", []),
    "tuned": ("configs/val_tuned.yaml", []),
    **{f"split_{i}": ("configs/val_base.yaml", [f"split.seed={i}"]) for i in range(1, 11)},
    **{f"perm_{p}": (f"configs/val_perm_{p}.yaml", []) for p in PERMS},
    "temporal": ("configs/val_temporal.yaml", []),
    "masktime": ("configs/val_masktime.yaml", []),
    "nodup": ("configs/val_nodup.yaml", []),
    **{f"proxy_{t}": (f"configs/val_proxy_{t}.yaml", []) for t in ("001", "003", "025", "random_small")},
    **{f"rounds_{r}": (f"configs/val_rounds_{r}.yaml", []) for r in (1, 3, 10)},
}
# default priority order of the plan (Steps 1-5)
DEFAULT_ORDER = (["paper", "tuned"] + [f"split_{i}" for i in range(1, 6)] +
                 [f"perm_{p}" for p in PERMS] + ["temporal", "masktime", "nodup"] +
                 [f"proxy_{t}" for t in ("001", "003", "025", "random_small")] +
                 [f"rounds_{r}" for r in (1, 3, 10)] + [f"split_{i}" for i in range(6, 11)])


@dataclass
class Unit:
    group: str
    seed: int
    scenario: str
    config: str
    sets: list[str]
    weight: float
    out: Path
    log: Path
    attempts: int = 0
    started: float = 0.0
    proc: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"{self.group}:{self.seed}:{self.scenario}"


def child_env(tree: Path) -> dict:
    env = {**os.environ, "PYTHONPATH": f"{tree / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}".rstrip(os.pathsep),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    return env


def load_cfg(tree: Path, config: str, sets: list[str]) -> dict:
    sys.path.insert(0, str(tree / "src"))
    from heterofl.utils.config import load_config
    return load_config(tree / config, sets)


def est_weight(cfg: dict, scenario: str) -> float:
    """Rough relative cost: KD runs dominate (RF straggler); larger when ablations apply."""
    methods = cfg["methods"]
    scen = cfg["scenarios"][scenario]
    if scen.get("methods"):
        methods = scen["methods"]
    w = 1.0 * ("heterogeneous_kd" in methods) * 100 + 40 * ("centralized" in methods) + \
        20 * ("fedavg" in methods) + 20 * ("local" in methods)
    ab = cfg.get("ablations", {})
    if ab.get("enabled") and not scen.get("ablation") and scenario in (ab.get("scenarios") or cfg["scenarios"]):
        w += 100 * sum(1 for r in ab["runs"].values() if r.get("method") == "heterogeneous_kd") + \
            1 * sum(1 for r in ab["runs"].values() if r.get("method") != "heterogeneous_kd")
    w *= 1 + 0.03 * cfg["federation"]["num_rounds"]
    return w


def build_queue(tree: Path, results: Path, groups: list[str], only: set[str] | None) -> list[Unit]:
    units: list[Unit] = []
    for g in groups:
        config, sets = GROUPS[g]
        cfg = load_cfg(tree, config, sets)
        for seed, scen in itertools.product(cfg["seeds"], [s for s in cfg["scenarios"]]):
            u = Unit(g, int(seed), scen, config, sets, est_weight(cfg, scen),
                     results / g / f"unit_s{seed}_{scen}", results / g / f"unit_s{seed}_{scen}.log")
            if only is None or u.name in only:
                units.append(u)
    return units


PREWARM = """
import sys, json
from heterofl.utils.config import load_config
from heterofl.data.pipeline import prepare_bundle, prepare_scenario
cfg = load_config(sys.argv[1], sys.argv[2:])
b = prepare_bundle(cfg)
for s in cfg["seeds"]:
    for sc in cfg["scenarios"]:
        prepare_scenario(cfg, b, int(s), sc)
print("prewarmed", b.manifest["tag"])
"""


def prewarm(tree: Path, py: str, groups: list[str]) -> None:
    """Serially build every bundle/partition the queue needs (no concurrent cache writes later)."""
    for g in groups:
        config, sets = GROUPS[g]
        t0 = time.time()
        r = subprocess.run([py, "-c", PREWARM, str(tree / config), *sets], cwd=tree, env=child_env(tree),
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"prewarm failed for {g}:\n{r.stderr[-2000:]}")
        print(f"[prewarm] {g:<18} {time.time() - t0:5.1f}s", flush=True)


def launch(u: Unit, tree: Path, py: str) -> None:
    u.out.mkdir(parents=True, exist_ok=True)
    cmd = [py, "-m", "heterofl.experiments.runner", "--config", u.config, "--seeds", str(u.seed),
           "--scenarios", u.scenario, "--no-report", "--require-clean-source",
           "--set", f"results_dir={u.out}", *[x for s in u.sets for x in ("--set", s)]]
    fh = u.log.open("a")
    u.proc = subprocess.Popen(cmd, cwd=tree, env=child_env(tree), stdout=fh, stderr=subprocess.STDOUT,
                              start_new_session=True)
    u.started = time.time()
    u.attempts += 1


def stop_all(running: list[Unit]) -> None:
    for u in running:                         # SIGINT -> runner's KeyboardInterrupt -> kills Ray tree
        if u.proc and u.proc.poll() is None:
            os.killpg(u.proc.pid, signal.SIGINT)
    t0 = time.time()
    while time.time() - t0 < 60 and any(u.proc.poll() is None for u in running if u.proc):
        time.sleep(1)
    for u in running:
        if u.proc and u.proc.poll() is None:
            os.killpg(u.proc.pid, signal.SIGKILL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", type=Path, required=True, help="frozen git worktree to run from")
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--results-root", type=Path, default=None, help="default: <tree>/results")
    ap.add_argument("--groups", default="all", help="comma list or 'all' (plan order)")
    ap.add_argument("--units", default=None, help="comma list of group:seed:scenario to restrict to")
    ap.add_argument("-P", "--parallel", type=int, default=8)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-prewarm", action="store_true")
    ap.add_argument("--retries", type=int, default=1)
    a = ap.parse_args()

    tree = a.tree.resolve()
    results = (a.results_root or tree / "results").resolve() / a.campaign
    groups = DEFAULT_ORDER if a.groups == "all" else a.groups.split(",")
    for g in groups:
        if g not in GROUPS:
            ap.error(f"unknown group {g!r}; known: {sorted(GROUPS)}")
    only = set(a.units.split(",")) if a.units else None
    queue = build_queue(tree, results, groups, only)
    order = {g: i for i, g in enumerate(groups)}
    # priority: plan order of groups first, then heaviest units first within a group
    queue.sort(key=lambda u: (order[u.group], -u.weight))
    pending = [u for u in queue if not (u.out / ".unit_done").is_file()]
    print(f"{len(queue)} units, {len(queue) - len(pending)} already done, {len(pending)} to run, P={a.parallel}",
          flush=True)
    if a.dry_run:
        for u in queue:
            print(f"{u.name:<34} w={u.weight:6.0f} {'DONE' if u not in pending else ''}")
        return 0
    results.mkdir(parents=True, exist_ok=True)
    (results / "queue.json").write_text(json.dumps(
        [{"unit": u.name, "weight": u.weight, "config": u.config, "sets": u.sets} for u in queue], indent=1))
    if not a.no_prewarm:
        prewarm(tree, a.python, sorted({u.group for u in pending}, key=order.get))

    running: list[Unit] = []
    failed: list[Unit] = []
    done = 0
    t_start = time.time()
    last_print = 0.0

    def _term(signum, frame):
        print(f"\nsignal {signum}: stopping {len(running)} running units cleanly", flush=True)
        stop_all(running)
        sys.exit(143)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    queue_iter = list(pending)
    while queue_iter or running:
        while queue_iter and len(running) < a.parallel:
            u = queue_iter.pop(0)
            launch(u, tree, a.python)
            running.append(u)
            print(f"[start ] {u.name}", flush=True)
        for u in list(running):
            rc = u.proc.poll()
            if rc is None:
                continue
            running.remove(u)
            dt = time.time() - u.started
            if rc == 0:
                (u.out / ".unit_done").write_text(time.strftime("%FT%T"))
                done += 1
                print(f"[done  ] {u.name} ({dt / 60:.1f} min)", flush=True)
            elif u.attempts <= a.retries:
                print(f"[retry ] {u.name} exit {rc} after {dt / 60:.1f} min; see {u.log}", flush=True)
                queue_iter.insert(0, u)
            else:
                failed.append(u)
                print(f"[FAILED] {u.name} exit {rc}; see {u.log}", flush=True)
        if time.time() - last_print > 120:
            last_print = time.time()
            el = (time.time() - t_start) / 60
            rem_w = sum(u.weight for u in queue_iter) + sum(u.weight for u in running) / 2
            done_w = sum(u.weight for u in pending if (u.out / ".unit_done").is_file())
            eta = el * rem_w / done_w if done_w else float("nan")
            print(f"[status] {done}/{len(pending)} done, {len(running)} running, {len(queue_iter)} queued, "
                  f"{len(failed)} failed | {el:.0f} min elapsed, ETA ~{eta:.0f} min | load "
                  f"{os.getloadavg()[0]:.1f}", flush=True)
            (results / "progress.json").write_text(json.dumps(
                {"done": done, "running": [u.name for u in running], "queued": len(queue_iter),
                 "failed": [u.name for u in failed], "elapsed_min": el}))
        time.sleep(2)
    print(f"finished: {done} done, {len(failed)} failed, {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
