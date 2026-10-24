"""Experiment runner: the full (seed x scenario x method) matrix plus ablations.

    python -m heterofl.experiments.runner --config configs/smoke.yaml
    python -m heterofl.experiments.runner --config configs/paper.yaml
    python -m heterofl.experiments.runner --config configs/paper.yaml \
        --seeds 42 --scenarios iid --methods local ensemble --set kd.lambda_kd=1.0

Resume semantics: a run directory whose ``status.json`` says ``completed`` is
skipped. Anything else (failed, interrupted) is moved to ``_archive/`` with a
timestamp -- diagnostics are preserved, never deleted -- and re-run. Completed
runs are only re-run with ``--overwrite`` (they are archived too, not deleted).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from heterofl.data.loader import DatasetNotFoundError
from heterofl.data.partitioning import PartitionError
from heterofl.data.pipeline import DataBundle, ScenarioData, prepare_bundle, prepare_scenario
from heterofl.experiments import centralized, ensemble, heterogeneous, homogeneous, local
from heterofl.experiments.common import RunContext, write_outputs
from heterofl.utils.config import deep_merge, dump_config, load_config, resolve_path
from heterofl.utils.io import read_json, write_csv, write_json
from heterofl.utils.logging import FileLog, get_logger, setup_logging
from heterofl.utils.reproducibility import environment_info, set_global_seed

log = get_logger("runner")

METHOD_ORDER = ("local", "centralized", "ensemble", "fedavg", "heterogeneous_kd")
METHOD_MODULES = {"local": local, "centralized": centralized, "ensemble": ensemble,
                  "fedavg": homogeneous, "heterogeneous_kd": heterogeneous}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Runner:
    def __init__(self, cfg: dict, *, overwrite: bool = False):
        self.cfg = cfg
        self.overwrite = overwrite
        self.root = resolve_path(cfg, cfg["results_dir"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.invocation_id = f"{cfg['profile']}-{_now()}-{uuid.uuid4().hex[:6]}"
        self.outcomes: list[dict] = []
        self.env = environment_info(cfg["project_root"])

    # ------------------------------------------------------------------ bookkeeping
    def _ledger(self, event: dict) -> None:
        with (self.root / "ledger.jsonl").open("a") as fh:
            fh.write(json.dumps({"time": _now(), "invocation": self.invocation_id, **event}) + "\n")

    def _archive(self, out: Path, reason: str) -> None:
        arch = out.parent / "_archive"
        arch.mkdir(parents=True, exist_ok=True)
        dest = arch / f"{out.name}-{reason}-{_now()}"
        shutil.move(str(out), str(dest))
        log.warning("Archived %s -> %s", out, dest)

    def prepare(self) -> DataBundle:
        bundle = prepare_bundle(self.cfg)
        write_json(self.root / "environment.json", self.env)
        dump_config(self.cfg, self.root / "config.yaml")
        shutil.copy(bundle.dir / "dataset_audit.json", self.root / "dataset_audit.json")
        shutil.copy(bundle.dir / "split_manifest.json", self.root / "split_manifest.json")
        return bundle

    # ------------------------------------------------------------------ one run
    def run_one(self, cfg: dict, method: str, seed: int, scenario: str, data: ScenarioData,
                out: Path, variant: str = "primary") -> str:
        status_path = out / "status.json"
        if status_path.is_file():
            prev = read_json(status_path).get("status")
            if prev == "completed" and not self.overwrite:
                log.info("Skipping completed %s", out)
                self.outcomes.append({"path": str(out), "status": "skipped-completed"})
                return "skipped"
            self._archive(out, "overwritten" if prev == "completed" else f"stale-{prev}")
        elif out.exists():
            self._archive(out, "incomplete")

        out.mkdir(parents=True, exist_ok=True)
        run_id = f"{self.invocation_id}/{seed}/{scenario}/{variant}/{method}"
        started = time.time()
        write_json(status_path, {"status": "running", "run_id": run_id, "started": started})
        dump_config(cfg, out / "config.yaml")
        write_json(out / "manifest.json", {
            "run_id": run_id, "method": method, "variant": variant, "seed": seed,
            "scenario": scenario, "publishable": bool(cfg.get("publishable", False)),
            "synthetic_data": bool(cfg["data"].get("synthetic", False)),
            "partition_manifest": data.manifest,
            "split_manifest": data.bundle.manifest,
            "environment": self.env})
        ctx = RunContext(cfg=cfg, seed=seed, scenario=scenario, method=method, data=data,
                         out_dir=out, run_id=run_id, variant=variant)
        self._ledger({"event": "start", "run_id": run_id, "path": str(out)})
        with FileLog(out / "run.log"):
            log.info("=== %s | seed=%d | scenario=%s | variant=%s ===", method, seed, scenario, variant)
            try:
                set_global_seed(seed)
                METHOD_MODULES[method].run(ctx)
                write_outputs(ctx)
                status = {"status": "completed", "run_id": run_id, "started": started,
                          "finished": time.time(), "wall_seconds": time.time() - started}
            except Exception as exc:  # recorded, never hidden; other runs continue
                log.error("Run failed: %r\n%s", exc, traceback.format_exc())
                status = {"status": "failed", "run_id": run_id, "started": started,
                          "finished": time.time(), "error": repr(exc),
                          "traceback": traceback.format_exc()}
        write_json(status_path, status)
        self._ledger({"event": status["status"], "run_id": run_id, "path": str(out),
                      "error": status.get("error")})
        self.outcomes.append({"path": str(out), "status": status["status"],
                              "error": status.get("error")})
        return status["status"]

    # ------------------------------------------------------------------ matrix
    def run_matrix(self, seeds, scenarios, methods, *, ablations: bool = True,
                   primary: bool = True) -> None:
        bundle = self.prepare()
        ab_cfg = self.cfg.get("ablations") or {}
        for seed in seeds:
            for scenario in scenarios:
                sdir = self.root / f"seed_{seed}" / scenario
                try:
                    data = prepare_scenario(self.cfg, bundle, seed, scenario)
                except PartitionError as exc:
                    log.error("Partitioning failed for seed=%d scenario=%s: %s", seed, scenario, exc)
                    sdir.mkdir(parents=True, exist_ok=True)
                    write_json(sdir / "partition_status.json", {"status": "failed", "error": str(exc)})
                    self.outcomes.append({"path": str(sdir), "status": "failed", "error": str(exc)})
                    continue
                sdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(data.dir / "partition_report.csv", sdir / "partition_report.csv")
                shutil.copy(data.dir / "partition_manifest.json", sdir / "partition_manifest.json")
                if primary:
                    allowed = self.cfg["scenarios"][scenario].get("methods", METHOD_ORDER)
                    for method in [m for m in METHOD_ORDER if m in methods and m in allowed]:
                        self.run_one(self.cfg, method, seed, scenario, data, sdir / method)
                if ablations and ab_cfg.get("enabled", False):
                    for name, ab in ab_cfg.get("runs", {}).items():
                        ab_seeds = ab.get("seeds", ab_cfg.get("seeds", seeds))
                        ab_scen = ab.get("scenarios", ab_cfg.get("scenarios", scenarios))
                        if seed not in ab_seeds or scenario not in ab_scen:
                            continue
                        eff = deep_merge(self.cfg, ab.get("overrides", {}))
                        self.run_one(eff, ab["method"], seed, scenario, data,
                                     sdir / "ablations" / name, variant=name)
        write_csv(self.root / f"outcomes_{self.invocation_id}.csv", self.outcomes)

    # ------------------------------------------------------------------ flwr run helper
    def prepare_debug_job(self, mode: str, seed: int, scenario: str) -> Path:
        from heterofl.experiments.federated_common import build_job

        bundle = self.prepare()
        data = prepare_scenario(self.cfg, bundle, seed, scenario)
        out = self.root / "debug" / f"{mode}-seed{seed}-{scenario}-{_now()}"
        out.mkdir(parents=True)
        ctx = RunContext(cfg=self.cfg, seed=seed, scenario=scenario, method=mode, data=data,
                         out_dir=out, run_id=f"debug/{out.name}")
        return build_job(ctx, mode).path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heterogeneous federated committees: experiment runner")
    p.add_argument("--config", required=True, help="profile YAML (configs/smoke.yaml, configs/paper.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set kd.lambda_kd=1.0 (repeatable)")
    p.add_argument("--seeds", type=int, nargs="+", help="subset of configured seeds")
    p.add_argument("--scenarios", nargs="+", help="subset of configured scenarios")
    p.add_argument("--methods", nargs="+", help="subset of configured methods")
    p.add_argument("--overwrite", action="store_true",
                   help="re-run completed runs (old results are archived, not deleted)")
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--ablations-only", action="store_true")
    p.add_argument("--no-report", action="store_true", help="skip automatic report generation")
    p.add_argument("--prepare-job", choices=("fedavg", "kd"),
                   help="only write a job.json for `flwr run` debugging and print the command")
    args = p.parse_args(argv)

    setup_logging()
    try:
        cfg = load_config(args.config, args.set)
    except Exception as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    seeds = args.seeds or list(cfg["seeds"])
    scenarios = args.scenarios or list(cfg["scenarios"])
    methods = args.methods or list(cfg["methods"])
    for s in seeds:
        if s not in cfg["seeds"]:
            print(f"Seed {s} is not in the config's seeds {cfg['seeds']}", file=sys.stderr)
            return 2
    for s in scenarios:
        if s not in cfg["scenarios"]:
            print(f"Unknown scenario {s!r}; configured: {list(cfg['scenarios'])}", file=sys.stderr)
            return 2

    runner = Runner(cfg, overwrite=args.overwrite)
    try:
        if args.prepare_job:
            job = runner.prepare_debug_job(args.prepare_job, seeds[0], scenarios[0])
            print(f"Wrote {job}\nRun it with:\n  flwr run . --stream --run-config "
                  f"'job-path=\"{job}\"'")
            return 0
        with FileLog(runner.root / "runner.log"):
            log.info("Invocation %s: seeds=%s scenarios=%s methods=%s", runner.invocation_id,
                     seeds, scenarios, methods)
            runner.run_matrix(seeds, scenarios, methods, ablations=not args.no_ablations,
                              primary=not args.ablations_only)
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    failed = [o for o in runner.outcomes if o["status"] == "failed"]
    done = [o for o in runner.outcomes if o["status"] == "completed"]
    skipped = [o for o in runner.outcomes if o["status"].startswith("skipped")]
    print(f"\nRuns: {len(done)} completed, {len(skipped)} skipped (already completed), "
          f"{len(failed)} failed")
    for o in failed:
        print(f"  FAILED {o['path']}: {o['error']}")

    if not args.no_report and cfg.get("report", {}).get("auto", True):
        from heterofl.evaluation.report import generate_report

        try:
            out = generate_report(runner.root, cfg)
            print(f"Report written to {out}")
        except Exception as exc:
            log.error("Report generation failed: %r\n%s", exc, traceback.format_exc())
            print(f"Report generation failed: {exc}", file=sys.stderr)
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
