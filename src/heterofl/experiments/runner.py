"""Experiment runner: the full (seed x scenario x method) matrix plus ablations.

    python -m heterofl.experiments.runner --config configs/smoke.yaml
    python -m heterofl.experiments.runner --config configs/paper.yaml
    python -m heterofl.experiments.runner --config configs/paper.yaml \
        --seeds 42 --scenarios iid --methods local ensemble --set kd.lambda_kd=1.0

Identity and resume (see ``heterofl.experiments.identity`` and docs/reproducibility.md):

* A results directory belongs to ONE experiment identity (semantic configuration +
  dataset SHA-256 + library versions), recorded in ``experiment.json``. Requesting
  a different identity for an existing directory fails before anything is written;
  use a new ``results_dir``. Directories without ``experiment.json`` (results from
  before identity recording) are legacy and are never extended or reused.
* Before any run starts, every planned run is checked. A run directory whose
  ``status.json`` says ``completed`` is skipped only if its ``run_identity.json``
  matches the requested run identity AND it was produced by the same source revision
  (``--allow-source-change`` relaxes the latter, and records that it did). Any
  mismatch aborts the invocation with an error naming the differing keys.
* Anything else (failed, interrupted) is moved to ``_archive/`` with a timestamp --
  diagnostics are preserved, never deleted -- and re-run. Completed runs are only
  re-run with ``--overwrite`` (they are archived too, not deleted).
* An exclusive lock (``.heterofl.lock``) prevents two processes from writing the
  same results directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from heterofl.data.loader import DatasetNotFoundError
from heterofl.data.partitioning import PartitionError
from heterofl.data.pipeline import DataBundle, ScenarioData, prepare_bundle, prepare_scenario
from heterofl.experiments import centralized, ensemble, fedxgb, heterogeneous, homogeneous, local
from heterofl.experiments.common import RunContext, write_outputs
from heterofl.experiments.identity import (
    IDENTITY_SCHEMA,
    ConfigMismatchError,
    LegacyResultsError,
    ResultsIdentityError,
    ResultsLock,
    SourceMismatchError,
    SourceNotVerifiedError,
    describe_mismatch,
    experiment_identity,
    identity_hash,
    partition_fingerprint,
    run_identity,
)
from heterofl.utils.config import config_hash, deep_merge, dump_config, load_config, resolve_path
from heterofl.utils.io import read_json, write_csv, write_json
from heterofl.utils.logging import FileLog, get_logger, setup_logging
from heterofl.utils.reproducibility import (
    configure_torch_threads,
    environment_info,
    same_source,
    set_global_seed,
    source_patch,
)

log = get_logger("runner")

METHOD_ORDER = ("local", "centralized", "ensemble", "fedavg", "heterogeneous_kd",
                "fedxgb_bagging")
METHOD_MODULES = {"local": local, "centralized": centralized, "ensemble": ensemble,
                  "fedavg": homogeneous, "heterogeneous_kd": heterogeneous,
                  "fedxgb_bagging": fedxgb}

SKIP_STATUS = {"skip": "skipped-completed", "skip-other-source": "skipped-completed-other-source"}

#: Files a completed run must contain to be reusable.
REQUIRED_RUN_OUTPUTS = ("metrics.csv", "metrics.json", "run_identity.json", "config.yaml")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


@dataclass
class PlannedRun:
    cfg: dict
    method: str
    seed: int
    scenario: str
    data: ScenarioData
    out: Path
    variant: str = "primary"


class Runner:
    def __init__(self, cfg: dict, *, overwrite: bool = False, allow_source_change: bool = False,
                 require_clean_source: bool = False):
        self.cfg = cfg
        self.overwrite = overwrite
        self.allow_source_change = allow_source_change
        self.root = resolve_path(cfg, cfg["results_dir"])
        self.invocation_id = f"{cfg['profile']}-{_now()}-{uuid.uuid4().hex[:6]}"
        self.outcomes: list[dict] = []
        self.env = environment_info(cfg["project_root"])
        self.source = self.env.get("git") or {}
        if require_clean_source and not self.source.get("source_verified"):
            raise SourceNotVerifiedError(
                f"--require-clean-source: the working tree is not a clean commit "
                f"(commit={self.source.get('commit')}, dirty={self.source.get('dirty')}); "
                "commit your changes first")
        self.dataset_sha256: str | None = None
        self.experiment_hash: str | None = None
        self._lock = ResultsLock(self.root)

    # ------------------------------------------------------------------ results dir
    def check_results_dir(self) -> None:
        """Read-only check: refuse legacy result trees before writing anything into them."""
        if not self.root.exists() or (self.root / "experiment.json").is_file():
            return
        markers = [p.name for p in (self.root / "config.yaml", self.root / "ledger.jsonl",
                                    self.root / "environment.json") if p.exists()]
        markers += [p.name for p in sorted(self.root.glob("seed_*"))]
        if markers:
            raise LegacyResultsError(
                f"{self.root} contains results without an experiment identity "
                f"(found {markers[:5]}): they predate configuration hashing, so it cannot "
                "be shown that they match the requested configuration. They are kept "
                "untouched as legacy results. Write new results elsewhere, e.g. "
                f"--set results_dir={self.root.parent.name}/{self.root.name}_v2")

    def close(self) -> None:
        self._lock.release()

    @property
    def publication_reproducible(self) -> bool:
        return (bool(self.cfg.get("publishable")) and not self.cfg["data"].get("synthetic")
                and bool(self.source.get("source_verified")))

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
        self.check_results_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self._lock.acquire():
            log.warning("File locking is unavailable on this platform; do not run two "
                        "processes on %s at the same time", self.root)
        bundle = prepare_bundle(self.cfg)
        self.dataset_sha256 = bundle.manifest["dataset"]["sha256"]
        ident = experiment_identity(self.cfg, dataset_sha256=self.dataset_sha256,
                                    packages=self.env["packages"])
        self.experiment_hash = identity_hash(ident)
        exp_path = self.root / "experiment.json"
        if exp_path.is_file():
            rec = read_json(exp_path)
            if rec.get("experiment_hash") != self.experiment_hash:
                raise ConfigMismatchError(
                    f"{self.root} holds results of experiment {rec.get('experiment_hash')}, "
                    f"but the requested configuration is {self.experiment_hash}. Differing: "
                    f"{describe_mismatch(rec.get('identity', {}), ident)}. Results of different "
                    "configurations are never mixed in one directory; choose a new "
                    "results_dir (e.g. --set results_dir=...).")
        else:
            write_json(exp_path, {
                "schema": IDENTITY_SCHEMA, "experiment_hash": self.experiment_hash,
                "config_hash": config_hash(self.cfg), "dataset_sha256": self.dataset_sha256,
                "created": _iso(time.time()), "created_by_invocation": self.invocation_id,
                "source_at_creation": self.source, "identity": ident})
            dump_config(self.cfg, self.root / "config.yaml")
            write_json(self.root / "environment.json", self.env)
            shutil.copy(bundle.dir / "dataset_audit.json", self.root / "dataset_audit.json")
            shutil.copy(bundle.dir / "split_manifest.json", self.root / "split_manifest.json")
        return bundle

    def record_invocation(self) -> None:
        """Per-invocation config/environment (and the source diff if the tree is dirty).

        Written only after the pre-flight identity checks passed, so a refused
        invocation leaves nothing behind but its line in runner.log.
        """
        inv = self.root / "invocations" / self.invocation_id
        dump_config(self.cfg, inv / "config.yaml")
        write_json(inv / "environment.json", self.env)
        if self.source.get("dirty"):
            patch = source_patch(self.cfg["project_root"])
            if patch:
                (inv / "source.patch").write_bytes(patch)
            log.warning("Source tree is dirty (commit %s, diff %s): results of this invocation "
                        "are NOT publication-reproducible; the tracked-file diff is saved in %s",
                        self.source.get("commit"), self.source.get("diff_sha256"), inv)

    # ------------------------------------------------------------------ identity / reuse
    def _identity(self, run: PlannedRun) -> dict:
        return run_identity(run.cfg, method=run.method, seed=run.seed, scenario=run.scenario,
                            variant=run.variant, dataset_sha256=self.dataset_sha256,
                            packages=self.env["packages"],
                            partition_sha256=partition_fingerprint(run.data.manifest))

    def _decide(self, run: PlannedRun, identity: dict) -> str:
        """``run`` | ``skip`` | ``skip-other-source``; raises if reuse would be unsafe."""
        out = run.out
        status_path = out / "status.json"
        if not status_path.is_file():
            return "run"
        st = read_json(status_path)
        if st.get("status") != "completed" or self.overwrite:
            return "run"
        id_path = out / "run_identity.json"
        if not id_path.is_file() or "run_hash" not in st:
            raise LegacyResultsError(
                f"{out} is a completed run without a recorded identity (legacy result); it "
                "cannot be shown to match the requested configuration. Use a new results_dir "
                "or --overwrite (archives it, never deletes).")
        stored = read_json(id_path)
        rhash = identity_hash(identity)
        if stored.get("run_hash") != rhash or st.get("run_hash") != rhash:
            raise ConfigMismatchError(
                f"{out} was completed under a different run identity "
                f"({stored.get('run_hash')} vs requested {rhash}); differing: "
                f"{describe_mismatch(stored.get('identity', {}), identity)}. It is not reused; "
                "use a new results_dir, or --overwrite to archive and re-run it.")
        missing = [f for f in REQUIRED_RUN_OUTPUTS if not (out / f).is_file()]
        if missing:
            raise ConfigMismatchError(f"{out} says completed but lacks {missing}; "
                                      "inspect it, then re-run with --overwrite")
        if not same_source(stored.get("source"), self.source):
            if not self.allow_source_change:
                raise SourceMismatchError(
                    f"{out} was produced by source {stored.get('source', {}).get('commit')} "
                    f"(dirty={stored.get('source', {}).get('dirty')}, diff "
                    f"{stored.get('source', {}).get('diff_sha256')}); the current source is "
                    f"{self.source.get('commit')} (dirty={self.source.get('dirty')}, diff "
                    f"{self.source.get('diff_sha256')}). Re-run with --overwrite, or pass "
                    "--allow-source-change to reuse it (recorded in the outcome log).")
            return "skip-other-source"
        return "skip"

    # ------------------------------------------------------------------ one run
    def run_one(self, cfg: dict, method: str, seed: int, scenario: str, data: ScenarioData,
                out: Path, variant: str = "primary") -> str:
        run = PlannedRun(cfg, method, seed, scenario, data, out, variant)
        identity = self._identity(run)
        decision = self._decide(run, identity)
        status_path = out / "status.json"
        if decision.startswith("skip"):
            log.info("Skipping completed %s (identity verified%s)", out,
                     "" if decision == "skip" else "; produced by a different source revision")
            self.outcomes.append({"path": str(out), "status": SKIP_STATUS[decision],
                                  "run_hash": identity_hash(identity)})
            return "skipped"
        if status_path.is_file():
            prev = read_json(status_path).get("status")
            self._archive(out, "overwritten" if prev == "completed" else f"stale-{prev}")
        elif out.exists():
            self._archive(out, "incomplete")

        out.mkdir(parents=True, exist_ok=True)
        run_id = f"{self.invocation_id}/{seed}/{scenario}/{variant}/{method}"
        rhash = identity_hash(identity)
        chash = identity["config_hash"]
        started = time.time()
        common = {"run_id": run_id, "run_hash": rhash, "config_hash": chash,
                  "experiment_hash": self.experiment_hash,
                  "source_commit": self.source.get("commit"),
                  "source_dirty": self.source.get("dirty"),
                  "started": started, "started_utc": _iso(started)}
        write_json(status_path, {"status": "running", **common})
        dump_config(cfg, out / "config.yaml")
        provenance = {"run_hash": rhash, "config_hash": chash,
                      "experiment_hash": self.experiment_hash,
                      "dataset_sha256": self.dataset_sha256, "source": self.source,
                      "source_verified": bool(self.source.get("source_verified")),
                      "publication_reproducible": self.publication_reproducible}
        write_json(out / "run_identity.json", {"schema": IDENTITY_SCHEMA, "run_hash": rhash,
                                               "config_hash": chash, "run_id": run_id,
                                               "invocation": self.invocation_id,
                                               "source": self.source, "identity": identity})
        write_json(out / "manifest.json", {
            "run_id": run_id, "method": method, "variant": variant, "seed": seed,
            "scenario": scenario, "publishable": bool(cfg.get("publishable", False)),
            "synthetic_data": bool(cfg["data"].get("synthetic", False)),
            "provenance": provenance,
            "partition_manifest": data.manifest,
            "split_manifest": data.bundle.manifest,
            "environment": self.env})
        ctx = RunContext(cfg=cfg, seed=seed, scenario=scenario, method=method, data=data,
                         out_dir=out, run_id=run_id, variant=variant, identity=identity)
        self._ledger({"event": "start", "run_id": run_id, "run_hash": rhash, "path": str(out)})
        with FileLog(out / "run.log"):
            log.info("=== %s | seed=%d | scenario=%s | variant=%s | run %s ===", method, seed,
                     scenario, variant, rhash[:12])
            try:
                set_global_seed(seed)
                METHOD_MODULES[method].run(ctx)
                write_outputs(ctx)
                finished = time.time()
                status = {"status": "completed", **common, "finished": finished,
                          "finished_utc": _iso(finished), "wall_seconds": finished - started}
            except Exception as exc:  # recorded, never hidden; other runs continue
                log.error("Run failed: %r\n%s", exc, traceback.format_exc())
                finished = time.time()
                status = {"status": "failed", **common, "finished": finished,
                          "finished_utc": _iso(finished), "error": repr(exc),
                          "traceback": traceback.format_exc()}
        write_json(status_path, status)
        self._ledger({"event": status["status"], "run_id": run_id, "run_hash": rhash,
                      "path": str(out), "error": status.get("error")})
        self.outcomes.append({"path": str(out), "status": status["status"],
                              "error": status.get("error"), "run_hash": rhash})
        return status["status"]

    # ------------------------------------------------------------------ matrix
    def _check_scenario_dir(self, sdir: Path, data: ScenarioData) -> None:
        mpath = sdir / "partition_manifest.json"
        if mpath.is_file() and partition_fingerprint(read_json(mpath)) != partition_fingerprint(
                data.manifest):
            raise ConfigMismatchError(
                f"{sdir} holds runs for a different partition than the one requested now "
                "(scenario definition, partition constraints or data changed); use a new "
                "results_dir")

    def plan(self, bundle: DataBundle, seeds, scenarios, methods, *, ablations: bool = True,
             primary: bool = True) -> list[PlannedRun]:
        ab_cfg = self.cfg.get("ablations") or {}
        planned: list[PlannedRun] = []
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
                self._check_scenario_dir(sdir, data)
                if primary:
                    allowed = self.cfg["scenarios"][scenario].get("methods", METHOD_ORDER)
                    for method in [m for m in METHOD_ORDER if m in methods and m in allowed]:
                        planned.append(PlannedRun(self.cfg, method, seed, scenario, data,
                                                  sdir / method))
                if ablations and ab_cfg.get("enabled", False):
                    for name, ab in ab_cfg.get("runs", {}).items():
                        ab_seeds = ab.get("seeds", ab_cfg.get("seeds", seeds))
                        ab_scen = ab.get("scenarios", ab_cfg.get("scenarios", scenarios))
                        if seed not in ab_seeds or scenario not in ab_scen:
                            continue
                        eff = deep_merge(self.cfg, ab.get("overrides", {}))
                        planned.append(PlannedRun(eff, ab["method"], seed, scenario, data,
                                                  sdir / "ablations" / name, variant=name))
        # Pre-flight: verify every completed run BEFORE executing anything, so a
        # mismatch never leaves a half-updated results tree behind.
        for run in planned:
            self._decide(run, self._identity(run))
        return planned

    def run_matrix(self, seeds, scenarios, methods, *, ablations: bool = True,
                   primary: bool = True) -> None:
        bundle = self.prepare()
        planned = self.plan(bundle, seeds, scenarios, methods, ablations=ablations,
                            primary=primary)
        self.record_invocation()
        copied: set[Path] = set()
        for run in planned:
            sdir = self.root / f"seed_{run.seed}" / run.scenario
            if sdir not in copied:
                sdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(run.data.dir / "partition_report.csv", sdir / "partition_report.csv")
                shutil.copy(run.data.dir / "partition_manifest.json",
                            sdir / "partition_manifest.json")
                copied.add(sdir)
            self.run_one(run.cfg, run.method, run.seed, run.scenario, run.data, run.out,
                         run.variant)
        write_csv(self.root / f"outcomes_{self.invocation_id}.csv", self.outcomes)

    # ------------------------------------------------------------------ flwr run helper
    def prepare_debug_job(self, mode: str, seed: int, scenario: str) -> Path:
        from heterofl.experiments.federated_common import build_job

        bundle = self.prepare()
        self.record_invocation()
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
    p.add_argument("--allow-source-change", action="store_true",
                   help="reuse completed runs produced by a different source revision "
                        "(recorded as skipped-completed-other-source)")
    p.add_argument("--require-clean-source", action="store_true",
                   help="refuse to run unless the working tree is a clean git commit")
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
    for m in methods:
        if m not in cfg["methods"]:
            print(f"Method {m!r} is not in the config's methods {cfg['methods']}", file=sys.stderr)
            return 2
    configure_torch_threads(int(cfg["models"].get("mlp", {}).get("torch_threads", 1)))

    try:
        runner = Runner(cfg, overwrite=args.overwrite,
                        allow_source_change=args.allow_source_change,
                        require_clean_source=args.require_clean_source)
        runner.check_results_dir()
    except ResultsIdentityError as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        return 4
    try:
        if args.prepare_job:
            job = runner.prepare_debug_job(args.prepare_job, seeds[0], scenarios[0])
            print(f"Wrote {job}\nRun it with:\n  flwr run . --stream --run-config "
                  f"'job-path=\"{job}\"'")
            return 0
        runner.root.mkdir(parents=True, exist_ok=True)
        with FileLog(runner.root / "runner.log"):
            log.info("Invocation %s: seeds=%s scenarios=%s methods=%s", runner.invocation_id,
                     seeds, scenarios, methods)
            try:
                runner.run_matrix(seeds, scenarios, methods, ablations=not args.no_ablations,
                                  primary=not args.ablations_only)
            except ResultsIdentityError as exc:
                log.error("Refusing to run: %s", exc)  # audit trail; nothing else was written
                raise
    except DatasetNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ResultsIdentityError as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        return 4
    finally:
        runner.close()

    failed = [o for o in runner.outcomes if o["status"] == "failed"]
    done = [o for o in runner.outcomes if o["status"] == "completed"]
    skipped = [o for o in runner.outcomes if o["status"].startswith("skipped")]
    other_src = [o for o in skipped if o["status"] == "skipped-completed-other-source"]
    print(f"\nRuns: {len(done)} completed, {len(skipped)} skipped (already completed, identity "
          f"verified{f'; {len(other_src)} from another source revision' if other_src else ''}), "
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
