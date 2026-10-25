"""Safe resume: completed results are reused only under an identical identity.

Uses the in-process ``local`` condition (no Flower) on synthetic, non-publishable data.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import yaml

from heterofl.experiments.identity import (
    ConfigMismatchError,
    LegacyResultsError,
    ResultsLock,
    ResultsLockedError,
    SourceMismatchError,
    dependency_mismatch,
    identity_hash,
)
from heterofl.experiments.runner import PlannedRun, Runner, main
from heterofl.utils.config import load_config
from heterofl.utils.io import read_json, write_json

from .conftest import ROOT, TINY_MODELS


def _sets(tmp_path, csv):
    return [f"data.path={csv}", f"data.processed_dir={tmp_path / 'processed'}",
            f"data.partitions_dir={tmp_path / 'partitions'}", f"results_dir={tmp_path / 'results'}",
            "partitioning.min_fraud_per_client=5", "partitioning.min_legit_per_client=100",
            "federation.client_num_cpus=1", "ablations.enabled=false", *TINY_MODELS]


def _args(tmp_path, csv, extra=(), methods=("local",), config=None):
    args = ["--config", str(config or ROOT / "configs" / "smoke_synthetic.yaml"), "--no-report",
            "--methods", *methods]
    for s in [*_sets(tmp_path, csv), *extra]:
        args += ["--set", s]
    return args


#: written by every invocation, including a refused one: the lock (owner pid) and the
#: append-only audit log, which records the refusal
AUDIT_FILES = {ResultsLock.FILENAME, "runner.log"}


def _digest_tree(root, skip=AUDIT_FILES):
    """SHA-256 of every result file under root (to prove a refused invocation changed nothing)."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name not in skip}


def _outcomes(tmp_path):
    files = (tmp_path / "results").glob("outcomes_*.csv")
    return pd.read_csv(max(files, key=lambda f: f.stat().st_mtime_ns))


@pytest.fixture()
def completed(tmp_path, synthetic_csv):
    """One completed local run in tmp_path/results."""
    assert main(_args(tmp_path, synthetic_csv)) == 0
    run = tmp_path / "results" / "seed_42" / "moderate_non_iid" / "local"
    assert read_json(run / "status.json")["status"] == "completed"
    return run


# 1 ---------------------------------------------------------------------------------------
def test_identical_configuration_resumes(tmp_path, synthetic_csv, completed):
    st = read_json(completed / "status.json")
    ident = read_json(completed / "run_identity.json")
    assert st["run_hash"] == ident["run_hash"] == identity_hash(ident["identity"])
    assert st["config_hash"] == ident["config_hash"] and st["started_utc"] and st["finished_utc"]
    exp = read_json(tmp_path / "results" / "experiment.json")
    assert exp["experiment_hash"] == st["experiment_hash"]
    man = read_json(completed / "manifest.json")["provenance"]
    assert man["dataset_sha256"] == exp["dataset_sha256"] and "commit" in man["source"]
    assert man["publication_reproducible"] is False  # synthetic data
    before = _digest_tree(completed)
    assert main(_args(tmp_path, synthetic_csv)) == 0
    assert (_outcomes(tmp_path)["status"] == "skipped-completed").all()
    assert _digest_tree(completed) == before


# 2-7 -------------------------------------------------------------------------------------
@pytest.mark.parametrize("override", [
    "kd.lambda_kd=1.0",                          # 2: distillation strength
    "models.xgboost.max_depth=2",                # 4: model hyperparameter
    "ensemble.weighting=validation_ap",          # 5: aggregation
    "kd.teacher_weighting=validation_ap",        # 5: aggregation (teachers)
    "split.seed=7",                              # 6: split seed
    "partitioning.min_fraud_per_client=6",       # 7: partition configuration
    "data.subsample_fraction=0.9",
])
def test_changed_configuration_is_refused_and_results_untouched(tmp_path, synthetic_csv,
                                                                completed, override):
    root = tmp_path / "results"
    before = _digest_tree(root)
    assert main(_args(tmp_path, synthetic_csv, [override])) == 4
    assert _digest_tree(root) == before
    assert "Refusing to run" in (root / "runner.log").read_text(encoding="utf-8")


def test_changed_models_yaml_is_refused(tmp_path, synthetic_csv, completed):    # 3
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    models["random_forest"]["min_samples_leaf"] = 3
    alt = tmp_path / "models_alt.yaml"
    alt.write_text(yaml.safe_dump(models), encoding="utf-8")
    before = _digest_tree(tmp_path / "results")
    assert main(_args(tmp_path, synthetic_csv, [f"models_config={alt}"])) == 4
    assert read_json(completed / "status.json")["status"] == "completed"
    assert _digest_tree(tmp_path / "results") == before


def test_changed_scenario_definition_is_refused(tmp_path, synthetic_csv, completed):  # 7
    before = _digest_tree(tmp_path / "results")
    assert main(_args(tmp_path, synthetic_csv, ["scenarios.moderate_non_iid.alpha=0.3"])) == 4
    assert _digest_tree(tmp_path / "results") == before


def test_changed_run_seed_identity_is_refused(tmp_path, synthetic_csv, completed):  # 6
    cfg = load_config(ROOT / "configs" / "smoke_synthetic.yaml", _sets(tmp_path, synthetic_csv))
    runner = Runner(cfg)
    try:
        bundle = runner.prepare()
        [run] = runner.plan(bundle, [42], ["moderate_non_iid"], ["local"], ablations=False)
        ident = runner._identity(run)
        assert runner._decide(run, ident) == "skip"
        other = {**ident, "seed": 123}            # e.g. results copied between seed dirs
        with pytest.raises(ConfigMismatchError, match="seed"):
            runner._decide(run, other)
    finally:
        runner.close()


def test_non_semantic_changes_still_resume(tmp_path, synthetic_csv, completed):
    assert main(_args(tmp_path, synthetic_csv, ["federation.job_timeout=77",
                                                "report.auto=false"])) == 0
    assert (_outcomes(tmp_path)["status"] == "skipped-completed").all()


# 8 is covered by tests/test_config.py::test_equivalent_configurations_hash_identically

# 9 ---------------------------------------------------------------------------------------
def test_interrupted_run_resumes_safely(tmp_path, synthetic_csv, completed):
    st = read_json(completed / "status.json")
    write_json(completed / "status.json", {**st, "status": "running"})   # simulated crash
    (completed / "metrics.csv").unlink()
    assert main(_args(tmp_path, synthetic_csv)) == 0
    new = read_json(completed / "status.json")
    assert new["status"] == "completed" and new["run_hash"] == st["run_hash"]
    assert new["run_id"] != st["run_id"]
    archived = list((completed.parent / "_archive").glob("local-stale-running-*"))
    assert archived and not (archived[0] / "metrics.csv").exists()


def test_overwrite_archives_and_reruns(tmp_path, synthetic_csv, completed):
    old_id = read_json(completed / "status.json")["run_id"]
    assert main(_args(tmp_path, synthetic_csv) + ["--overwrite"]) == 0
    assert read_json(completed / "status.json")["run_id"] != old_id
    assert list((completed.parent / "_archive").glob("local-overwritten-*"))


# 10 --------------------------------------------------------------------------------------
def _legacy_tree(root):
    run = root / "seed_42" / "moderate_non_iid" / "local"
    run.mkdir(parents=True)
    write_json(run / "status.json", {"status": "completed", "run_id": "old"})
    (run / "metrics.csv").write_text("system,test_average_precision\nbank_A,0.9\n", encoding="utf-8")
    (root / "config.yaml").write_text("profile: legacy\n", encoding="utf-8")
    (root / "ledger.jsonl").write_text("{}\n", encoding="utf-8")
    return run


def test_legacy_results_without_identity_are_never_reused(tmp_path, synthetic_csv):
    root = tmp_path / "results"
    _legacy_tree(root)
    before = _digest_tree(root)
    assert main(_args(tmp_path, synthetic_csv)) == 4
    assert _digest_tree(root, skip=set()) == before   # not even a lock file or runner.log
    with pytest.raises(LegacyResultsError):
        Runner(load_config(ROOT / "configs" / "smoke_synthetic.yaml",
                           _sets(tmp_path, synthetic_csv))).check_results_dir()


def test_completed_run_without_identity_inside_new_root_is_refused(tmp_path, synthetic_csv,
                                                                   completed):
    (completed / "run_identity.json").unlink()
    with pytest.raises(LegacyResultsError):
        cfg = load_config(ROOT / "configs" / "smoke_synthetic.yaml", _sets(tmp_path, synthetic_csv))
        runner = Runner(cfg)
        try:
            runner.plan(runner.prepare(), [42], ["moderate_non_iid"], ["local"], ablations=False)
        finally:
            runner.close()
    assert main(_args(tmp_path, synthetic_csv)) == 4


# source revision -------------------------------------------------------------------------
def test_other_source_revision_blocks_reuse_unless_allowed(tmp_path, synthetic_csv, completed):
    cfg = load_config(ROOT / "configs" / "smoke_synthetic.yaml", _sets(tmp_path, synthetic_csv))
    plan_args = ([42], ["moderate_non_iid"], ["local"])
    runner = Runner(cfg)
    runner.source = {**runner.source, "commit": "0" * 40}
    try:
        bundle = runner.prepare()
        with pytest.raises(SourceMismatchError, match="allow-source-change"):
            runner.plan(bundle, *plan_args, ablations=False)
    finally:
        runner.close()
    runner = Runner(cfg, allow_source_change=True)
    runner.source = {**runner.source, "commit": "0" * 40}
    try:
        [run] = runner.plan(runner.prepare(), *plan_args, ablations=False)
        assert runner._decide(run, runner._identity(run)) == "skip-other-source"
    finally:
        runner.close()


# locking ---------------------------------------------------------------------------------
def test_results_directory_lock_is_exclusive(tmp_path):
    a, b = ResultsLock(tmp_path), ResultsLock(tmp_path)
    assert a.acquire()
    with pytest.raises(ResultsLockedError, match="another process"):
        b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_concurrent_invocation_is_refused(tmp_path, synthetic_csv, completed):
    lock = ResultsLock(tmp_path / "results")
    lock.acquire()
    try:
        assert main(_args(tmp_path, synthetic_csv)) == 4
    finally:
        lock.release()


# dependencies ----------------------------------------------------------------------------
def test_ensemble_dependency_check_ignores_only_unread_sections(tmp_path, synthetic_csv,
                                                                completed):
    local = read_json(completed / "run_identity.json")["identity"]
    own = json.loads(json.dumps(local))
    own.update(method="ensemble", variant="ensemble_val_weighted")
    own["config"]["ensemble"]["weighting"] = "validation_ap"
    own["config"]["kd"]["lambda_kd"] = 0.0
    assert dependency_mismatch(local, own, ("ensemble", "kd")) == []
    own["config"]["models"]["xgboost"]["max_depth"] = 1
    assert dependency_mismatch(local, own, ("ensemble", "kd")) == ["config.models.xgboost.max_depth"]


def test_ensemble_refuses_local_run_from_another_configuration(tmp_path, synthetic_csv,
                                                               completed):
    ident = read_json(completed / "run_identity.json")
    ident["identity"]["config"]["models"]["xgboost"]["max_depth"] = 1
    write_json(completed / "run_identity.json", ident)
    assert main(_args(tmp_path, synthetic_csv, methods=("ensemble",))) == 1
    ens = completed.parent / "ensemble"
    st = read_json(ens / "status.json")
    assert st["status"] == "failed" and "different configuration" in st["error"]


def test_planned_run_dataclass_defaults():
    assert PlannedRun({}, "local", 1, "iid", None, None).variant == "primary"
