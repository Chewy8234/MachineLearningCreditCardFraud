"""End-to-end: the smoke profile (on synthetic data) runs every condition and reports.

Synthetic data -> outputs are flagged non-publishable; this tests plumbing only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from heterofl.experiments.runner import main
from heterofl.utils.io import read_json

from .conftest import ROOT, TINY_MODELS


def _args(tmp_path, csv, extra=()):
    sets = [f"data.path={csv}", f"data.processed_dir={tmp_path / 'processed'}",
            f"data.partitions_dir={tmp_path / 'partitions'}", f"results_dir={tmp_path / 'results'}",
            "partitioning.min_fraud_per_client=5", "partitioning.min_legit_per_client=100",
            "federation.client_num_cpus=1", "federation.job_timeout=900",
            "evaluation.bootstrap.n_resamples=50", "fedavg.local_epochs=1", *TINY_MODELS, *extra]
    args = ["--config", str(ROOT / "configs" / "smoke_synthetic.yaml")]
    for s in sets:
        args += ["--set", s]
    return args


@pytest.mark.slow
def test_smoke_profile_runs_all_conditions_and_reports(tmp_path, synthetic_csv):
    assert main(_args(tmp_path, synthetic_csv)) == 0
    res = tmp_path / "results"
    sdir = res / "seed_42" / "moderate_non_iid"
    for method in ("local", "centralized", "ensemble", "fedavg", "heterogeneous_kd"):
        d = sdir / method
        assert read_json(d / "status.json")["status"] == "completed", method
        for f in ("config.yaml", "manifest.json", "metrics.json", "metrics.csv", "runtime.json",
                  "run.log"):
            assert (d / f).is_file(), f"{method}/{f}"
        m = read_json(d / "metrics.json")
        assert m["publishable"] is False
        man = read_json(d / "manifest.json")
        assert man["synthetic_data"] is True
        # provenance: identity + source + dataset recorded with every run
        ident = read_json(d / "run_identity.json")
        assert ident["run_hash"] == read_json(d / "status.json")["run_hash"]
        assert man["provenance"]["run_hash"] == ident["run_hash"]
        assert man["provenance"]["publication_reproducible"] is False
        assert ident["identity"]["method"] == method and ident["identity"]["seed"] == 42
        # every condition used the very same partitions
        assert man["partition_manifest"]["sets"] == read_json(
            sdir / "local" / "manifest.json")["partition_manifest"]["sets"]
    for method in ("fedavg", "heterogeneous_kd"):
        assert (sdir / method / "rounds.csv").is_file()
    for ab in ("kd_lambda_0.0", "kd_full_committee", "kd_teacher_val_weighted", "ensemble_val_weighted"):
        assert read_json(sdir / "ablations" / ab / "status.json")["status"] == "completed", ab

    exp = read_json(res / "experiment.json")
    hashes = {read_json(sdir / m / "run_identity.json")["config_hash"]
              for m in ("local", "centralized", "ensemble", "fedavg", "heterogeneous_kd")}
    assert hashes == {exp["config_hash"]}   # primary runs share the experiment's config
    lam0 = read_json(sdir / "ablations" / "kd_lambda_0.0" / "run_identity.json")
    assert lam0["config_hash"] != exp["config_hash"]
    assert lam0["identity"]["config"]["kd"]["lambda_kd"] == 0.0
    # RF distillation records its effective bootstrap composition every round
    meta = read_json(sdir / "heterogeneous_kd" / "federation" / "clients" / "bank_B" /
                     "round_01" / "meta.json")
    assert meta["fit_info"]["rf_expected_proxy_draws"] > 0
    assert meta["fit_info"]["rf_bootstrap_draws_per_tree"] > meta["fit_info"]["n_local"]

    metrics = pd.concat(pd.read_csv(p) for p in sdir.glob("*/metrics.csv"))
    assert metrics["partition_sha256"].nunique() == 1
    assert {"bank_A", "bank_B", "bank_C", "centralized_xgboost", "ensemble_equal", "fedavg_global",
            "kd_bank_A", "kd_committee"} <= set(metrics["system"])
    assert metrics["test_average_precision"].between(0, 1).all()
    diag = read_json(sdir / "heterogeneous_kd" / "diagnostics.json")
    for b in "ABC":  # KD round 0 reproduces the local-only models
        assert diag["round0_vs_local_only"][b]["max_abs_diff_val_probs_vs_local"] < 1e-6

    # report: comparison table, figures, summary
    assert (res / "tables" / "table3_main.tex").is_file()
    assert (res / "tables" / "table3_main.csv").is_file()
    assert (res / "figures" / "fig2_ap_comparison.png").is_file()
    assert (res / "figures" / "fig2_ap_comparison.pdf").is_file()
    assert "NOT FOR PUBLICATION" in (res / "reports" / "summary.md").read_text(encoding="utf-8")

    # resume: a second invocation skips everything that already completed
    assert main(_args(tmp_path, synthetic_csv) + ["--no-report"]) == 0
    status = pd.read_csv(max(res.glob("outcomes_*.csv"), key=lambda f: f.stat().st_mtime_ns))
    assert (status["status"] == "skipped-completed").all()


@pytest.mark.slow
def test_failed_run_is_recorded_and_safely_rerun(tmp_path, synthetic_csv):
    base = _args(tmp_path, synthetic_csv, ["ablations.enabled=false"])
    rc = main(base + ["--methods", "local", "ensemble", "heterogeneous_kd", "--no-report",
                      "--set", "debug.fail.partition_id=2", "--set", "debug.fail.round=0"])
    assert rc == 1
    kd = tmp_path / "results" / "seed_42" / "moderate_non_iid" / "heterogeneous_kd"
    st = read_json(kd / "status.json")
    assert st["status"] == "failed" and "Injected failure" in (kd / "federation.log").read_text(encoding="utf-8")
    assert not (kd / "metrics.csv").exists()  # no invented metrics
    # rerun without the fault: failed run archived (diagnostics kept), new run completes
    assert main(base + ["--methods", "local", "ensemble", "heterogeneous_kd", "--no-report"]) == 0
    assert read_json(kd / "status.json")["status"] == "completed"
    archived = list((kd.parent / "_archive").glob("heterogeneous_kd-stale-failed-*"))
    assert archived and (archived[0] / "federation.log").is_file()
