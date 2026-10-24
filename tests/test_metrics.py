"""Metrics, threshold selection and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from heterofl.evaluation.metrics import (
    average_precision,
    evaluate_system,
    pr_auc_trapezoid,
    recall_at_fpr,
    roc_auc,
    select_threshold,
    threshold_metrics,
)
from heterofl.evaluation.statistics import (
    bootstrap_ap_ci,
    fast_average_precision,
    paired_bootstrap_ap_diff,
    stratified_bootstrap_weights,
    summarize_across_seeds,
)


@pytest.fixture()
def yp():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.02).astype(int)
    p = np.clip(0.3 * y + rng.random(3000) * 0.7, 0, 1)
    return y, p


def test_ap_and_roc_match_sklearn(yp):
    y, p = yp
    assert average_precision(y, p) == pytest.approx(average_precision_score(y, p))
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_trapezoidal_pr_auc_is_reported_separately(yp):
    y, p = yp
    assert pr_auc_trapezoid(y, p) != pytest.approx(average_precision(y, p), abs=1e-9)


def test_fast_ap_matches_sklearn_including_ties(yp):
    y, p = yp
    assert fast_average_precision(y, p) == pytest.approx(average_precision_score(y, p))
    pt = np.round(p, 1)  # heavy ties
    assert fast_average_precision(y, pt) == pytest.approx(average_precision_score(y, pt))


def test_metrics_with_single_class_are_nan():
    assert np.isnan(average_precision(np.zeros(5), np.linspace(0, 1, 5)))


def test_threshold_metrics_confusion_matrix():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(0.5)


def test_recall_at_fixed_fpr():
    y = np.array([0] * 1000 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 1000), np.linspace(0.2, 1, 10)])
    r = recall_at_fpr(y, p, 0.001)
    assert 0 < r <= 1
    assert recall_at_fpr(y, p, 1.0) == 1.0


def test_threshold_selected_on_validation_only(yp):
    y, p = yp
    yv, pv, yt, pt = y[:1500], p[:1500], y[1500:], p[1500:]
    ev = evaluate_system(yv, pv, yt, pt)
    assert ev["validation_threshold"] == select_threshold(yv, pv)
    # permuting/flipping test labels must not move the threshold
    ev2 = evaluate_system(yv, pv, 1 - yt, pt)
    assert ev2["validation_threshold"] == ev["validation_threshold"]
    assert ev["test"]["at_val_threshold"]["threshold"] == ev["validation_threshold"]
    assert ev["test"]["at_0.5"]["threshold"] == 0.5


def test_select_threshold_maximises_validation_f1(yp):
    y, p = yp
    thr = select_threshold(y, p)
    best = threshold_metrics(y, p, thr)["f1"]
    for t in np.linspace(0.05, 0.95, 19):
        assert threshold_metrics(y, p, t)["f1"] <= best + 1e-12


def test_stratified_bootstrap_keeps_class_counts(yp):
    y, _ = yp
    for w in stratified_bootstrap_weights(y, 5, 0):
        assert w[y == 1].sum() == y.sum()
        assert w[y == 0].sum() == (y == 0).sum()


def test_bootstrap_ci_and_paired_difference(yp):
    y, p = yp
    ci = bootstrap_ap_ci(y, p, 200, seed=1)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    same = paired_bootstrap_ap_diff(y, p, p, 100, seed=1)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    noisy = np.clip(p + np.random.default_rng(1).normal(0, 0.3, len(p)), 0, 1)
    d = paired_bootstrap_ap_diff(y, p, noisy, 200, seed=1)
    assert d["diff"] > 0 and d["ci_low"] <= d["diff"] <= d["ci_high"]


def test_summarize_across_seeds():
    df = pd.DataFrame({"system": ["a"] * 3 + ["b"] * 3, "seed": [1, 2, 3] * 2,
                       "ap": [0.5, 0.6, 0.7, 0.1, 0.1, 0.1]})
    s = summarize_across_seeds(df, ["system"], ["ap"]).set_index("system")
    assert s.loc["a", "ap_mean"] == pytest.approx(0.6)
    assert s.loc["a", "ap_std"] == pytest.approx(0.1)
    assert s.loc["a", "ap_ci_low"] < 0.6 < s.loc["a", "ap_ci_high"]
    assert s.loc["b", "ap_n"] == 3


# --------------------------------------------------------------------------- statistical reporting
def _pairs_df():
    return pd.DataFrame({
        "scenario": ["s"] * 6, "seed": [1, 2, 3] * 2, "system": ["a"] * 3 + ["b"] * 3,
        "ap": [0.8, 0.7, 0.9, 0.6, 0.7, 0.5],
        "partition_sha256": ["p1", "p2", "p3"] * 2})


def test_paired_differences_are_a_minus_b_matched_by_seed():
    from heterofl.evaluation.statistics import paired_seed_differences

    df = _pairs_df().sample(frac=1, random_state=0)            # row order must not matter
    r = paired_seed_differences(df, "a", "b", "ap", ["scenario"], system_col="system").iloc[0]
    assert r["n_pairs"] == 3 and r["n_a_better"] == 2
    assert r["mean_diff"] == pytest.approx(np.mean([0.2, 0.0, 0.4]))
    assert r["min_diff"] == pytest.approx(0.0) and r["max_diff"] == pytest.approx(0.4)


def test_paired_differences_refuse_ambiguous_or_mismatched_pairs():
    from heterofl.evaluation.statistics import PairingError, paired_seed_differences

    df = _pairs_df()
    dup = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(PairingError, match="several rows"):
        paired_seed_differences(dup, "a", "b", "ap", ["scenario"], system_col="system")
    other = df.copy()
    other.loc[3, "partition_sha256"] = "elsewhere"           # b, seed 1, other partition
    with pytest.raises(PairingError, match="different partitions"):
        paired_seed_differences(other, "a", "b", "ap", ["scenario"], system_col="system")


def test_missing_runs_are_dropped_not_counted_as_zero():
    from heterofl.evaluation.statistics import paired_seed_differences

    df = _pairs_df()
    df = df[~((df["system"] == "b") & (df["seed"] == 3))]    # b failed for seed 3
    df.loc[df.index[0], "ap"] = np.nan                         # a, seed 1: undefined metric
    r = paired_seed_differences(df, "a", "b", "ap", ["scenario"], system_col="system").iloc[0]
    assert r["n_pairs"] == 1 and r["mean_diff"] == pytest.approx(0.0)
    s = summarize_across_seeds(_pairs_df().assign(ap=[0.8, np.nan, 0.9, 0.6, 0.7, 0.5]),
                               ["system"], ["ap"]).set_index("system")
    assert s.loc["a", "ap_n"] == 2 and s.loc["a", "ap_mean"] == pytest.approx(0.85)


def test_bootstrap_is_reproducible_and_seed_dependent(yp):
    y, p = yp
    assert bootstrap_ap_ci(y, p, 100, seed=5) == bootstrap_ap_ci(y, p, 100, seed=5)
    assert bootstrap_ap_ci(y, p, 100, seed=5)["ci_low"] != bootstrap_ap_ci(y, p, 100, seed=6)["ci_low"]
    # paired: both systems see the same resamples, so a - b == -(b - a) exactly
    q = np.clip(p + 0.05, 0, 1)[::-1]
    ab, ba = paired_bootstrap_ap_diff(y, p, q, 100, seed=2), paired_bootstrap_ap_diff(y, q, p, 100, seed=2)
    assert ab["diff"] == pytest.approx(-ba["diff"])
    assert ab["ci_low"] == pytest.approx(-ba["ci_high"])


def test_ap_is_not_trapezoidal_pr_auc(yp):
    y, p = yp
    assert average_precision(y, p) != pytest.approx(pr_auc_trapezoid(y, p), abs=1e-6)


def test_report_provenance_flags_legacy_and_dirty_runs(tmp_path):
    from heterofl.evaluation.report import provenance_summary

    row = {"status": "completed", "legacy_no_identity": False, "source_commit": "abc",
           "source_dirty": False, "source_diff_sha256": None, "source_verified": True}
    clean = pd.DataFrame([row, row])
    assert provenance_summary(tmp_path, clean)["verified"] is False          # no experiment.json
    (tmp_path / "experiment.json").write_text('{"experiment_hash": "e", "config_hash": "c"}')
    assert provenance_summary(tmp_path, clean)["verified"] is True
    dirty = pd.DataFrame([row, {**row, "source_dirty": True, "source_diff_sha256": "d",
                                "source_verified": False}])
    prov = provenance_summary(tmp_path, dirty)
    assert not prov["verified"] and any("clean git commit" in m for m in prov["problems"])
    assert any("2 different source revisions" in m for m in prov["problems"])
    legacy = pd.DataFrame([{**row, "legacy_no_identity": True, "source_verified": False}])
    assert any("without a recorded" in m for m in provenance_summary(tmp_path, legacy)["problems"])
