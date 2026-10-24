"""Dataset audit: schema, value validity, label validity, duplicates, class balance.

The audit *reports*; it never removes or relabels records. Fatal problems
(missing columns, missing/non-finite values, non-binary labels) raise
``DataValidationError`` so that experiments cannot run on corrupted data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from heterofl.data.loader import (
    EXPECTED_FRAUD,
    EXPECTED_ROWS,
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)
from heterofl.utils.logging import get_logger

log = get_logger("data.validation")


class DataValidationError(ValueError):
    """The dataset failed a fatal validation check."""


def audit_dataset(df: pd.DataFrame, expect_ulb: bool = True) -> dict:
    """Compute an audit report. Raises DataValidationError on fatal issues."""
    report: dict = {"n_rows": int(len(df)), "n_columns": int(df.shape[1])}
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    report["missing_columns"] = missing_cols
    report["extra_columns"] = extra_cols
    if missing_cols:
        errors.append(f"missing required columns {missing_cols}")
        raise DataValidationError("; ".join(errors))

    na_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    report["missing_values_per_column"] = {k: int(v) for k, v in na_counts.items() if v}
    report["missing_values_total"] = int(na_counts.sum())
    if report["missing_values_total"]:
        errors.append(f"{report['missing_values_total']} missing values")

    feats = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    nonfinite = int((~np.isfinite(feats)).sum())
    report["non_finite_feature_values"] = nonfinite
    if nonfinite:
        errors.append(f"{nonfinite} non-finite feature values")
    report["negative_amounts"] = int((df["Amount"] < 0).sum())
    if report["negative_amounts"]:
        errors.append(f"{report['negative_amounts']} negative Amount values")
    report["negative_times"] = int((df["Time"] < 0).sum())
    if report["negative_times"]:
        errors.append(f"{report['negative_times']} negative Time values")

    labels = df[TARGET_COLUMN]
    unique_labels = sorted(pd.unique(labels.dropna()).tolist())
    report["label_values"] = unique_labels
    if set(unique_labels) - {0, 1}:
        errors.append(f"non-binary labels {unique_labels}")
    if not np.all(np.equal(np.mod(labels.dropna(), 1), 0)):
        errors.append("non-integer labels")

    n_fraud = int((labels == 1).sum())
    n_legit = int((labels == 0).sum())
    report["class_distribution"] = {
        "legitimate": n_legit,
        "fraud": n_fraud,
        "fraud_rate": n_fraud / max(len(df), 1),
    }
    if n_fraud == 0 or n_legit == 0:
        errors.append("both classes must be present")

    dup_full = df.duplicated(keep="first")
    dup_feat = df.duplicated(subset=list(FEATURE_COLUMNS), keep=False)
    feat_groups = df[dup_feat].groupby(list(FEATURE_COLUMNS))[TARGET_COLUMN].nunique()
    report["duplicates"] = {
        "exact_duplicate_rows": int(dup_full.sum()),
        "exact_duplicate_fraud_rows": int((dup_full & (labels == 1)).sum()),
        "feature_duplicate_groups_with_conflicting_labels": int((feat_groups > 1).sum()),
        "policy": "retained (not removed); cross-split duplicates are reported in each "
                  "partition manifest",
    }
    if report["duplicates"]["exact_duplicate_rows"]:
        warnings.append(f"{report['duplicates']['exact_duplicate_rows']} exact duplicate "
                        "rows retained")

    report["feature_summary"] = {
        "Amount": _describe(df["Amount"]),
        "Time": _describe(df["Time"]),
    }

    if expect_ulb:
        if len(df) != EXPECTED_ROWS:
            warnings.append(f"row count {len(df)} != expected ULB {EXPECTED_ROWS}")
        if n_fraud != EXPECTED_FRAUD:
            warnings.append(f"fraud count {n_fraud} != expected ULB {EXPECTED_FRAUD}")

    report["errors"] = errors
    report["warnings"] = warnings
    report["valid"] = not errors
    for w in warnings:
        log.warning("Dataset audit: %s", w)
    log.info("Dataset audit: %d rows, %d fraud (%.4f%%), %d exact duplicates",
             len(df), n_fraud, 100 * report["class_distribution"]["fraud_rate"],
             report["duplicates"]["exact_duplicate_rows"])
    if errors:
        raise DataValidationError("Dataset failed validation: " + "; ".join(errors))
    return report


def _describe(s: pd.Series) -> dict:
    return {
        "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
        "std": float(s.std()), "median": float(s.median()),
    }
