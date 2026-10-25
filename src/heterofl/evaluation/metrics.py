"""Evaluation metrics for rare-event binary classification.

Naming is deliberate:

* ``average_precision`` -- sklearn ``average_precision_score`` (step-wise sum
  ``sum_n (R_n - R_{n-1}) P_n``). This is the PRIMARY metric, reported as "AP".
* ``pr_auc_trapezoid``  -- trapezoidal area under the PR curve
  (``auc(recall, precision)``); reported separately and never called "AP".

Threshold policy: threshold-dependent metrics are reported (a) at 0.5 and
(b) at a threshold chosen on the VALIDATION split with a rule fixed before any
test evaluation (default: maximise F1). The test labels are never used to
choose a threshold, a round, or a hyperparameter.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

THRESHOLD_RULES = ("max_f1",)


def _check(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(f"y and p must be 1-D and equal length, got {y.shape} vs {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("Predicted probabilities contain non-finite values")
    return y, p


def has_both_classes(y: np.ndarray) -> bool:
    return 0 < int(np.sum(y)) < len(y)


def average_precision(y, p) -> float:
    y, p = _check(y, p)
    return float(average_precision_score(y, p)) if has_both_classes(y) else float("nan")


def pr_auc_trapezoid(y, p) -> float:
    y, p = _check(y, p)
    if not has_both_classes(y):
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, p)
    return float(auc(recall, precision))


def roc_auc(y, p) -> float:
    y, p = _check(y, p)
    return float(roc_auc_score(y, p)) if has_both_classes(y) else float("nan")


def recall_at_fpr(y, p, target_fpr: float) -> float:
    """Largest TPR achievable with FPR <= target_fpr (no interpolation)."""
    y, p = _check(y, p)
    if not has_both_classes(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def select_threshold(y_val, p_val, rule: str = "max_f1") -> float:
    """Choose a decision threshold on VALIDATION data. Positive iff p >= threshold."""
    y, p = _check(y_val, p_val)
    if rule != "max_f1":
        raise ValueError(f"Unknown threshold rule {rule!r}; valid: {THRESHOLD_RULES}")
    if not has_both_classes(y):
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]  # last point has no threshold
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))  # first maximum = lowest threshold among ties
    return float(thresholds[best])


def threshold_metrics(y, p, threshold: float) -> dict:
    y, p = _check(y, p)
    pred = (p >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "balanced_accuracy": (recall + tnr) / 2, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": (tp + tn) / len(y) if len(y) else float("nan")}


def threshold_free_metrics(y, p, fpr_targets=(0.001, 0.01)) -> dict:
    out = {"average_precision": average_precision(y, p),
           "pr_auc_trapezoid": pr_auc_trapezoid(y, p),
           "roc_auc": roc_auc(y, p),
           "n": int(len(y)), "n_positive": int(np.sum(y))}
    for t in fpr_targets:
        out[f"recall_at_fpr_{t:g}"] = recall_at_fpr(y, p, t)
    return out


def evaluate_system(y_val, p_val, y_test, p_test, rule: str = "max_f1",
                    fpr_targets=(0.001, 0.01)) -> dict:
    """Full evaluation of one system. The threshold is chosen on validation only."""
    thr = select_threshold(y_val, p_val, rule)
    return {
        "threshold_rule": f"{rule} on validation",
        "validation_threshold": thr,
        "validation": {**threshold_free_metrics(y_val, p_val, fpr_targets),
                       "at_0.5": threshold_metrics(y_val, p_val, 0.5),
                       "at_val_threshold": threshold_metrics(y_val, p_val, thr)},
        "test": {**threshold_free_metrics(y_test, p_test, fpr_targets),
                 "at_0.5": threshold_metrics(y_test, p_test, 0.5),
                 "at_val_threshold": threshold_metrics(y_test, p_test, thr)},
    }


def flatten_evaluation(ev: dict) -> dict:
    """Flatten ``evaluate_system`` output into one CSV-friendly row."""
    row = {"validation_threshold": ev["validation_threshold"]}
    for split in ("validation", "test"):
        for k, v in ev[split].items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if kk != "confusion_matrix":
                        row[f"{split}_{k}_{kk}"] = vv
            else:
                row[f"{split}_{k}"] = v
    return row
