"""Server-side aggregation of per-example proxy predictions into teacher signals.

Primary (leave-one-out) teacher for client k over the ordered proxy set::

    T_k(x) = sum_{j != k} w_j p_j(x) / sum_{j != k} w_j

With equal weights and three clients this is T_A = (p_B + p_C) / 2, etc.
The full-committee ablation includes the student's own prediction (j = k).

This is per-example proxy distillation (FedMD-style), *not* class-average
FedDistill: every proxy example gets its own teacher probability. No model
parameters are aggregated.
"""

from __future__ import annotations

import numpy as np

from heterofl.models.base import check_probabilities


class AggregationError(ValueError):
    """Invalid or inconsistent prediction vectors."""


def validate_prediction_vectors(preds: dict[str, np.ndarray], expected_len: int) -> dict[str, np.ndarray]:
    if not preds:
        raise AggregationError("No prediction vectors received")
    out = {}
    for k, p in preds.items():
        try:
            out[k] = check_probabilities(p, expected_len, f"proxy predictions of {k}")
        except ValueError as exc:
            raise AggregationError(str(exc)) from exc
    return out


def _normalise_weights(members: list[str], weights: dict[str, float] | None) -> np.ndarray:
    if weights is None:
        w = np.ones(len(members))
    else:
        w = np.array([float(weights[m]) for m in members], dtype=np.float64)
        if np.any(~np.isfinite(w)) or np.any(w < 0):
            raise AggregationError(f"Invalid aggregation weights {weights}")
        if w.sum() <= 0:
            w = np.ones(len(members))
    return w / w.sum()


def weighted_average(preds: dict[str, np.ndarray], members: list[str],
                     weights: dict[str, float] | None = None) -> np.ndarray:
    if not members:
        raise AggregationError("Cannot average an empty committee")
    w = _normalise_weights(members, weights)
    stacked = np.stack([np.asarray(preds[m], dtype=np.float64) for m in members])
    return np.clip(w @ stacked, 0.0, 1.0)


def leave_one_out_teachers(preds: dict[str, np.ndarray],
                           weights: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    clients = sorted(preds)
    if len(clients) < 2:
        raise AggregationError("Leave-one-out teachers need at least two clients")
    return {k: weighted_average(preds, [j for j in clients if j != k], weights) for k in clients}


def full_committee_teachers(preds: dict[str, np.ndarray],
                            weights: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    clients = sorted(preds)
    t = weighted_average(preds, clients, weights)
    return {k: t.copy() for k in clients}


def compute_teachers(preds: dict[str, np.ndarray], mode: str, expected_len: int,
                     weights: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    preds = validate_prediction_vectors(preds, expected_len)
    if mode == "leave_one_out":
        return leave_one_out_teachers(preds, weights)
    if mode == "full_committee":
        return full_committee_teachers(preds, weights)
    raise AggregationError(f"Unknown teacher mode {mode!r}")


def validation_ap_weights(val_ap: dict[str, float], floor: float = 1e-3) -> dict[str, float]:
    """Weights proportional to each member's validation Average Precision.

    Derived from validation labels only. A small floor keeps every member in
    the committee; NaN APs (e.g. no positives) fall back to the floor.
    """
    w = {k: max(float(v), floor) if np.isfinite(v) else floor for k, v in val_ap.items()}
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}
