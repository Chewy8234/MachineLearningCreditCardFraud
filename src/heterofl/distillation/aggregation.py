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
from heterofl.utils.logging import get_logger

log = get_logger("distillation.aggregation")


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
    """Normalised weights for ``members``; ``None`` means equal weights (the primary rule).

    Explicit weights must be finite, non-negative and have a positive sum over the
    members being averaged. Anything else is an error: a degenerate weight vector is
    never silently replaced by equal weights.
    """
    if weights is None:
        w = np.ones(len(members))
        return w / w.sum()
    missing = [m for m in members if m not in weights]
    if missing:
        raise AggregationError(f"No aggregation weight for {missing} (weights: {weights})")
    try:
        w = np.array([float(weights[m]) for m in members], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"Non-numeric aggregation weights {weights}") from exc
    if np.any(~np.isfinite(w)):
        raise AggregationError(f"Non-finite aggregation weights {weights}")
    if np.any(w < 0):
        raise AggregationError(f"Negative aggregation weights {weights}")
    if not w.sum() > 0:
        raise AggregationError(f"Aggregation weights for {members} sum to zero ({weights}); "
                               "refusing to fall back to equal weights silently")
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


def validation_ap_weights(val_ap: dict[str, float], floor: float = 1e-3,
                          diagnostics: dict | None = None) -> dict[str, float]:
    """Weights proportional to each member's validation Average Precision.

    The APs are computed on the shared, labelled validation split (by the harness for
    the ensemble, by each client for KD teachers -- the server then receives only
    these scalar APs, not validation labels). A documented floor keeps every member
    in the committee: a member whose AP is undefined (NaN -- e.g. no validation
    positives -- or the client's ``-1`` "undefined" sentinel) or below ``floor``
    gets weight ``floor`` instead. Every such substitution is logged as a warning and,
    if ``diagnostics`` is given, recorded there as ``{"floored": {member: reported_ap}}``
    so it is visible in the run's outputs.
    """
    if not val_ap:
        raise AggregationError("No validation APs to weight by")
    if not floor > 0:
        raise AggregationError("validation_ap_weights needs a positive floor")
    w, floored = {}, {}
    for k, v in val_ap.items():
        v = float(v)
        if np.isfinite(v) and v >= floor:
            w[k] = v
        else:
            w[k] = floor
            floored[k] = v if np.isfinite(v) else None
    if floored:
        log.warning("validation-AP weighting: member(s) %s had undefined or < %g AP %s; "
                    "using the floor weight %g", sorted(floored), floor, floored, floor)
    if diagnostics is not None:
        diagnostics["floored"] = floored
        diagnostics["floor"] = floor
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}
