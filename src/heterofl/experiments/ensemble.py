"""Experiment 4 -- heterogeneous ensemble WITHOUT knowledge distillation.

The members are exactly the local-only models of Experiment 1 (same seeds,
same partitions); there is no training-time exchange. At evaluation time their
fraud probabilities are averaged with equal weights (primary) or with weights
proportional to each member's validation AP (ablation D).
"""

from __future__ import annotations

from pathlib import Path

from heterofl.distillation.aggregation import validation_ap_weights, weighted_average
from heterofl.evaluation.metrics import average_precision
from heterofl.experiments.common import RunContext, record_system
from heterofl.utils.io import load_npz, read_json


class MissingDependencyError(RuntimeError):
    """The local-only run this ensemble depends on is not completed."""


def committee_weights(p_val: dict, y_val, weighting: str) -> dict[str, float] | None:
    if weighting == "equal":
        return None
    return validation_ap_weights({k: average_precision(y_val, p) for k, p in p_val.items()})


def run(ctx: RunContext) -> None:
    local_dir: Path = ctx.results_root() / "local"
    status = local_dir / "status.json"
    if not status.is_file() or read_json(status).get("status") != "completed":
        raise MissingDependencyError(f"Ensemble requires a completed local run at {local_dir}")
    banks = [c["bank"] for c in ctx.clients]
    preds = {b: load_npz(local_dir / "predictions" / f"bank_{b}.npz") for b in banks}
    p_val = {b: d["p_val"] for b, d in preds.items()}
    p_test = {b: d["p_test"] for b, d in preds.items()}
    _, y_val = ctx.data.validation()
    weighting = ctx.cfg["ensemble"].get("weighting", "equal")
    weights = committee_weights(p_val, y_val, weighting)
    members = sorted(banks)
    ens_val = weighted_average(p_val, members, weights)
    ens_test = weighted_average(p_test, members, weights)
    runtime = read_json(local_dir / "runtime.json")
    infer = sum(runtime[f"bank_{b}"]["test_inference_seconds"] for b in banks)
    train = sum(runtime[f"bank_{b}"]["train_seconds"] for b in banks)
    ctx.runtime["ensemble"] = {"members_train_seconds": train, "test_inference_seconds": infer,
                               "weights": weights or {b: 1 / len(banks) for b in banks},
                               "members_source": str(local_dir)}
    record_system(ctx, system=f"ensemble_{weighting}",
                  label=("Heterogeneous ensemble (equal)" if weighting == "equal"
                         else "Heterogeneous ensemble (val-AP weighted)"),
                  model_type="committee", p_val=ens_val, p_test=ens_test,
                  extra={"weighting": weighting, "train_seconds": train,
                         "test_inference_seconds": infer, "communication_bytes": 0,
                         "weights": str(weights or "equal")})
