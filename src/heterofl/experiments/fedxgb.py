"""Homogeneous federated XGBoost: federated tree bagging (Flower ``FedXgbBagging`` scheme).

A traditional *parameter-free* homogeneous FL baseline for the XGBoost family (the
counterpart of FedAvg for the MLP): every round each bank boosts
``fedxgb.trees_per_round`` new trees from the current global model on its private
rows, and the server appends all new trees (``heterofl.federation.xgb_bagging``).
It uses the shared ``models.xgboost`` hyperparameters and ``federation.num_rounds``
(5 in the paper profile; 5 rounds x 3 banks x 20 trees = 300 trees, the size of one
local or centralized XGBoost model).

The protocol is executed in-process (no Flower transport): each bank's update reads
only that bank's partition and the server step only concatenates trees. Round
selection uses validation only; the test split is scored once, after selection.
"""

from __future__ import annotations

from heterofl.evaluation.metrics import average_precision
from heterofl.experiments.common import RunContext, record_system, timed_predict
from heterofl.experiments.federated_common import select_round
from heterofl.federation.xgb_bagging import FedXgbBagging
from heterofl.utils.io import write_csv


def run(ctx: RunContext) -> None:
    banks = {c["bank"]: ctx.data.bank(c["bank"]) for c in ctx.clients}
    num_rounds = int(ctx.cfg["federation"]["num_rounds"])
    k = int(ctx.cfg["fedxgb"]["trees_per_round"])
    Xv, y_val = ctx.data.validation()
    fx = FedXgbBagging(banks, ctx.models_cfg["xgboost"], ctx.seed, ctx.n_jobs, num_rounds, k)

    rows, val_ap, fit_total, bytes_up, bytes_down = [], {}, 0.0, 0, 0
    for r, p, info in fx.rounds({"val": Xv}):
        val_ap[r] = average_precision(y_val, p["val"])
        fit_total += sum(info["fit_seconds"].values())
        bytes_up += info["bytes_up"]
        bytes_down += info["bytes_down"]
        rows.append({"seed": ctx.seed, "scenario": ctx.scenario, "method": ctx.method,
                     "variant": ctx.variant, "round": r, "system": "fedxgb_global",
                     "val_average_precision": val_ap[r], "n_trees": info["n_trees"],
                     "bytes_up": info["bytes_up"], "bytes_down": info["bytes_down"],
                     "round_seconds": info["round_seconds"],
                     **{f"bank_{b}_fit_seconds": s for b, s in info["fit_seconds"].items()}})
    write_csv(ctx.out_dir / "rounds.csv", rows)

    policy = ctx.cfg["evaluation"].get("model_selection", "final_round")
    chosen = select_round(val_ap, policy, sorted(val_ap))
    fx.save(ctx.out_dir / "models" / "fedxgb_global", upto_round=chosen)
    p_val = fx.predict_proba(Xv, upto_round=chosen)
    Xt, _ = ctx.data.test()
    p_test, infer_s = timed_predict(_Global(fx, chosen), Xt)
    n = {b: len(y) for b, (_, y) in banks.items()}
    ctx.runtime["fedxgb_bagging"] = {
        "client_fit_seconds_total": fit_total, "test_inference_seconds": infer_s,
        "bytes_up": bytes_up, "bytes_down": bytes_down,
        "bytes_note": "serialised JSON size of the trees the protocol sends (estimated, "
                      "in-process run); b0 needs two count scalars per bank in addition",
        "n_trees_global": chosen * len(banks) * k, "trees_per_round_per_bank": k,
        "b0": fx.b0, "n_train": n, "transport": "in-process (no Flower messages)"}
    record_system(ctx, system="fedxgb_global", label="Federated XGBoost (bagging, homogeneous)",
                  model_type="xgboost", p_val=p_val, p_test=p_test,
                  extra={"selected_round": chosen, "model_selection": policy,
                         "train_seconds": fit_total, "test_inference_seconds": infer_s,
                         "communication_bytes": bytes_up + bytes_down})


class _Global:
    """``predict_proba`` adapter for the global model of one round (for timing)."""

    def __init__(self, fx: FedXgbBagging, upto: int):
        self.fx, self.upto = fx, upto

    def predict_proba(self, X):
        return self.fx.predict_proba(X, upto_round=self.upto)
