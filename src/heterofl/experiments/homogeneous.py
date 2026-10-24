"""Experiment 3 -- homogeneous FedAvg (genuine Flower federation, shared MLP).

All three banks train the same MLP architecture. Each round the server sends
the global parameters, every bank runs ``fedavg.local_epochs`` epochs on its
private data and returns parameters, and the server aggregates with
example-count-weighted FedAvg. Total local epochs = num_rounds * local_epochs.
"""

from __future__ import annotations

import numpy as np

from heterofl.evaluation.metrics import average_precision
from heterofl.experiments.common import RunContext, record_system, timed_predict
from heterofl.experiments.federated_common import build_job, job_dir, launch, select_round
from heterofl.models.mlp import MLPModel
from heterofl.utils.io import read_json, write_csv


def run(ctx: RunContext) -> None:
    job = build_job(ctx, "fedavg")
    launch(ctx, job)
    jd = job_dir(ctx)
    summary = read_json(jd / "server" / "rounds.json")

    _, y_val = ctx.data.validation()
    rows, val_ap = [], {}
    for r in summary["rounds"]:
        rnd = int(r["round"])
        gv = r.get("global_validation", {})
        p_val = np.load(jd / "server" / f"global_round_{rnd:02d}" / "val_probs.npy")
        val_ap[rnd] = average_precision(y_val, p_val)
        row = {"seed": ctx.seed, "scenario": ctx.scenario, "method": ctx.method,
               "variant": ctx.variant, "round": rnd, "system": "fedavg_global",
               "val_average_precision": val_ap[rnd], "val_roc_auc": gv.get("val-roc-auc"),
               "val_precision@0.5": gv.get("val-precision@0.5"),
               "val_recall@0.5": gv.get("val-recall@0.5"), "val_f1@0.5": gv.get("val-f1@0.5"),
               "bytes_up": r.get("bytes_up", 0), "bytes_down": r.get("bytes_down", 0),
               "round_seconds": r.get("round_seconds")}
        for bank, m in (r.get("client_metrics") or {}).items():
            row[f"bank_{bank}_train_loss"] = m.get("train-loss")
            row[f"bank_{bank}_local_val_ap"] = m.get("local-val-ap")
            row[f"bank_{bank}_fit_seconds"] = m.get("fit-seconds")
            row[f"bank_{bank}_weight"] = (r.get("weights") or {}).get(bank)
        rows.append(row)
    write_csv(ctx.out_dir / "rounds.csv", rows)

    policy = ctx.cfg["evaluation"].get("model_selection", "final_round")
    candidates = [r for r in val_ap if r >= 1]
    chosen = select_round(val_ap, policy, candidates)
    model = MLPModel.load(jd / "server" / f"global_round_{chosen:02d}" / "model.pt",
                          n_jobs=ctx.n_jobs)
    Xv, _ = ctx.data.validation()
    Xt, _ = ctx.data.test()
    p_val, _ = timed_predict(model, Xv)
    p_test, infer_s = timed_predict(model, Xt)
    client_fit = sum(m.get("fit-seconds", 0.0) for r in summary["rounds"]
                     for m in (r.get("client_metrics") or {}).values())
    comm = summary["total_bytes_up"] + summary["total_bytes_down"]
    ctx.runtime["fedavg"] = {"federation_wall_seconds": summary["wall_seconds"],
                             "client_fit_seconds_total": client_fit,
                             "test_inference_seconds": infer_s,
                             "bytes_up": summary["total_bytes_up"],
                             "bytes_down": summary["total_bytes_down"],
                             "num_parameters": summary["num_parameters"]}
    record_system(ctx, system="fedavg_global", label="FedAvg MLP (homogeneous)", model_type="mlp",
                  p_val=p_val, p_test=p_test,
                  extra={"selected_round": chosen, "model_selection": policy,
                         "train_seconds": client_fit, "test_inference_seconds": infer_s,
                         "federation_wall_seconds": summary["wall_seconds"],
                         "communication_bytes": comm})
