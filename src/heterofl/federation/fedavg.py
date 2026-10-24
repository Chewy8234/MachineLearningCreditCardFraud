"""Homogeneous baseline: Flower FedAvg over a shared PyTorch MLP.

Flower's ``FedAvg`` computes ``sum_k (n_k / sum_j n_j) * theta_k`` with ``n_k``
taken from each reply's ``num-examples`` metric. ``StrictFedAvg`` keeps that
aggregation untouched but (a) refuses to aggregate if any of the expected
clients is missing or replied with an error -- Flower's default would silently
average the remaining clients -- and (b) records communication volume and
per-client training metrics for every round.

The global model is evaluated after each round on the validation split
(server-side ``evaluate_fn``), and saved, so the harness can apply the model
selection policy afterwards. No test data is available here.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from heterofl.evaluation.metrics import average_precision, roc_auc, threshold_metrics
from heterofl.federation import messages as M
from heterofl.federation.job import FederatedJob, FederationError
from heterofl.models.mlp import MLPModel
from heterofl.models.model_factory import model_seed
from heterofl.utils.io import load_npz, write_json
from heterofl.utils.logging import get_logger

log = get_logger("federation.fedavg")


class StrictFedAvg(FedAvg):
    def __init__(self, expected_clients: int, node_to_bank: dict[int, str], **kwargs):
        super().__init__(fraction_train=1.0, fraction_evaluate=0.0,
                         min_train_nodes=expected_clients, min_evaluate_nodes=0,
                         min_available_nodes=expected_clients, **kwargs)
        self.expected_clients = expected_clients
        self.node_to_bank = node_to_bank
        self.rounds: dict[int, dict] = {}
        self._t_round: float = 0.0

    def configure_train(self, server_round, arrays, config, grid) -> Iterable[Message]:
        msgs = list(super().configure_train(server_round, arrays, config, grid))
        if len(msgs) != self.expected_clients:
            raise FederationError(f"FedAvg round {server_round}: sampled {len(msgs)} clients, "
                                  f"expected {self.expected_clients}")
        self._t_round = time.perf_counter()
        self.rounds[server_round] = {
            "round": server_round,
            "bytes_down": int(sum(M.payload_bytes(m.content) for m in msgs)),
            "messages_down": len(msgs),
        }
        return msgs

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        errors = [r for r in replies if r.has_error()]
        if errors:
            reasons = "; ".join(f"node {r.metadata.src_node_id}: {r.error.reason}" for r in errors)
            raise FederationError(f"FedAvg round {server_round}: client error(s): {reasons}")
        if len(replies) != self.expected_clients:
            raise FederationError(f"FedAvg round {server_round}: received {len(replies)} "
                                  f"replies, expected {self.expected_clients}")
        clients = {}
        for r in replies:
            bank = self.node_to_bank.get(r.metadata.src_node_id)
            if bank is None:
                raise FederationError(f"Reply from unknown node {r.metadata.src_node_id}")
            m = dict(r.content[M.METRICS])
            clients[bank] = m
        arrays, metrics = super().aggregate_train(server_round, replies)
        rec = self.rounds[server_round]
        rec.update({
            "bytes_up": int(sum(M.payload_bytes(r.content) for r in replies)),
            "messages_up": len(replies),
            "client_metrics": clients,
            "weights": {b: m["num-examples"] / sum(c["num-examples"] for c in clients.values())
                        for b, m in clients.items()},
            "round_seconds": time.perf_counter() - self._t_round,
        })
        return arrays, metrics


def make_evaluate_fn(job: FederatedJob, server_dir: Path, rounds: dict[int, dict]):
    """Server-side validation of the global model after each round (incl. round 0)."""
    view = load_npz(Path(job.view_dir) / "validation.npz")
    Xv, yv = view["X"], view["y"]

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        model = MLPModel(job.models["mlp"], model_seed(job.seed, "fedavg_global", "mlp"),
                         n_features=Xv.shape[1], n_jobs=job.num_cpus_per_client)
        model.set_state_dict(arrays.to_torch_state_dict())
        p = model.predict_proba(Xv)
        rd = server_dir / f"global_round_{server_round:02d}"
        rd.mkdir(parents=True, exist_ok=True)
        model.save(rd / "model")
        np.save(rd / "val_probs.npy", p)
        tm = threshold_metrics(yv, p, 0.5)
        res = {"val-ap": average_precision(yv, p), "val-roc-auc": roc_auc(yv, p),
               "val-precision@0.5": tm["precision"], "val-recall@0.5": tm["recall"],
               "val-f1@0.5": tm["f1"]}
        rounds.setdefault(server_round, {"round": server_round})["global_validation"] = res
        log.info("FedAvg global model round %d: val AP=%.4f ROC-AUC=%.4f", server_round,
                 res["val-ap"], res["val-roc-auc"])
        return MetricRecord({k: float(v) if np.isfinite(v) else -1.0 for k, v in res.items()})

    return evaluate


def run_fedavg(grid: Grid, job: FederatedJob, node_to_bank: dict[int, str]) -> dict:
    server_dir = Path(job.job_dir) / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    init = MLPModel(job.models["mlp"], model_seed(job.seed, "fedavg_global", "mlp"))
    strategy = StrictFedAvg(len(job.clients), node_to_bank)
    t0 = time.perf_counter()
    strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(init.get_state_dict()),
        num_rounds=job.num_rounds,
        timeout=job.round_timeout,
        train_config=ConfigRecord({"job-path": str(job.path)}),
        evaluate_fn=make_evaluate_fn(job, server_dir, strategy.rounds),
    )
    rounds = [strategy.rounds[r] for r in sorted(strategy.rounds)]
    for r in rounds:
        r.setdefault("bytes_up", 0)
        r.setdefault("bytes_down", 0)
    summary = {
        "mode": "fedavg",
        "num_rounds": job.num_rounds,
        "num_parameters": init.num_parameters(),
        "rounds": rounds,
        "total_bytes_up": int(sum(r["bytes_up"] for r in rounds)),
        "total_bytes_down": int(sum(r["bytes_down"] for r in rounds)),
        "wall_seconds": time.perf_counter() - t0,
        "aggregation": "FedAvg: weighted mean of client parameters, weight = num-examples",
    }
    write_json(server_dir / "rounds.json", summary)
    return summary
