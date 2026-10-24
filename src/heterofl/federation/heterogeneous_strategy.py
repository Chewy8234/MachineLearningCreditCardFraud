"""Custom Flower strategy for heterogeneous prediction-based knowledge distillation.

Protocol (FedMD-inspired, per-example proxy distillation)::

    round 0  server --kd_init-->  every bank trains its own model on private data
             bank   --p_k(proxy)--> server            (ordered proxy predictions)
    round r  server: T_k = mean_{j != k} p_j          (leave-one-out teachers)
             server --T_k-->  bank k                  (each bank gets ITS teacher)
             bank k: update with private labels + soft targets T_k on the proxy set
             bank   --p_k(proxy)--> server

The server never sees model parameters, never builds a global model, and has
no access to proxy labels, private data, or the test split. Every reply is
validated: all expected banks must answer without error, identities must match
the round-0 identification, proxy-set hashes must match, and prediction
vectors must be finite probabilities of the right length.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid

from heterofl.distillation.aggregation import (
    AggregationError,
    compute_teachers,
    validate_prediction_vectors,
    validation_ap_weights,
)
from heterofl.federation import messages as M
from heterofl.federation.job import ClientSpec, FederatedJob, FederationError
from heterofl.utils.io import write_json
from heterofl.utils.logging import get_logger

log = get_logger("federation.kd")


def wait_for_nodes(grid: Grid, expected: int, timeout: float) -> list[int]:
    t0 = time.time()
    while True:
        ids = sorted(grid.get_node_ids())
        if len(ids) >= expected:
            return ids
        if time.time() - t0 > timeout:
            raise FederationError(f"Only {len(ids)} of {expected} nodes connected "
                                  f"after {timeout:.0f}s")
        time.sleep(0.5)


def identify_clients(grid: Grid, job: FederatedJob) -> dict[int, ClientSpec]:
    """Query every node for its partition id; return node_id -> ClientSpec.

    Extra nodes (e.g. a 10-node local SuperLink used with ``flwr run``) whose
    partition id has no bank are ignored; missing banks are an error.
    """
    node_ids = wait_for_nodes(grid, len(job.clients), job.node_wait_timeout)
    cfg = ConfigRecord({"job-path": str(job.path)})
    msgs = [Message(content=RecordDict({M.CONFIG: cfg}), dst_node_id=n,
                    message_type=M.MSG_IDENTIFY, group_id="identify") for n in node_ids]
    replies = list(grid.send_and_receive(msgs, timeout=job.round_timeout))
    mapping: dict[int, ClientSpec] = {}
    for r in replies:
        if r.has_error():
            reason = r.error.reason
            if "No client is assigned" in reason:
                continue
            raise FederationError(f"identify failed on node {r.metadata.src_node_id}: {reason}")
        info = r.content[M.INFO]
        spec = job.client(int(info["partition-id"]))
        if info["bank"] != spec.bank or info["model-type"] != job.model_type_for(spec):
            raise FederationError(f"Node {r.metadata.src_node_id} reported inconsistent identity")
        mapping[r.metadata.src_node_id] = spec
    banks = sorted(s.bank for s in mapping.values())
    if banks != sorted(c.bank for c in job.clients):
        raise FederationError(f"Identified banks {banks}, expected "
                              f"{sorted(c.bank for c in job.clients)}")
    log.info("Identified clients: %s", {n: f"bank_{s.bank}/{job.model_type_for(s)}"
                                        for n, s in mapping.items()})
    return mapping


class HeterogeneousKDStrategy:
    """Server-side orchestration of the heterogeneous KD protocol."""

    def __init__(self, job: FederatedJob):
        if job.mode != "kd":
            raise ValueError("HeterogeneousKDStrategy requires a job with mode='kd'")
        self.job = job
        self.kd = job.kd
        self.server_dir = Path(job.job_dir) / "server"
        self.rounds: list[dict] = []
        self.n_proxy: int | None = None

    # ----------------------------------------------------------------- one round
    def _send_round(self, grid: Grid, mapping: dict[int, ClientSpec], rnd: int,
                    teachers: dict[str, np.ndarray] | None) -> tuple[dict, dict, dict]:
        msgs = []
        for node_id, spec in mapping.items():
            cfg = {"job-path": str(self.job.path), "server-round": rnd,
                   "proxy-sha256": self.job.expected_proxy_sha256,
                   "teacher-for-bank": spec.bank}
            content = RecordDict({M.CONFIG: ConfigRecord(cfg)})
            if teachers is not None:
                content[M.TEACHER] = M.probs_to_record(teachers[spec.bank])
            msgs.append(Message(content=content, dst_node_id=node_id,
                                message_type=M.MSG_KD_INIT if rnd == 0 else M.MSG_KD_DISTILL,
                                group_id=str(rnd)))
        bytes_down = sum(M.payload_bytes(m.content) for m in msgs)
        t0 = time.perf_counter()
        replies = list(grid.send_and_receive(msgs, timeout=self.job.round_timeout))
        wall = time.perf_counter() - t0

        if len(replies) != len(mapping):
            got = {r.metadata.src_node_id for r in replies}
            missing = [f"bank_{s.bank}" for n, s in mapping.items() if n not in got]
            raise FederationError(f"KD round {rnd}: missing replies from {missing}")
        preds, metrics = {}, {}
        for r in replies:
            spec = mapping.get(r.metadata.src_node_id)
            if spec is None:
                raise FederationError(f"KD round {rnd}: reply from unknown node")
            if r.has_error():
                raise FederationError(f"KD round {rnd}: bank {spec.bank} failed: {r.error.reason}")
            info = r.content[M.INFO]
            if info["bank"] != spec.bank or int(info["partition-id"]) != spec.partition_id:
                raise FederationError(f"KD round {rnd}: identity changed for node "
                                      f"{r.metadata.src_node_id}")
            if info["proxy-sha256"] != self.job.expected_proxy_sha256:
                raise FederationError(f"KD round {rnd}: bank {spec.bank} used a different "
                                      "proxy set/order")
            if M.ARRAYS in r.content:
                raise FederationError("KD replies must not contain model parameters")
            preds[spec.bank] = M.record_to_probs(r.content[M.PROXY_PREDICTIONS])
            metrics[spec.bank] = {k: v for k, v in dict(r.content[M.METRICS]).items()}
        if self.n_proxy is None:
            self.n_proxy = len(next(iter(preds.values())))
        try:
            preds = validate_prediction_vectors(preds, self.n_proxy)
        except AggregationError as exc:
            raise FederationError(f"KD round {rnd}: {exc}") from exc
        bytes_up = sum(M.payload_bytes(r.content) for r in replies)
        comm = {"bytes_down": int(bytes_down), "bytes_up": int(bytes_up),
                "messages_down": len(msgs), "messages_up": len(replies), "round_seconds": wall}
        return preds, metrics, comm

    def _weights(self, metrics: dict[str, dict]) -> dict[str, float] | None:
        if self.kd["teacher_weighting"] == "equal":
            return None
        return validation_ap_weights({b: float(m["val-ap"]) for b, m in metrics.items()})

    # ----------------------------------------------------------------- full run
    def run(self, grid: Grid, mapping: dict[int, ClientSpec]) -> dict:
        self.server_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        preds, metrics, comm = self._send_round(grid, mapping, 0, None)
        self._log_round(0, metrics, comm, None, None)
        for rnd in range(1, self.job.num_rounds + 1):
            weights = self._weights(metrics)
            teachers = compute_teachers(preds, self.kd["teacher"], self.n_proxy, weights)
            stats = {b: {"mean": float(t.mean()), "max": float(t.max()),
                         "sum": float(t.sum()), "n_above_0.5": int((t >= 0.5).sum())}
                     for b, t in teachers.items()}
            preds, metrics, comm = self._send_round(grid, mapping, rnd, teachers)
            self._log_round(rnd, metrics, comm, stats, weights)
        summary = {
            "mode": "kd",
            "teacher": self.kd["teacher"],
            "teacher_weighting": self.kd["teacher_weighting"],
            "lambda_kd": float(self.kd["lambda_kd"]),
            "num_rounds": self.job.num_rounds,
            "n_proxy": self.n_proxy,
            "rounds": self.rounds,
            "total_bytes_up": int(sum(r["bytes_up"] for r in self.rounds)),
            "total_bytes_down": int(sum(r["bytes_down"] for r in self.rounds)),
            "wall_seconds": time.perf_counter() - t_start,
            "aggregation": "per-example proxy probabilities; no parameter aggregation",
        }
        write_json(self.server_dir / "rounds.json", summary)
        return summary

    def _log_round(self, rnd, metrics, comm, teacher_stats, weights) -> None:
        rec = {"round": rnd, **comm, "client_metrics": metrics,
               "teacher_stats": teacher_stats, "teacher_weights": weights}
        self.rounds.append(rec)
        log.info("KD round %d: %s | up=%d B down=%d B (%.1fs)", rnd,
                 ", ".join(f"{b}: val-AP={m['val-ap']:.4f}" for b, m in sorted(metrics.items())),
                 comm["bytes_up"], comm["bytes_down"], comm["round_seconds"])
