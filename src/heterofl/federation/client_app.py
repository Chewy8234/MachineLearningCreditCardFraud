"""Flower ClientApp: one simulated bank per SuperNode.

Identity: a SuperNode's ``node_config['partition-id']`` (0, 1, 2) selects its
bank and model type from ``job.json``; identity is re-reported in every reply
and checked by the server.

Handlers
--------
``query.identify``   report partition id, bank and model type
``train``            FedAvg local update of the shared MLP (Flower FedAvg default)
``train.kd_init``    round 0: train the bank's own model on private data only,
                     return its probabilities on the ordered proxy set
``train.kd_distill`` round r >= 1: receive this bank's teacher vector, update
                     the model with local labels + soft proxy targets, return
                     new proxy probabilities

Raw private training rows never leave the client: replies contain only proxy
probabilities (KD) or MLP parameters (FedAvg) plus scalar metrics.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from heterofl.distillation.trainers import distill_update, initial_fit
from heterofl.evaluation.metrics import average_precision
from heterofl.federation import messages as M
from heterofl.federation.job import ClientSpec, FederatedJob, FederationError, load_job
from heterofl.federation.state import STATE_KEY, ClientStateStore
from heterofl.models.mlp import MLPModel
from heterofl.models.model_factory import create_model, model_seed
from heterofl.utils.io import load_npz
from heterofl.utils.logging import get_logger
from heterofl.utils.reproducibility import derive_seed, sha256_array

log = get_logger("federation.client")

app = ClientApp()


# --------------------------------------------------------------------------- data
@lru_cache(maxsize=16)
def _bank_data(view_dir: str, bank: str) -> tuple[np.ndarray, np.ndarray]:
    d = load_npz(Path(view_dir) / f"bank_{bank}.npz")
    return d["X"], d["y"]


@lru_cache(maxsize=4)
def _proxy(view_dir: str) -> tuple[np.ndarray, str]:
    d = load_npz(Path(view_dir) / "proxy.npz")
    if "y" in d:  # the view must never contain proxy labels
        raise FederationError("proxy.npz contains labels; refusing to run")
    return d["X"], sha256_array(d["row_ids"].astype(np.int64))


@lru_cache(maxsize=4)
def _validation(view_dir: str) -> tuple[np.ndarray, np.ndarray]:
    d = load_npz(Path(view_dir) / "validation.npz")
    return d["X"], d["y"]


# --------------------------------------------------------------------------- helpers
def _resolve(msg: Message, context: Context) -> tuple[FederatedJob, ClientSpec]:
    job = load_job(str(msg.content[M.CONFIG]["job-path"]))
    pid = int(context.node_config["partition-id"])
    return job, job.client(pid)


def _maybe_inject_failure(job: FederatedJob, spec: ClientSpec, rnd: int) -> None:
    """Test hook: ``job.debug.fail = {partition_id, round}`` makes this client raise."""
    fail = job.debug.get("fail")
    if fail and int(fail["partition_id"]) == spec.partition_id and int(fail["round"]) == rnd:
        raise RuntimeError(f"Injected failure: partition {spec.partition_id}, round {rnd}")


def _identity(job: FederatedJob, spec: ClientSpec) -> dict:
    return {"partition-id": spec.partition_id, "bank": spec.bank,
            "model-type": job.model_type_for(spec)}


def _scalar_metrics(d: dict) -> dict:
    return {k: float(v) for k, v in d.items()
            if isinstance(v, (int, float, np.floating, np.integer)) and v is not None
            and np.isfinite(v)}


# --------------------------------------------------------------------------- identify
@app.query("identify")
def identify(msg: Message, context: Context) -> Message:
    job, spec = _resolve(msg, context)
    content = RecordDict({M.INFO: ConfigRecord({**_identity(job, spec), "mode": job.mode,
                                                 "node-id": int(context.node_id)})})
    return Message(content=content, reply_to=msg)


# --------------------------------------------------------------------------- FedAvg
@app.train()
def fedavg_train(msg: Message, context: Context) -> Message:
    """FedAvg local update: load global MLP weights, train locally, return weights."""
    job, spec = _resolve(msg, context)
    rnd = int(msg.content[M.CONFIG]["server-round"])
    _maybe_inject_failure(job, spec, rnd)
    X, y = _bank_data(job.view_dir, spec.bank)
    model = MLPModel(job.models["mlp"], derive_seed(job.seed, "fedavg-client", spec.bank, rnd),
                     n_features=X.shape[1], n_jobs=job.num_cpus_per_client)
    model.set_state_dict(msg.content[M.ARRAYS].to_torch_state_dict())
    info = model.fit(X, y, epochs=int(job.fedavg["local_epochs"]))
    Xv, yv = _validation(job.view_dir)
    local_val_ap = average_precision(yv, model.predict_proba(Xv))

    prev = int(context.state[STATE_KEY]["round"]) if STATE_KEY in context.state else 0
    if rnd <= prev:
        raise FederationError(f"bank {spec.bank}: round {rnd} not after previous {prev}")
    context.state[STATE_KEY] = ConfigRecord({"bank": spec.bank, "round": rnd})

    metrics = {
        # "num-examples" must come first: Flower weights FedAvg by this key
        "num-examples": int(len(y)),
        "partition-id": spec.partition_id,
        "train-loss": float(info["final_supervised_loss"]),
        "fit-seconds": float(info["fit_seconds"]),
        "local-val-ap": local_val_ap if np.isfinite(local_val_ap) else -1.0,
        "n-fraud": int(y.sum()),
    }
    log.info("FedAvg round %d bank %s: n=%d loss=%.4f local-val-AP=%.4f", rnd, spec.bank,
             len(y), metrics["train-loss"], local_val_ap)
    content = RecordDict({M.ARRAYS: ArrayRecord(model.get_state_dict()),
                          M.METRICS: MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


# --------------------------------------------------------------------------- KD
def _kd_reply(msg: Message, job: FederatedJob, spec: ClientSpec, model, rnd: int,
              info: dict, teacher: np.ndarray | None, context: Context,
              fit_seconds: float) -> Message:
    Xp, proxy_sha = _proxy(job.view_dir)
    Xv, yv = _validation(job.view_dir)
    t0 = time.perf_counter()
    p_proxy = model.predict_proba(Xp)
    proxy_predict_seconds = time.perf_counter() - t0
    p_val = model.predict_proba(Xv)
    val_ap = average_precision(yv, p_val)

    store = ClientStateStore(job.job_dir, spec.bank)
    path, digest = store.save_round(model, rnd, val_probs=p_val, proxy_probs=p_proxy,
                                    teacher=teacher, info=info)
    ClientStateStore.record(context, bank=spec.bank, rnd=rnd, path=path, digest=digest,
                            model_type=model.model_type)

    _, y = _bank_data(job.view_dir, spec.bank)
    metrics = {"num-examples": int(len(y)), "partition-id": spec.partition_id, "round": rnd,
               "val-ap": val_ap if np.isfinite(val_ap) else -1.0,
               "fit-seconds": fit_seconds, "proxy-predict-seconds": proxy_predict_seconds,
               **{f"fit-{k}": v for k, v in _scalar_metrics(info).items()
                  if k not in ("fit_seconds",)}}
    info_rec = {**_identity(job, spec), "proxy-sha256": proxy_sha, "model-sha256": digest}
    log.info("KD round %d bank %s (%s): val-AP=%.4f fit=%.1fs", rnd, spec.bank,
             model.model_type, val_ap, fit_seconds)
    content = RecordDict({M.PROXY_PREDICTIONS: M.probs_to_record(p_proxy),
                          M.METRICS: MetricRecord(metrics),
                          M.INFO: ConfigRecord(info_rec)})
    return Message(content=content, reply_to=msg)


@app.train("kd_init")
def kd_init(msg: Message, context: Context) -> Message:
    """Round 0: independent local training on private labelled data only."""
    job, spec = _resolve(msg, context)
    _maybe_inject_failure(job, spec, 0)
    X, y = _bank_data(job.view_dir, spec.bank)
    model = create_model(spec.model_type, job.models,
                         model_seed(job.seed, f"bank_{spec.bank}", spec.model_type),
                         n_jobs=job.num_cpus_per_client, n_features=X.shape[1])
    t0 = time.perf_counter()
    info = initial_fit(model, X, y)
    return _kd_reply(msg, job, spec, model, 0, info, None, context, time.perf_counter() - t0)


@app.train("kd_distill")
def kd_distill(msg: Message, context: Context) -> Message:
    """Round r >= 1: distil from this bank's teacher vector on the proxy set."""
    job, spec = _resolve(msg, context)
    cfg = msg.content[M.CONFIG]
    rnd = int(cfg["server-round"])
    _maybe_inject_failure(job, spec, rnd)
    Xp, proxy_sha = _proxy(job.view_dir)
    if str(cfg["proxy-sha256"]) != proxy_sha:
        raise FederationError(f"bank {spec.bank}: proxy set mismatch with server")
    if str(cfg["teacher-for-bank"]) != spec.bank:
        raise FederationError(f"bank {spec.bank} received the teacher for {cfg['teacher-for-bank']}")
    teacher = M.record_to_probs(msg.content[M.TEACHER])
    if teacher.shape != (len(Xp),):
        raise FederationError(f"teacher length {teacher.shape} != proxy size {len(Xp)}")

    store = ClientStateStore(job.job_dir, spec.bank)
    seed = model_seed(job.seed, f"bank_{spec.bank}", spec.model_type)
    model = store.load_previous(context, expected_round=rnd - 1, model_type=spec.model_type,
                                models_cfg=job.models, seed=seed,
                                n_jobs=job.num_cpus_per_client)
    X, y = _bank_data(job.view_dir, spec.bank)
    t0 = time.perf_counter()
    info = distill_update(model, X, y, Xp, teacher, job.kd)
    return _kd_reply(msg, job, spec, model, rnd, info, teacher, context, time.perf_counter() - t0)
