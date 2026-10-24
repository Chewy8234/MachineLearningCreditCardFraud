"""Message types, record keys and payload helpers for the Flower Message API.

Heterogeneous KD exchanges *prediction vectors* (``ArrayRecord`` under key
``"probs"`` / ``"teacher"``), never model parameters. FedAvg exchanges MLP
parameters (``ArrayRecord`` under ``"arrays"``), as Flower's FedAvg expects.
"""

from __future__ import annotations

import numpy as np
from flwr.app import Array, ArrayRecord, RecordDict

# message types ("<category>.<action>" routes to @app.<category>("<action>"))
MSG_IDENTIFY = "query.identify"
MSG_KD_INIT = "train.kd_init"
MSG_KD_DISTILL = "train.kd_distill"
MSG_FEDAVG_TRAIN = "train"  # Flower's FedAvg sends the default train action

# record keys
CONFIG = "config"
METRICS = "metrics"
INFO = "info"
PROXY_PREDICTIONS = "proxy_predictions"
TEACHER = "teacher"
ARRAYS = "arrays"
PROBS = "probs"

PREDICTION_DTYPE = np.float32


def probs_to_record(p: np.ndarray) -> ArrayRecord:
    return ArrayRecord({PROBS: Array(np.ascontiguousarray(p, dtype=PREDICTION_DTYPE))})


def record_to_probs(record: ArrayRecord) -> np.ndarray:
    if PROBS not in record:
        raise KeyError(f"ArrayRecord has no '{PROBS}' entry (keys: {list(record.keys())})")
    return record[PROBS].numpy().astype(np.float64)


def payload_bytes(content: RecordDict) -> int:
    """Serialised size of the array payloads in a message (dominant traffic).

    Config/metric records (a few hundred bytes of scalars) are not counted.
    """
    return int(sum(rec.count_bytes() for rec in content.array_records.values()))
