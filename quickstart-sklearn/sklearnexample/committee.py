"""Experiment 4: heterogeneous federated committee with FedDistill-style distillation.

Each bank keeps its own model type (bank 0 XGBoost, bank 1 Random Forest, bank 2 MLP).
No model parameters are shared. Each round:
  1. A bank trains its model on its private data. After round 1 the targets are soft:
     (1 - alpha) * true label + alpha * committee's average prediction for that class.
  2. It sends the server its per-class sum of logits and per-class counts.
  3. The server combines them into the committee's average logit for each class
     (legit, fraud) and sends those two numbers back for the next round.
"""

import pickle
from collections.abc import Iterable

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearnexample.models import MLP_HIDDEN, in_bank_order
from sklearnexample.task import SEED

EPS = 1e-4  # keeps logits finite when a tree model predicts exactly 0 or 1


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def bank_model(partition_id: int, run_config) -> str:
    return run_config["committee-models"].split(",")[partition_id].strip()


def soft_targets(y, global_logits: np.ndarray, alpha: float) -> np.ndarray:
    """Blend true labels with the committee's class-average prediction."""
    if global_logits.size == 0:  # round 1: no committee knowledge yet
        return y.astype(float)
    return (1 - alpha) * y + alpha * _sigmoid(global_logits[y])


def _class_weights(name: str, y) -> np.ndarray:
    """Same imbalance handling as the local baselines, expressed as sample weights."""
    n_fraud, n_legit = (y == 1).sum(), (y == 0).sum()
    if name == "xgboost":  # scale_pos_weight = legit / fraud
        return np.where(y == 1, n_legit / n_fraud, 1.0)
    if name == "rf":  # class_weight="balanced"
        return np.where(y == 1, len(y) / (2 * n_fraud), len(y) / (2 * n_legit))
    return np.ones(len(y))  # MLP baseline is unweighted


def fit_soft(name: str, X, y, targets):
    """Train a classifier on soft targets.

    A row with target t becomes two weighted copies: (x, 1) with weight t and
    (x, 0) with weight 1 - t. Weighted cross-entropy on these copies equals
    cross-entropy against the soft target, so every model type can learn from it.
    """
    base = _class_weights(name, y)
    X2 = np.vstack([X, X])
    y2 = np.concatenate([np.ones(len(y), int), np.zeros(len(y), int)])
    w2 = np.concatenate([base * targets, base * (1 - targets)])
    keep = w2 > 0  # with hard labels this is exactly the original data
    X2, y2, w2 = X2[keep], y2[keep], w2[keep]

    if name == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=SEED,
        )
    elif name == "rf":
        model = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=SEED
        )
    elif name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            early_stopping=True,
            max_iter=200,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown committee model '{name}'")
    model.fit(X2, y2, sample_weight=w2)
    return model


def class_logit_stats(model, X, y) -> list[np.ndarray]:
    """Per-class sum of logits and per-class counts: the only thing a bank shares."""
    z = _logit(model.predict_proba(X)[:, 1])
    sums = np.array([z[y == 0].sum(), z[y == 1].sum()])
    counts = np.array([(y == 0).sum(), (y == 1).sum()], dtype=float)
    return [sums, counts]


def dump_model(model) -> ArrayRecord:
    return ArrayRecord([np.frombuffer(pickle.dumps(model), dtype=np.uint8)])


def load_model(record: ArrayRecord):
    return pickle.loads(record.to_numpy_ndarrays()[0].tobytes())


class FedDistill(FedAvg):
    """Aggregate class-average logits instead of model parameters."""

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(
            in_bank_order(replies), is_train=True
        )
        if not valid_replies:
            return None, None
        contents = [msg.content for msg in valid_replies]
        stats = [c["arrays"].to_numpy_ndarrays() for c in contents]
        # Exact pooled mean per class: total logit sum / total count
        sums = np.sum([s[0] for s in stats], axis=0)
        counts = np.sum([s[1] for s in stats], axis=0)
        global_logits = sums / counts
        arrays = ArrayRecord([global_logits])
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        metrics["global_logit_legit"] = float(global_logits[0])
        metrics["global_logit_fraud"] = float(global_logits[1])
        return arrays, metrics
