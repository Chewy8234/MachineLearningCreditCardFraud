"""Homogeneous federated versions of each model family.

Every bank trains the same model type; only how the server combines them differs:
  logreg   - FedAvg over coefficients
  mlp      - FedAvg over layer weights
  xgboost  - FedXgbBagging: each bank adds trees to one shared booster per round
  rf       - federated forest: the banks' trees are pooled into one forest
"""

import pickle
import warnings
from collections.abc import Callable, Iterable

import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg, FedXgbBagging
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearnexample.task import (
    FEATURES,
    SEED,
    create_log_reg_and_instantiate_parameters,
    get_model_params,
    set_model_params,
)

MLP_HIDDEN = (64, 32)
ProbaFn = Callable[[np.ndarray], np.ndarray]  # X -> fraud probability


# ---------- byte helpers (XGBoost and RF travel as raw bytes) ----------


def _to_array(data: bytes) -> list[np.ndarray]:
    return [np.frombuffer(data, dtype=np.uint8)]


def _to_bytes(ndarrays: list[np.ndarray]) -> bytes:
    return ndarrays[0].tobytes()


# ---------- MLP ----------


def initial_mlp_params() -> list[np.ndarray]:
    """Glorot-uniform weights, the same scheme sklearn uses, from a fixed seed."""
    rng = np.random.default_rng(SEED)
    sizes = [len(FEATURES), *MLP_HIDDEN, 1]
    coefs, intercepts = [], []
    for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        # sklearn uses a smaller bound for the logistic output layer
        factor = 2.0 if i == len(sizes) - 2 else 6.0
        bound = np.sqrt(factor / (fan_in + fan_out))
        coefs.append(rng.uniform(-bound, bound, (fan_in, fan_out)))
        intercepts.append(rng.uniform(-bound, bound, fan_out))
    return coefs + intercepts


def _mlp_from_params(params: list[np.ndarray], X_sample, y_sample) -> MLPClassifier:
    model = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, random_state=SEED)
    # sklearn only builds the network on its first fit, so prime it on one row
    model.partial_fit(X_sample[:1], y_sample[:1], classes=[0, 1])
    n = len(params) // 2
    model.coefs_ = [p.copy() for p in params[:n]]
    model.intercepts_ = [p.copy() for p in params[n:]]
    del model._optimizer  # fresh Adam state for the received weights
    return model


# ---------- XGBoost ----------


def xgb_params(y_train) -> dict:
    return {
        "objective": "binary:logistic",
        "eta": 0.1,
        "max_depth": 6,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "seed": SEED,
    }


def _load_booster(data: bytes) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(bytearray(data))
    return bst


# ---------- Random Forest ----------


def merge_forests(forests: list[RandomForestClassifier]) -> RandomForestClassifier:
    """Pool every bank's trees into one forest that averages their votes."""
    merged = forests[0]
    merged.estimators_ = [tree for forest in forests for tree in forest.estimators_]
    merged.n_estimators = len(merged.estimators_)
    return merged


class FedForest(FedAvg):
    """Federated forest: concatenate the trees each bank sends, never average."""

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        contents = [msg.content for msg in valid_replies]
        forests = [
            pickle.loads(_to_bytes(c["arrays"].to_numpy_ndarrays())) for c in contents
        ]
        merged = merge_forests(forests)
        arrays = ArrayRecord(_to_array(pickle.dumps(merged)))
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        return arrays, metrics


# ---------- one interface for client_app and server_app ----------


def initial_arrays(name: str, penalty: str) -> list[np.ndarray]:
    """Global model the server starts from."""
    if name == "logreg":
        return get_model_params(create_log_reg_and_instantiate_parameters(penalty))
    if name == "mlp":
        return initial_mlp_params()
    # Tree models start empty; the banks grow the first trees
    return _to_array(b"")


def make_strategy(name: str, min_nodes: int) -> FedAvg:
    kwargs = dict(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=min_nodes,
        min_evaluate_nodes=min_nodes,
        min_available_nodes=min_nodes,
    )
    if name == "xgboost":
        return FedXgbBagging(**kwargs)
    if name == "rf":
        return FedForest(**kwargs)
    return FedAvg(**kwargs)


def train_local(
    name: str, ndarrays: list[np.ndarray], X_train, y_train, run_config
) -> tuple[list[np.ndarray], ProbaFn]:
    """Train on one bank's data starting from the global model.

    Returns what to send to the server, and a fraud-probability function for the
    locally trained model.
    """
    if name == "logreg":
        model = create_log_reg_and_instantiate_parameters(run_config["penalty"])
        set_model_params(model, ndarrays)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        return get_model_params(model), lambda X: model.predict_proba(X)[:, 1]

    if name == "mlp":
        model = _mlp_from_params(ndarrays, X_train, y_train)
        rng = np.random.default_rng(SEED)
        for _ in range(int(run_config["local-epochs"])):
            order = rng.permutation(len(X_train))
            model.partial_fit(X_train[order], y_train[order])
        return model.coefs_ + model.intercepts_, lambda X: model.predict_proba(X)[:, 1]

    if name == "xgboost":
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = xgb_params(y_train)
        global_model = _to_bytes(ndarrays)
        if global_model == b"":
            bst = xgb.train(params, dtrain, num_boost_round=1)
        else:
            bst = _load_booster(global_model)
            bst.set_param(params)
            bst.update(dtrain, bst.num_boosted_rounds())
        # Send only the tree grown this round; the server appends it (bagging)
        n = bst.num_boosted_rounds()
        new_tree = bst[n - 1 : n]
        proba = lambda X: bst.predict(xgb.DMatrix(X))
        return _to_array(new_tree.save_raw("json")), proba

    if name == "rf":
        model = RandomForestClassifier(
            n_estimators=int(run_config["rf-trees-per-bank"]),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        return _to_array(pickle.dumps(model)), lambda X: model.predict_proba(X)[:, 1]

    raise ValueError(f"Unknown model '{name}'")


def load_global(name: str, ndarrays: list[np.ndarray], run_config) -> ProbaFn:
    """Fraud-probability function for the global model the server sent."""
    if name == "logreg":
        model = create_log_reg_and_instantiate_parameters(run_config["penalty"])
        set_model_params(model, ndarrays)
        return lambda X: model.predict_proba(X)[:, 1]
    if name == "mlp":
        dummy_X = np.zeros((1, len(FEATURES)))
        model = _mlp_from_params(ndarrays, dummy_X, np.array([0]))
        return lambda X: model.predict_proba(X)[:, 1]
    if name == "xgboost":
        bst = _load_booster(_to_bytes(ndarrays))
        return lambda X: bst.predict(xgb.DMatrix(X))
    if name == "rf":
        model = pickle.loads(_to_bytes(ndarrays))
        return lambda X: model.predict_proba(X)[:, 1]
    raise ValueError(f"Unknown model '{name}'")


def save_global(name: str, ndarrays: list[np.ndarray]) -> str:
    """Write the final global model to disk and return the file name."""
    if name in ("xgboost", "rf"):
        path = "xgboost_model.json" if name == "xgboost" else "rf_model.pkl"
        with open(path, "wb") as f:
            f.write(_to_bytes(ndarrays))
        return path
    # Weight-based models: save the weight arrays in order
    path = f"{name}_model.npz"
    np.savez(path, *ndarrays)
    return path
