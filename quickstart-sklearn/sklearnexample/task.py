from pathlib import Path

import numpy as np
import pandas as pd
from flwr.common import NDArrays
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# This information is needed to create a correct scikit-learn model
UNIQUE_LABELS = [0, 1]  # 0 = legit, 1 = fraud
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
SEED = 42  # used for every shuffle, split and model


def get_model_params(model: LogisticRegression) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(model: LogisticRegression, params: NDArrays) -> LogisticRegression:
    """Set the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.array([i for i in range(n_classes)])

    # Binary classification stores a single row of weights, not one per class
    n_rows = 1 if n_classes == 2 else n_classes
    model.coef_ = np.zeros((n_rows, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_rows,))


def create_log_reg_and_instantiate_parameters(penalty):
    model = LogisticRegression(
        penalty=penalty,
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
        solver="saga",
        class_weight="balanced",  # fraud is only ~0.17% of transactions
        random_state=SEED,
    )
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model, n_features=len(FEATURES), n_classes=len(UNIQUE_LABELS))
    return model


df = None  # Cache the dataset


def load_data(partition_id: int, num_partitions: int, data_path: str):
    """Load the data for the given partition."""
    global df
    if df is None:
        path = Path(data_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Could not find {path.resolve()}. Set `data-path` in pyproject.toml."
            )
        df = pd.read_csv(path)
        # Amount ranges into the thousands; log-scale it so it doesn't dominate
        df["Amount"] = np.log1p(df["Amount"])
        # Shuffle once with a fixed seed so every client sees the same split
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Each client (e.g. a bank) gets its own slice of the transactions
    bounds = np.linspace(0, len(df), num_partitions + 1, dtype=int)
    partition = df.iloc[bounds[partition_id] : bounds[partition_id + 1]]
    X = partition[FEATURES].values
    y = partition[TARGET].values
    # Split the on-edge data: 70% train, 10% validation, 20% test,
    # keeping the fraud ratio the same in all three
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.125, stratify=y_rest, random_state=SEED
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def best_threshold(y_val, y_val_proba) -> float:
    """Pick the decision threshold that maximizes F1 on the validation set."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_val, y_val_proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    # The last precision/recall pair has no threshold
    return float(thresholds[np.argmax(f1[:-1])])


def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Fraud-focused metrics shared by every experiment."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
