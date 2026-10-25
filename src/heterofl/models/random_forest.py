"""Bank B: scikit-learn RandomForestClassifier."""

from __future__ import annotations

import time
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from heterofl.models.base import (
    FraudModel,
    any_of,
    check_probabilities,
    int_range,
    one_of,
    real_range,
    validate_params,
)


def _is_mapping(v):
    return None if isinstance(v, dict) else "must be a mapping"


#: Supported hyperparameters (``None`` is passed through: it is meaningful, e.g.
#: ``max_depth: null`` grows trees fully).
PARAM_SPECS = {
    "n_estimators": int_range(1),
    "max_depth": int_range(1, allow_none=True),
    "min_samples_leaf": any_of(int_range(1), real_range(0, 1, low_open=True, high_open=True)),
    "min_samples_split": any_of(int_range(2), real_range(0, 1, low_open=True)),
    "max_features": any_of(one_of("sqrt", "log2", allow_none=True), int_range(1),
                           real_range(0, 1, low_open=True)),
    "bootstrap": one_of(True, False),
    "max_samples": any_of(int_range(1, allow_none=True), real_range(0, 1, low_open=True)),
    "criterion": one_of("gini", "entropy", "log_loss"),
    "class_weight": any_of(one_of("balanced", "balanced_subsample", allow_none=True), _is_mapping),
}
PARAM_KEYS = tuple(PARAM_SPECS)


def bootstrap_draws_per_tree(params: dict, n_rows: int, sample_weight: np.ndarray | None) -> int:
    """Rows drawn (with replacement) for each tree, as scikit-learn 1.9.1 computes it.

    Mirrors ``sklearn.ensemble._bootstrap._get_n_samples_bootstrap``: ``None`` -> the
    number of rows; an int -> itself; a float -> ``max(int(f * W), 1)`` where ``W`` is
    the row count without weights and ``sum(sample_weight)`` with weights (changed in
    1.9). Verified against the fitted estimator in tests/test_models.py.
    """
    if not params.get("bootstrap", True):
        return int(n_rows)
    ms = params.get("max_samples")
    if ms is None:
        return int(n_rows)
    if isinstance(ms, Integral):
        return int(ms)
    total = float(n_rows) if sample_weight is None else float(np.sum(sample_weight))
    return max(int(float(ms) * total), 1) if isinstance(ms, Real) else int(n_rows)


class RandomForestModel(FraudModel):
    """Random forest. It cannot be updated incrementally: every ``fit`` call grows a
    completely new forest (the federation therefore *retrains* it each round).

    Sample weights (scikit-learn >= 1.7 semantics, verified for 1.9.1): with
    ``bootstrap=True`` each tree's bootstrap sample is drawn with probability
    proportional to ``sample_weight`` (frequency semantics), not uniformly, and the
    tree is then fitted on the resulting row *counts* (the weights are not applied a
    second time). When weights are given we set ``max_samples=1.0``, so each tree
    draws ``int(sum(sample_weight))`` rows; for the soft-target set that is
    ``N_local * (1 + lambda_kd)``: in expectation the same number of local draws
    as a local-only forest, plus ``lambda_kd * N_local`` proxy draws whose labels
    follow the teacher probability. ``sample_weight=None`` is the ordinary forest.
    """

    model_type = "random_forest"
    file_suffix = ".joblib"

    @staticmethod
    def validate_params(params: dict) -> None:
        validate_params("random_forest", params, PARAM_SPECS)

    def __init__(self, params: dict, seed: int, n_jobs: int = 1):
        self.validate_params(params)
        self.params = dict(params)
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.clf: RandomForestClassifier | None = None

    def effective_params(self, sample_weight: np.ndarray | None) -> dict:
        params = dict(self.params)
        if sample_weight is not None and params.get("bootstrap", True) \
                and params.get("max_samples") is None:
            params["max_samples"] = 1.0  # draw sum(sample_weight) rows (see class doc)
        return params

    def fit(self, X, y, sample_weight=None) -> dict:
        t0 = time.perf_counter()
        params = self.effective_params(sample_weight)
        self.clf = RandomForestClassifier(random_state=self.seed, n_jobs=self.n_jobs, **params)
        self.clf.fit(X, y, sample_weight=sample_weight)
        if list(self.clf.classes_) != [0, 1]:
            raise ValueError(f"Random forest saw classes {self.clf.classes_}; need both 0 and 1")
        return {"fit_seconds": time.perf_counter() - t0, "n_rows": int(len(y)),
                "n_trees": len(self.clf.estimators_), "update": "retrain",
                "bootstrap_draws_per_tree": bootstrap_draws_per_tree(params, len(y), sample_weight),
                "weighted_bootstrap": sample_weight is not None}

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model is not fitted")
        return check_probabilities(self.clf.predict_proba(X)[:, 1], len(X), "rf proba")

    def save(self, path: Path) -> Path:
        if self.clf is None:
            raise RuntimeError("Cannot save an unfitted Random Forest model")
        path = self._with_suffix(path)
        joblib.dump(self.clf, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path, params: dict, seed: int, n_jobs: int = 1) -> "RandomForestModel":
        """Load a forest saved by ``save``.

        SECURITY: ``joblib.load`` unpickles, i.e. it can execute arbitrary code. Only
        load checkpoints this project wrote itself (see docs/reproducibility.md,
        "Checkpoint trust"); callers validate that the path lies inside the run's own
        directory before calling this.
        """
        obj = cls(params, seed, n_jobs)
        obj.clf = joblib.load(path)
        if not isinstance(obj.clf, RandomForestClassifier):
            raise TypeError(f"{path} does not contain a RandomForestClassifier")
        obj.clf.n_jobs = n_jobs
        return obj
