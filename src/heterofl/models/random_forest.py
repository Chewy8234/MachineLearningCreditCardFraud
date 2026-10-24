"""Bank B: scikit-learn RandomForestClassifier."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from heterofl.models.base import FraudModel, check_probabilities

PARAM_KEYS = ("n_estimators", "max_depth", "min_samples_leaf", "min_samples_split",
              "max_features", "bootstrap", "max_samples", "criterion", "class_weight")


class RandomForestModel(FraudModel):
    """Random forest. It cannot be updated incrementally: every ``fit`` call grows a
    completely new forest (the federation therefore *retrains* it each round).

    Sample weights (scikit-learn >= 1.7 semantics, verified for 1.9.1): with
    ``bootstrap=True`` each tree's bootstrap sample is drawn with probability
    proportional to ``sample_weight`` (frequency semantics), not uniformly. When
    weights are given we set ``max_samples=1.0``, so each tree draws
    ``sum(sample_weight)`` rows; for the soft-target set that is
    ``N_local * (1 + lambda_kd)``: in expectation the same number of local draws
    as a local-only forest, plus ``lambda_kd * N_local`` proxy draws whose labels
    follow the teacher probability. ``sample_weight=None`` is the ordinary forest.
    """

    model_type = "random_forest"
    file_suffix = ".joblib"

    def __init__(self, params: dict, seed: int, n_jobs: int = 1):
        self.params = {k: v for k, v in params.items() if k in PARAM_KEYS}
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.clf: RandomForestClassifier | None = None

    def fit(self, X, y, sample_weight=None) -> dict:
        t0 = time.perf_counter()
        params = dict(self.params)
        if sample_weight is not None and params.get("bootstrap", True) \
                and params.get("max_samples") is None:
            params["max_samples"] = 1.0  # draw sum(sample_weight) rows (see class doc)
        self.clf = RandomForestClassifier(random_state=self.seed, n_jobs=self.n_jobs, **params)
        self.clf.fit(X, y, sample_weight=sample_weight)
        if list(self.clf.classes_) != [0, 1]:
            raise ValueError(f"Random forest saw classes {self.clf.classes_}; need both 0 and 1")
        return {"fit_seconds": time.perf_counter() - t0, "n_rows": int(len(y)),
                "n_trees": len(self.clf.estimators_), "update": "retrain"}

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model is not fitted")
        return check_probabilities(self.clf.predict_proba(X)[:, 1], len(X), "rf proba")

    def save(self, path: Path) -> Path:
        path = self._with_suffix(path)
        joblib.dump(self.clf, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path, params: dict, seed: int, n_jobs: int = 1) -> "RandomForestModel":
        obj = cls(params, seed, n_jobs)
        obj.clf = joblib.load(path)
        obj.clf.n_jobs = n_jobs
        return obj
