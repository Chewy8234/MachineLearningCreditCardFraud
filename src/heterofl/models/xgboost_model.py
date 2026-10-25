"""Bank A: gradient-boosted trees (XGBoost, scikit-learn API)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from heterofl.models.base import (
    FraudModel,
    check_probabilities,
    int_range,
    one_of,
    real_range,
    validate_params,
)

#: Supported hyperparameters. ``None`` means "library default" and is dropped.
PARAM_SPECS = {
    "n_estimators": int_range(1, allow_none=True),
    "max_depth": int_range(0, allow_none=True),
    "learning_rate": real_range(0, low_open=True, allow_none=True),
    "subsample": real_range(0, 1, low_open=True, allow_none=True),
    "colsample_bytree": real_range(0, 1, low_open=True, allow_none=True),
    "min_child_weight": real_range(0, allow_none=True),
    "reg_lambda": real_range(0, allow_none=True),
    "reg_alpha": real_range(0, allow_none=True),
    "gamma": real_range(0, allow_none=True),
    "max_bin": int_range(2, allow_none=True),
    "tree_method": one_of("hist", "approx", "exact", "auto", allow_none=True),
    "scale_pos_weight": real_range(0, low_open=True, allow_none=True),
}
PARAM_KEYS = tuple(PARAM_SPECS)


class XGBoostModel(FraudModel):
    """``binary:logistic`` XGBoost. ``sample_weight`` scales each row's log-loss term,
    which is what makes the weighted soft-target expansion an exact soft cross-entropy
    (see ``heterofl.distillation.soft_targets``)."""

    model_type = "xgboost"
    file_suffix = ".json"

    @staticmethod
    def validate_params(params: dict) -> None:
        validate_params("xgboost", params, PARAM_SPECS)

    def __init__(self, params: dict, seed: int, n_jobs: int = 1):
        self.validate_params(params)
        self.params = {k: v for k, v in params.items() if v is not None}
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.clf: xgb.XGBClassifier | None = None

    def _new(self, **override) -> xgb.XGBClassifier:
        kw = {"objective": "binary:logistic", "eval_metric": "logloss",
              "random_state": self.seed, "n_jobs": self.n_jobs, **self.params, **override}
        return xgb.XGBClassifier(**kw)

    def fit(self, X, y, sample_weight=None) -> dict:
        """Train from scratch with ``n_estimators`` trees."""
        t0 = time.perf_counter()
        self.clf = self._new()
        self.clf.fit(X, y, sample_weight=sample_weight, verbose=False)
        return {"fit_seconds": time.perf_counter() - t0, "n_rows": int(len(y)),
                "n_trees": int(self.clf.get_booster().num_boosted_rounds()),
                "update": "retrain"}

    def continue_fit(self, X, y, sample_weight=None, extra_trees: int = 50) -> dict:
        """Warm start: append ``extra_trees`` boosting rounds to the existing booster."""
        if self.clf is None:
            raise RuntimeError("continue_fit requires a fitted model")
        t0 = time.perf_counter()
        prev = self.clf.get_booster()
        new = self._new(n_estimators=int(extra_trees))
        new.fit(X, y, sample_weight=sample_weight, xgb_model=prev, verbose=False)
        self.clf = new
        return {"fit_seconds": time.perf_counter() - t0, "n_rows": int(len(y)),
                "n_trees": int(new.get_booster().num_boosted_rounds()),
                "update": "continue"}

    def predict_proba(self, X) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model is not fitted")
        return check_probabilities(self.clf.predict_proba(X)[:, 1], len(X), "xgboost proba")

    def save(self, path: Path) -> Path:
        if self.clf is None:
            raise RuntimeError("Cannot save an unfitted XGBoost model")
        path = self._with_suffix(path)
        self.clf.save_model(path)
        return path

    @classmethod
    def load(cls, path: Path, params: dict, seed: int, n_jobs: int = 1) -> "XGBoostModel":
        """Load a native XGBoost JSON model (a data format, not executable code)."""
        obj = cls(params, seed, n_jobs)
        obj.clf = obj._new()
        obj.clf.load_model(path)
        return obj
