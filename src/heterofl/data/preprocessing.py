"""Shared feature preprocessing.

One transformation is used by *every* model and *every* experimental condition:

1. ``Amount -> log1p(Amount)`` (a fixed, data-independent transform), then
2. per-feature standardisation with mean/std estimated on the **public
   proxy/reference split only** (features, never labels).

Rationale: FedAvg requires identical preprocessing on every client, and the
statistics must not secretly pool private client data. The proxy split is,
by the simulation's assumption, public, so estimating the scaler there is
permitted; validation and test data are never used for fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from heterofl.data.loader import FEATURE_COLUMNS

AMOUNT_IDX = FEATURE_COLUMNS.index("Amount")


@dataclass
class FeaturePreprocessor:
    log_amount: bool = True
    mean_: np.ndarray | None = field(default=None, repr=False)
    scale_: np.ndarray | None = field(default=None, repr=False)
    fitted_on: str | None = None
    n_fit_rows: int | None = None

    def _base(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(f"Expected (n, {len(FEATURE_COLUMNS)}) features, got {X.shape}")
        if self.log_amount:
            X[:, AMOUNT_IDX] = np.log1p(X[:, AMOUNT_IDX])
        return X

    def fit(self, X_reference: np.ndarray, source: str) -> "FeaturePreprocessor":
        """Fit on the permitted reference features; ``source`` is recorded for audit."""
        Xb = self._base(X_reference)
        self.mean_ = Xb.mean(axis=0)
        std = Xb.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        self.fitted_on = source
        self.n_fit_rows = int(len(Xb))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform")
        return ((self._base(X) - self.mean_) / self.scale_).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "log_amount": self.log_amount,
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "fitted_on": self.fitted_on,
            "n_fit_rows": self.n_fit_rows,
            "feature_columns": list(FEATURE_COLUMNS),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeaturePreprocessor":
        obj = cls(log_amount=d["log_amount"], fitted_on=d.get("fitted_on"),
                  n_fit_rows=d.get("n_fit_rows"))
        obj.mean_ = None if d["mean"] is None else np.asarray(d["mean"])
        obj.scale_ = None if d["scale"] is None else np.asarray(d["scale"])
        return obj
