"""Common interface for the three model families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class FraudModel(ABC):
    """A binary fraud classifier returning P(fraud) as a 1-D float array."""

    model_type: str = ""
    file_suffix: str = ""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> dict:
        """Train from scratch (or as documented by the subclass); return diagnostics."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class = 1) for each row, shape ``(n,)``, values in [0, 1]."""

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Persist to ``path`` (suffix added if missing) and return the final path."""

    def _with_suffix(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path if path.suffix == self.file_suffix else path.with_suffix(self.file_suffix)


def check_probabilities(p: np.ndarray, n: int | None = None, name: str = "probabilities") -> np.ndarray:
    """Validate a probability vector: 1-D, expected length, finite, within [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {p.shape}")
    if n is not None and p.shape[0] != n:
        raise ValueError(f"{name} has length {p.shape[0]}, expected {n}")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains non-finite values")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError(f"{name} outside [0, 1]: min={p.min()}, max={p.max()}")
    return p
