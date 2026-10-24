"""Common interface for the three model families."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

#: A parameter check returns an error message, or ``None`` if the value is valid.
ParamCheck = Callable[[Any], "str | None"]


def int_range(low: int, high: int | None = None, allow_none: bool = False) -> ParamCheck:
    def check(v):
        if v is None and allow_none:
            return None
        if isinstance(v, bool) or not isinstance(v, Integral) or v < low or (
                high is not None and v > high):
            return f"must be an integer in [{low}, {high if high is not None else 'inf'}]"
        return None
    return check


def real_range(low: float, high: float | None = None, *, low_open: bool = False,
               high_open: bool = False, allow_none: bool = False) -> ParamCheck:
    def check(v):
        if v is None and allow_none:
            return None
        if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v):
            return "must be a finite number"
        if v < low or (low_open and v == low) or (high is not None and (
                v > high or (high_open and v == high))):
            return (f"must be in {'(' if low_open else '['}{low}, "
                    f"{high if high is not None else 'inf'}{')' if high_open or high is None else ']'}")
        return None
    return check


def one_of(*values, allow_none: bool = False) -> ParamCheck:
    def check(v):
        if (v is None and allow_none) or v in values:
            return None
        return f"must be one of {list(values)}"
    return check


def any_of(*checks: ParamCheck) -> ParamCheck:
    def check(v):
        errors = [c(v) for c in checks]
        return None if any(e is None for e in errors) else " or ".join(e for e in errors if e)
    return check


def validate_params(model_type: str, params: dict, specs: dict[str, ParamCheck],
                    required: tuple[str, ...] = ()) -> None:
    """Reject unknown, missing or out-of-range hyperparameters with a descriptive ``ValueError``.

    A typo (``n_estimator``) must stop the experiment instead of silently falling
    back to the library default.
    """
    if not isinstance(params, dict):
        raise ValueError(f"{model_type} hyperparameters must be a mapping, got {type(params).__name__}")
    unknown = sorted(set(params) - set(specs))
    if unknown:
        raise ValueError(f"Unknown {model_type} hyperparameter(s) {unknown}; "
                         f"valid names: {sorted(specs)}")
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(f"Missing required {model_type} hyperparameter(s) {missing}")
    for key, value in params.items():
        err = specs[key](value)
        if err:
            raise ValueError(f"{model_type}.{key}={value!r} {err}")


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
