"""Construct and load models by type name."""

from __future__ import annotations

from pathlib import Path

from heterofl.models.base import FraudModel
from heterofl.models.mlp import MLPModel
from heterofl.models.random_forest import RandomForestModel
from heterofl.models.xgboost_model import XGBoostModel
from heterofl.utils.reproducibility import derive_seed

MODEL_CLASSES: dict[str, type[FraudModel]] = {
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "mlp": MLPModel,
}


def model_seed(run_seed: int, role: str, model_type: str) -> int:
    """Seed for a model instance. ``role`` is e.g. ``bank_A`` or ``centralized``.

    The local-only model of bank A and the round-0 model of bank A inside the
    federated KD run use the same role, hence identical seeds and training.
    """
    return derive_seed(run_seed, "model", role, model_type)


def validate_model_params(model_type: str, params: dict) -> None:
    """Raise ``ValueError`` for an unknown model type or invalid hyperparameters."""
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type {model_type!r}; valid: {sorted(MODEL_CLASSES)}")
    MODEL_CLASSES[model_type].validate_params(params)


def create_model(model_type: str, models_cfg: dict, seed: int, n_jobs: int = 1,
                 n_features: int = 30) -> FraudModel:
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type {model_type!r}; valid: {sorted(MODEL_CLASSES)}")
    if model_type not in models_cfg:
        raise ValueError(f"No hyperparameters configured for model type {model_type!r}")
    params = models_cfg[model_type]
    if model_type == "mlp":
        return MLPModel(params, seed, n_features=n_features, n_jobs=n_jobs)
    if model_type in MODEL_CLASSES:
        return MODEL_CLASSES[model_type](params, seed, n_jobs)
    raise ValueError(f"Unknown model type {model_type!r}")


def load_model(model_type: str, path: Path, models_cfg: dict, seed: int,
               n_jobs: int = 1) -> FraudModel:
    """Load a trusted, self-written checkpoint (see docs/reproducibility.md)."""
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type {model_type!r}; valid: {sorted(MODEL_CLASSES)}")
    cls = MODEL_CLASSES[model_type]
    if Path(path).suffix != cls.file_suffix:
        raise ValueError(f"{path}: expected a {cls.file_suffix} checkpoint for {model_type}")
    return cls.load(Path(path), models_cfg.get(model_type, {}), seed, n_jobs)


def model_path(base: Path, model_type: str) -> Path:
    return Path(base).with_suffix(MODEL_CLASSES[model_type].file_suffix)
