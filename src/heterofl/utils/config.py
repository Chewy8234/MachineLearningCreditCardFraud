"""YAML experiment configuration: loading, merging, overrides and validation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

VALID_METHODS = ("local", "centralized", "ensemble", "fedavg", "heterogeneous_kd")
VALID_MODEL_TYPES = ("xgboost", "random_forest", "mlp")
VALID_TEACHERS = ("leave_one_out", "full_committee")
VALID_WEIGHTINGS = ("equal", "validation_ap")


class ConfigError(ValueError):
    """Raised for an invalid or inconsistent experiment configuration."""


REMOVE = "__remove__"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where ``override`` is recursively merged into ``base``.

    Lists and scalars are replaced; the value ``"__remove__"`` deletes a key
    (``null`` cannot be used for that, since e.g. ``max_depth: null`` is meaningful).
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if value == REMOVE:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg[a][b][c] = value`` for ``dotted_key='a.b.c'``."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; the value is parsed as YAML (so 0.5, true, [1,2] work)."""
    if "=" not in text:
        raise ConfigError(f"Override must look like key.path=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a profile YAML, merge the referenced models YAML, apply overrides, validate.

    Relative paths inside the config are resolved against the project root (the
    parent of the ``configs/`` directory), so commands work from the repo root.
    """
    path = Path(path).resolve()
    cfg = _load_with_base(path)

    root = path.parent.parent if path.parent.name == "configs" else path.parent
    cfg.setdefault("project_root", str(root))

    models_ref = cfg.get("models_config")
    if models_ref:
        models_path = _resolve(root, models_ref)
        with models_path.open() as fh:
            base_models = yaml.safe_load(fh) or {}
        cfg["models"] = deep_merge(base_models, cfg.get("models", {}))

    for item in overrides or []:
        key, value = parse_override(item)
        set_by_path(cfg, key, value)

    cfg["config_path"] = str(path)
    validate_config(cfg)
    return cfg


def _load_with_base(path: Path, depth: int = 0) -> dict:
    """Load YAML; a ``base_config`` key (relative to this file) is merged underneath."""
    if depth > 5:
        raise ConfigError("base_config chain is too deep (cycle?)")
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base_config", None)
    if base:
        cfg = deep_merge(_load_with_base((path.parent / base).resolve(), depth + 1), cfg)
    return cfg


def _resolve(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (root / p).resolve()


def resolve_path(cfg: dict, value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    return _resolve(Path(cfg["project_root"]), value)


def validate_config(cfg: dict) -> None:
    """Fail early with a clear message instead of deep inside an experiment."""
    required = ["profile", "results_dir", "data", "split", "seeds", "scenarios",
                "partitioning", "clients", "methods", "models", "federation",
                "fedavg", "kd", "ensemble", "evaluation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"Config is missing required sections: {missing}")

    split = cfg["split"]
    for key in ("test_fraction", "validation_fraction", "proxy_fraction"):
        frac = split.get(key)
        if not isinstance(frac, (int, float)) or not 0.0 < frac < 1.0:
            raise ConfigError(f"split.{key} must be in (0, 1), got {frac!r}")
    if split.get("proxy_sampling", "stratified") not in ("stratified", "random"):
        raise ConfigError("split.proxy_sampling must be 'stratified' or 'random'")

    clients = cfg["clients"]
    if len(clients) != cfg["partitioning"]["num_clients"]:
        raise ConfigError("len(clients) must equal partitioning.num_clients")
    for c in clients:
        if c.get("model") not in VALID_MODEL_TYPES:
            raise ConfigError(f"Unknown client model type {c.get('model')!r}")
    if len({c["bank"] for c in clients}) != len(clients):
        raise ConfigError("Client bank identifiers must be unique")

    for m in cfg["methods"]:
        if m not in VALID_METHODS:
            raise ConfigError(f"Unknown method {m!r}; valid: {VALID_METHODS}")
    if "ensemble" in cfg["methods"] and "local" not in cfg["methods"]:
        raise ConfigError("'ensemble' reuses the local-only models; include 'local'")

    for name, sc in cfg["scenarios"].items():
        if sc.get("type") not in ("iid", "dirichlet"):
            raise ConfigError(f"Scenario {name}: type must be 'iid' or 'dirichlet'")
        if sc["type"] == "dirichlet" and not float(sc.get("alpha", 0)) > 0:
            raise ConfigError(f"Scenario {name}: dirichlet alpha must be > 0")

    _validate_kd(cfg["kd"], "kd")
    if cfg["ensemble"].get("weighting", "equal") not in VALID_WEIGHTINGS:
        raise ConfigError(f"ensemble.weighting must be one of {VALID_WEIGHTINGS}")
    if cfg["evaluation"].get("model_selection", "final_round") not in (
        "final_round", "best_validation_ap"
    ):
        raise ConfigError("evaluation.model_selection must be final_round|best_validation_ap")
    if int(cfg["federation"]["num_rounds"]) < 1:
        raise ConfigError("federation.num_rounds must be >= 1")

    for name, ab in (cfg.get("ablations", {}) or {}).get("runs", {}).items():
        if ab.get("method") not in ("heterogeneous_kd", "ensemble"):
            raise ConfigError(f"Ablation {name}: method must be heterogeneous_kd or ensemble")
        merged = deep_merge(cfg, ab.get("overrides", {}))
        _validate_kd(merged["kd"], f"ablations.runs.{name}.kd")


def _validate_kd(kd: dict, where: str) -> None:
    if float(kd.get("lambda_kd", -1)) < 0:
        raise ConfigError(f"{where}.lambda_kd must be >= 0")
    if kd.get("teacher") not in VALID_TEACHERS:
        raise ConfigError(f"{where}.teacher must be one of {VALID_TEACHERS}")
    if kd.get("teacher_weighting") not in VALID_WEIGHTINGS:
        raise ConfigError(f"{where}.teacher_weighting must be one of {VALID_WEIGHTINGS}")
    if kd.get("tree_update") not in ("retrain", "continue"):
        raise ConfigError(f"{where}.tree_update must be retrain|continue")
    if not 0 < float(kd.get("prob_clip", 0)) < 0.5:
        raise ConfigError(f"{where}.prob_clip must be in (0, 0.5)")


def dump_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
