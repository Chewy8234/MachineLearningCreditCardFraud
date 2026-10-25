"""YAML experiment configuration: loading, merging, overrides, validation and hashing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

VALID_METHODS = ("local", "centralized", "ensemble", "fedavg", "heterogeneous_kd",
                 "fedxgb_bagging")
ABLATION_METHODS = ("heterogeneous_kd", "ensemble", "fedavg")
VALID_MODEL_TYPES = ("xgboost", "random_forest", "mlp")
VALID_TEACHERS = ("leave_one_out", "full_committee")
VALID_WEIGHTINGS = ("equal", "validation_ap")


class ConfigError(ValueError):
    """Raised for an invalid or inconsistent experiment configuration."""


REMOVE = "__remove__"

# --------------------------------------------------------------------------- schema
#: Every key the code reads, per section. Anything else is a typo or an unsupported
#: setting and is rejected, so it can never be silently ignored.
TOP_LEVEL_KEYS = frozenset({
    "profile", "publishable", "results_dir", "models_config", "data", "split",
    "preprocessing", "seeds", "scenarios", "partitioning", "clients", "methods", "models",
    "federation", "fedavg", "fedxgb", "kd", "ensemble", "evaluation", "ablations", "report",
    "debug",
    "project_root", "config_path",
})
SECTION_KEYS = {
    "data": {"path", "processed_dir", "partitions_dir", "synthetic", "subsample_fraction",
             "subsample_seed", "drop_duplicates"},
    "split": {"seed", "test_fraction", "validation_fraction", "proxy_fraction", "proxy_sampling"},
    "preprocessing": {"log_amount"},
    "partitioning": {"num_clients", "min_fraud_per_client", "min_legit_per_client", "max_retries"},
    "federation": {"num_rounds", "client_num_cpus", "client_num_gpus", "round_timeout",
                   "job_timeout"},
    "fedavg": {"local_epochs"},
    "fedxgb": {"trees_per_round"},   # optional: only read by the fedxgb_bagging method
    "kd": {"lambda_kd", "teacher", "teacher_weighting", "prob_clip", "tree_update",
           "xgb_trees_per_round", "mlp_epochs_per_round", "proxy_batch_size"},
    "ensemble": {"weighting"},
    "evaluation": {"threshold_rule", "fixed_fpr_targets", "model_selection", "bootstrap"},
    "report": {"auto", "bootstrap_seed_for_curves"},
    "ablations": {"enabled", "seeds", "scenarios", "runs"},
    "debug": {"fail"},
}
SCENARIO_KEYS = {"type", "alpha", "ablation", "methods"}
CLIENT_KEYS = {"bank", "model"}
ABLATION_RUN_KEYS = {"method", "overrides", "seeds", "scenarios"}
BOOTSTRAP_KEYS = {"n_resamples", "confidence", "seed"}

#: Defaults the code applies when a key is absent. They are written into the loaded
#: config, so the effective configuration (and its hash) is the same whether a
#: default is spelled out in YAML or left implicit.
DEFAULTS: dict[str, Any] = {
    "publishable": False,
    "data": {"synthetic": False, "subsample_fraction": None, "subsample_seed": 0,
             "drop_duplicates": False},
    "split": {"proxy_sampling": "stratified"},
    "preprocessing": {"log_amount": True},
    "federation": {"client_num_gpus": 0.0, "round_timeout": 3600, "job_timeout": None},
    "kd": {"tree_update": "retrain", "xgb_trees_per_round": 50, "proxy_batch_size": None},
    "ensemble": {"weighting": "equal"},
    "evaluation": {"threshold_rule": "max_f1", "fixed_fpr_targets": [0.001, 0.01],
                   "model_selection": "final_round",
                   "bootstrap": {"n_resamples": 1000, "confidence": 0.95, "seed": 0}},
    "report": {"auto": True},
    "ablations": {"enabled": False, "runs": {}},
    "debug": {},
}
MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "mlp": {"dropout": 0.0, "weight_decay": 0.0, "device": "cpu", "torch_threads": 1},
}


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


_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; the value is parsed as YAML (so 0.5, true, [1,2] work).

    Exponent notation without a decimal point (``1e-6``) is a string in YAML 1.1;
    it is converted to a float here so ``--set kd.prob_clip=1e-6`` works as expected.
    """
    if "=" not in text:
        raise ConfigError(f"Override must look like key.path=value, got {text!r}")
    key, raw = text.split("=", 1)
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _FLOAT_RE.match(value.strip()):
        value = float(value)
    return key.strip(), value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a profile YAML, merge the referenced models YAML, apply overrides, validate.

    Relative paths inside the config are resolved against the project root (the
    parent of the ``configs/`` directory), so commands work from the repo root.
    Command-line overrides are validated exactly like YAML keys (an unknown key
    such as ``kd.lamda_kd`` is an error, not a new, ignored setting).
    """
    path = Path(path).resolve()
    cfg = _load_with_base(path)

    root = path.parent.parent if path.parent.name == "configs" else path.parent
    cfg.setdefault("project_root", str(root))

    parsed = [parse_override(item) for item in overrides or []]
    # an overridden models file must be in place before it is merged
    for key, value in parsed:
        if key == "models_config":
            cfg["models_config"] = value
    models_ref = cfg.get("models_config")
    if models_ref:
        models_path = _resolve(root, models_ref)
        with models_path.open() as fh:
            base_models = yaml.safe_load(fh) or {}
        cfg["models"] = deep_merge(base_models, cfg.get("models", {}))

    for key, value in parsed:
        if key != "models_config":
            set_by_path(cfg, key, value)

    cfg["config_path"] = str(path)
    apply_defaults(cfg)
    validate_config(cfg)
    return cfg


def apply_defaults(cfg: dict) -> dict:
    """Write the code's implicit defaults into ``cfg`` (in place) and return it."""
    for key, default in DEFAULTS.items():
        if isinstance(default, dict):
            node = cfg.setdefault(key, {})
            if isinstance(node, dict):
                cfg[key] = deep_merge(default, node)
        else:
            cfg.setdefault(key, default)
    models = cfg.get("models")
    if isinstance(models, dict):
        for mtype, default in MODEL_DEFAULTS.items():
            if isinstance(models.get(mtype), dict):
                models[mtype] = {**default, **models[mtype]}
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


# --------------------------------------------------------------------------- validation
def _unknown(keys, allowed, where: str) -> None:
    bad = sorted(set(keys) - set(allowed))
    if bad:
        raise ConfigError(f"Unknown key(s) {bad} in {where}; valid keys: {sorted(allowed)}")


def _number(value, where: str, *, integer: bool = False, low=None, high=None,
            low_open: bool = False, high_open: bool = False, allow_none: bool = False):
    """Type- and range-check a numeric setting (bool and numeric strings are rejected)."""
    if value is None and allow_none:
        return
    kind = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ConfigError(f"{where} must be {'an integer' if integer else 'a number'}, "
                          f"got {value!r} ({type(value).__name__})")
    if not math.isfinite(float(value)):
        raise ConfigError(f"{where} must be finite, got {value!r}")
    if low is not None and (value < low or (low_open and value == low)):
        raise ConfigError(f"{where} must be {'>' if low_open else '>='} {low}, got {value!r}")
    if high is not None and (value > high or (high_open and value == high)):
        raise ConfigError(f"{where} must be {'<' if high_open else '<='} {high}, got {value!r}")


def validate_config(cfg: dict) -> None:
    """Fail early with a clear message instead of deep inside an experiment."""
    required = ["profile", "results_dir", "data", "split", "seeds", "scenarios",
                "partitioning", "clients", "methods", "models", "federation",
                "fedavg", "kd", "ensemble", "evaluation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"Config is missing required sections: {missing}")
    _unknown(cfg, TOP_LEVEL_KEYS, "the top level")
    for section, allowed in SECTION_KEYS.items():
        if section in cfg:
            if not isinstance(cfg[section], dict):
                raise ConfigError(f"Section {section!r} must be a mapping")
            _unknown(cfg[section], allowed, f"section {section!r}")
    _unknown(cfg["evaluation"].get("bootstrap") or {}, BOOTSTRAP_KEYS, "evaluation.bootstrap")

    split = cfg["split"]
    for key in ("test_fraction", "validation_fraction", "proxy_fraction"):
        frac = split.get(key)
        if isinstance(frac, bool) or not isinstance(frac, (int, float)) or not 0.0 < frac < 1.0:
            raise ConfigError(f"split.{key} must be in (0, 1), got {frac!r}")
    _number(split.get("seed"), "split.seed", integer=True, low=0)
    if split.get("proxy_sampling", "stratified") not in ("stratified", "random"):
        raise ConfigError("split.proxy_sampling must be 'stratified' or 'random'")
    _number(cfg["data"].get("subsample_fraction"), "data.subsample_fraction", low=0,
            low_open=True, high=1, allow_none=True)

    if not isinstance(cfg["seeds"], list) or not cfg["seeds"]:
        raise ConfigError("seeds must be a non-empty list of integers")
    for s in cfg["seeds"]:
        _number(s, "seeds[]", integer=True, low=0)

    part = cfg["partitioning"]
    for key in ("num_clients", "min_fraud_per_client", "min_legit_per_client", "max_retries"):
        _number(part.get(key), f"partitioning.{key}", integer=True,
                low=1 if key in ("num_clients", "max_retries") else 0)

    clients = cfg["clients"]
    if len(clients) != part["num_clients"]:
        raise ConfigError("len(clients) must equal partitioning.num_clients")
    for c in clients:
        _unknown(c, CLIENT_KEYS, f"client {c}")
        if c.get("model") not in VALID_MODEL_TYPES:
            raise ConfigError(f"Unknown client model type {c.get('model')!r}; "
                              f"valid: {VALID_MODEL_TYPES}")
    if len({c["bank"] for c in clients}) != len(clients):
        raise ConfigError("Client bank identifiers must be unique")

    for m in cfg["methods"]:
        if m not in VALID_METHODS:
            raise ConfigError(f"Unknown method {m!r}; valid: {VALID_METHODS}")
    if "ensemble" in cfg["methods"] and "local" not in cfg["methods"]:
        raise ConfigError("'ensemble' reuses the local-only models; include 'local'")

    for name, sc in cfg["scenarios"].items():
        if not isinstance(sc, dict):
            raise ConfigError(f"Scenario {name} must be a mapping")
        _unknown(sc, SCENARIO_KEYS, f"scenario {name!r}")
        if sc.get("type") not in ("iid", "dirichlet"):
            raise ConfigError(f"Scenario {name}: type must be 'iid' or 'dirichlet'")
        if sc["type"] == "dirichlet":
            _number(sc.get("alpha"), f"scenario {name}: dirichlet alpha", low=0, low_open=True)
        for m in sc.get("methods", []) or []:
            if m not in VALID_METHODS:
                raise ConfigError(f"Scenario {name}: unknown method {m!r}")

    fed = cfg["federation"]
    _number(fed.get("num_rounds"), "federation.num_rounds", integer=True, low=1)
    _number(fed.get("client_num_cpus"), "federation.client_num_cpus", integer=True, low=1)
    _number(fed.get("client_num_gpus"), "federation.client_num_gpus", low=0)
    _number(fed.get("round_timeout"), "federation.round_timeout", low=0, low_open=True)
    _number(fed.get("job_timeout"), "federation.job_timeout", low=0, low_open=True,
            allow_none=True)
    _number(cfg["fedavg"].get("local_epochs"), "fedavg.local_epochs", integer=True, low=1)
    if "fedxgb_bagging" in cfg["methods"]:
        _number((cfg.get("fedxgb") or {}).get("trees_per_round"), "fedxgb.trees_per_round",
                integer=True, low=1)

    validate_models(cfg)
    _validate_kd(cfg["kd"], "kd", cfg)
    if cfg["ensemble"].get("weighting", "equal") not in VALID_WEIGHTINGS:
        raise ConfigError(f"ensemble.weighting must be one of {VALID_WEIGHTINGS}")
    ev = cfg["evaluation"]
    if ev.get("threshold_rule", "max_f1") != "max_f1":
        raise ConfigError("evaluation.threshold_rule must be 'max_f1'")
    if ev.get("model_selection", "final_round") not in ("final_round", "best_validation_ap"):
        raise ConfigError("evaluation.model_selection must be final_round|best_validation_ap")
    for t in ev.get("fixed_fpr_targets", []):
        _number(t, "evaluation.fixed_fpr_targets[]", low=0, low_open=True, high=1)
    boot = ev.get("bootstrap") or {}
    _number(boot.get("n_resamples", 1000), "evaluation.bootstrap.n_resamples", integer=True, low=1)
    _number(boot.get("confidence", 0.95), "evaluation.bootstrap.confidence", low=0,
            low_open=True, high=1, high_open=True)

    fail = (cfg.get("debug") or {}).get("fail")
    if fail is not None and (not isinstance(fail, dict) or set(fail) != {"partition_id", "round"}):
        raise ConfigError("debug.fail must be {partition_id: int, round: int}")

    for name, ab in (cfg.get("ablations", {}) or {}).get("runs", {}).items():
        _unknown(ab, ABLATION_RUN_KEYS, f"ablations.runs.{name}")
        if ab.get("method") not in ABLATION_METHODS:
            raise ConfigError(f"Ablation {name}: method must be one of {ABLATION_METHODS}")
        merged = deep_merge(cfg, ab.get("overrides", {}))
        for section in ab.get("overrides", {}):
            if section not in TOP_LEVEL_KEYS:
                raise ConfigError(f"Ablation {name}: unknown override section {section!r}")
            if section in SECTION_KEYS and isinstance(merged[section], dict):
                _unknown(merged[section], SECTION_KEYS[section],
                         f"ablations.runs.{name}.overrides.{section}")
        validate_models(merged, f"ablations.runs.{name}.")
        _validate_kd(merged["kd"], f"ablations.runs.{name}.kd", merged)


def _validate_kd(kd: dict, where: str, cfg: dict) -> None:
    _number(kd.get("lambda_kd"), f"{where}.lambda_kd", low=0)
    if kd.get("teacher") not in VALID_TEACHERS:
        raise ConfigError(f"{where}.teacher must be one of {VALID_TEACHERS}")
    if kd.get("teacher_weighting") not in VALID_WEIGHTINGS:
        raise ConfigError(f"{where}.teacher_weighting must be one of {VALID_WEIGHTINGS}")
    if kd.get("tree_update") not in ("retrain", "continue"):
        raise ConfigError(f"{where}.tree_update must be retrain|continue")
    _number(kd.get("prob_clip"), f"{where}.prob_clip", low=0, low_open=True, high=0.5,
            high_open=True)
    _number(kd.get("mlp_epochs_per_round"), f"{where}.mlp_epochs_per_round", integer=True, low=1)
    _number(kd.get("xgb_trees_per_round"), f"{where}.xgb_trees_per_round", integer=True, low=1)
    _number(kd.get("proxy_batch_size"), f"{where}.proxy_batch_size", integer=True, low=1,
            allow_none=True)
    if kd.get("tree_update") == "continue" and any(
            c.get("model") == "random_forest" for c in cfg.get("clients", [])):
        raise ConfigError(f"{where}.tree_update='continue' is not supported for Random Forest "
                          "clients (a forest can only be retrained)")


def validate_models(cfg: dict, where: str = "") -> None:
    """Every model type in use has a config, and every hyperparameter is known and in range."""
    from heterofl.models.model_factory import validate_model_params  # avoid an import cycle

    models = cfg.get("models")
    if not isinstance(models, dict):
        raise ConfigError(f"{where}models must be a mapping of model type -> hyperparameters")
    _unknown(models, VALID_MODEL_TYPES, f"{where}models")
    needed = {c["model"] for c in cfg["clients"]}
    if "fedavg" in cfg.get("methods", []):
        needed.add("mlp")
    if "fedxgb_bagging" in cfg.get("methods", []):
        needed.add("xgboost")
    missing = sorted(needed - set(models))
    if missing:
        raise ConfigError(f"{where}models has no configuration for {missing} "
                          "(required by the configured clients/methods)")
    for mtype, params in models.items():
        try:
            validate_model_params(mtype, params)
        except ValueError as exc:
            raise ConfigError(f"{where}models.{mtype}: {exc}") from exc


# --------------------------------------------------------------------------- identity
#: Settings that cannot change any experimental output, excluded from the hash:
#:   locations (results/cache directories, the dataset *path* -- its content is
#:   identified by SHA-256 instead), interpreter bookkeeping, timeouts (they can only
#:   make a run fail), report-only settings (``report``, ``evaluation.bootstrap``:
#:   used by the report generator, never by a run) and fault injection (``debug``:
#:   can only make a run fail, never alter a completed one).
NON_SEMANTIC_KEYS: tuple[tuple[str, ...], ...] = (
    ("project_root",), ("config_path",), ("results_dir",), ("models_config",), ("report",),
    ("debug",), ("data", "path"), ("data", "processed_dir"), ("data", "partitions_dir"),
    ("federation", "round_timeout"), ("federation", "job_timeout"),
    ("evaluation", "bootstrap"),
)
#: Settings that only *select* which runs execute. Each run's own seed, scenario
#: definition, method and ablation variant are part of its identity instead.
RUN_SELECTION_KEYS: tuple[str, ...] = ("seeds", "methods", "scenarios", "ablations")


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ConfigError(f"Non-finite value {obj!r} cannot be part of a configuration hash")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_json(obj: Any) -> str:
    """Stable serialisation: sorted keys, no whitespace, numpy scalars as Python values.

    Types are preserved (``1`` and ``1.0`` differ), because some libraries treat them
    differently (scikit-learn ``max_features=1`` vs ``1.0``).
    """
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def semantic_config(cfg: dict, ignore: tuple[str, ...] = ()) -> dict:
    """The part of ``cfg`` that can influence results (see ``NON_SEMANTIC_KEYS``).

    ``ignore`` drops further top-level sections, for dependency checks between runs
    (e.g. the ensemble does not care about ``kd``).
    """
    out = copy.deepcopy(cfg)
    for path in NON_SEMANTIC_KEYS:
        node = out
        for part in path[:-1]:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(path[-1], None)
    for key in (*RUN_SELECTION_KEYS, *ignore):
        out.pop(key, None)
    return _canonical(out)


def config_hash(cfg: dict) -> str:
    """SHA-256 of the canonical semantic configuration."""
    return sha256_json(semantic_config(cfg))


def config_diff(a: Any, b: Any, prefix: str = "") -> list[str]:
    """Dotted keys whose values differ between two nested structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            key = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(key)
            else:
                out += config_diff(a[k], b[k], key)
        return out
    return [] if canonical_json(a) == canonical_json(b) else [prefix or "<root>"]


def dump_config(cfg: dict, path: Path) -> None:
    """Write YAML atomically (temp file + rename), so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
