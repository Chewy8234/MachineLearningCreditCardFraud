"""Configuration validation, resolved defaults and deterministic configuration hashing."""

from __future__ import annotations

import copy

import pytest
import yaml

from heterofl.experiments.identity import run_identity
from heterofl.utils.config import (
    ConfigError,
    config_diff,
    config_hash,
    dump_config,
    load_config,
    validate_config,
)

from .conftest import ROOT

SMOKE = ROOT / "configs" / "smoke_synthetic.yaml"


@pytest.mark.parametrize("name", ["paper", "smoke", "smoke_synthetic", "paper_random_proxy"])
def test_shipped_configs_are_valid(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["models"]["xgboost"]["n_estimators"] >= 1


# --------------------------------------------------------------------------- strict keys
@pytest.mark.parametrize("override, fragment", [
    ("models.xgboost.n_estimator=10", "n_estimator"),          # typo of n_estimators
    ("models.random_forest.max_feature=sqrt", "max_feature"),
    ("models.mlp.hidden_size=[8]", "hidden_size"),
    ("kd.lamda_kd=1.0", "lamda_kd"),                             # CLI typo, unknown section key
    ("federation.num_round=3", "num_round"),
    ("evaluation.bootstrap.n_resample=10", "n_resample"),
    ("not_a_section.x=1", "not_a_section"),
])
def test_unknown_keys_are_rejected_not_ignored(override, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_config(SMOKE, [override])


def test_unknown_model_parameter_error_lists_valid_names():
    with pytest.raises(ConfigError) as exc:
        load_config(SMOKE, ["models.xgboost.n_estimator=10"])
    assert "n_estimators" in str(exc.value) and "valid names" in str(exc.value)


@pytest.mark.parametrize("override", [
    "models.xgboost.learning_rate=-0.1", "models.xgboost.subsample=1.5",
    "models.xgboost.n_estimators=0", "models.xgboost.tree_method=fast",
    "models.random_forest.max_depth=0", "models.random_forest.criterion=mse",
    "models.random_forest.bootstrap=maybe", "models.mlp.dropout=1.0",
    "models.mlp.hidden_sizes=[16, 0]", "models.mlp.batch_size=0", "models.mlp.epochs=2.5",
    "kd.lambda_kd=-1", "kd.prob_clip=0.6", "kd.mlp_epochs_per_round=0",
    "federation.num_rounds=0", "split.test_fraction=1.2", "fedavg.local_epochs=0",
    "kd.prob_clip='0.001'",                                      # numeric strings are not numbers
])
def test_out_of_range_or_mistyped_values_are_rejected(override):
    with pytest.raises(ConfigError):
        load_config(SMOKE, [override])


def test_missing_model_config_and_required_parameter():
    cfg = load_config(SMOKE)
    bad = copy.deepcopy(cfg)
    del bad["models"]["random_forest"]
    with pytest.raises(ConfigError, match="random_forest"):
        validate_config(bad)
    bad = copy.deepcopy(cfg)
    del bad["models"]["mlp"]["hidden_sizes"]
    with pytest.raises(ConfigError, match="hidden_sizes"):
        validate_config(bad)
    bad = copy.deepcopy(cfg)
    bad["clients"][0]["model"] = "lightgbm"
    with pytest.raises(ConfigError, match="lightgbm"):
        validate_config(bad)


def test_rf_continue_policy_rejected_at_load_time():
    with pytest.raises(ConfigError, match="Random Forest"):
        load_config(SMOKE, ["kd.tree_update=continue"])


def test_ablation_overrides_are_validated():
    with pytest.raises(ConfigError, match="lamda_kd"):
        load_config(SMOKE, ["ablations.runs.bad.method=heterogeneous_kd",
                            "ablations.runs.bad.overrides.kd.lamda_kd=0.2"])


def test_exponent_override_is_a_float():
    assert load_config(SMOKE, ["kd.prob_clip=1e-6"])["kd"]["prob_clip"] == pytest.approx(1e-6)


def test_models_config_override_takes_effect(tmp_path):
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    models["xgboost"]["max_depth"] = 2
    alt = tmp_path / "models_alt.yaml"
    alt.write_text(yaml.safe_dump(models), encoding="utf-8")
    cfg = load_config(SMOKE, [f"models_config={alt}"])
    # the profile's own models: section still overrides the file (smoke sets depth 4)
    assert cfg["models"]["xgboost"]["max_depth"] == 4
    assert cfg["models_config"] == str(alt)
    models["xgboost"]["gamma"] = 0.5
    alt.write_text(yaml.safe_dump(models), encoding="utf-8")
    assert load_config(SMOKE, [f"models_config={alt}"])["models"]["xgboost"]["gamma"] == 0.5


# --------------------------------------------------------------------------- defaults + hash
def test_defaults_are_materialised():
    cfg = load_config(SMOKE)
    assert cfg["ensemble"]["weighting"] == "equal"
    assert cfg["evaluation"]["threshold_rule"] == "max_f1"
    assert cfg["models"]["mlp"]["torch_threads"] == 1
    assert cfg["federation"]["job_timeout"] == 1800  # explicit value kept


def _write_profile(tmp_path, name: str, body: dict) -> str:
    d = tmp_path / "configs"
    d.mkdir(exist_ok=True)
    (d / "models.yaml").write_text((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"),
                                  encoding="utf-8")
    p = d / name
    p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return str(p)


def test_equivalent_configurations_hash_identically(tmp_path):
    flat = load_config(ROOT / "configs" / "paper.yaml")
    body = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    # (1) same content, keys in a different order, (2) defaults spelled out explicitly
    reordered = dict(reversed(list(body.items())))
    reordered["kd"] = dict(reversed(list(reordered["kd"].items())))
    explicit = copy.deepcopy(body)
    explicit["ensemble"]["weighting"] = "equal"
    explicit["evaluation"]["threshold_rule"] = "max_f1"
    explicit["kd"]["xgb_trees_per_round"] = 50
    explicit["publishable"] = True
    hashes = {config_hash(load_config(_write_profile(tmp_path, f"{n}.yaml", b)))
              for n, b in (("reordered", reordered), ("explicit", explicit))}
    # (3) the same experiment expressed through base_config inheritance
    _write_profile(tmp_path, "paper.yaml", body)
    child = _write_profile(tmp_path, "child.yaml", {"base_config": "paper.yaml",
                                                    "kd": {"lambda_kd": 0.5}})
    hashes.add(config_hash(load_config(child)))
    # project_root/config_path differ (tmp vs repo) but are non-semantic
    hashes.add(config_hash(flat))
    assert len(hashes) == 1


@pytest.mark.parametrize("override", [
    "kd.lambda_kd=1.0", "models.xgboost.max_depth=5", "models.random_forest.n_estimators=41",
    "models.mlp.learning_rate=0.002", "kd.teacher_weighting=validation_ap",
    "kd.teacher=full_committee", "ensemble.weighting=validation_ap", "federation.num_rounds=3",
    "split.seed=7", "partitioning.min_fraud_per_client=6", "data.subsample_fraction=0.5",
    "kd.tree_update=retrain", "publishable=true",
])
def test_semantic_changes_change_the_hash(override):
    base = load_config(SMOKE)
    changed = load_config(SMOKE, [override])
    if override == "kd.tree_update=retrain":   # already the value: must NOT change
        assert config_hash(changed) == config_hash(base)
    else:
        assert config_hash(changed) != config_hash(base)
        assert config_diff(base, changed)


@pytest.mark.parametrize("override", [
    "results_dir=elsewhere", "data.processed_dir=/tmp/x", "data.partitions_dir=/tmp/y",
    "federation.job_timeout=99", "federation.round_timeout=5", "report.auto=false",
    "evaluation.bootstrap.n_resamples=7", "seeds=[42, 123]", "methods=[local]",
    "debug.fail.partition_id=1",
])
def test_non_semantic_changes_do_not_change_the_hash(override):
    extra = ["debug.fail.round=0"] if override.startswith("debug") else []
    assert config_hash(load_config(SMOKE, [override, *extra])) == config_hash(load_config(SMOKE))


def test_run_identity_covers_seed_and_scenario_definition():
    cfg = load_config(SMOKE)
    kw = dict(method="local", scenario="moderate_non_iid", variant="primary",
              dataset_sha256="d", packages={}, partition_sha256="p")
    a = run_identity(cfg, seed=42, **kw)
    assert run_identity(cfg, seed=42, **kw) == a
    assert run_identity(cfg, seed=43, **kw) != a
    alt = load_config(SMOKE, ["scenarios.moderate_non_iid.alpha=0.3"])
    assert config_hash(alt) == config_hash(cfg)       # scenarios only select runs ...
    assert run_identity(alt, seed=42, **kw) != a      # ... but the run's own scenario counts
    sel = load_config(SMOKE, ["scenarios.moderate_non_iid.methods=[local]"])
    assert run_identity(sel, seed=42, **kw) == a      # a method filter is not semantic


def test_dump_config_is_atomic(tmp_path):
    cfg = load_config(SMOKE)
    target = tmp_path / "config.yaml"
    dump_config(cfg, target)
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["profile"] == cfg["profile"]
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]  # no temp files left
