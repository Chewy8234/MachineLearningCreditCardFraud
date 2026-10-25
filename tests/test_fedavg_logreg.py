"""FedAvg over logistic regression (method ``fedavg_logreg``).

Tests marked ``slow`` launch real Flower simulations (Ray) on synthetic data.
"""

from __future__ import annotations

import pytest

from heterofl.experiments import homogeneous
from heterofl.federation.fedavg import fedavg_mlp_params
from heterofl.federation.job import FederatedJob
from heterofl.models.mlp import MLPModel
from heterofl.utils.config import ConfigError, load_config
from heterofl.utils.io import read_json

from .conftest import ROOT
from .test_federation import _run_ctx


def _job(tmp_path, fedavg: dict) -> FederatedJob:
    return FederatedJob(mode="fedavg", job_dir=str(tmp_path), view_dir=str(tmp_path), seed=1,
                        num_rounds=1,
                        clients=[{"partition_id": 0, "bank": "A", "model_type": "xgboost"}],
                        models={"mlp": {"hidden_sizes": [64, 32], "dropout": 0.1,
                                        "learning_rate": 1e-3, "batch_size": 64, "epochs": 1}},
                        fedavg=fedavg, kd={}, expected_proxy_sha256="x")


def test_logistic_regression_is_an_mlp_without_hidden_layers(tmp_path):
    params = fedavg_mlp_params(_job(tmp_path, {"model": "logistic_regression"}))
    assert params["hidden_sizes"] == [] and params["dropout"] == 0.0
    assert MLPModel(params, seed=1, n_features=30).num_parameters() == 31  # 30 coef + intercept
    # the default (and an unset model) is the unchanged FedAvg MLP
    assert fedavg_mlp_params(_job(tmp_path, {}))["hidden_sizes"] == [64, 32]
    with pytest.raises(ValueError, match="fedavg.model"):
        fedavg_mlp_params(_job(tmp_path, {"model": "svm"}))


def test_profile_differs_from_paper_only_in_methods():
    paper = load_config(ROOT / "configs" / "paper.yaml")
    lr = load_config(ROOT / "configs" / "fedavg_logreg.yaml")
    assert lr["methods"] == ["fedavg_logreg"]
    for key in ("data", "split", "seeds", "partitioning", "clients", "models", "federation",
                "fedavg", "evaluation"):
        assert lr[key] == paper[key], key


def test_unknown_fedavg_model_is_rejected():
    with pytest.raises(ConfigError, match="fedavg.model"):
        load_config(ROOT / "configs" / "smoke.yaml", ["fedavg.model=svm"])


@pytest.mark.slow
def test_fedavg_logreg_run_trains_and_evaluates_a_linear_model(prepared, tmp_path):
    ctx = _run_ctx(prepared, tmp_path, "fedavg_logreg")
    homogeneous.run(ctx)
    server = ctx.out_dir / "federation" / "server"
    assert read_json(server / "status.json")["status"] == "completed"
    summ = read_json(server / "rounds.json")
    assert summ["fedavg_model"] == "logistic_regression"
    assert summ["num_parameters"] == 31
    assert [r["system"] for r in ctx.rows] == ["fedavg_logreg_global"]
    assert ctx.rows[0]["model_type"] == "logistic_regression"
    assert 0.0 <= ctx.rows[0]["test_average_precision"] <= 1.0
    model = MLPModel.load(server / "global_round_01" / "model.pt")
    assert model.num_parameters() == 31
