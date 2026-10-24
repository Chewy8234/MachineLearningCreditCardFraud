"""The three model families: training, probabilities, persistence, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from heterofl.evaluation.metrics import roc_auc
from heterofl.models.base import check_probabilities
from heterofl.models.mlp import MLPModel
from heterofl.models.model_factory import create_model, load_model, model_seed
from heterofl.models.random_forest import RandomForestModel

MODELS = {"xgboost": {"n_estimators": 30, "max_depth": 3, "tree_method": "hist"},
          "random_forest": {"n_estimators": 20, "max_features": "sqrt", "bootstrap": True},
          "mlp": {"hidden_sizes": [16, 8], "dropout": 0.1, "learning_rate": 0.003,
                  "batch_size": 128, "epochs": 15, "device": "cpu"}}


@pytest.mark.parametrize("mtype", list(MODELS))
def test_trains_and_predicts_valid_probabilities(mtype, toy_xy, tmp_path):
    X, y = toy_xy
    m = create_model(mtype, MODELS, seed=1, n_jobs=1)
    info = m.fit(X, y)
    assert info["fit_seconds"] >= 0
    p = m.predict_proba(X)
    check_probabilities(p, len(X))
    assert roc_auc(y, p) > 0.8  # learns the planted signal
    path = m.save(tmp_path / mtype)
    m2 = load_model(mtype, path, MODELS, seed=1)
    np.testing.assert_allclose(m2.predict_proba(X), p, atol=1e-7)


@pytest.mark.parametrize("mtype", list(MODELS))
def test_training_is_deterministic_given_seed(mtype, toy_xy):
    X, y = toy_xy
    a, b = create_model(mtype, MODELS, 3), create_model(mtype, MODELS, 3)
    a.fit(X, y)
    b.fit(X, y)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_check_probabilities_rejects_bad_vectors():
    for bad in (np.array([0.1, np.nan]), np.array([0.1, 1.2]), np.array([[0.1]])):
        with pytest.raises(ValueError):
            check_probabilities(bad)
    with pytest.raises(ValueError):
        check_probabilities(np.array([0.1, 0.2]), n=3)


def test_mlp_state_dict_roundtrip(toy_xy):
    X, y = toy_xy
    a = MLPModel(MODELS["mlp"], 1)
    a.fit(X, y, epochs=2)
    b = MLPModel(MODELS["mlp"], 99)
    b.set_state_dict(a.get_state_dict())
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
    assert a.num_parameters() == sum(v.numel() for v in a.get_state_dict().values())


def test_random_forest_weighted_bootstrap_draws_sum_of_weights(toy_xy):
    X, y = toy_xy
    m = RandomForestModel({"n_estimators": 5}, 1)
    m.fit(X, y, sample_weight=np.full(len(y), 1.5))
    assert m.clf._n_samples_bootstrap == int(1.5 * len(y))
    m.fit(X, y)  # unweighted: ordinary forest
    assert m.clf._n_samples_bootstrap == len(y)


def test_model_seed_shared_between_local_and_kd_round0():
    assert model_seed(42, "bank_A", "xgboost") == model_seed(42, "bank_A", "xgboost")
    assert model_seed(42, "bank_A", "xgboost") != model_seed(123, "bank_A", "xgboost")
    assert model_seed(42, "bank_A", "xgboost") != model_seed(42, "centralized", "xgboost")
