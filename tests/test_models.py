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


# --------------------------------------------------------------------------- strict params
@pytest.mark.parametrize("mtype, bad", [
    ("xgboost", {"n_estimator": 10}), ("xgboost", {"random_state": 1}),
    ("random_forest", {"n_estimator": 10}), ("random_forest", {"n_jobs": 4}),
    ("mlp", {**MODELS["mlp"], "hidden_size": [4]}),
])
def test_unknown_hyperparameters_raise_with_valid_names(mtype, bad):
    with pytest.raises(ValueError, match="valid names"):
        create_model(mtype, {mtype: bad}, seed=1)


def test_missing_required_mlp_parameter():
    params = {k: v for k, v in MODELS["mlp"].items() if k != "epochs"}
    with pytest.raises(ValueError, match="epochs"):
        MLPModel(params, 1)


def test_unknown_model_type_and_missing_config():
    with pytest.raises(ValueError, match="Unknown model type"):
        create_model("lightgbm", MODELS, 1)
    with pytest.raises(ValueError, match="No hyperparameters"):
        create_model("xgboost", {}, 1)


@pytest.mark.parametrize("mtype", ["xgboost", "random_forest"])
def test_saving_an_unfitted_tree_model_raises(mtype, tmp_path):
    with pytest.raises(RuntimeError, match="unfitted"):
        create_model(mtype, MODELS, 1).save(tmp_path / mtype)
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- torch global state
def test_mlp_does_not_touch_global_torch_state(toy_xy, tmp_path, monkeypatch):
    import torch

    X, y = toy_xy
    # never actually raise the thread count here: multi-threaded torch next to XGBoost's
    # OpenMP runtime deadlocks on macOS (the reason for the single-thread policy)
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    torch.manual_seed(1234)
    expected = torch.rand(5)                 # what an unrelated consumer would draw next
    torch.manual_seed(1234)
    m = MLPModel(MODELS["mlp"], 1)            # construction
    m.fit(X, y, epochs=1)                     # training (dropout)
    m.predict_proba(X)                        # evaluation
    MLPModel.load(m.save(tmp_path / "m"))     # checkpoint loading
    assert calls == []                        # no process-wide thread changes
    np.testing.assert_array_equal(torch.rand(5).numpy(), expected.numpy())


def test_mlp_initialisation_matches_the_seeded_global_rng_formula():
    """Same initial weights as before the RNG isolation (seed global RNG, build net)."""
    import torch

    from heterofl.models.mlp import MLPNet
    from heterofl.utils.reproducibility import derive_seed

    m = MLPModel(MODELS["mlp"], 7, n_features=30)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(derive_seed(7, "mlp-init"))
        ref = MLPNet(30, MODELS["mlp"]["hidden_sizes"], MODELS["mlp"]["dropout"])
    for k, v in ref.state_dict().items():
        assert torch.equal(m.net.state_dict()[k], v)


def test_mlp_training_is_independent_of_outside_rng_use(toy_xy):
    import torch

    X, y = toy_xy
    a = MLPModel(MODELS["mlp"], 3)
    torch.manual_seed(0)
    a.fit(X, y, epochs=2)
    b = MLPModel(MODELS["mlp"], 3)
    torch.manual_seed(999)
    torch.rand(1000)                          # unrelated RNG use between runs
    b.fit(X, y, epochs=2)
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


# --------------------------------------------------------------------------- checkpoint trust
def test_mlp_load_refuses_pickled_code(tmp_path):
    import pickle

    import torch

    class Evil:
        def __reduce__(self):
            return (print, ("executed",))

    path = tmp_path / "evil.pt"
    torch.save({"state_dict": {}, "params": Evil(), "seed": 1, "n_features": 30}, path)
    with pytest.raises(pickle.UnpicklingError):
        MLPModel.load(path)


def test_load_model_checks_suffix(tmp_path, toy_xy):
    X, y = toy_xy
    m = create_model("xgboost", MODELS, 1)
    m.fit(X, y)
    path = m.save(tmp_path / "m")
    with pytest.raises(ValueError, match=".joblib"):
        load_model("random_forest", path, MODELS, 1)


# --------------------------------------------------------------------------- RF weighted bootstrap
def _distill_set(toy_xy, lam):
    from heterofl.distillation.soft_targets import build_distillation_set

    X, y = toy_xy
    teacher = np.random.default_rng(3).uniform(0.0, 0.3, 500)
    return build_distillation_set(X[:1000], y[:1000], X[1000:], teacher, lam), teacher


@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_rf_bootstrap_size_is_sum_of_weights(toy_xy, lam):
    from heterofl.models.random_forest import bootstrap_draws_per_tree

    ds, _ = _distill_set(toy_xy, lam)
    m = RandomForestModel({"n_estimators": 5}, 1)
    info = m.fit(ds.X, ds.y, ds.sample_weight)
    assert m.clf._n_samples_bootstrap == int(1000 * (1 + lam))   # scikit-learn's own value
    assert info["bootstrap_draws_per_tree"] == m.clf._n_samples_bootstrap
    assert bootstrap_draws_per_tree(m.effective_params(None), 1000, None) == 1000


def test_rf_weighted_bootstrap_draws_follow_the_weights(toy_xy):
    """Count the rows scikit-learn actually draws for every tree (installed 1.9.1)."""
    from sklearn.ensemble._forest import _generate_sample_indices

    lam = 0.5
    ds, teacher = _distill_set(toy_xy, lam)
    m = RandomForestModel({"n_estimators": 60}, 2)
    m.fit(ds.X, ds.y, ds.sample_weight)
    n, n_boot = len(ds.y), m.clf._n_samples_bootstrap
    local, proxy, proxy_pos = [], [], []
    for tree in m.clf.estimators_:
        idx = _generate_sample_indices(tree.random_state, n, n_boot, ds.sample_weight)
        local.append(np.sum(~ds.is_proxy[idx]))
        proxy.append(np.sum(ds.is_proxy[idx]))
        proxy_pos.append(np.sum(ds.is_proxy[idx] & (ds.y[idx] == 1)))
    # local draws: the local-only budget (N_local) in expectation; proxy: lambda * N_local
    assert np.mean(local) == pytest.approx(1000, rel=0.02)
    assert np.mean(proxy) == pytest.approx(lam * 1000, rel=0.04)
    # proxy labels are drawn in proportion to the teacher probabilities
    assert np.sum(proxy_pos) / np.sum(proxy) == pytest.approx(
        np.clip(teacher, 1e-6, 1 - 1e-6).mean(), abs=0.01)


def test_rf_sampling_budget_is_logged_and_scales_only_proxy_draws(toy_xy):
    from heterofl.distillation.trainers import distill_update

    X, y = toy_xy
    teacher = np.full(500, 0.2)
    infos = {}
    for lam in (0.0, 0.5, 1.0):
        m = RandomForestModel({"n_estimators": 5}, 4)
        infos[lam] = distill_update(m, X[:1000], y[:1000], X[1000:], teacher,
                                    {"lambda_kd": lam, "prob_clip": 1e-6, "tree_update": "retrain"})
    for lam, info in infos.items():
        assert info["rf_expected_local_draws"] == pytest.approx(1000, rel=1e-3)
        assert info["rf_expected_proxy_draws"] == pytest.approx(lam * 1000, rel=1e-3)
        assert info["rf_expected_proxy_positive_draws"] == pytest.approx(0.2 * lam * 1000, rel=1e-3)
    assert infos[0.0]["weighted_bootstrap"] is False      # lambda = 0: the local-only forest
