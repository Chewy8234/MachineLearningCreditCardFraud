"""Soft-target expansion, weight normalisation and model-specific distillation."""

from __future__ import annotations

import numpy as np
import pytest

from heterofl.distillation.soft_targets import (
    build_distillation_set,
    expand_soft_targets,
    proxy_weight_per_example,
)
from heterofl.distillation.trainers import distill_update, initial_fit
from heterofl.models.model_factory import create_model

from .test_models import MODELS


def test_expansion_labels_and_weights():
    Xp = np.arange(12, dtype=np.float32).reshape(4, 3)
    t = np.array([0.0, 0.25, 0.9, 1.0])
    X2, y2, w2 = expand_soft_targets(Xp, t, weight_per_example=2.0, prob_clip=1e-6)
    assert X2.shape == (8, 3)
    np.testing.assert_array_equal(X2[:4], Xp)
    np.testing.assert_array_equal(X2[4:], Xp)
    np.testing.assert_array_equal(y2, [0, 0, 0, 0, 1, 1, 1, 1])
    p = np.clip(t, 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(w2[:4], 2.0 * (1 - p))
    np.testing.assert_allclose(w2[4:], 2.0 * p)
    np.testing.assert_allclose(w2[:4] + w2[4:], 2.0)  # each example's total weight is w


@pytest.mark.parametrize("lam", [0.1, 0.5, 1.0])
def test_effective_proxy_weight_matches_lambda(lam):
    rng = np.random.default_rng(0)
    Xl, yl = rng.normal(size=(300, 5)), (rng.random(300) < 0.05).astype(int)
    Xp, t = rng.normal(size=(1000, 5)), rng.random(1000)
    ds = build_distillation_set(Xl, yl, Xp, t, lam)
    assert ds.info["total_proxy_weight"] == pytest.approx(lam * 300)
    assert ds.info["total_local_weight"] == 300
    assert ds.sample_weight[~ds.is_proxy].sum() == 300
    assert ds.sample_weight[ds.is_proxy].sum() == pytest.approx(lam * 300)
    # proxy size does not change the total proxy influence
    ds2 = build_distillation_set(Xl, yl, Xp[:100], t[:100], lam)
    assert ds2.info["total_proxy_weight"] == pytest.approx(lam * 300)
    # true local labels are kept unchanged and separate from distilled targets
    np.testing.assert_array_equal(ds.y[~ds.is_proxy], yl)
    assert ds.info["effective_proxy_positive_weight"] == pytest.approx(
        proxy_weight_per_example(300, 1000, lam) * np.clip(t, 1e-6, 1 - 1e-6).sum())


def test_lambda_zero_returns_local_data_without_weights():
    Xl, yl = np.zeros((10, 3)), np.array([0] * 9 + [1])
    ds = build_distillation_set(Xl, yl, np.ones((5, 3)), np.full(5, 0.5), 0.0)
    assert ds.sample_weight is None and len(ds.y) == 10 and not ds.is_proxy.any()


def test_expansion_is_exact_soft_cross_entropy():
    """Weighted log-loss of the two copies == w * soft BCE (basis of XGBoost exactness)."""
    rng = np.random.default_rng(1)
    p, q, w = rng.random(50), rng.uniform(0.01, 0.99, 50), 0.7
    _, y2, w2 = expand_soft_targets(np.zeros((50, 1)), p, w, prob_clip=1e-12)
    q2 = np.concatenate([q, q])
    weighted = -(w2 * (y2 * np.log(q2) + (1 - y2) * np.log(1 - q2))).sum()
    soft = -(w * (p * np.log(q) + (1 - p) * np.log(1 - q))).sum()
    assert weighted == pytest.approx(soft)


def test_numerical_stability_near_zero_and_one():
    t = np.array([0.0, 1e-30, 1 - 1e-16, 1.0])
    _, _, w = expand_soft_targets(np.zeros((4, 2)), t, 1.0, prob_clip=1e-6)
    assert np.all(np.isfinite(w)) and np.all(w > 0)
    with pytest.raises(ValueError):
        expand_soft_targets(np.zeros((2, 2)), np.array([0.5, np.nan]), 1.0)
    with pytest.raises(ValueError):
        expand_soft_targets(np.zeros((2, 2)), np.array([0.5]), 1.0)


KD = {"lambda_kd": 1.0, "prob_clip": 1e-6, "tree_update": "retrain", "mlp_epochs_per_round": 3,
      "proxy_batch_size": 128, "xgb_trees_per_round": 10}


@pytest.mark.parametrize("mtype", list(MODELS))
def test_distillation_moves_proxy_predictions_towards_teacher(mtype, toy_xy):
    X, y = toy_xy
    Xl, yl, Xp = X[:1000], y[:1000], X[1000:]
    m0 = create_model(mtype, MODELS, 5)
    initial_fit(m0, Xl, yl)
    base = m0.predict_proba(Xp)
    teacher = np.full(len(Xp), 0.9)  # an extreme teacher makes the effect unambiguous
    m1 = create_model(mtype, MODELS, 5)
    initial_fit(m1, Xl, yl)
    info = distill_update(m1, Xl, yl, Xp, teacher, KD)
    after = m1.predict_proba(Xp)
    assert after.mean() > base.mean() + 0.05
    assert np.all(np.isfinite(after))
    if mtype != "mlp":
        assert info["total_proxy_weight"] == pytest.approx(KD["lambda_kd"] * 1000)


@pytest.mark.parametrize("mtype", ["xgboost", "random_forest"])
def test_tree_lambda_zero_reproduces_local_model(mtype, toy_xy):
    X, y = toy_xy
    a = create_model(mtype, MODELS, 9)
    initial_fit(a, X[:1000], y[:1000])
    b = create_model(mtype, MODELS, 9)
    distill_update(b, X[:1000], y[:1000], X[1000:], np.full(500, 0.9), {**KD, "lambda_kd": 0.0})
    np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


def test_xgboost_continue_policy_appends_trees(toy_xy):
    X, y = toy_xy
    m = create_model("xgboost", MODELS, 1)
    m.fit(X[:1000], y[:1000])
    info = distill_update(m, X[:1000], y[:1000], X[1000:], np.full(500, 0.2),
                          {**KD, "tree_update": "continue"})
    assert info["n_trees"] == MODELS["xgboost"]["n_estimators"] + KD["xgb_trees_per_round"]


def test_random_forest_cannot_continue(toy_xy):
    X, y = toy_xy
    m = create_model("random_forest", MODELS, 1)
    m.fit(X, y)
    with pytest.raises(ValueError, match="retrained"):
        distill_update(m, X, y, X[:10], np.full(10, 0.5), {**KD, "tree_update": "continue"})
