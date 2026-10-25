"""Model-family matrix (heterofl.experiments.model_matrix). SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from heterofl.experiments import model_matrix as mm


class _OneBank:
    """Minimal ScenarioData stand-in with a single bank."""

    def __init__(self, X, y):
        self.banks = ["A"]
        self._X, self._y = X, y

    def bank(self, bank):
        return self._X, self._y

    def validation(self):
        return self._X[:500], self._y[:500]

    def test(self):
        return self._X[500:], self._y[500:]


def test_xgb_bagging_with_one_bank_equals_plain_boosting(toy_xy):
    """With one bank and no row/column sampling, federated bagging is ordinary boosting."""
    X, y = toy_xy
    cfg = {"models": {"xgboost": {"n_estimators": 5, "max_depth": 3, "learning_rate": 0.3,
                                  "subsample": 1.0, "colsample_bytree": 1.0,
                                  "tree_method": "hist"}},
           "evaluation": {"threshold_rule": "max_f1"}}
    data = _OneBank(X, y)
    h = mm.Harness(cfg, data)
    *_, (_, _, p_test) = mm._federated_xgb_bagging(cfg, data, 0, 1, h, 4)

    b0 = mm._logit(y.mean())
    booster = xgb.train(mm._xgb_native_params(cfg, 0, 1),
                        xgb.DMatrix(X, label=y, base_margin=np.full(len(y), b0)),
                        num_boost_round=4)
    ref = booster.predict(xgb.DMatrix(X[500:], base_margin=np.full(len(X) - 500, b0)))
    np.testing.assert_allclose(p_test, ref, rtol=1e-5, atol=1e-6)


def test_matrix_runs_and_reports(prepared, tmp_path):
    cfg, _, _ = prepared
    out = tmp_path / "matrix"
    mm.run_unit(cfg, 42, "moderate_non_iid", out, mm.EXPERIMENTS, n_jobs=1, num_rounds=2,
                overwrite=False)
    d = out / "seed_42" / "moderate_non_iid"
    local = pd.read_csv(d / "local.csv")
    assert len(local) == len(mm.MODELS) * 3
    fl = pd.read_csv(d / "federated.csv")
    assert set(fl.model) == set(mm.MODELS) and set(fl["round"]) == {2}

    # committee round 0 is the local model of each bank's assigned family
    kd = pd.read_csv(d / "committee.csv")
    for c in cfg["clients"]:
        r0 = kd[(kd.bank == c["bank"]) & (kd.phase == "round0")].average_precision.item()
        loc = local[(local.bank == c["bank"]) & (local.model == c["model"])]
        assert r0 == loc.average_precision.item()

    text = mm.write_report(out, cfg).read_text(encoding="utf-8")
    for needle in ("1. Local only", "2. Centralized", "3. Homogeneous FL",
                   "4. Committee (KD)", "Committee per bank", "Test PR-AUC by round"):
        assert needle in text
