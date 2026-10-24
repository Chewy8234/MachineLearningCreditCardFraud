"""Model-specific local training and distillation updates.

State policy per model family (documented in docs/methodology.md):

* MLP           -- *incremental*: each round fine-tunes the current weights for
                   ``kd.mlp_epochs_per_round`` epochs on ``L_sup + lambda_kd * L_kd``.
* XGBoost       -- ``kd.tree_update='retrain'`` (default): a new booster with the
                   same seed and ``n_estimators`` is trained on local + expanded
                   proxy rows. ``'continue'`` appends ``kd.xgb_trees_per_round``
                   trees to the current booster instead.
* Random Forest -- always *retrained* from scratch (no incremental fitting exists);
                   ``tree_update='continue'`` is rejected for it rather than
                   silently emulated.
"""

from __future__ import annotations

import numpy as np

from heterofl.distillation.soft_targets import build_distillation_set
from heterofl.models.base import FraudModel
from heterofl.models.mlp import MLPModel
from heterofl.models.random_forest import RandomForestModel, bootstrap_draws_per_tree
from heterofl.models.xgboost_model import XGBoostModel
from heterofl.utils.logging import get_logger

log = get_logger("distillation.trainers")


def rf_sampling_budget(model: RandomForestModel, ds) -> dict:
    """Expected composition of each tree's weighted bootstrap sample.

    Rows are drawn with probability proportional to their weight, so in expectation a
    tree draws ``draws * W_local / W_total`` local rows and ``draws * W_proxy / W_total``
    proxy rows (``W_proxy_pos`` of which carry label 1). With ``max_samples=1.0`` and
    ``W_local = N_local``, ``W_proxy = lambda * N_local``, this is ``N_local`` local
    draws -- the local-only budget -- plus ``lambda * N_local`` proxy draws.
    """
    params = model.effective_params(ds.sample_weight)
    draws = bootstrap_draws_per_tree(params, len(ds.y), ds.sample_weight)
    if ds.sample_weight is None:
        return {"rf_bootstrap_draws_per_tree": draws, "rf_expected_local_draws": float(draws),
                "rf_expected_proxy_draws": 0.0, "rf_expected_proxy_positive_draws": 0.0}
    w = ds.sample_weight
    total = float(w.sum())
    proxy_pos = ds.is_proxy & (ds.y == 1)
    return {"rf_bootstrap_draws_per_tree": draws,
            "rf_expected_local_draws": draws * float(w[~ds.is_proxy].sum()) / total,
            "rf_expected_proxy_draws": draws * float(w[ds.is_proxy].sum()) / total,
            "rf_expected_proxy_positive_draws": draws * float(w[proxy_pos].sum()) / total}


def initial_fit(model: FraudModel, X: np.ndarray, y: np.ndarray) -> dict:
    """Round-0 / local-only training on private labelled data only."""
    info = model.fit(X, y)
    info["phase"] = "local"
    return info


def distill_update(model: FraudModel, X_local: np.ndarray, y_local: np.ndarray,
                   X_proxy: np.ndarray, teacher: np.ndarray, kd_cfg: dict) -> dict:
    lam = float(kd_cfg["lambda_kd"])
    clip = float(kd_cfg["prob_clip"])
    if isinstance(model, MLPModel):
        info = model.fit_distill(X_local, y_local, X_proxy, teacher, lam,
                                 epochs=int(kd_cfg["mlp_epochs_per_round"]),
                                 proxy_batch_size=kd_cfg.get("proxy_batch_size"),
                                 prob_clip=clip)
        info.update({"n_local": int(len(y_local)), "n_proxy": int(len(X_proxy)),
                     "loss_normalisation": "mean(L_sup) + lambda_kd * mean(L_kd)"})
        return info

    ds = build_distillation_set(X_local, y_local, X_proxy, teacher, lam, clip)
    policy = kd_cfg.get("tree_update", "retrain")
    if policy == "continue":
        if not isinstance(model, XGBoostModel):
            raise ValueError(f"tree_update='continue' is not supported for {model.model_type}; "
                             "Random Forest can only be retrained")
        info = model.continue_fit(ds.X, ds.y, ds.sample_weight,
                                  extra_trees=int(kd_cfg.get("xgb_trees_per_round", 50)))
    else:
        info = model.fit(ds.X, ds.y, ds.sample_weight)
    info.update(ds.info)
    if isinstance(model, RandomForestModel):
        info.update(rf_sampling_budget(model, ds))
        log.info("RF distillation (lambda=%g): %d bootstrap draws/tree, expected %.1f local "
                 "(%d local rows) + %.1f proxy (%.1f labelled 1)", lam,
                 info["rf_bootstrap_draws_per_tree"], info["rf_expected_local_draws"],
                 ds.info["n_local"], info["rf_expected_proxy_draws"],
                 info["rf_expected_proxy_positive_draws"])
    return info
