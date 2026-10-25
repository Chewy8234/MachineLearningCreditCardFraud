"""Soft-target expansion for tree models (XGBoost, Random Forest).

Tree learners in XGBoost/scikit-learn require hard 0/1 labels. A soft target
``p`` for proxy example ``x`` is represented by two weighted copies::

    (x, label=1, weight = w * p)      (x, label=0, weight = w * (1 - p))

**XGBoost** minimises a sample-weighted log-loss, so the two copies contribute
``w * [-p log q(x) - (1-p) log(1-q(x))]`` -- exactly the soft-target binary
cross-entropy used for the MLP. The expansion is therefore exact for XGBoost.

**Random Forest** (scikit-learn 1.9.1, ``bootstrap=True``) draws each tree's
bootstrap sample with probability proportional to the weights (see
``RandomForestModel``). A proxy example's class-1 / class-0 copies are therefore
drawn in expected ratio ``p : 1-p``; since both copies share ``x`` they reach
the same leaf, and the leaf's class fraction -- averaged over trees -- moves
towards ``p``. In effect each tree sees *hard labels sampled from the teacher
distribution*, and the forest averages over those samples. This is a practical
*approximation* of soft-target training, not an exact objective: for small
``w * p`` a proxy positive copy is rarely drawn at all, and impurity-based
splitting and leaf-size limits also intervene.

Weight normalisation
--------------------
Private local rows have weight 1 each (total ``N_local``). Each proxy example
receives total weight ``w = lambda_kd * N_local / N_proxy`` split across its
two copies, so the *total* proxy weight is ``lambda_kd * N_local`` independent of
the proxy-set size. This mirrors the MLP objective ``mean L_sup + lambda_kd * mean
L_kd``. Hard pseudo-labelling (``p > 0.5 -> 1``) is deliberately not used: with
0.17% fraud it would turn almost every proxy example into a negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from heterofl.models.base import check_probabilities


def proxy_weight_per_example(n_local_weight: float, n_proxy: int, lambda_kd: float) -> float:
    if n_proxy <= 0:
        raise ValueError("Proxy set is empty")
    return float(lambda_kd) * float(n_local_weight) / float(n_proxy)


def expand_soft_targets(X_proxy: np.ndarray, teacher: np.ndarray, weight_per_example: float,
                        prob_clip: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(X, y, w)`` with 2*M rows: M class-0 copies followed by M class-1 copies."""
    p = check_probabilities(teacher, len(X_proxy), "teacher")
    p = np.clip(p, prob_clip, 1.0 - prob_clip)
    X2 = np.concatenate([X_proxy, X_proxy], axis=0)
    y2 = np.concatenate([np.zeros(len(p), dtype=np.int64), np.ones(len(p), dtype=np.int64)])
    w2 = np.concatenate([weight_per_example * (1.0 - p), weight_per_example * p])
    return X2, y2, w2


@dataclass
class DistillationSet:
    X: np.ndarray
    y: np.ndarray
    #: ``None`` when no proxy rows are added (lambda_kd = 0): the model is then fitted
    #: exactly like the local-only model (for scikit-learn forests, passing all-ones
    #: weights changes the bootstrap path and hence the trees).
    sample_weight: np.ndarray | None
    is_proxy: np.ndarray
    info: dict


def build_distillation_set(X_local: np.ndarray, y_local: np.ndarray, X_proxy: np.ndarray | None,
                           teacher: np.ndarray | None, lambda_kd: float,
                           prob_clip: float = 1e-6) -> DistillationSet:
    """Concatenate private rows (true labels, weight 1) with expanded proxy rows."""
    y_local = np.asarray(y_local).astype(np.int64)
    if set(np.unique(y_local)) - {0, 1}:
        raise ValueError("Local labels must be binary 0/1")
    w_local = np.ones(len(y_local), dtype=np.float64)
    if lambda_kd == 0 or X_proxy is None or teacher is None:
        return DistillationSet(np.asarray(X_local), y_local, None,
                               np.zeros(len(y_local), dtype=bool),
                               {"n_local": len(y_local), "n_proxy": 0, "lambda_kd": 0.0,
                                "total_local_weight": float(w_local.sum()),
                                "total_proxy_weight": 0.0, "proxy_weight_per_example": 0.0})
    w = proxy_weight_per_example(w_local.sum(), len(X_proxy), lambda_kd)
    Xp, yp, wp = expand_soft_targets(X_proxy, teacher, w, prob_clip)
    info = {"n_local": int(len(y_local)), "n_proxy": int(len(X_proxy)),
            "lambda_kd": float(lambda_kd), "total_local_weight": float(w_local.sum()),
            "total_proxy_weight": float(wp.sum()), "proxy_weight_per_example": w,
            "effective_proxy_positive_weight": float(wp[yp == 1].sum())}
    return DistillationSet(
        X=np.concatenate([np.asarray(X_local), Xp], axis=0),
        y=np.concatenate([y_local, yp]),
        sample_weight=np.concatenate([w_local, wp]),
        is_proxy=np.concatenate([np.zeros(len(y_local), bool), np.ones(len(yp), bool)]),
        info=info)
