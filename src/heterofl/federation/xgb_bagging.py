"""Federated XGBoost by tree bagging (the scheme of Flower's ``FedXgbBagging`` strategy).

Protocol, per round ``r = 1..R``::

    server --global booster-->  every bank k
    bank k: boost ``trees_per_round`` NEW trees on its private rows, starting from the
            global model's margin (i.e. the trees fit the global model's residuals
            on bank k's data)
    bank k --its new trees-->  server
    server: global booster <- global booster + new trees of every bank (concatenation)

Nothing is averaged: XGBoost trees have no shared parameter space, so FedAvg does
not apply. Trees are *appended*, and the global margin is ``b0 + sum of all trees``.
After ``R`` rounds the global model holds ``K * R * trees_per_round`` trees. The
banks' new trees within one round are fitted independently to the same global
residual, so their summed step can overshoot -- a known property of the scheme.

``b0`` is the logit of the pooled fraud rate, computed from each bank's example
and fraud counts (two scalars per bank; no rows leave a bank).

This module runs the protocol in-process: each bank's update reads only that
bank's rows, the server step only concatenates boosters, but no Flower messages
are exchanged. Communication is accounted from the serialised (JSON) size of the
trees that the protocol would send.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from heterofl.utils.io import write_json
from heterofl.utils.reproducibility import derive_seed

#: sklearn-style XGBoost hyperparameter -> native ``xgb.train`` parameter
_NATIVE = (("max_depth", "max_depth"), ("subsample", "subsample"),
           ("colsample_bytree", "colsample_bytree"), ("min_child_weight", "min_child_weight"),
           ("reg_lambda", "lambda"), ("reg_alpha", "alpha"), ("gamma", "gamma"),
           ("max_bin", "max_bin"), ("tree_method", "tree_method"),
           ("scale_pos_weight", "scale_pos_weight"))


def xgb_native_params(params: dict, seed: int, n_jobs: int) -> dict:
    """Native training parameters equivalent to the project's ``models.xgboost`` block."""
    native = {"objective": "binary:logistic", "eval_metric": "logloss", "seed": seed,
              "nthread": n_jobs, "eta": params.get("learning_rate", 0.3)}
    for src, dst in _NATIVE:
        if params.get(src) is not None:
            native[dst] = params[src]
    return native


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _sigmoid(m: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-m))


class FedXgbBagging:
    """Federated bagging over ``banks`` = {bank: (X, y)}; call ``rounds()`` to train."""

    def __init__(self, banks: dict[str, tuple[np.ndarray, np.ndarray]], params: dict,
                 seed: int, n_jobs: int, num_rounds: int, trees_per_round: int):
        if not banks:
            raise ValueError("Federated bagging needs at least one bank")
        self.banks = banks
        self.params = params
        self.seed = int(seed)
        self.n_jobs = int(n_jobs)
        self.num_rounds = int(num_rounds)
        self.trees_per_round = int(trees_per_round)
        n = sum(len(y) for _, y in banks.values())
        self.b0 = logit(sum(float(y.sum()) for _, y in banks.values()) / n)
        #: boosters[r-1][i] = the trees bank i appended in round r
        self.boosters: list[list[xgb.Booster]] = []

    def rounds(self, eval_sets: dict[str, np.ndarray] | None = None):
        """Train round by round; yield ``(round, {name: probabilities}, info)``.

        ``eval_sets`` are scored with the global model after every round (the margin
        is accumulated incrementally, exactly as the concatenated booster computes it).
        """
        eval_sets = eval_sets or {}
        banks = self.banks
        dmat = {b: xgb.DMatrix(X, label=y) for b, (X, y) in banks.items()}
        margin = {b: np.full(len(banks[b][1]), self.b0) for b in banks}
        zero = {b: xgb.DMatrix(banks[b][0], base_margin=np.zeros(len(banks[b][1])))
                for b in banks}
        dev = {k: xgb.DMatrix(X, base_margin=np.zeros(len(X))) for k, X in eval_sets.items()}
        mev = {k: np.full(len(X), self.b0) for k, X in eval_sets.items()}
        global_bytes = 0
        for r in range(1, self.num_rounds + 1):
            t0 = time.perf_counter()
            new, fit_seconds = [], {}
            for b in banks:
                dmat[b].set_base_margin(margin[b])
                t_fit = time.perf_counter()
                new.append(xgb.train(
                    xgb_native_params(self.params, derive_seed(self.seed, "fed-xgb", b, r),
                                      self.n_jobs),
                    dmat[b], num_boost_round=self.trees_per_round))
                fit_seconds[b] = time.perf_counter() - t_fit
            for booster in new:
                for b in banks:
                    margin[b] += booster.predict(zero[b], output_margin=True)
                for k in eval_sets:
                    mev[k] += booster.predict(dev[k], output_margin=True)
            self.boosters.append(new)
            up = sum(len(bst.save_raw("json")) for bst in new)
            # round 1 starts from b0 only; later rounds send the current global model
            down = len(banks) * global_bytes
            global_bytes += up
            info = {"round": r, "fit_seconds": fit_seconds, "bytes_up": up, "bytes_down": down,
                    "n_trees": r * len(banks) * self.trees_per_round,
                    "round_seconds": time.perf_counter() - t0}
            yield r, {k: _sigmoid(m) for k, m in mev.items()}, info

    def predict_proba(self, X: np.ndarray, upto_round: int | None = None) -> np.ndarray:
        """Global model of round ``upto_round`` (default: last trained round)."""
        upto = len(self.boosters) if upto_round is None else int(upto_round)
        d = xgb.DMatrix(X, base_margin=np.zeros(len(X)))
        m = np.full(len(X), self.b0)
        for rnd in self.boosters[:upto]:
            for booster in rnd:
                m += booster.predict(d, output_margin=True)
        return _sigmoid(m)

    def save(self, directory: Path, upto_round: int | None = None) -> Path:
        """Write every appended booster (JSON) plus ``meta.json`` with ``b0`` and the order."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        upto = len(self.boosters) if upto_round is None else int(upto_round)
        files = []
        for r, rnd in enumerate(self.boosters[:upto], start=1):
            for b, booster in zip(self.banks, rnd, strict=True):
                name = f"round_{r:02d}_bank_{b}.json"
                booster.save_model(str(directory / name))
                files.append(name)
        write_json(directory / "meta.json", {
            "scheme": "federated XGBoost bagging (tree concatenation)", "b0": self.b0,
            "rounds": upto, "trees_per_round_per_bank": self.trees_per_round,
            "banks": list(self.banks), "boosters_in_order": files,
            "predict": "sigmoid(b0 + sum of the boosters' margins, in the listed order)"})
        return directory
