"""Uncertainty quantification appropriate for a rare-event test set.

Two distinct sources of uncertainty are kept separate:

1. **Test-sample uncertainty** (for a fixed trained model): class-stratified
   bootstrap over test examples. Positives and negatives are resampled
   separately, so every replicate keeps exactly the observed number of frauds
   (otherwise some replicates would contain very few positives and AP would be
   dominated by resampling noise in the class ratio).
2. **Training variability** (partition + initialisation): spread across seeds.
   All seeds are evaluated on the SAME test set, so seed results are not
   independent datasets; we report mean, standard deviation and a t-interval
   across seeds as a descriptive summary of training variability only, and
   make no significance claims from them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class _SortedAP:
    """Average precision with per-example integer weights on a fixed sort order.

    Bootstrap replicates are represented as multiplicity vectors, which lets us
    sort once and evaluate each replicate in O(n). Ties in the score are grouped
    exactly as in ``sklearn.metrics.average_precision_score``.
    """

    def __init__(self, y: np.ndarray, p: np.ndarray):
        self.order = np.argsort(-np.asarray(p, dtype=np.float64), kind="mergesort")
        ps = np.asarray(p, dtype=np.float64)[self.order]
        self.ys = np.asarray(y, dtype=np.float64)[self.order]
        self.ends = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]

    def __call__(self, w: np.ndarray | None = None) -> float:
        ws = np.ones_like(self.ys) if w is None else np.asarray(w, dtype=np.float64)[self.order]
        tps = np.cumsum(ws * self.ys)[self.ends]
        fps = np.cumsum(ws * (1.0 - self.ys))[self.ends]
        if tps[-1] <= 0:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(tps + fps > 0, tps / (tps + fps), 1.0)
        recall = tps / tps[-1]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_average_precision(y, p) -> float:
    return _SortedAP(np.asarray(y), np.asarray(p))()


def stratified_bootstrap_weights(y: np.ndarray, n_resamples: int, seed: int):
    """Yield ``n_resamples`` multiplicity vectors; each resamples each class with replacement.

    A generator (not a matrix) to keep memory O(n). The same ``seed`` yields the
    same replicates, which is what makes paired comparisons paired.
    """
    y = np.asarray(y)
    pos = np.nonzero(y == 1)[0]
    neg = np.nonzero(y == 0)[0]
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        w = np.zeros(len(y), dtype=np.float64)
        w[pos] = np.bincount(rng.integers(0, len(pos), len(pos)), minlength=len(pos))
        w[neg] = np.bincount(rng.integers(0, len(neg), len(neg)), minlength=len(neg))
        yield w


def bootstrap_ap_ci(y, p, n_resamples: int = 1000, confidence: float = 0.95,
                    seed: int = 0) -> dict:
    """Percentile CI of AP under class-stratified bootstrap of the test examples."""
    ap = _SortedAP(np.asarray(y), np.asarray(p))
    reps = np.array([ap(w) for w in stratified_bootstrap_weights(y, n_resamples, seed)])
    a = (1 - confidence) / 2
    return {"point": ap(), "ci_low": float(np.nanquantile(reps, a)),
            "ci_high": float(np.nanquantile(reps, 1 - a)), "bootstrap_std": float(np.nanstd(reps)),
            "n_resamples": int(n_resamples), "confidence": confidence,
            "scheme": "class-stratified bootstrap over test examples"}


def paired_bootstrap_ap_diff(y, p_a, p_b, n_resamples: int = 1000, confidence: float = 0.95,
                             seed: int = 0) -> dict:
    """CI for AP(a) - AP(b), resampling the same test examples for both systems."""
    ap_a = _SortedAP(np.asarray(y), np.asarray(p_a))
    ap_b = _SortedAP(np.asarray(y), np.asarray(p_b))
    diffs = np.array([ap_a(w) - ap_b(w)
                      for w in stratified_bootstrap_weights(y, n_resamples, seed)])
    a = (1 - confidence) / 2
    return {"diff": ap_a() - ap_b(), "ci_low": float(np.nanquantile(diffs, a)),
            "ci_high": float(np.nanquantile(diffs, 1 - a)),
            "fraction_a_better": float(np.nanmean(diffs > 0)),
            "n_resamples": int(n_resamples), "confidence": confidence}


def summarize_across_seeds(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str],
                           confidence: float = 0.95) -> pd.DataFrame:
    """Mean, sample std, n and t-interval across seeds (descriptive; seeds share a test set)."""
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key, strict=True))
        for m in metric_cols:
            v = g[m].astype(float).dropna().to_numpy()
            n = len(v)
            mean = float(v.mean()) if n else float("nan")
            std = float(v.std(ddof=1)) if n > 1 else float("nan")
            half = (float(stats.t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n))
                    if n > 1 else float("nan"))
            row.update({f"{m}_mean": mean, f"{m}_std": std, f"{m}_n": n,
                        f"{m}_ci_low": mean - half, f"{m}_ci_high": mean + half})
        rows.append(row)
    return pd.DataFrame(rows)


class PairingError(ValueError):
    """Rows cannot be paired unambiguously (duplicates, or different partitions)."""


def paired_seed_differences(df: pd.DataFrame, method_a: str, method_b: str, metric: str,
                            by: list[str], pair_on: str = "seed",
                            system_col: str = "system_label",
                            identity_col: str = "partition_sha256") -> pd.DataFrame:
    """Per-group summary of seed-paired differences a - b (descriptive only).

    Rows are paired on ``by + [pair_on]`` (scenario and seed). Each key must occur at
    most once per system (``PairingError`` otherwise), and when ``identity_col`` is
    present the two members of a pair must come from the same realised partition.
    Pairs with a missing value on either side are dropped and not counted in
    ``n_pairs`` (never imputed).
    """
    keys = by + [pair_on]
    sides = {}
    for name, system in (("a", method_a), ("b", method_b)):
        part = df[df[system_col] == system]
        dup = part.duplicated(keys, keep=False)
        if dup.any():
            raise PairingError(f"{system}: several rows for the same {keys}: "
                               f"{part.loc[dup, keys].drop_duplicates().to_dict('records')[:5]}")
        sides[name] = part.set_index(keys)
    joined = pd.concat({"a": sides["a"][metric], "b": sides["b"][metric]}, axis=1)
    if identity_col in df.columns:
        ida, idb = sides["a"][identity_col], sides["b"][identity_col]
        both = ida.index.intersection(idb.index)
        bad = [k for k in both if pd.notna(ida[k]) and pd.notna(idb[k]) and ida[k] != idb[k]]
        if bad:
            raise PairingError(f"{method_a} vs {method_b}: paired rows come from different "
                               f"partitions at {bad[:5]}")
    joined = joined.dropna()
    joined["diff"] = joined["a"] - joined["b"]
    rows = []
    for key, g in joined.groupby(level=by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        d = g["diff"].to_numpy()
        rows.append({**dict(zip(by, key, strict=True)), "a": method_a, "b": method_b,
                     "metric": metric, "n_pairs": len(d), "mean_diff": float(d.mean()),
                     "std_diff": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                     "min_diff": float(d.min()), "max_diff": float(d.max()),
                     "n_a_better": int((d > 0).sum())})
    return pd.DataFrame(rows)
