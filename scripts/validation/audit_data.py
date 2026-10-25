"""Phase 3 data-integrity audit, recomputed independently from the raw CSV.

Does NOT trust the pipeline's own manifests: it re-derives everything from
``creditcard.csv`` + the saved split indices.

Checks
  1. split index sets are disjoint, cover the dataset, and have the claimed class counts
  2. the scaler statistics equal mean/std of the *proxy* features only (recomputed)
  3. each (seed, scenario) partition view: banks disjoint from proxy/val/test, union == client pool
  4. proxy.npz contains no labels; no test rows in any federation view
  5. exact-duplicate analysis (feature vector incl. Time) between test and training data
  6. temporal structure: is Time a feature? (random split + Time lets trees interpolate fraud bursts)
     -> reports fraud-rate autocorrelation in time and the time-distribution of each split
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main(root: Path, csv: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {}
    df = pd.read_csv(csv)
    y = df["Class"].to_numpy()
    n = len(df)
    for bundle in sorted(p for p in (root / "data/processed").iterdir() if p.is_dir()):
        sp = np.load(bundle / "splits.npz")
        row_ids = np.load(bundle / "row_ids.npy")
        yb = np.load(bundle / "y.npy")
        names = ["test", "validation", "proxy", "client_pool"]
        sets = {k: sp[k] for k in names}
        allidx = np.concatenate(list(sets.values()))
        rep["disjoint_and_cover"] = bool(len(np.unique(allidx)) == len(allidx) == len(yb))
        rep["row_ids_identity"] = bool(np.array_equal(row_ids, np.arange(n)))
        rep["class_counts"] = {k: [int(len(v)), int(yb[v].sum())] for k, v in sets.items()}
        # scaler recomputation
        X = df.drop(columns=["Class"]).to_numpy(dtype=np.float64)
        Xp = X[row_ids[sets["proxy"]]].copy()
        Xp[:, -1] = np.log1p(Xp[:, -1])
        pre = json.loads((bundle / "preprocessor.json").read_text())
        rep["scaler_matches_proxy_only"] = bool(
            np.allclose(Xp.mean(0), pre["mean"], rtol=1e-9) and
            np.allclose(np.where(Xp.std(0) > 1e-12, Xp.std(0), 1.0), pre["scale"], rtol=1e-9))
        Xall = X.copy()
        Xall[:, -1] = np.log1p(Xall[:, -1])
        rep["scaler_differs_from_all_data_scaler"] = bool(
            not np.allclose(Xall.mean(0), pre["mean"], rtol=1e-6))
        # duplicates (all 30 raw features)
        h = pd.util.hash_pandas_object(df.drop(columns=["Class"]), index=False).to_numpy()
        train_h = set(h[np.concatenate([sets["client_pool"], sets["proxy"]])].tolist())
        te = sets["test"]
        dup = np.array([h[i] in train_h for i in te])
        rep["duplicates"] = {
            "n_test_rows_with_train_duplicate": int(dup.sum()),
            "n_test_frauds_with_train_duplicate": int((dup & (y[te] == 1)).sum()),
            "n_test_frauds": int(y[te].sum()),
            "total_exact_duplicate_rows_in_csv": int(df.duplicated().sum()),
            "frauds_among_duplicates_in_csv": int(df.duplicated(keep=False)[y == 1].sum()),
        }
        np.save(out / "test_has_train_duplicate.npy", dup)
        # time structure
        t = df["Time"].to_numpy()
        rep["time_ks_test_vs_pool"] = {
            k: float(stats.ks_2samp(t[sets[k]], t[sets["client_pool"]]).statistic)
            for k in ("test", "validation", "proxy")}
        order = np.argsort(t, kind="stable")
        ys = y[order]
        bins = np.array_split(ys, 200)
        rates = np.array([b.mean() for b in bins])
        rep["fraud_rate_time_bins"] = {
            "n_bins": 200, "bins_with_zero_fraud": int((rates == 0).sum()),
            "max_bin_rate": float(rates.max()),
            "lag1_autocorr": float(np.corrcoef(rates[:-1], rates[1:])[0, 1])}
        # nearest-neighbour-in-time leakage proxy: a fraud's closest-in-time neighbour in train
        tr = np.sort(t[sets["client_pool"]])
        tt = t[te][y[te] == 1]
        pos = np.clip(np.searchsorted(tr, tt), 1, len(tr) - 1)
        gap = np.minimum(np.abs(tt - tr[pos - 1]), np.abs(tr[pos] - tt))
        rep["test_fraud_to_nearest_train_row_seconds"] = {
            "median": float(np.median(gap)), "p90": float(np.quantile(gap, 0.9))}
    # partition views
    views = []
    for sd in sorted((root / "data/partitions").glob("*/seed_*/*")):
        v = sd / "view"
        if not v.is_dir():
            continue
        banks = {p.stem: np.load(p) for p in v.glob("bank_*.npz")}
        proxy = np.load(v / "proxy.npz")
        val = np.load(v / "validation.npz")
        ids = {k: set(b["row_ids"].tolist()) for k, b in banks.items()}
        ids["proxy"] = set(proxy["row_ids"].tolist())
        ids["validation"] = set(val["row_ids"].tolist())
        test_ids = set(row_ids[sets["test"]].tolist())
        pool = set(row_ids[sets["client_pool"]].tolist())
        union = set().union(*[ids[k] for k in banks])
        names2 = list(ids)
        pair_overlap = sum(len(ids[a] & ids[b]) for i, a in enumerate(names2) for b in names2[i + 1:])
        views.append({"view": str(sd.relative_to(root / "data/partitions")),
                      "pair_overlap": pair_overlap,
                      "test_rows_in_view": sum(len(s & test_ids) for s in ids.values()),
                      "union_equals_pool": union == pool,
                      "proxy_has_labels": "y" in proxy.files})
    rep["views"] = views
    rep["views_all_ok"] = all(v["pair_overlap"] == 0 and v["test_rows_in_view"] == 0 and
                              v["union_equals_pool"] and not v["proxy_has_labels"] for v in views)
    (out / "data_audit.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "views"}, indent=2))
    print(f"views checked: {len(views)}, all ok: {rep['views_all_ok']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--csv", type=Path, default=Path("data/raw/creditcard.csv"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    main(a.root.resolve(), a.csv, a.out)
