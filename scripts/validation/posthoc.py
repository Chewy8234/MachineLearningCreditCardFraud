"""Post-hoc analyses on saved per-transaction predictions (no retraining).

For every completed (scenario, seed) of a results directory it loads the three local models
(`local/bank_X`) and the three KD models (`heterogeneous_kd/kd_bank_X`) plus the saved committee
systems, and evaluates:

* calibration            Brier, log-loss, ECE (equal-mass bins), per member and per committee
* combiners              mean (paper) | geometric-mean | rank-mean | max | logit-mean | Platt-mean
                         (per-member Platt fitted on validation) | stacked LR (fitted on validation)
* cost / alert budget    precision@k alerts, expected cost at FN:FP = 100:1 and 500:1 with the
                         threshold chosen on validation (honest) vs oracle on test (bound)
* duplicate sensitivity  AP on test rows that have an exact twin in the training data vs not

Outputs CSVs plus ``posthoc.md`` in ``--out``. Duplicate flags come from ``audit_data.py``
(``test_has_train_duplicate.npy``; valid for the primary random split only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    """Expected calibration error with equal-mass bins (equal-width is useless at 0.17% prevalence)."""
    order = np.argsort(p)
    ece_ = 0.0
    for chunk in np.array_split(order, bins):
        ece_ += len(chunk) / len(y) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece_)


def load(d: Path, name: str):
    z = np.load(d / "predictions" / f"{name}.npz")
    return z["p_val"], z["p_test"], z["y_val"], z["y_test"]


def combiners(pv: dict, pt: dict, yv):
    banks = sorted(pv)
    Pv = np.stack([pv[b] for b in banks], 1)
    Pt = np.stack([pt[b] for b in banks], 1)
    out = {"mean": Pt.mean(1),
           "geo_mean": np.exp(np.log(np.clip(Pt, EPS, 1)).mean(1)),
           "logit_mean": 1 / (1 + np.exp(-logit(Pt).mean(1))),
           "max": Pt.max(1)}
    n = len(Pt)
    out["rank_mean"] = np.mean([rankdata(Pt[:, k]) / n for k in range(Pt.shape[1])], axis=0)
    # per-member Platt on validation, then mean of calibrated probabilities
    cal = []
    for k in range(Pv.shape[1]):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(Pv[:, [k]]), yv)
        cal.append(lr.predict_proba(logit(Pt[:, [k]]))[:, 1])
    out["platt_mean"] = np.mean(cal, axis=0)
    # stacked logistic regression on member logits (fitted on validation only)
    st = LogisticRegression(C=1.0, max_iter=2000).fit(logit(Pv), yv)
    out["stacked_lr"] = st.predict_proba(logit(Pt))[:, 1]
    return out


def cost_at(y, p, thr, c_fn, c_fp=1.0):
    pred = p >= thr
    return float(c_fn * ((~pred) & (y == 1)).sum() + c_fp * (pred & (y == 0)).sum())


def best_cost_threshold(y, p, c_fn):
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.99999, 800)))
    costs = [cost_at(y, p, t, c_fn) for t in cand]
    i = int(np.argmin(costs))
    return float(cand[i]), float(costs[i])


def prec_at_k(y, p, k):
    top = np.argsort(-p)[:k]
    return float(y[top].mean())


def main(results: Path, out: Path, dup_file: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dup = np.load(dup_file) if dup_file and dup_file.is_file() else None
    cal_rows, comb_rows, cost_rows, dup_rows = [], [], [], []
    for sdir in sorted(results.glob("seed_*/*")):
        seed, scen = int(sdir.parent.name.split("_")[1]), sdir.name
        loc, kd = sdir / "local", sdir / "heterogeneous_kd"
        if not (loc / "status.json").is_file() or not (kd / "status.json").is_file():
            continue
        if '"completed"' not in (loc / "status.json").read_text() or \
                '"completed"' not in (kd / "status.json").read_text():
            continue
        for tag, d, prefix in [("local", loc, "bank_"), ("kd", kd, "kd_bank_")]:
            pv, pt = {}, {}
            for b in "ABC":
                pv[b], pt[b], yv, yt = load(d, f"{prefix}{b}")
                cal_rows.append({"seed": seed, "scenario": scen, "group": tag, "system": f"{prefix}{b}",
                                 "ap": average_precision_score(yt, pt[b]),
                                 "brier": brier_score_loss(yt, pt[b]),
                                 "logloss": log_loss(yt, np.clip(pt[b], EPS, 1 - EPS)),
                                 "ece": ece(yt, pt[b]), "mean_pred": pt[b].mean(), "prevalence": yt.mean()})
            cmb = combiners(pv, pt, yv)
            for name, p in cmb.items():
                comb_rows.append({"seed": seed, "scenario": scen, "group": tag, "combiner": name,
                                  "ap": average_precision_score(yt, p), "brier": brier_score_loss(yt, np.clip(p, 0, 1)),
                                  "ece": ece(yt, p)})
            # cost / alert budget for the paper combiner and the best single member (by val AP)
            Pv_mean = np.mean([pv[b] for b in "ABC"], axis=0)
            members = {"committee_mean": (Pv_mean, cmb["mean"])}
            bv = max("ABC", key=lambda b: average_precision_score(yv, pv[b]))
            members[f"best_single_by_val({bv})"] = (pv[bv], pt[bv])
            for sysn, (pvv, ptt) in members.items():
                row = {"seed": seed, "scenario": scen, "group": tag, "system": sysn}
                for k in (50, 100, 200):
                    row[f"precision@{k}"] = prec_at_k(yt, ptt, k)
                for c_fn in (100, 500):
                    thr_v, _ = best_cost_threshold(yv, pvv, c_fn)
                    _, oracle = best_cost_threshold(yt, ptt, c_fn)
                    row[f"cost_fn{c_fn}_valthr"] = cost_at(yt, ptt, thr_v, c_fn)
                    row[f"cost_fn{c_fn}_oracle"] = oracle
                    row[f"cost_fn{c_fn}_all_negative"] = float(c_fn * yt.sum())
                cost_rows.append(row)
            if dup is not None and len(dup) == len(yt) and tag == "kd":
                for sysn, p in {**{f"kd_bank_{b}": pt[b] for b in "ABC"}, "kd_committee_mean": cmb["mean"]}.items():
                    for lab, mask in (("with_train_duplicate", dup), ("no_train_duplicate", ~dup)):
                        if yt[mask].sum() > 0:
                            dup_rows.append({"seed": seed, "scenario": scen, "system": sysn, "subset": lab,
                                             "n": int(mask.sum()), "n_fraud": int(yt[mask].sum()),
                                             "ap": average_precision_score(yt[mask], p[mask])})
    cal, comb, cost = map(pd.DataFrame, (cal_rows, comb_rows, cost_rows))
    cal.to_csv(out / "calibration.csv", index=False)
    comb.to_csv(out / "combiners.csv", index=False)
    cost.to_csv(out / "cost.csv", index=False)
    md = ["# Post-hoc analysis\n", f"cells analysed: {comb[['seed', 'scenario']].drop_duplicates().shape[0]}\n"]
    if len(cal):
        md += ["## Calibration (mean over cells)\n",
               cal.groupby(["group", "system"])[["ap", "brier", "logloss", "ece", "mean_pred", "prevalence"]]
               .mean().round(5).to_markdown(), ""]
        md += ["## Combiners: test AP (mean over cells; paired diff vs `mean`)\n"]
        piv = comb.pivot_table(index=["seed", "scenario", "group"], columns="combiner", values="ap")
        diff = piv.sub(piv["mean"], axis=0)
        tab = pd.DataFrame({"ap_mean": piv.groupby("group").mean().stack(),
                            "diff_vs_mean": diff.groupby("group").mean().stack(),
                            "n_better": (diff > 0).groupby("group").sum().stack()})
        md += [tab.round(4).to_markdown(), ""]
        md += ["## Cost / alert budget (mean over cells; lower cost is better)\n",
               cost.groupby(["group", "system"]).mean(numeric_only=True).drop(columns="seed").round(2).to_markdown(), ""]
    if dup_rows:
        dd = pd.DataFrame(dup_rows)
        dd.to_csv(out / "duplicates.csv", index=False)
        md += ["## AP on test rows with / without an exact training twin (mean over cells)\n",
               dd.groupby(["system", "subset"])[["n", "n_fraud", "ap"]].mean().round(4).to_markdown(), ""]
    (out / "posthoc.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dup-file", type=Path, default=None)
    a = ap.parse_args()
    main(a.results, a.out, a.dup_file)
