"""Cluster-aware analysis across repeated outer splits (and any other group of result directories).

Unit of replication = the outer split (a fresh test/validation/proxy draw). Inside a split, partition
seeds and scenarios are nested. For each comparison we compute the paired difference in test AP per
(split, scenario, partition seed) cell, average over partition seeds within (split, scenario), and then
test/estimate across splits:

* exact sign-flip permutation test over split means (two-sided)
* percentile cluster-bootstrap CI over splits
* Holm adjustment across the comparison family (per scenario)
* a mixed model  diff ~ scenario + (1 | split)  for the pooled estimate (statsmodels)

Usage:  campaign_analysis.py --out DIR --group split42=results/full_repro --group split1=results/val_split_1 ...
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

AP = "test_average_precision"
SCEN = ["iid", "moderate_non_iid", "severe_non_iid"]


def collect(path: Path) -> pd.DataFrame:
    allm = path / "reports" / "all_metrics.csv"      # committed paper_results layout
    frames = [pd.read_csv(allm)] if allm.is_file() else \
        [pd.read_csv(f) for f in list(path.glob("seed_*/*/**/metrics.csv")) +
                         list(path.glob("unit_*/seed_*/*/**/metrics.csv"))
                         if "_archive" not in f.parts]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d[d.scenario.isin(SCEN)]


def sel(d: pd.DataFrame, variant: str, system: str) -> pd.Series:
    s = d[(d.variant == variant) & (d.system == system)]
    return s.set_index(["split", "scenario", "seed"])[AP]


def signflip_p(x: np.ndarray) -> float:
    n = len(x)
    obs = abs(x.mean())
    cnt = sum(abs((np.array(s) * x).mean()) >= obs - 1e-15 for s in itertools.product([1, -1], repeat=n))
    return cnt / 2 ** n


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    m, run, adj = len(ps), 0.0, np.empty(len(ps))
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x: np.ndarray, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


COMPARISONS = {
    "KD committee - lambda0 committee": (("primary", "kd_committee"), ("kd_lambda_0.0", "kd_committee")),
    "KD committee - ensemble_equal": (("primary", "kd_committee"), ("primary", "ensemble_equal")),
    "KD committee - FedAvg MLP": (("primary", "kd_committee"), ("primary", "fedavg_global")),
    "ensemble_equal - FedAvg MLP": (("primary", "ensemble_equal"), ("primary", "fedavg_global")),
    "KD committee - best local (oracle)": ("oracle", None),
    "KD committee - best local (val-chosen)": ("valchosen", None),
    "lambda0 committee - ensemble_equal": (("kd_lambda_0.0", "kd_committee"), ("primary", "ensemble_equal")),
    "KD bank C(MLP) - lambda0 C": (("primary", "kd_bank_C"), ("kd_lambda_0.0", "kd_bank_C")),
    "KD bank A(XGB) - lambda0 A": (("primary", "kd_bank_A"), ("kd_lambda_0.0", "kd_bank_A")),
    "KD bank B(RF) - lambda0 B": (("primary", "kd_bank_B"), ("kd_lambda_0.0", "kd_bank_B")),
    "lambda0 bank C(MLP) - local C": (("kd_lambda_0.0", "kd_bank_C"), ("primary", "bank_C")),
    "KD bank C(MLP) - local C": (("primary", "kd_bank_C"), ("primary", "bank_C")),
    "KD bank A(XGB) - local A": (("primary", "kd_bank_A"), ("primary", "bank_A")),
    "KD bank B(RF) - local B": (("primary", "kd_bank_B"), ("primary", "bank_B")),
}


def cell_diffs(d: pd.DataFrame, name: str, spec) -> pd.Series:
    if spec[0] == "oracle":
        loc = pd.concat([sel(d, "primary", f"bank_{b}") for b in "ABC"], axis=1).max(axis=1)
        return (sel(d, "primary", "kd_committee") - loc).dropna()
    if spec[0] == "valchosen":
        v = d[(d.variant == "primary") & d.system.isin([f"bank_{b}" for b in "ABC"])]
        key = ["split", "scenario", "seed"]
        pick = v.loc[v.groupby(key)["validation_average_precision"].idxmax()].set_index(key)[AP]
        return (sel(d, "primary", "kd_committee") - pick).dropna()
    return (sel(d, *spec[0]) - sel(d, *spec[1])).dropna()


def analyse(d: pd.DataFrame, out: Path) -> str:
    rows = []
    for name, spec in COMPARISONS.items():
        diffs = cell_diffs(d, name, spec)
        if diffs.empty:
            continue
        for scope in SCEN + ["pooled"]:
            x = diffs if scope == "pooled" else diffs.xs(scope, level="scenario") if scope in diffs.index.get_level_values("scenario") else None
            if x is None or len(x) == 0:
                continue
            sp = x.groupby(level=0).mean() if scope != "pooled" else x.groupby(level="split").mean()
            v = sp.values
            lo, hi = boot_ci(v) if len(v) > 1 else (np.nan, np.nan)
            rows.append({"comparison": name, "scope": scope, "n_splits": len(v), "n_cells": len(x),
                         "mean_diff": v.mean(), "sd_between_splits": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "ci_lo": lo, "ci_hi": hi, "n_splits_pos": int((v > 0).sum()),
                         "p_signflip": signflip_p(v) if len(v) > 1 else np.nan})
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for scope, idx in t.groupby("scope").groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p_signflip"].fillna(1).values)
    t.to_csv(out / "split_level_comparisons.csv", index=False)
    md = ["## Split-level paired AP comparisons (cluster = outer split)\n", t.round(4).to_markdown(index=False), ""]
    # mixed model on pooled cell differences for the headline contrasts
    try:
        import statsmodels.formula.api as smf
        mm = []
        for name in ("KD committee - lambda0 committee", "KD committee - ensemble_equal",
                     "KD committee - FedAvg MLP", "ensemble_equal - FedAvg MLP"):
            diffs = cell_diffs(d, name, COMPARISONS[name]).rename("diff").reset_index()
            if diffs["split"].nunique() < 3:
                continue
            fit = smf.mixedlm("diff ~ 1", diffs, groups=diffs["split"]).fit(reml=True)
            mm.append({"comparison": name, "estimate": fit.params["Intercept"], "se": fit.bse["Intercept"],
                       "z": fit.tvalues["Intercept"], "p": fit.pvalues["Intercept"],
                       "n_splits": diffs["split"].nunique(), "n_cells": len(diffs)})
        if mm:
            m = pd.DataFrame(mm)
            m["p_holm"] = holm(m.p.values)
            m.to_csv(out / "mixed_model.csv", index=False)
            md += ["## Mixed model `diff ~ 1 + (1|split)` over all cells\n", m.round(4).to_markdown(index=False), ""]
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        md.append(f"(mixed model failed: {exc})")
    # level table
    lv = d[d.variant == "primary"].groupby(["scenario", "system"])[AP].agg(["mean", "std", "count"])
    lv.to_csv(out / "ap_levels.csv")
    md += ["## Mean test AP by system (over all cells)\n", lv.round(4).to_markdown(), ""]
    return "\n".join(md)


def main(groups: dict[str, Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, path in groups.items():
        c = collect(path)
        if len(c):
            c["split"] = name
            parts.append(c)
    d = pd.concat(parts, ignore_index=True)
    d.to_csv(out / "all_cells.csv", index=False)
    text = analyse(d, out)
    (out / "campaign_analysis.md").write_text(text)
    print(f"splits: {sorted(d['split'].unique())}  rows: {len(d)}")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group", action="append", required=True, help="name=results_dir (repeatable)")
    a = ap.parse_args()
    main({g.split("=", 1)[0]: Path(g.split("=", 1)[1]) for g in a.group}, a.out)
