"""Phase 1 re-analysis of per-run metrics (no training).

Reads ``<results>/reports/all_metrics.csv`` and ``<results>/tables/table1_partitions.csv``
and writes tables to ``--out``. Answers, with the *right controls*:

* does KD beat its lambda=0 control (same extra training, no distillation)?
* does the KD committee beat the plain ensemble / FedAvg?
* are seed-level intervals valid (bounded) and are conclusions robust to multiplicity?
* is per-bank KD gain explained by bank size / fraud count (data-scarcity confound)?
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

AP = "test_average_precision"
PRIMARY_SCEN = ["iid", "moderate_non_iid", "severe_non_iid"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def signflip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation test on paired differences (mean statistic)."""
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    cnt = 0
    for s in itertools.product([1, -1], repeat=n):
        cnt += abs((np.array(s) * d).mean()) >= obs - 1e-15
    return cnt / 2 ** n


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def bounded_ci(x: np.ndarray, conf=0.95):
    """t-interval on the logit scale (always inside (0,1)); n<2 -> nan."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    z = logit(x)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * z.std(ddof=1) / np.sqrt(n)
    return (float(expit(z.mean() - h)), float(expit(z.mean() + h)))


def wide(df: pd.DataFrame, variant: str, system: str, name: str) -> pd.Series:
    s = df[(df.variant == variant) & (df.system == system)].set_index(["scenario", "seed"])[AP]
    return s.rename(name)


def main(results: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(results / "reports" / "all_metrics.csv")
    d = d[d.scenario.isin(PRIMARY_SCEN)]
    lines: list[str] = []

    # ---- 1. paired comparisons over the 9 (scenario, seed) cells -----------------------
    comps = {
        "KD committee - lambda0 committee":
            (("primary", "kd_committee"), ("kd_lambda_0.0", "kd_committee")),
        "KD committee - ensemble_equal":
            (("primary", "kd_committee"), ("primary", "ensemble_equal")),
        "KD committee - FedAvg MLP":
            (("primary", "kd_committee"), ("primary", "fedavg_global")),
        "ensemble_equal - FedAvg MLP":
            (("primary", "ensemble_equal"), ("primary", "fedavg_global")),
        "KD committee - best local (oracle on test)": None,
        "KD committee - centralized XGB (ref)":
            (("primary", "kd_committee"), ("primary", "centralized_xgboost")),
    }
    rows = []
    for name, spec in comps.items():
        if spec is None:
            a = wide(d, "primary", "kd_committee", "a")
            loc = pd.concat([wide(d, "primary", f"bank_{b}", b) for b in "ABC"], axis=1).max(axis=1)
            diff = (a - loc).dropna()
        else:
            a = wide(d, *spec[0], "a")
            b = wide(d, *spec[1], "b")
            diff = (a - b).dropna()
        for scope, dd in [("all 9 cells", diff)] + [(s, diff.xs(s, level="scenario")) for s in PRIMARY_SCEN]:
            rows.append({"comparison": name, "scope": scope, "n": len(dd), "mean_diff": dd.mean(),
                         "sd": dd.std(ddof=1), "min": dd.min(), "max": dd.max(),
                         "n_pos": int((dd > 0).sum()), "p_signflip": signflip_p(dd.values)})
    comp = pd.DataFrame(rows)
    fam = comp[comp.scope == "all 9 cells"].index
    comp.loc[fam, "p_holm"] = holm(comp.loc[fam, "p_signflip"].tolist())
    comp.to_csv(out / "paired_comparisons.csv", index=False)
    lines += ["## Paired AP comparisons (exact sign-flip permutation; cells = scenario x seed)\n",
              comp.round(4).to_markdown(index=False), ""]

    # ---- 2. per-bank KD effect vs lambda=0 control, and vs local --------------------------
    rows = []
    for b in "ABC":
        kd = wide(d, "primary", f"kd_bank_{b}", "kd")
        l0 = wide(d, "kd_lambda_0.0", f"kd_bank_{b}", "l0")
        loc = wide(d, "primary", f"bank_{b}", "loc")
        for nm, x, y in [("KD - local", kd, loc), ("KD - lambda0", kd, l0), ("lambda0 - local", l0, loc)]:
            diff = (x - y).dropna()
            rows.append({"bank": b, "contrast": nm, "n": len(diff), "mean_diff": diff.mean(),
                         "n_pos": int((diff > 0).sum()), "p_signflip": signflip_p(diff.values)})
    pb = pd.DataFrame(rows)
    pb["p_holm"] = holm(pb.p_signflip.tolist())
    pb.to_csv(out / "per_bank_kd_effect.csv", index=False)
    lines += ["## Per-bank KD effect (control = lambda 0: same epochs / retraining, no teacher)\n",
              pb.round(4).to_markdown(index=False), ""]

    # ---- 3. data-scarcity confound ----------------------------------------------------
    part = pd.read_csv(results / "tables" / "table1_partitions.csv")
    part = part[part.section == "scenario"] if "section" in part else part
    part = part.rename(columns={"partition_seed": "seed"})[
        ["scenario", "seed", "bank", "n_transactions", "n_fraud"]]
    reg = []
    for b in "ABC":
        kd = wide(d, "primary", f"kd_bank_{b}", "kd")
        loc = wide(d, "primary", f"bank_{b}", "loc")
        g = (kd - loc).rename("gain").reset_index()
        g["bank"] = b
        reg.append(g)
    reg = pd.concat(reg).merge(part, on=["scenario", "seed", "bank"])
    reg["log_n"] = np.log10(reg.n_transactions)
    reg["log_fraud"] = np.log10(reg.n_fraud)
    mt = reg.bank.map({"A": "xgboost", "B": "random_forest", "C": "mlp"})
    reg["model"] = mt
    reg.to_csv(out / "kd_gain_vs_bank_size.csv", index=False)
    sp_n = stats.spearmanr(reg.log_n, reg.gain)
    sp_f = stats.spearmanr(reg.log_fraud, reg.gain)
    lines += ["## KD gain (KD bank - local bank) vs bank size / fraud count (n=%d bank-runs)\n" % len(reg),
              f"Spearman(log10 rows, gain) = {sp_n.statistic:.3f} (p={sp_n.pvalue:.3f}); "
              f"Spearman(log10 frauds, gain) = {sp_f.statistic:.3f} (p={sp_f.pvalue:.3f}).\n",
              "Per bank/model Spearman (rows vs gain):\n"]
    for m, g in reg.groupby("model"):
        r = stats.spearmanr(g.log_n, g.gain)
        lines.append(f"- {m}: rho={r.statistic:.3f} (p={r.pvalue:.3f}, n={len(g)}); mean gain={g.gain.mean():+.4f}")
    lines.append("")
    small = reg[reg.n_fraud < 40]
    big = reg[reg.n_fraud >= 100]
    lines.append(f"Banks with <40 frauds: mean gain {small.gain.mean():+.4f} (n={len(small)}); "
                 f">=100 frauds: {big.gain.mean():+.4f} (n={len(big)}).\n")

    # ---- 4. seed-level intervals: invalid t-interval vs bounded -------------------------
    rows = []
    for (scen, sysn, var), g in d[d.variant == "primary"].groupby(["scenario", "system", "variant"]):
        x = g[AP].values
        if len(x) < 2:
            continue
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        lo, hi = bounded_ci(x)
        rows.append({"scenario": scen, "system": sysn, "mean": x.mean(), "t_lo": x.mean() - h,
                     "t_hi": x.mean() + h, "t_exceeds_1": bool(x.mean() + h > 1),
                     "logit_lo": lo, "logit_hi": hi})
    ci = pd.DataFrame(rows)
    ci.to_csv(out / "bounded_seed_intervals.csv", index=False)
    lines += [f"## Seed-level t-intervals: {int(ci.t_exceeds_1.sum())}/{len(ci)} exceed AP=1.0 "
              "(invalid); logit-scale intervals are bounded and in `bounded_seed_intervals.csv`.\n"]

    # ---- 5. ranking stability ---------------------------------------------------------
    syss = ["bank_A", "bank_B", "bank_C", "ensemble_equal", "fedavg_global", "kd_committee"]
    rk = []
    for scen in PRIMARY_SCEN:
        mats = {}
        for sd in sorted(d.seed.unique()):
            v = d[(d.scenario == scen) & (d.seed == sd) & (d.variant == "primary") & d.system.isin(syss)]
            mats[sd] = v.set_index("system")[AP].reindex(syss)
        for s1, s2 in itertools.combinations(mats, 2):
            tau = stats.kendalltau(mats[s1], mats[s2]).statistic
            rk.append({"scenario": scen, "seeds": f"{s1}-{s2}", "kendall_tau": tau})
    rk = pd.DataFrame(rk)
    rk.to_csv(out / "ranking_stability.csv", index=False)
    lines += ["## Ranking stability across seeds (Kendall tau of 6 systems by test AP)\n",
              rk.round(3).to_markdown(index=False), f"\nMean tau = {rk.kendall_tau.mean():.3f}\n"]

    # ---- 6. best system per cell -----------------------------------------------------
    w = d[(d.variant == "primary") & d.system.isin(syss)].pivot_table(
        index=["scenario", "seed"], columns="system", values=AP)
    best = w.idxmax(axis=1).value_counts()
    lines += ["## Which system has the highest test AP per (scenario, seed) cell (9 cells)\n",
              best.to_frame("n_cells").to_markdown(), ""]

    (out / "reanalysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    main(a.results, a.out)
