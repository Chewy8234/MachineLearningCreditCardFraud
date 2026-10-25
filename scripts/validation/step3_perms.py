"""Step 3: is the per-bank KD effect a model-family effect or a data-size effect?

Bank-run table over all 6 bank->model assignments (paper XRM + 5 permutations), 3 scenarios x 3 seeds.
gain_local = KD bank AP - local bank AP ; gain_l0 = KD bank AP - lambda0 bank AP (same extra training, no teacher).
Models: gain ~ C(model) + log10(rows) + log10(frauds), cluster-robust by (scenario,seed) cell.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

R = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
groups = {"XRM": R / "paper", **{p: R / f"perm_{p}" for p in ["MRX", "MXR", "RMX", "RXM", "XMR"]}}
fam = {"X": "xgboost", "R": "random_forest", "M": "mlp"}
rows = []
for perm, path in groups.items():
    mt = {b: fam[c] for b, c in zip("ABC", perm)}
    for m in path.glob("unit_*/seed_*/*/*/metrics.csv"):
        pass
    frames = [pd.read_csv(f) for f in path.glob("unit_*/seed_*/*/**/metrics.csv") if "_archive" not in f.parts]
    d = pd.concat(frames); d = d[d.scenario.isin(["iid", "moderate_non_iid", "severe_non_iid"])]
    ap = lambda var, sysn: d[(d.variant == var) & (d.system == sysn)].set_index(["scenario", "seed"])["test_average_precision"]
    for unit in path.glob("unit_*"):
        for pr in unit.glob("seed_*/*/partition_report.csv"):
            pass
    for b in "ABC":
        kd, l0, loc = ap("primary", f"kd_bank_{b}"), ap("kd_lambda_0.0", f"kd_bank_{b}"), ap("primary", f"bank_{b}")
        for (sc, sd) in kd.index:
            pr = pd.read_csv(next(path.glob(f"unit_s{sd}_{sc}/seed_{sd}/{sc}/partition_report.csv")))
            r = pr[pr.bank == b].iloc[0]
            if (sc, sd) in l0.index and (sc, sd) in loc.index:
                rows.append({"perm": perm, "bank": b, "model": mt[b], "scenario": sc, "seed": sd,
                             "rows": r.n_transactions, "frauds": r.n_fraud,
                             "gain_local": kd[(sc, sd)] - loc[(sc, sd)], "gain_l0": kd[(sc, sd)] - l0[(sc, sd)],
                             "l0_minus_local": l0[(sc, sd)] - loc[(sc, sd)], "cell": f"{sc}|{sd}|{perm}"})
t = pd.DataFrame(rows); t["log_rows"] = np.log10(t.rows); t["log_frauds"] = np.log10(t.frauds)
t["cellkey"] = t.scenario + "|" + t.seed.astype(str)
t.to_csv(OUT / "bank_runs.csv", index=False)
print(f"bank-runs: {len(t)}")
print(t.groupby("model")[["gain_local", "gain_l0", "l0_minus_local"]].agg(["mean", "std"]).round(4))
print(t.groupby(["model", "scenario"])[["gain_local", "gain_l0"]].mean().round(4))
for y in ("gain_local", "gain_l0"):
    f = smf.ols(f"{y} ~ C(model) + log_rows + log_frauds", t).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(t.cellkey)[0]})
    print(f"\n== {y} ~ model + log10 rows + log10 frauds (cluster-robust by scenario|seed)"); print(f.summary().tables[1])
# does the same MODEL show gain independent of which BANK (size) it sits on? rank by size within cell
t["size_rank"] = t.groupby(["perm", "scenario", "seed"])["rows"].rank(ascending=False)
print("\nmean gain_l0 by model x size rank (1=largest bank):")
print(t.pivot_table(index="model", columns="size_rank", values="gain_l0", aggfunc="mean").round(4))
print(t.pivot_table(index="model", columns="size_rank", values="gain_local", aggfunc="mean").round(4))
