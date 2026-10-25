"""Step 6: run posthoc.py over every unit of the selected groups and aggregate (mean over cells, then over groups)."""
import io, sys, contextlib
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import posthoc

R = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
dup = Path(sys.argv[3]) if len(sys.argv) > 3 else None
groups = ["paper", "tuned"] + [f"split_{i}" for i in range(1, 11)]
parts = {"calibration": [], "combiners": [], "cost": [], "duplicates": []}
tmp = OUT / "_tmp"
for g in groups:
    for unit in sorted((R / g).glob("unit_s*_*")):
        if unit.is_file() or unit.suffix == ".log":
            continue
        o = tmp / g / unit.name
        with contextlib.redirect_stdout(io.StringIO()):
            posthoc.main(unit, o, dup if g == "paper" else None)
        for k in parts:
            f = o / f"{k}.csv"
            if f.is_file():
                d = pd.read_csv(f); d["cgroup"] = g; d["unit"] = unit.name; parts[k].append(d)
for k, v in parts.items():
    if v:
        pd.concat(v).to_csv(OUT / f"{k}_all.csv", index=False)
cal = pd.read_csv(OUT / "calibration_all.csv"); comb = pd.read_csv(OUT / "combiners_all.csv"); cost = pd.read_csv(OUT / "cost_all.csv")
md = []
def sec(t, df): md.extend([f"## {t}\n", df.to_markdown(), ""])
main = comb[comb.cgroup.isin(["paper"] + [f"split_{i}" for i in range(1, 11)])]
sec("Calibration, mean over all default-hyperparameter cells", cal[cal.cgroup != "tuned"].groupby(["group", "system"])[["ap", "brier", "logloss", "ece", "mean_pred", "prevalence"]].mean().round(5))
piv = main.pivot_table(index=["cgroup", "unit", "seed", "scenario", "group"], columns="combiner", values="ap")
diff = piv.sub(piv["mean"], axis=0)
sec("Combiners: test AP; diff vs paper mean (all %d cells x {local, kd})" % (len(piv) // 2),
    pd.DataFrame({"ap": piv.groupby("group").mean().stack(), "diff_vs_mean": diff.groupby("group").mean().stack(),
                  "frac_cells_better": (diff > 0).groupby("group").mean().stack()}).round(4))
sec("Cost / alert budget (mean over cells, default hyperparameters; lower cost better)",
    cost[cost.cgroup != "tuned"].groupby(["group", "system"]).mean(numeric_only=True).drop(columns="seed").round(1))
if (OUT / "duplicates_all.csv").is_file():
    dd = pd.read_csv(OUT / "duplicates_all.csv")
    sec("AP on test rows with / without an exact training twin (split 42 paper group)", dd.groupby(["system", "subset"])[["n", "n_fraud", "ap"]].mean().round(4))
(OUT / "step6.md").write_text("\n".join(md)); print("\n".join(md))
