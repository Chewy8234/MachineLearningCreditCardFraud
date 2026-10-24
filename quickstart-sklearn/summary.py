"""Print every experiment's results from the CSVs in results/.

Run from this folder after the experiments:  python summary.py
"""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS = ["precision", "recall", "f1", "pr_auc"]
MODEL_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "MLP": "MLP",
}
FED_FILES = {
    "Logistic Regression": "federated_logreg.csv",
    "MLP": "federated_mlp.csv",
    "XGBoost": "federated_xgboost.csv",
    "Random Forest": "federated_rf.csv",
}
COMMITTEE_BANKS = ["XGBoost", "Random Forest", "MLP"]  # bank 0, 1, 2


def selected_round(d: pd.DataFrame) -> pd.Series:
    """The round with the best validation PR-AUC (test is never used to choose)."""
    if "val_pr_auc" not in d:
        return d.iloc[-1]
    return d.loc[d["val_pr_auc"].idxmax()]


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    section("DATASET SPLITS (per bank)")
    print(pd.read_csv(RESULTS_DIR / "splits.csv").to_string(index=False))

    baselines = pd.read_csv(RESULTS_DIR / "baselines.csv")
    baselines["model"] = baselines["model"].map(MODEL_NAMES)
    rows = []
    for (exp, model), g in baselines.groupby(["experiment", "model"]):
        label = "1 Local only" if exp == "local" else "2 Centralized"
        rows.append({"model": model, "experiment": label, **g[METRICS].mean()})
    for model, file in FED_FILES.items():
        path = RESULTS_DIR / file
        if path.exists():
            best = selected_round(pd.read_csv(path))
            rows.append(
                {"model": model, "experiment": "3 Homogeneous FL",
                 "round": int(best["round"]), **best[METRICS]}
            )

    committee_path = RESULTS_DIR / "federated_committee.csv"
    committee = pd.read_csv(committee_path) if committee_path.exists() else None
    if committee is not None:
        best = selected_round(committee)
        rows.append(
            {"model": "XGB + RF + MLP", "experiment": "4 Committee",
             "round": int(best["round"]), **best[METRICS]}
        )

    section("ALL EXPERIMENTS (test set, mean across the 3 banks)")
    table = pd.DataFrame(rows).set_index(["experiment", "model"]).sort_index()
    table[METRICS] = table[METRICS].astype(float).round(3)
    table["round"] = table["round"].map(lambda r: "" if pd.isna(r) else int(r))
    print(table[METRICS + ["round"]].to_string())
    print("round = federated round reported, chosen by best validation PR-AUC")

    section("PR-AUC PER BANK: local only vs centralized")
    per_bank = baselines.pivot_table(
        index=["model", "experiment"], columns="bank", values="pr_auc"
    )
    per_bank.columns = [f"bank {b}" for b in per_bank.columns]
    print(per_bank.round(3).to_string())

    section("PR-AUC BY ROUND: homogeneous FL")
    for model, file in FED_FILES.items():
        path = RESULTS_DIR / file
        if path.exists():
            d = pd.read_csv(path)
            cols = ["round", *(["val_pr_auc"] if "val_pr_auc" in d else []), *METRICS]
            print(f"\n{model} ({len(d)} rounds, reported round "
                  f"{int(selected_round(d)['round'])})")
            print(d[cols].round(3).to_string(index=False))

    if committee is not None:
        section("COMMITTEE BY ROUND (each bank scored with its own model)")
        cols = ["round", *(["val_pr_auc"] if "val_pr_auc" in committee else []), *METRICS]
        cols += [f"bank{b}_pr_auc" for b in range(3) if f"bank{b}_pr_auc" in committee]
        view = committee[cols].rename(
            columns={
                f"bank{b}_pr_auc": f"bank{b} {name} pr_auc"
                for b, name in enumerate(COMMITTEE_BANKS)
            }
        )
        print(view.round(3).to_string(index=False))

        best = selected_round(committee)
        section(f"COMMITTEE vs LOCAL ONLY, per bank (reported round {int(best['round'])})")
        local = baselines[baselines["experiment"] == "local"]
        for b, name in enumerate(COMMITTEE_BANKS):
            before = local[(local["bank"] == b) & (local["model"] == name)]["pr_auc"].iloc[0]
            after = best[f"bank{b}_pr_auc"]
            print(f"bank {b} {name:14s} local {before:.3f} -> committee {after:.3f} "
                  f"({after - before:+.3f})")


if __name__ == "__main__":
    main()
