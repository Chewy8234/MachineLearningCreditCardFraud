"""Experiments 1 and 2: local-only models per bank, and a centralized benchmark.

Uses the same 3-bank split and per-bank test sets as the Flower app, so results
are directly comparable with the federated experiments.

Run from this folder:  python baselines.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearnexample.task import SEED, best_threshold, compute_metrics, load_data

NUM_BANKS = 3
DATA_PATH = Path(__file__).resolve().parent.parent / "creditcard.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def make_models(y_train) -> dict:
    """The model families compared in the paper, each tuned for class imbalance."""
    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=fraud_ratio,  # upweight the rare fraud class
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=SEED,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            max_iter=200,
            random_state=SEED,
        ),
    }


def main():
    banks = [load_data(i, NUM_BANKS, str(DATA_PATH)) for i in range(NUM_BANKS)]
    rows = []

    # Dataset split table: size and fraud count of every bank's train/val/test
    split_rows = []
    for bank_id, bank in enumerate(banks):
        for split, y in zip(["train", "val", "test"], bank[1::2]):
            split_rows.append(
                {"bank": bank_id, "split": split, "transactions": len(y),
                 "fraud": int(y.sum()), "fraud_pct": round(100 * y.mean(), 3)}
            )
    splits = pd.DataFrame(split_rows)
    print(splits.to_string(index=False), end="\n\n")

    def evaluate(model, X_val, y_val, X_test, y_test):
        # Threshold is chosen on validation; test is only used for the final score
        threshold = best_threshold(y_val, model.predict_proba(X_val)[:, 1])
        metrics = compute_metrics(y_test, model.predict_proba(X_test)[:, 1], threshold)
        return {"threshold": threshold, **metrics}

    # Experiment 1: each bank trains only on its own data
    for bank_id, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(banks):
        for name, model in make_models(y_train).items():
            print(f"[local] bank {bank_id} - {name}")
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_val, y_val, X_test, y_test)
            rows.append({"experiment": "local", "model": name, "bank": bank_id, **metrics})

    # Experiment 2: all banks' training data pooled in one place
    X_all = np.vstack([b[0] for b in banks])
    y_all = np.concatenate([b[1] for b in banks])
    X_val_all = np.vstack([b[2] for b in banks])
    y_val_all = np.concatenate([b[3] for b in banks])
    for name, model in make_models(y_all).items():
        print(f"[centralized] {name}")
        model.fit(X_all, y_all)
        # Score on each bank's own test set, same as the federated runs
        for bank_id, (_, _, _, _, X_test, y_test) in enumerate(banks):
            metrics = evaluate(model, X_val_all, y_val_all, X_test, y_test)
            rows.append(
                {"experiment": "centralized", "model": name, "bank": bank_id, **metrics}
            )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "baselines.csv", index=False)
    splits.to_csv(RESULTS_DIR / "splits.csv", index=False)

    summary = results.groupby(["experiment", "model"])[
        ["precision", "recall", "f1", "pr_auc"]
    ].mean()
    print("\nMean across the 3 banks' test sets:")
    print(summary.round(3).to_string())
    print(f"\nPer-bank results saved to {RESULTS_DIR / 'baselines.csv'}")


if __name__ == "__main__":
    main()
