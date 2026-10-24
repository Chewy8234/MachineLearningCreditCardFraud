"""Synthetic creditcard.csv-shaped data for unit/smoke tests ONLY.

NOT PUBLISHABLE. These records are random draws from Gaussian class-conditional
distributions with the same schema as the ULB dataset. Any result computed on
them says nothing about fraud detection; run outputs are flagged
``publishable: false`` when ``data.synthetic`` is set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heterofl.data.loader import FEATURE_COLUMNS, TARGET_COLUMN

SYNTHETIC_MARKER = "SYNTHETIC-NON-PUBLISHABLE"


def make_synthetic_creditcard(n_rows: int = 6000, fraud_rate: float = 0.02,
                              seed: int = 0, separation: float = 1.5) -> pd.DataFrame:
    """Return a DataFrame with the ULB schema and a learnable (but fake) signal."""
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_rows * fraud_rate)), 2)
    y = np.zeros(n_rows, dtype=np.int64)
    y[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    v = rng.standard_normal((n_rows, 28))
    shift = np.zeros(28)
    shift[:6] = separation * np.array([1.0, -1.0, 0.8, -0.8, 0.6, -0.6])
    v[y == 1] += shift
    time = np.sort(rng.uniform(0, 172_800, n_rows))
    amount = np.round(rng.lognormal(3.0, 1.2, n_rows) * np.where(y == 1, 1.5, 1.0), 2)

    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df[TARGET_COLUMN] = y
    assert list(df.columns[:-1]) == list(FEATURE_COLUMNS)
    return df


def write_synthetic_creditcard(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_creditcard(**kwargs).to_csv(path, index=False)
    path.with_suffix(".SYNTHETIC.txt").write_text(
        f"{SYNTHETIC_MARKER}: {path.name} is randomly generated test data.\n")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write a NON-PUBLISHABLE synthetic creditcard.csv")
    ap.add_argument("--out", default="data/raw/synthetic_creditcard.csv")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--fraud-rate", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = write_synthetic_creditcard(a.out, n_rows=a.rows, fraud_rate=a.fraud_rate, seed=a.seed)
    print(f"Wrote {p} ({SYNTHETIC_MARKER})")
