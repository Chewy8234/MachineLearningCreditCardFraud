"""Loading the ULB Credit Card Fraud Detection dataset (``creditcard.csv``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from heterofl.utils.logging import get_logger
from heterofl.utils.reproducibility import sha256_file

log = get_logger("data.loader")

FEATURE_COLUMNS: tuple[str, ...] = ("Time", *(f"V{i}" for i in range(1, 29)), "Amount")
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURE_COLUMNS, TARGET_COLUMN)

#: SHA-256 of the canonical Kaggle ``mlg-ulb/creditcardfraud`` creditcard.csv.
KNOWN_ULB_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD = 492

DATASET_INSTRUCTIONS = """\
The ULB Credit Card Fraud Detection dataset was not found at:
    {path}

Obtain it from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
either by downloading creditcard.csv manually, or with kagglehub:

    python -c "import kagglehub, shutil; p = kagglehub.dataset_download('mlg-ulb/creditcardfraud'); \\
shutil.copy(p + '/creditcard.csv', 'data/raw/creditcard.csv')"

Place the file at data/raw/creditcard.csv, then re-run:

    python -m heterofl.experiments.runner --config configs/smoke.yaml

To exercise the pipeline WITHOUT the real data (non-publishable synthetic data):

    python -m heterofl.experiments.runner --config configs/smoke_synthetic.yaml
"""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when creditcard.csv is missing, with instructions to obtain it."""


def load_creditcard(path: str | Path) -> pd.DataFrame:
    """Read creditcard.csv with explicit dtypes. Does not drop or alter records."""
    path = Path(path)
    if not path.is_file():
        raise DatasetNotFoundError(DATASET_INSTRUCTIONS.format(path=path))
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    for col in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col!r} is not numeric (dtype {df[col].dtype})")
    log.info("Loaded %s: %d rows x %d columns", path, len(df), df.shape[1])
    return df


def dataset_fingerprint(path: str | Path) -> dict:
    """File hash plus whether it matches the canonical ULB release."""
    digest = sha256_file(path)
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest,
        "matches_known_ulb_release": digest == KNOWN_ULB_SHA256,
        "size_bytes": Path(path).stat().st_size,
    }


def features_and_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw feature matrix (float64, FEATURE_COLUMNS order) and int8 labels."""
    X = df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy()
    return X, y.astype(np.int8)
