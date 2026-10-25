"""Shared fixtures. All data here is SYNTHETIC and NON-PUBLISHABLE."""

from __future__ import annotations

from pathlib import Path

import heterofl  # noqa: F401  (imported first: fixes the xgboost/torch OpenMP load order)
import numpy as np
import pytest

from heterofl.data.synthetic import write_synthetic_creditcard
from heterofl.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]

TINY_MODELS = [
    "models.xgboost.n_estimators=20", "models.xgboost.max_depth=3",
    "models.random_forest.n_estimators=15", "models.mlp.hidden_sizes=[16, 8]",
    "models.mlp.epochs=3", "models.mlp.batch_size=256",
]


def make_config(tmp: Path, csv: Path, extra: list[str] | None = None) -> dict:
    """The synthetic smoke profile, redirected entirely into ``tmp``."""
    overrides = [
        f"data.path={csv}", "data.synthetic=true", "data.subsample_fraction=null",
        f"data.processed_dir={tmp / 'processed'}", f"data.partitions_dir={tmp / 'partitions'}",
        f"results_dir={tmp / 'results'}", "partitioning.min_fraud_per_client=5",
        "partitioning.min_legit_per_client=100", "federation.client_num_cpus=1",
        "federation.job_timeout=600", "evaluation.bootstrap.n_resamples=50",
        "fedavg.local_epochs=1", *TINY_MODELS, *(extra or []),
    ]
    return load_config(ROOT / "configs" / "smoke_synthetic.yaml", overrides)


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthetic")
    return write_synthetic_creditcard(d / "creditcard.csv", n_rows=6000, fraud_rate=0.03, seed=1)


@pytest.fixture()
def cfg(tmp_path, synthetic_csv) -> dict:
    return make_config(tmp_path, synthetic_csv)


@pytest.fixture()
def prepared(cfg):
    from heterofl.data.pipeline import prepare_bundle, prepare_scenario

    bundle = prepare_bundle(cfg)
    data = prepare_scenario(cfg, bundle, 42, "moderate_non_iid")
    return cfg, bundle, data


@pytest.fixture()
def toy_xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 30)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=1500) > 2.0).astype(np.int64)
    return X, y
