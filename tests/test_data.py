"""Dataset loading, auditing and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heterofl.data.loader import (
    FEATURE_COLUMNS,
    DatasetNotFoundError,
    features_and_target,
    load_creditcard,
)
from heterofl.data.preprocessing import AMOUNT_IDX, FeaturePreprocessor
from heterofl.data.synthetic import make_synthetic_creditcard
from heterofl.data.validation import DataValidationError, audit_dataset


def test_missing_file_explains_how_to_get_data(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="kaggle"):
        load_creditcard(tmp_path / "creditcard.csv")


def test_missing_column_rejected(tmp_path):
    df = make_synthetic_creditcard(200, seed=0).drop(columns=["V7"])
    p = tmp_path / "c.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="V7"):
        load_creditcard(p)


def test_audit_reports_class_distribution_and_duplicates_without_removing():
    df = make_synthetic_creditcard(1000, fraud_rate=0.05, seed=0)
    df = pd.concat([df, df.iloc[:7]], ignore_index=True)
    rep = audit_dataset(df, expect_ulb=False)
    assert rep["valid"]
    assert rep["n_rows"] == 1007  # nothing dropped
    assert rep["duplicates"]["exact_duplicate_rows"] == 7
    assert rep["class_distribution"]["fraud"] == int((df["Class"] == 1).sum())
    assert rep["class_distribution"]["legitimate"] + rep["class_distribution"]["fraud"] == 1007


@pytest.mark.parametrize("corrupt", ["nan", "label", "negative_amount", "inf"])
def test_audit_rejects_invalid_values(corrupt):
    df = make_synthetic_creditcard(300, seed=0)
    if corrupt == "nan":
        df.loc[3, "V2"] = np.nan
    elif corrupt == "label":
        df.loc[3, "Class"] = 2
    elif corrupt == "negative_amount":
        df.loc[3, "Amount"] = -1.0
    else:
        df.loc[3, "V5"] = np.inf
    with pytest.raises(DataValidationError):
        audit_dataset(df, expect_ulb=False)


def test_audit_warns_when_not_ulb_shaped():
    rep = audit_dataset(make_synthetic_creditcard(300, seed=0), expect_ulb=True)
    assert any("expected ULB" in w for w in rep["warnings"])


def test_features_and_target_order():
    df = make_synthetic_creditcard(50, seed=0)
    X, y = features_and_target(df)
    assert X.shape == (50, len(FEATURE_COLUMNS))
    np.testing.assert_array_equal(X[:, AMOUNT_IDX], df["Amount"].to_numpy())
    assert set(np.unique(y)) <= {0, 1}


def test_preprocessor_fits_only_reference_and_is_reusable():
    rng = np.random.default_rng(0)
    ref = np.abs(rng.normal(size=(100, 30)))
    other = np.abs(rng.normal(loc=5, size=(100, 30)))
    pre = FeaturePreprocessor().fit(ref, source="proxy")
    before = pre.mean_.copy()
    pre.transform(other)  # transforming never refits
    np.testing.assert_array_equal(pre.mean_, before)
    Z = pre.transform(ref)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-5)
    restored = FeaturePreprocessor.from_dict(pre.to_dict())
    np.testing.assert_allclose(restored.transform(other), pre.transform(other))
    assert pre.to_dict()["fitted_on"] == "proxy"


def test_preprocessor_requires_fit():
    with pytest.raises(RuntimeError):
        FeaturePreprocessor().transform(np.zeros((2, 30)))
