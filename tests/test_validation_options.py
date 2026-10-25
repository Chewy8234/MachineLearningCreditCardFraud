"""Validation-campaign options: temporal split and Time masking."""

from __future__ import annotations

import numpy as np
import pytest

from heterofl.data.partitioning import assert_disjoint, split_dataset
from heterofl.data.pipeline import prepare_bundle
from heterofl.utils.config import ConfigError
from tests.conftest import make_config


@pytest.fixture()
def y_time():
    rng = np.random.default_rng(0)
    y = np.zeros(20000, dtype=np.int8)
    y[rng.choice(20000, 300, replace=False)] = 1
    t = rng.permutation(20000).astype(float)  # row order unrelated to time
    return y, t


def _temporal(y, t, **kw):
    return split_dataset(y, test_fraction=0.2, validation_fraction=0.15, proxy_fraction=0.1,
                         proxy_sampling="stratified", seed=1, mode="temporal", time=t, **kw)


def test_temporal_split_orders_train_validation_test_and_is_disjoint(y_time):
    y, t = y_time
    s = _temporal(y, t)
    assert_disjoint(s.as_dict())
    assert sum(len(v) for v in s.as_dict().values()) == len(y)
    train = np.concatenate([s.client_pool, s.proxy])
    assert t[train].max() < t[s.validation].min()
    assert t[s.validation].max() < t[s.test].min()
    assert len(s.test) == int(round(0.2 * len(y)))


def test_temporal_split_is_seed_independent_for_test_and_validation(y_time):
    y, t = y_time
    a = split_dataset(y, test_fraction=0.2, validation_fraction=0.15, proxy_fraction=0.1,
                      proxy_sampling="stratified", seed=1, mode="temporal", time=t)
    b = split_dataset(y, test_fraction=0.2, validation_fraction=0.15, proxy_fraction=0.1,
                      proxy_sampling="stratified", seed=2, mode="temporal", time=t)
    assert np.array_equal(a.test, b.test) and np.array_equal(a.validation, b.validation)


def test_temporal_requires_time_and_unknown_mode_rejected(y_time):
    y, t = y_time
    with pytest.raises(ValueError, match="Time"):
        split_dataset(y, test_fraction=0.2, validation_fraction=0.15, proxy_fraction=0.1,
                      proxy_sampling="stratified", seed=1, mode="temporal")
    with pytest.raises(ValueError, match="mode"):
        split_dataset(y, test_fraction=0.2, validation_fraction=0.15, proxy_fraction=0.1,
                      proxy_sampling="stratified", seed=1, mode="bogus", time=t)


def test_invalid_mode_in_config_rejected(tmp_path, synthetic_csv):
    with pytest.raises(ConfigError, match="split.mode"):
        make_config(tmp_path, synthetic_csv, ["split.mode=bogus"])


def test_default_mode_keeps_bundle_tag_unchanged(tmp_path, synthetic_csv):
    base = prepare_bundle(make_config(tmp_path / "a", synthetic_csv))
    explicit = prepare_bundle(make_config(tmp_path / "b", synthetic_csv, ["split.mode=random"]))
    assert base.manifest["tag"] == explicit.manifest["tag"]
    assert np.array_equal(base.splits["test"], explicit.splits["test"])


def test_mask_time_zeroes_time_but_preserves_split(tmp_path, synthetic_csv):
    base = prepare_bundle(make_config(tmp_path / "a", synthetic_csv))
    masked = prepare_bundle(make_config(tmp_path / "b", synthetic_csv, ["preprocessing.mask_time=true"]))
    assert base.manifest["tag"] != masked.manifest["tag"]
    assert np.array_equal(base.splits["test"], masked.splits["test"])
    assert np.allclose(np.asarray(masked.X)[:, 0], 0.0)
    assert np.asarray(base.X)[:, 0].std() > 0.5


def test_temporal_bundle_has_time_ordered_test(tmp_path, synthetic_csv):
    import pandas as pd
    b = prepare_bundle(make_config(tmp_path, synthetic_csv, ["split.mode=temporal"]))
    t = pd.read_csv(synthetic_csv)["Time"].to_numpy()
    train = np.concatenate([b.splits["client_pool"], b.splits["proxy"]])
    assert t[train].max() <= t[b.splits["validation"]].min()
    assert t[b.splits["validation"]].max() <= t[b.splits["test"]].min()
