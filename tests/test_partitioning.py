"""Splitting and IID / non-IID client partitioning."""

from __future__ import annotations

import numpy as np
import pytest

from heterofl.data.partitioning import (
    LeakageError,
    PartitionError,
    assert_disjoint,
    make_partition,
    partition_dirichlet,
    partition_iid,
    split_dataset,
    stratified_subsample,
)


@pytest.fixture()
def y():
    rng = np.random.default_rng(0)
    y = np.zeros(20000, dtype=np.int8)
    y[rng.choice(20000, 300, replace=False)] = 1
    return y


def _split(y, seed=42, sampling="stratified"):
    return split_dataset(y, test_fraction=0.2, validation_fraction=0.15, proxy_fraction=0.1,
                         proxy_sampling=sampling, seed=seed)


def test_split_sizes_disjoint_and_cover(y):
    s = _split(y)
    sets = s.as_dict()
    assert_disjoint(sets)
    assert sum(len(v) for v in sets.values()) == len(y)
    assert len(s.test) == pytest.approx(0.2 * len(y), abs=1)
    dev = 0.8 * len(y)
    assert len(s.validation) == pytest.approx(0.15 * dev, abs=1)
    assert len(s.proxy) == pytest.approx(0.1 * 0.85 * dev, abs=2)


def test_split_preserves_fraud_counts_and_stratifies(y):
    s = _split(y)
    total = sum(int(y[v].sum()) for v in s.as_dict().values())
    assert total == int(y.sum())
    for name, idx in s.as_dict().items():
        assert y[idx].mean() == pytest.approx(y.mean(), rel=0.1), name


def test_split_is_deterministic_and_seed_dependent(y):
    a, b, c = _split(y, 42), _split(y, 42), _split(y, 7)
    for k in a.as_dict():
        np.testing.assert_array_equal(a.as_dict()[k], b.as_dict()[k])
    assert not np.array_equal(a.test, c.test)


def test_random_proxy_option_runs(y):
    s = _split(y, sampling="random")
    assert_disjoint(s.as_dict())


def test_stratified_subsample(y):
    idx = stratified_subsample(y, 0.25, 0)
    assert len(idx) == pytest.approx(5000, abs=1)
    assert int(y[idx].sum()) == pytest.approx(75, abs=1)


def test_assert_disjoint_detects_overlap():
    with pytest.raises(LeakageError):
        assert_disjoint({"a": np.array([1, 2, 3]), "b": np.array([3, 4])})
    with pytest.raises(LeakageError):
        assert_disjoint({"a": np.array([1, 1])})


def test_iid_partition_is_balanced(y):
    pool = np.arange(len(y))
    parts = partition_iid(pool, y, 3, seed=1)
    assert_disjoint({str(i): p for i, p in enumerate(parts)})
    sizes = [len(p) for p in parts]
    frauds = [int(y[p].sum()) for p in parts]
    assert max(sizes) - min(sizes) <= 2
    assert max(frauds) - min(frauds) <= 1
    assert sum(frauds) == int(y.sum())


def test_dirichlet_partition_respects_constraints_and_is_deterministic(y):
    pool = np.arange(len(y))
    p1, info = partition_dirichlet(pool, y, 3, 0.1, 5, min_fraud=10, min_legit=500, max_retries=2000)
    p2, _ = partition_dirichlet(pool, y, 3, 0.1, 5, min_fraud=10, min_legit=500, max_retries=2000)
    for a, b in zip(p1, p2, strict=True):
        np.testing.assert_array_equal(a, b)
    assert all(int(y[p].sum()) >= 10 for p in p1)
    assert all(int((y[p] == 0).sum()) >= 500 for p in p1)
    assert sum(int(y[p].sum()) for p in p1) == int(y.sum())
    assert info["accepted_attempt"] < 2000


def test_dirichlet_yields_heterogeneous_fraud_rates(y):
    pool = np.arange(len(y))
    parts, _ = partition_dirichlet(pool, y, 3, 0.1, 3, min_fraud=5, min_legit=200, max_retries=2000)
    rates = [y[p].mean() for p in parts]
    iid_rates = [y[p].mean() for p in partition_iid(pool, y, 3, 3)]
    assert np.std(rates) > 5 * np.std(iid_rates)


def test_impossible_configuration_fails_loudly(y):
    pool = np.arange(len(y))
    with pytest.raises(PartitionError, match="Infeasible"):
        partition_dirichlet(pool, y, 3, 0.5, 0, min_fraud=200, min_legit=10, max_retries=5)
    with pytest.raises(PartitionError, match="no draw"):
        partition_dirichlet(pool, y, 3, 0.01, 0, min_fraud=95, min_legit=6000, max_retries=3)


def test_make_partition_covers_pool_exactly(y):
    pool = np.sort(np.random.default_rng(0).choice(len(y), 12000, replace=False))
    for sc in ({"type": "iid"}, {"type": "dirichlet", "alpha": 0.5}):
        parts, _ = make_partition(pool, y, sc, 3, 1, min_fraud=5, min_legit=100, max_retries=500)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), pool)
