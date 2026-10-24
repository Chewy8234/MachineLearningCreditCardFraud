"""Leakage control: these tests fail if any split contaminates another."""

from __future__ import annotations

import numpy as np
import pytest

from heterofl.data.partitioning import LeakageError, assert_disjoint
from heterofl.data.pipeline import prepare_scenario
from heterofl.data.preprocessing import FeaturePreprocessor
from heterofl.utils.io import load_npz, read_json


def test_all_sets_mutually_disjoint(prepared):
    cfg, bundle, data = prepared
    sets = {f"bank_{b}": load_npz(data.view_dir / f"bank_{b}.npz")["row_ids"] for b in data.banks}
    sp = bundle.splits
    rid = bundle.row_ids
    sets.update({k: rid[sp[k]] for k in ("proxy", "validation", "test")})
    assert_disjoint(sets)
    assert data.manifest["overlap_check"]["all_disjoint"]
    assert bundle.manifest["overlap_check"]["all_disjoint"]
    assert data.manifest["coverage_check"]


def test_proxy_view_has_no_labels_and_test_is_not_in_view(prepared):
    _, bundle, data = prepared
    proxy = load_npz(data.view_dir / "proxy.npz")
    assert set(proxy) == {"X", "row_ids"}
    test_ids = set(bundle.row_ids[bundle.splits["test"]].tolist())
    for f in data.view_dir.glob("*.npz"):
        ids = set(load_npz(f)["row_ids"].tolist())
        assert not ids & test_ids, f"test rows leaked into {f.name}"
    assert not (data.view_dir / "test.npz").exists()


def test_scaler_is_fitted_on_proxy_features_only(prepared):
    cfg, bundle, _ = prepared
    import pandas as pd

    from heterofl.data.loader import features_and_target

    pre = FeaturePreprocessor.from_dict(read_json(bundle.dir / "preprocessor.json"))
    assert "proxy" in pre.fitted_on
    X_raw, _ = features_and_target(pd.read_csv(cfg["data"]["path"]))
    ref = FeaturePreprocessor().fit(X_raw[bundle.splits["proxy"]], source="check")
    np.testing.assert_allclose(pre.mean_, ref.mean_)
    np.testing.assert_allclose(pre.scale_, ref.scale_)
    wrong = FeaturePreprocessor().fit(X_raw[bundle.splits["test"]], source="test")
    assert not np.allclose(pre.mean_, wrong.mean_)


def test_split_identical_across_seeds_and_scenarios(prepared):
    cfg, bundle, data = prepared
    other = prepare_scenario(cfg, bundle, 123, "moderate_non_iid")
    v1, v2 = load_npz(data.view_dir / "validation.npz"), load_npz(other.view_dir / "validation.npz")
    np.testing.assert_array_equal(v1["row_ids"], v2["row_ids"])
    assert data.manifest["proxy_row_id_sha256"] == other.manifest["proxy_row_id_sha256"]
    # client pool is identical; only its partition differs
    pool1 = np.sort(np.concatenate([load_npz(data.view_dir / f"bank_{b}.npz")["row_ids"] for b in data.banks]))
    pool2 = np.sort(np.concatenate([load_npz(other.view_dir / f"bank_{b}.npz")["row_ids"] for b in data.banks]))
    np.testing.assert_array_equal(pool1, pool2)


def test_partition_reuse_is_deterministic(prepared):
    cfg, bundle, data = prepared
    import shutil

    before = {b: load_npz(data.view_dir / f"bank_{b}.npz")["row_ids"] for b in data.banks}
    shutil.rmtree(data.dir)
    again = prepare_scenario(cfg, bundle, 42, "moderate_non_iid")
    for b in data.banks:
        np.testing.assert_array_equal(before[b], load_npz(again.view_dir / f"bank_{b}.npz")["row_ids"])


def test_client_refuses_a_proxy_file_with_labels(tmp_path):
    from heterofl.federation.client_app import _proxy
    from heterofl.federation.job import FederationError
    from heterofl.utils.io import save_npz

    save_npz(tmp_path / "proxy.npz", X=np.zeros((3, 30)), row_ids=np.arange(3), y=np.zeros(3))
    with pytest.raises(FederationError):
        _proxy(str(tmp_path))


def test_pooled_centralized_data_equals_union_of_banks(prepared):
    _, bundle, data = prepared
    X, y = data.pooled_clients()
    assert len(y) == len(bundle.splits["client_pool"])
    assert int(y.sum()) == int(bundle.y[bundle.splits["client_pool"]].sum())


def test_overlap_injection_is_detected():
    with pytest.raises(LeakageError):
        assert_disjoint({"bank_A": np.arange(10), "test": np.arange(9, 20)})
