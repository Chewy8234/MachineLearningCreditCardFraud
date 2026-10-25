"""Server-side teacher aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from heterofl.distillation.aggregation import (
    AggregationError,
    compute_teachers,
    full_committee_teachers,
    leave_one_out_teachers,
    validation_ap_weights,
    weighted_average,
)


@pytest.fixture()
def preds():
    rng = np.random.default_rng(0)
    return {"A": rng.random(100), "B": rng.random(100), "C": rng.random(100)}


def test_leave_one_out_is_mean_of_the_other_two(preds):
    t = leave_one_out_teachers(preds)
    np.testing.assert_allclose(t["A"], (preds["B"] + preds["C"]) / 2)
    np.testing.assert_allclose(t["B"], (preds["A"] + preds["C"]) / 2)
    np.testing.assert_allclose(t["C"], (preds["A"] + preds["B"]) / 2)


def test_no_client_receives_its_own_or_a_wrong_teacher(preds):
    t = leave_one_out_teachers(preds)
    for k in preds:
        others = [j for j in preds if j != k]
        expected = np.mean([preds[j] for j in others], axis=0)
        np.testing.assert_allclose(t[k], expected)
        for j in preds:
            if j != k:
                assert not np.allclose(t[k], t[j])


def test_full_committee_includes_self(preds):
    t = full_committee_teachers(preds)
    mean = (preds["A"] + preds["B"] + preds["C"]) / 3
    for k in preds:
        np.testing.assert_allclose(t[k], mean)


def test_weighted_leave_one_out(preds):
    w = {"A": 0.2, "B": 0.3, "C": 0.5}
    t = leave_one_out_teachers(preds, w)
    np.testing.assert_allclose(t["A"], (0.3 * preds["B"] + 0.5 * preds["C"]) / 0.8)


def test_compute_teachers_validates_vectors(preds):
    with pytest.raises(AggregationError):
        compute_teachers({**preds, "A": preds["A"][:-1]}, "leave_one_out", 100)
    with pytest.raises(AggregationError):
        compute_teachers({**preds, "B": np.full(100, np.nan)}, "leave_one_out", 100)
    with pytest.raises(AggregationError):
        compute_teachers({**preds, "C": np.full(100, 1.5)}, "leave_one_out", 100)
    with pytest.raises(AggregationError):
        compute_teachers(preds, "nonsense", 100)
    t = compute_teachers(preds, "leave_one_out", 100)
    assert set(t) == {"A", "B", "C"}


def test_validation_ap_weights():
    w = validation_ap_weights({"A": 0.8, "B": 0.4, "C": float("nan")})
    assert sum(w.values()) == pytest.approx(1)
    assert w["A"] == pytest.approx(2 * w["B"])
    assert 0 < w["C"] < 0.01


def test_weighted_average_bounds():
    p = {"A": np.array([0.0, 1.0]), "B": np.array([1.0, 1.0])}
    np.testing.assert_allclose(weighted_average(p, ["A", "B"]), [0.5, 1.0])
    with pytest.raises(AggregationError):
        weighted_average(p, [])


# --------------------------------------------------------------------------- weight validation
def test_valid_weights_are_normalised(preds):
    t = weighted_average(preds, ["A", "B"], {"A": 2.0, "B": 6.0, "C": 0.0})
    np.testing.assert_allclose(t, 0.25 * preds["A"] + 0.75 * preds["B"])


@pytest.mark.parametrize("weights, match", [
    ({"A": 0.0, "B": 0.0}, "sum to zero"),
    ({"A": -0.5, "B": 1.0}, "Negative"),
    ({"A": float("nan"), "B": 1.0}, "Non-finite"),
    ({"A": float("inf"), "B": 1.0}, "Non-finite"),
    ({"A": 1.0}, "No aggregation weight"),
    ({"A": "x", "B": 1.0}, "Non-numeric"),
])
def test_degenerate_weights_raise_instead_of_falling_back(preds, weights, match):
    with pytest.raises(AggregationError, match=match):
        weighted_average(preds, ["A", "B"], weights)


def test_zero_weight_leave_one_out_teacher_is_an_error(preds):
    # B and C both weight 0 -> A's teacher (mean of B, C) is undefined
    with pytest.raises(AggregationError, match="sum to zero"):
        leave_one_out_teachers(preds, {"A": 1.0, "B": 0.0, "C": 0.0})


def test_validation_ap_floor_is_observable(caplog):
    diag = {}
    with caplog.at_level("WARNING", logger="heterofl.distillation.aggregation"):
        w = validation_ap_weights({"A": 0.8, "B": -1.0, "C": float("nan")}, diagnostics=diag)
    assert diag["floored"] == {"B": -1.0, "C": None} and diag["floor"] == 1e-3
    assert "floor" in caplog.text
    assert w["B"] == w["C"] == pytest.approx(1e-3 / (0.8 + 2e-3))
    diag = {}
    validation_ap_weights({"A": 0.8, "B": 0.4}, diagnostics=diag)
    assert diag["floored"] == {}


def test_validation_ap_weights_reject_empty_or_bad_floor():
    with pytest.raises(AggregationError):
        validation_ap_weights({})
    with pytest.raises(AggregationError):
        validation_ap_weights({"A": 0.5}, floor=0.0)
