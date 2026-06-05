"""Tests for the v2.0 backend-agnostic public contract (issue #15).

Public *input* parameters accept ``ArrayLike`` (lists, tuples, ndarrays); because the
implementation normalizes at the boundary, the result must be **identical** regardless of the
input container. These tests pin the observable payoff of the ArrayLike sweep across a
representative slice of the public surface (metrics, statistical tests, regimes, persistence,
conformal prediction, stationarity, changepoint). The return types stay concrete ``np.ndarray``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from temporalcv import (
    SplitConformalPredictor,
    adf_test,
    classify_volatility_regime,
    compute_mae,
    compute_move_threshold,
    compute_rmse,
    detect_changepoints,
    dm_test,
)


def _close(a: Any, b: Any) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True))


def test_core_metrics_accept_list_tuple_array() -> None:
    preds = [1.0, 2.0, 3.0, 4.0, 5.0]
    acts = [1.1, 1.9, 3.2, 3.8, 5.1]
    arr_mae = compute_mae(np.array(preds), np.array(acts))
    arr_rmse = compute_rmse(np.array(preds), np.array(acts))
    assert _close(compute_mae(preds, acts), arr_mae)
    assert _close(compute_mae(tuple(preds), tuple(acts)), arr_mae)
    assert _close(compute_rmse(preds, acts), arr_rmse)


def test_dm_test_accepts_list_like() -> None:
    rng = np.random.default_rng(0)
    e1 = rng.standard_normal(60)
    e2 = e1 + rng.standard_normal(60) * 0.5
    r_list = dm_test(e1.tolist(), e2.tolist())
    r_arr = dm_test(e1, e2)
    assert _close(r_list.statistic, r_arr.statistic)
    assert _close(r_list.pvalue, r_arr.pvalue)


def test_classify_volatility_regime_accepts_list_like() -> None:
    rng = np.random.default_rng(1)
    vals = rng.standard_normal(60)
    r_list = classify_volatility_regime(vals.tolist())
    r_arr = classify_volatility_regime(vals)
    assert np.array_equal(np.asarray(r_list), np.asarray(r_arr))


def test_compute_move_threshold_accepts_list_like() -> None:
    acts = np.sin(np.linspace(0, 10, 80))
    assert _close(compute_move_threshold(acts.tolist()), compute_move_threshold(acts))


def test_split_conformal_accepts_list_like() -> None:
    rng = np.random.default_rng(2)
    preds = rng.standard_normal(50)
    acts = preds + rng.standard_normal(50) * 0.1
    c_list = SplitConformalPredictor(alpha=0.1)
    c_list.calibrate(preds.tolist(), acts.tolist())
    iv_list = c_list.predict_interval(preds.tolist())
    c_arr = SplitConformalPredictor(alpha=0.1)
    c_arr.calibrate(preds, acts)
    iv_arr = c_arr.predict_interval(preds)
    assert _close(iv_list.lower, iv_arr.lower)
    assert _close(iv_list.upper, iv_arr.upper)


def test_adf_test_accepts_list_like() -> None:
    rng = np.random.default_rng(3)
    series = np.cumsum(rng.standard_normal(100))
    r_list = adf_test(series.tolist())
    r_arr = adf_test(series)
    assert _close(r_list.statistic, r_arr.statistic)
    assert r_list.is_stationary == r_arr.is_stationary


def test_detect_changepoints_accepts_list_like() -> None:
    series = [0.0] * 30 + [5.0] * 30
    r_list = detect_changepoints(series)
    r_arr = detect_changepoints(np.array(series))
    assert r_list.n_segments == r_arr.n_segments
    assert np.array_equal(np.asarray(r_list.changepoints), np.asarray(r_arr.changepoints))


@pytest.mark.parametrize("container", [list, tuple])
def test_metrics_container_agnostic(container: type) -> None:
    rng = np.random.default_rng(4)
    preds = rng.standard_normal(40)
    acts = preds + rng.standard_normal(40) * 0.2
    expected = compute_mae(preds, acts)
    assert _close(compute_mae(container(preds.tolist()), container(acts.tolist())), expected)
