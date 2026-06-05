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
    check_residual_autocorrelation,
    classify_volatility_regime,
    compute_dm_influence,
    compute_hac_variance,
    compute_mae,
    compute_move_conditional_metrics,
    compute_move_threshold,
    compute_rmse,
    detect_changepoints,
    dm_test,
    pt_test,
)
from temporalcv.metrics.event import compute_pr_auc
from temporalcv.statistical_tests import compute_self_normalized_variance
from temporalcv.viz.intervals import PredictionIntervalDisplay


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


# =============================================================================
# Review remediation — lock the arithmetic-core, the two new as_array sites,
# and the three hand-fixed forwarding spots against future dropped-normalization.
# =============================================================================


@pytest.mark.parametrize("container", [list, tuple])
def test_pt_test_accepts_list_like(container: type) -> None:
    # First core op is boolean-mask assignment `classes[values > thr] = 1` — a raw list breaks.
    rng = np.random.default_rng(5)
    a = rng.standard_normal(50)
    p = a + rng.standard_normal(50) * 0.4
    assert _close(
        pt_test(container(a.tolist()), container(p.tolist())).statistic, pt_test(a, p).statistic
    )


def test_compute_move_conditional_metrics_accepts_list_like() -> None:
    # Boolean-mask selection `predictions[mask]` after thresholding — a raw list breaks.
    rng = np.random.default_rng(6)
    p = rng.standard_normal(80)
    a = p + rng.standard_normal(80) * 0.3
    r_list = compute_move_conditional_metrics(p.tolist(), a.tolist())
    r_arr = compute_move_conditional_metrics(p, a)
    assert _close(r_list.mae_up, r_arr.mae_up)
    assert r_list.n_up == r_arr.n_up


def test_compute_pr_auc_accepts_list_like() -> None:
    # Sorts/indexes pred_probs[order] for the PR curve — a raw list breaks.
    rng = np.random.default_rng(7)
    pr = rng.random(60)
    binary = (rng.random(60) > 0.5).astype(int)
    assert _close(
        compute_pr_auc(pr.tolist(), binary.tolist()).pr_auc, compute_pr_auc(pr, binary).pr_auc
    )


def test_new_as_array_variance_sites_accept_list_like() -> None:
    # The two functions that gained a NEW as_array(d) call in this PR.
    rng = np.random.default_rng(8)
    d = rng.standard_normal(70)
    assert _close(compute_hac_variance(d.tolist()), compute_hac_variance(d))
    assert _close(compute_self_normalized_variance(d.tolist()), compute_self_normalized_variance(d))


def test_compute_dm_influence_accepts_list_like() -> None:
    rng = np.random.default_rng(9)
    e1 = rng.standard_normal(60)
    e2 = e1 + rng.standard_normal(60) * 0.5
    r_list = compute_dm_influence(e1.tolist(), e2.tolist())
    r_arr = compute_dm_influence(e1, e2)
    assert np.array_equal(
        np.asarray(r_list.observation_influence), np.asarray(r_arr.observation_influence)
    )


def test_check_residual_autocorrelation_accepts_list_like() -> None:
    # Hand-fixed forwarding spot: normalization order + re-narrowing.
    rng = np.random.default_rng(10)
    resid = rng.standard_normal(100)
    assert (
        check_residual_autocorrelation(resid.tolist()).passed
        == check_residual_autocorrelation(resid).passed
    )


def test_prediction_interval_display_accepts_list_x() -> None:
    # Hand-fixed new as_array site: PredictionIntervalDisplay stored x raw before the sweep.
    rng = np.random.default_rng(11)
    preds = rng.standard_normal(20)
    x = list(range(20))
    disp = PredictionIntervalDisplay(preds, preds - 1.0, preds + 1.0, x=x)
    assert np.array_equal(np.asarray(disp.x), np.asarray(x))
