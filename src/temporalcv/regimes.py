"""
Regime Classification Module.

Classify market regimes for conditional performance analysis.

CRITICAL: Volatility must be computed on CHANGES (first differences), NOT levels.
Using levels mislabels steady drifts as "volatile" because:
- A series drifting steadily from 3.0 to 4.0 has high std of LEVELS
- But it has ZERO volatility of changes (constant increments)

Knowledge Tiers
---------------
[T1] Regime-switching theory provides framework (Hamilton 1989, 1994)
[T1] Rolling volatility as regime indicator (standard in finance literature)
[T1] Percentile-based thresholds for regime classification (non-parametric approach)
[T2] Volatility of CHANGES not levels (BUG-005 fix in myga-forecasting-v2)
[T2] 3-class direction (UP/DOWN/FLAT) enables fair persistence comparison (v2)
[T3] 13-week window assumes quarterly seasonality pattern
[T3] 33rd/67th percentiles for regime boundaries (symmetric terciles, not validated)
[T3] Minimum n=10 per regime for reliability (rule of thumb)

Example
-------
>>> from temporalcv.regimes import classify_volatility_regime, classify_direction_regime
>>>
>>> # Classify volatility using changes (correct)
>>> vol_regimes = classify_volatility_regime(changes, window=13, basis='changes')
>>>
>>> # Classify direction using thresholded signs
>>> threshold = compute_move_threshold(train_actuals)
>>> dir_regimes = classify_direction_regime(actuals, threshold)

References
----------
[T1] Hamilton, J.D. (1989). A new approach to the economic analysis of
     nonstationary time series and the business cycle. Econometrica, 57(2), 357-384.
[T1] Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
     Chapter 22: Regime-switching models.
[T1] Stock, J.H. & Watson, M.W. (2001). Forecasting output and inflation:
     The role of asset prices. Journal of Economic Literature, 41(3), 788-829.
[T2] Volatility basis='changes' correction: myga-forecasting-v2 BUG-005.
     Root cause: Using levels led to mislabeling steady drifts as volatile.
     Symptom: "HIGH volatility" regime had lowest prediction error.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Union, cast

import numpy as np
from numpy.typing import NDArray


def classify_volatility_regime(
    values: np.ndarray,
    window: int = 13,
    basis: Literal["changes", "levels"] = "changes",
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
) -> np.ndarray:
    """
    Classify volatility regime for each point using rolling window.

    CRITICAL: Default `basis='changes'` computes volatility on first differences,
    which is the methodologically correct approach. Using `basis='levels'` is
    provided for legacy comparison only.

    Parameters
    ----------
    values : np.ndarray
        Time series values. If basis='changes', these should be levels
        (differences will be computed). If basis='levels', used directly.
    window : int, default=13
        Rolling window for volatility calculation (13 weeks ~ 1 quarter)
    basis : {'changes', 'levels'}, default='changes'
        'changes' (correct): Compute volatility on first differences
        'levels' (legacy): Compute volatility on raw values
    low_percentile : float, default=33.0
        Percentile threshold for LOW volatility
    high_percentile : float, default=67.0
        Percentile threshold for HIGH volatility

    Returns
    -------
    np.ndarray
        Regime labels: 'LOW', 'MED', 'HIGH' for each point.
        Points with insufficient history are labeled 'MED'.

    Notes
    -----
    The correct pattern is volatility of CHANGES, not levels:
    - Steady drift (constant increases) has LOW volatility of changes
    - But HIGH std of levels (increasing values)

    Thresholds are computed from the full series, which is appropriate when
    classifying a single evaluation period. For walk-forward with multiple
    test periods, use training-only thresholds to prevent leakage.

    Examples
    --------
    >>> values = np.cumsum(np.random.randn(200) * 0.01) + 3.0
    >>> regimes = classify_volatility_regime(values, window=13, basis='changes')
    >>> print(np.unique(regimes, return_counts=True))

    See Also
    --------
    classify_direction_regime : Classify by direction (UP/DOWN/FLAT).
    run_gates_stratified : Run validation gates per-regime.
    """
    values = np.asarray(values)

    # Warn about near-zero data with levels basis
    if basis == "levels" and np.std(values) < 1e-8:
        warnings.warn(
            f"Very small std ({np.std(values):.2e}) with basis='levels'. "
            f"Regime classification may be degenerate. "
            f"Consider basis='changes' or normalizing data.",
            UserWarning,
            stacklevel=2,
        )

    if len(values) < window + 1:
        # Insufficient data - return all MED
        result: np.ndarray = np.array(["MED"] * len(values))
        return result

    # Compute series for volatility calculation
    if basis == "changes":
        # Compute first differences
        series_for_vol = np.diff(values)
        # Pad with NaN to maintain alignment (first point has no change)
        series_for_vol = np.concatenate([[np.nan], series_for_vol])
    else:
        series_for_vol = values.copy()

    # Compute rolling volatility using numpy (no pandas dependency)
    n = len(series_for_vol)
    rolling_vol = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = series_for_vol[i - window + 1 : i + 1]
        # Skip if any NaN in window
        if not np.any(np.isnan(window_data)):
            rolling_vol[i] = np.std(window_data, ddof=1)

    # Get valid volatility values for threshold computation
    valid_vol = rolling_vol[~np.isnan(rolling_vol)]

    if len(valid_vol) == 0:
        result_med: np.ndarray = np.array(["MED"] * n)
        return result_med

    # Compute thresholds
    vol_low = np.percentile(valid_vol, low_percentile)
    vol_high = np.percentile(valid_vol, high_percentile)

    # Classify each point
    regimes: List[str] = []
    for vol in rolling_vol:
        if np.isnan(vol):
            regimes.append("MED")  # Default for insufficient history
        elif vol <= vol_low:
            regimes.append("LOW")
        elif vol <= vol_high:
            regimes.append("MED")
        else:
            regimes.append("HIGH")

    result_final: np.ndarray = np.array(regimes)
    return result_final


def classify_direction_regime(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Classify direction using thresholded signs.

    This makes persistence (predicts 0) a meaningful baseline for
    direction accuracy metrics.

    Parameters
    ----------
    values : np.ndarray
        Values to classify (typically actual changes)
    threshold : float
        Move threshold (typically 70th percentile of |actuals| from training)

    Returns
    -------
    np.ndarray
        Direction labels: 'UP', 'DOWN', 'FLAT'

    Notes
    -----
    Using thresholded signs instead of raw signs:
    - |value| > threshold and value > 0 -> 'UP'
    - |value| > threshold and value < 0 -> 'DOWN'
    - |value| <= threshold -> 'FLAT'

    This provides a fair baseline for persistence model comparison.
    Without threshold, persistence (predicts 0) gets 0% direction accuracy,
    making all comparisons trivially "significant".

    Examples
    --------
    >>> actuals = np.array([0.1, -0.1, 0.02, -0.02, 0.0])
    >>> directions = classify_direction_regime(actuals, threshold=0.05)
    >>> print(directions)  # ['UP', 'DOWN', 'FLAT', 'FLAT', 'FLAT']

    See Also
    --------
    classify_volatility_regime : Classify by volatility level.
    compute_move_threshold : Compute threshold from training data.
    """
    values = np.asarray(values)

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    directions: List[str] = []
    for v in values:
        if abs(v) <= threshold:
            directions.append("FLAT")
        elif v > 0:
            directions.append("UP")
        else:
            directions.append("DOWN")

    result: np.ndarray = np.array(directions)
    return result


def get_combined_regimes(
    vol_regimes: np.ndarray,
    dir_regimes: np.ndarray,
) -> np.ndarray:
    """
    Combine volatility and direction into single label.

    Parameters
    ----------
    vol_regimes : np.ndarray
        Volatility regime labels ('LOW', 'MED', 'HIGH')
    dir_regimes : np.ndarray
        Direction regime labels ('UP', 'DOWN', 'FLAT')

    Returns
    -------
    np.ndarray
        Combined labels like 'HIGH-UP', 'LOW-FLAT', etc.

    Raises
    ------
    ValueError
        If arrays have different lengths

    Notes
    -----
    Combined regimes can have very low sample counts in some cells.
    Always check sample counts before interpreting conditional performance.

    Examples
    --------
    >>> vol = np.array(['HIGH', 'LOW', 'MED'])
    >>> dir_ = np.array(['UP', 'DOWN', 'FLAT'])
    >>> combined = get_combined_regimes(vol, dir_)
    >>> print(combined)  # ['HIGH-UP', 'LOW-DOWN', 'MED-FLAT']
    """
    vol_regimes = np.asarray(vol_regimes)
    dir_regimes = np.asarray(dir_regimes)

    if len(vol_regimes) != len(dir_regimes):
        raise ValueError(
            f"Arrays must have same length. "
            f"vol_regimes: {len(vol_regimes)}, dir_regimes: {len(dir_regimes)}"
        )

    result: np.ndarray = np.array([f"{v}-{d}" for v, d in zip(vol_regimes, dir_regimes)])
    return result


def get_regime_counts(regimes: np.ndarray) -> Dict[str, int]:
    """
    Get sample counts per regime.

    Parameters
    ----------
    regimes : np.ndarray
        Regime labels

    Returns
    -------
    dict
        Counts per regime, sorted by count descending

    Examples
    --------
    >>> regimes = np.array(['HIGH', 'LOW', 'LOW', 'MED', 'LOW'])
    >>> counts = get_regime_counts(regimes)
    >>> print(counts)  # {'LOW': 3, 'HIGH': 1, 'MED': 1}
    """
    regimes = np.asarray(regimes)
    unique, counts = np.unique(regimes, return_counts=True)

    # Sort by count descending
    sorted_idx = np.argsort(-counts)
    return {str(unique[i]): int(counts[i]) for i in sorted_idx}


def mask_low_n_regimes(
    regimes: np.ndarray,
    min_n: int = 10,
    mask_value: str = "MASKED",
) -> np.ndarray:
    """
    Mask regime labels with insufficient samples.

    Parameters
    ----------
    regimes : np.ndarray
        Regime labels
    min_n : int, default=10
        Minimum samples required per regime
    mask_value : str, default='MASKED'
        Value to use for masked regimes

    Returns
    -------
    np.ndarray
        Regimes with low-n cells masked

    Notes
    -----
    Use this to identify unreliable regime-conditional metrics.
    Cells with n < min_n should be masked before interpretation.

    Examples
    --------
    >>> regimes = np.array(['HIGH'] * 5 + ['LOW'] * 15)
    >>> masked = mask_low_n_regimes(regimes, min_n=10)
    >>> print(np.unique(masked))  # ['LOW', 'MASKED']
    """
    regimes = np.asarray(regimes)
    counts = get_regime_counts(regimes)

    low_n_regimes = {r for r, c in counts.items() if c < min_n}

    if not low_n_regimes:
        result_copy: np.ndarray = regimes.copy()
        return result_copy

    # Use object dtype to allow variable-length strings
    result: np.ndarray = regimes.astype(object).copy()
    for regime in low_n_regimes:
        result[result == regime] = mask_value

    return result


# =============================================================================
# Stratified Metrics
# =============================================================================


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StratifiedMetricsResult:
    """
    Metrics stratified by regime.

    Provides overall metrics and per-regime breakdown for MAE, RMSE, and sample counts.

    Attributes
    ----------
    overall_mae : float
        Mean Absolute Error across all samples.
    overall_rmse : float
        Root Mean Squared Error across all samples.
    n_total : int
        Total number of samples.
    by_regime : Dict[str, Dict[str, float]]
        Per-regime metrics. Each key is a regime label, value is dict with:
        - 'mae': Mean Absolute Error for regime
        - 'rmse': Root Mean Squared Error for regime
        - 'n': Sample count for regime
        - 'pct': Percentage of total samples
    masked_regimes : List[str]
        Regimes with n < min_n that were excluded from analysis.

    Examples
    --------
    >>> result = compute_stratified_metrics(preds, actuals, regimes)
    >>> print(f"Overall MAE: {result.overall_mae:.4f}")
    >>> for regime, metrics in result.by_regime.items():
    ...     print(f"{regime}: MAE={metrics['mae']:.4f}, n={metrics['n']}")
    """

    overall_mae: float
    overall_rmse: float
    n_total: int
    by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)
    masked_regimes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Generate human-readable summary of stratified metrics."""
        lines = [
            f"Overall: MAE={self.overall_mae:.4f}, RMSE={self.overall_rmse:.4f}, n={self.n_total}",
            "",
            "By Regime:",
        ]

        # Sort by n descending
        sorted_regimes = sorted(
            self.by_regime.items(), key=lambda x: x[1]["n"], reverse=True
        )

        for regime, metrics in sorted_regimes:
            lines.append(
                f"  {regime}: MAE={metrics['mae']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, n={metrics['n']} ({metrics['pct']:.1f}%)"
            )

        if self.masked_regimes:
            lines.append("")
            lines.append(f"Masked (n < min_n): {', '.join(self.masked_regimes)}")

        return "\n".join(lines)


def compute_stratified_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    regimes: np.ndarray,
    min_n: int = 10,
) -> StratifiedMetricsResult:
    """
    Compute MAE and RMSE stratified by regime.

    Provides automatic per-regime breakdown of prediction errors, essential for
    understanding model performance across different market conditions.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions.
    actuals : np.ndarray
        Actual values.
    regimes : np.ndarray
        Regime labels for each point (e.g., 'LOW', 'MED', 'HIGH' or 'UP', 'DOWN', 'FLAT').
        Must have same length as predictions and actuals.
    min_n : int, default=10
        Minimum samples required per regime. Regimes with fewer samples are
        masked and reported in `masked_regimes`.

    Returns
    -------
    StratifiedMetricsResult
        Dataclass with overall metrics, per-regime breakdown, and masked regimes.

    Raises
    ------
    ValueError
        If array lengths don't match or if predictions/actuals are empty.

    Notes
    -----
    This function is designed for post-hoc analysis of walk-forward results.
    For regime-conditional validation during training, use `run_gates_stratified()`.

    The `min_n` threshold helps avoid drawing conclusions from statistically
    unreliable subsets. Default of 10 is a conservative rule of thumb.

    Examples
    --------
    >>> # Classify regimes
    >>> vol_regimes = classify_volatility_regime(actuals, window=13, basis='changes')
    >>>
    >>> # Compute stratified metrics
    >>> result = compute_stratified_metrics(predictions, actuals, vol_regimes)
    >>> print(result.summary())
    Overall: MAE=0.0234, RMSE=0.0312, n=200
    By Regime:
      LOW: MAE=0.0156, RMSE=0.0201, n=67 (33.5%)
      MED: MAE=0.0245, RMSE=0.0298, n=66 (33.0%)
      HIGH: MAE=0.0301, RMSE=0.0437, n=67 (33.5%)

    See Also
    --------
    classify_volatility_regime : Classify by volatility level.
    classify_direction_regime : Classify by direction.
    mask_low_n_regimes : Mask regimes with insufficient samples.
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    regimes = np.asarray(regimes)

    # Validation
    if len(predictions) == 0 or len(actuals) == 0:
        raise ValueError("predictions and actuals cannot be empty")

    if len(predictions) != len(actuals):
        raise ValueError(
            f"predictions and actuals must have same length. "
            f"Got {len(predictions)} and {len(actuals)}"
        )

    if len(predictions) != len(regimes):
        raise ValueError(
            f"regimes must have same length as predictions. "
            f"Got {len(regimes)} regimes for {len(predictions)} predictions"
        )

    # Compute overall metrics
    errors = predictions - actuals
    overall_mae = float(np.mean(np.abs(errors)))
    overall_rmse = float(np.sqrt(np.mean(errors**2)))
    n_total = len(predictions)

    # Get unique regimes and counts
    unique_regimes = np.unique(regimes)
    regime_counts = get_regime_counts(regimes)

    # Identify masked regimes
    masked_regimes = [r for r, c in regime_counts.items() if c < min_n]

    # Compute per-regime metrics
    by_regime: Dict[str, Dict[str, float]] = {}

    for regime in unique_regimes:
        regime_str = str(regime)
        n_regime = regime_counts[regime_str]

        if n_regime < min_n:
            # Skip masked regimes
            continue

        mask = regimes == regime
        regime_errors = errors[mask]

        mae = float(np.mean(np.abs(regime_errors)))
        rmse = float(np.sqrt(np.mean(regime_errors**2)))
        pct = 100.0 * n_regime / n_total

        by_regime[regime_str] = {
            "mae": mae,
            "rmse": rmse,
            "n": n_regime,
            "pct": pct,
        }

    return StratifiedMetricsResult(
        overall_mae=overall_mae,
        overall_rmse=overall_rmse,
        n_total=n_total,
        by_regime=by_regime,
        masked_regimes=masked_regimes,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "classify_volatility_regime",
    "classify_direction_regime",
    "get_combined_regimes",
    "get_regime_counts",
    "mask_low_n_regimes",
    "StratifiedMetricsResult",
    "compute_stratified_metrics",
]
