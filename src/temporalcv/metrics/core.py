"""
Core Forecast Evaluation Metrics.

Foundational metrics for time series forecast evaluation:
- Point forecast metrics: RMSE, MSE, MAE, MAPE, SMAPE
- Scale-invariant metrics: MASE, Theil's U, MRAE
- Correlation metrics: Pearson r, Spearman ρ
- Naive/persistence baselines: compute_naive_error

Knowledge Tiers
---------------
[T1] All metrics are standard statistical formulations
[T1] MASE: Hyndman & Koehler (2006), "Another look at measures of forecast accuracy"
[T1] SMAPE: Armstrong (1985), bounded symmetric alternative to MAPE
[T1] Theil's U: Theil (1966), relative accuracy to naive forecast

Example
-------
>>> from temporalcv.metrics import compute_rmse, compute_mape, compute_mase
>>>
>>> rmse = compute_rmse(predictions, actuals)
>>> mape = compute_mape(predictions, actuals)
>>>
>>> # Scale-invariant comparison
>>> naive_mae = compute_naive_error(train_actuals)
>>> mase = compute_mase(predictions, actuals, naive_mae)

References
----------
[T1] Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of
     forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
[T1] Armstrong, J.S. (1985). Long-Range Forecasting: From Crystal Ball to
     Computer. Wiley.
[T1] Theil, H. (1966). Applied Economic Forecasting. North-Holland Publishing.
"""

from __future__ import annotations

from typing import Literal, Union, cast

import numpy as np
from numpy.typing import ArrayLike

# Type alias for numeric arrays
NumericArray = Union[np.ndarray, list, tuple]


def _validate_inputs(
    predictions: ArrayLike,
    actuals: ArrayLike,
    name: str = "inputs",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert inputs to numpy arrays.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values
    name : str
        Name for error messages

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated predictions and actuals

    Raises
    ------
    ValueError
        If inputs contain NaN or have mismatched lengths
    """
    preds = np.asarray(predictions, dtype=np.float64)
    acts = np.asarray(actuals, dtype=np.float64)

    if len(preds) != len(acts):
        raise ValueError(
            f"{name}: predictions and actuals must have same length "
            f"(got {len(preds)} and {len(acts)})"
        )

    if np.any(np.isnan(preds)):
        raise ValueError(f"{name}: predictions contain NaN values")

    if np.any(np.isnan(acts)):
        raise ValueError(f"{name}: actuals contain NaN values")

    return preds, acts


# =============================================================================
# Point Forecast Metrics
# =============================================================================


def compute_mae(predictions: ArrayLike, actuals: ArrayLike) -> float:
    """
    Compute Mean Absolute Error.

    MAE = mean(|y_hat - y|)

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values

    Returns
    -------
    float
        Mean absolute error

    Knowledge Tier: [T1] Standard error metric.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_mae")
    return float(np.mean(np.abs(preds - acts)))


def compute_mse(predictions: ArrayLike, actuals: ArrayLike) -> float:
    """
    Compute Mean Squared Error.

    MSE = mean((y_hat - y)²)

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values

    Returns
    -------
    float
        Mean squared error

    Knowledge Tier: [T1] Standard error metric.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_mse")
    return float(np.mean((preds - acts) ** 2))


def compute_rmse(predictions: ArrayLike, actuals: ArrayLike) -> float:
    """
    Compute Root Mean Squared Error.

    RMSE = sqrt(mean((y_hat - y)²))

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values

    Returns
    -------
    float
        Root mean squared error

    Knowledge Tier: [T1] Standard error metric.
    """
    return float(np.sqrt(compute_mse(predictions, actuals)))


def compute_mape(
    predictions: ArrayLike,
    actuals: ArrayLike,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute Mean Absolute Percentage Error.

    MAPE = 100 * mean(|y_hat - y| / |y|)

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values
    epsilon : float, default=1e-8
        Small value to prevent division by zero

    Returns
    -------
    float
        Mean absolute percentage error (as percentage, 0-100+)

    Notes
    -----
    MAPE has known issues:
    - Undefined when actuals = 0
    - Asymmetric: penalizes over-prediction more
    - Unbounded above 100%

    Consider SMAPE for a bounded alternative.

    Knowledge Tier: [T1] Standard percentage error metric.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_mape")
    denom = np.maximum(np.abs(acts), epsilon)
    return float(100.0 * np.mean(np.abs(preds - acts) / denom))


def compute_smape(predictions: ArrayLike, actuals: ArrayLike) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error.

    SMAPE = 100 * mean(2 * |y_hat - y| / (|y_hat| + |y|))

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values

    Returns
    -------
    float
        Symmetric MAPE (bounded 0-200%)

    Notes
    -----
    SMAPE is bounded [0, 200%] and symmetric around zero.
    When both prediction and actual are zero, that observation is excluded.

    Knowledge Tier: [T1] Armstrong (1985) symmetric alternative to MAPE.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_smape")

    denom = np.abs(preds) + np.abs(acts)
    # Exclude cases where both are zero
    mask = denom > 0
    if not np.any(mask):
        return 0.0

    numerator = 2.0 * np.abs(preds[mask] - acts[mask])
    return float(100.0 * np.mean(numerator / denom[mask]))


def compute_bias(predictions: ArrayLike, actuals: ArrayLike) -> float:
    """
    Compute mean signed error (bias).

    Bias = mean(y_hat - y)

    Positive bias indicates over-prediction on average.
    Negative bias indicates under-prediction on average.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values

    Returns
    -------
    float
        Mean signed error (positive = over-prediction)

    Knowledge Tier: [T1] Standard error metric.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_bias")
    return float(np.mean(preds - acts))


# =============================================================================
# Scale-Invariant Metrics
# =============================================================================


def compute_naive_error(
    values: ArrayLike,
    method: Literal["persistence", "mean"] = "persistence",
) -> float:
    """
    Compute naive forecast MAE for scale normalization.

    Used as denominator for MASE and other scale-free metrics.

    Parameters
    ----------
    values : ArrayLike
        Training series values
    method : {"persistence", "mean"}, default="persistence"
        - "persistence": Naive forecast (y[t] = y[t-1])
        - "mean": Mean forecast (y[t] = mean(y))

    Returns
    -------
    float
        MAE of naive forecast on training data

    Notes
    -----
    For persistence: MAE = mean(|y[t] - y[t-1]|) for t = 1, ..., n-1
    This represents the "cost of being naive" for MASE normalization.

    Knowledge Tier: [T1] Hyndman & Koehler (2006).
    """
    vals = np.asarray(values, dtype=np.float64)

    if len(vals) < 2:
        raise ValueError("compute_naive_error requires at least 2 values")

    if method == "persistence":
        # y[t] - y[t-1] for t >= 1
        return float(np.mean(np.abs(np.diff(vals))))
    elif method == "mean":
        mean_val = np.mean(vals)
        return float(np.mean(np.abs(vals - mean_val)))
    else:
        raise ValueError(f"method must be 'persistence' or 'mean', got {method}")


def compute_mase(
    predictions: ArrayLike,
    actuals: ArrayLike,
    naive_mae: float,
) -> float:
    """
    Compute Mean Absolute Scaled Error.

    MASE = MAE / naive_MAE

    Where naive_MAE is typically the in-sample MAE of the naive
    (persistence) forecast.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values
    naive_mae : float
        MAE of naive forecast on training data.
        Compute with `compute_naive_error(train_values)`.

    Returns
    -------
    float
        MASE value. <1 means better than naive, >1 means worse.

    Notes
    -----
    MASE is scale-free and can be used to compare accuracy across
    different time series with different scales.

    MASE < 1: Model beats naive forecast
    MASE = 1: Model equals naive forecast
    MASE > 1: Model worse than naive forecast

    Knowledge Tier: [T1] Hyndman & Koehler (2006).
    """
    if naive_mae <= 0:
        raise ValueError(f"naive_mae must be positive, got {naive_mae}")

    mae = compute_mae(predictions, actuals)
    return mae / naive_mae


def compute_mrae(
    predictions: ArrayLike,
    actuals: ArrayLike,
    naive_predictions: ArrayLike,
) -> float:
    """
    Compute Mean Relative Absolute Error.

    MRAE = mean(|y_hat - y| / |y_naive - y|)

    Parameters
    ----------
    predictions : ArrayLike
        Model predictions
    actuals : ArrayLike
        Actual values
    naive_predictions : ArrayLike
        Naive/baseline predictions (same length as predictions)

    Returns
    -------
    float
        MRAE value. <1 means better than naive.

    Notes
    -----
    MRAE compares each error to the naive error at that point.
    Points where naive_error = 0 are excluded.

    Knowledge Tier: [T1] Standard relative error metric.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_mrae")
    naive = np.asarray(naive_predictions, dtype=np.float64)

    if len(naive) != len(preds):
        raise ValueError("naive_predictions must have same length as predictions")

    model_errors = np.abs(preds - acts)
    naive_errors = np.abs(naive - acts)

    # Exclude points where naive error is zero
    mask = naive_errors > 0
    if not np.any(mask):
        return float("nan")

    return float(np.mean(model_errors[mask] / naive_errors[mask]))


def compute_theils_u(
    predictions: ArrayLike,
    actuals: ArrayLike,
    naive_predictions: ArrayLike | None = None,
) -> float:
    """
    Compute Theil's U statistic.

    U = RMSE(model) / RMSE(naive)

    If naive_predictions not provided, uses persistence (y[t-1]).

    Parameters
    ----------
    predictions : ArrayLike
        Model predictions
    actuals : ArrayLike
        Actual values
    naive_predictions : ArrayLike, optional
        Naive/baseline predictions. If None, uses persistence.

    Returns
    -------
    float
        Theil's U. <1 means better than naive, >1 means worse.

    Notes
    -----
    This is Theil's U2 (1966), comparing to a naive forecast.
    U < 1 indicates the model outperforms the naive forecast.

    Knowledge Tier: [T1] Theil (1966).
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_theils_u")

    if naive_predictions is None:
        # Use persistence: y[t-1] as prediction for y[t]
        # First prediction has no naive (exclude from calculation)
        if len(preds) < 2:
            raise ValueError("Need at least 2 observations for persistence baseline")
        naive = acts[:-1]  # y[0], y[1], ..., y[n-2] predict y[1], ..., y[n-1]
        preds = preds[1:]
        acts = acts[1:]
    else:
        naive = np.asarray(naive_predictions, dtype=np.float64)
        if len(naive) != len(preds):
            raise ValueError("naive_predictions must have same length as predictions")

    model_rmse = np.sqrt(np.mean((preds - acts) ** 2))
    naive_rmse = np.sqrt(np.mean((naive - acts) ** 2))

    if naive_rmse == 0:
        return float("inf") if model_rmse > 0 else 1.0

    return float(model_rmse / naive_rmse)


# =============================================================================
# Correlation Metrics
# =============================================================================


def compute_forecast_correlation(
    predictions: ArrayLike,
    actuals: ArrayLike,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float:
    """
    Compute correlation between predictions and actuals.

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values
    method : {"pearson", "spearman"}, default="pearson"
        Correlation method

    Returns
    -------
    float
        Correlation coefficient [-1, 1]

    Notes
    -----
    Correlation measures association but not accuracy. A model can
    have high correlation but large errors (wrong scale/offset).

    Knowledge Tier: [T1] Standard statistical measures.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_forecast_correlation")

    if len(preds) < 2:
        return float("nan")

    if method == "pearson":
        # Pearson correlation
        corr_matrix = np.corrcoef(preds, acts)
        return float(corr_matrix[0, 1])
    elif method == "spearman":
        # Spearman rank correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(preds, acts)
        return float(corr)
    else:
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method}")


def compute_r_squared(predictions: ArrayLike, actuals: ArrayLike) -> float:
    """
    Compute R² (coefficient of determination).

    R² = 1 - SS_res / SS_tot

    Where:
    - SS_res = sum((y - y_hat)²)  [residual sum of squares]
    - SS_tot = sum((y - mean(y))²)  [total sum of squares]

    Parameters
    ----------
    predictions : ArrayLike
        Predicted values
    actuals : ArrayLike
        Actual values

    Returns
    -------
    float
        R² value. Can be negative if model is worse than mean.

    Notes
    -----
    R² = 1: Perfect predictions
    R² = 0: Model equals mean forecast
    R² < 0: Model worse than mean forecast

    Knowledge Tier: [T1] Standard statistical measure.
    """
    preds, acts = _validate_inputs(predictions, actuals, "compute_r_squared")

    ss_res: float = float(np.sum((acts - preds) ** 2))
    ss_tot: float = float(np.sum((acts - np.mean(acts)) ** 2))

    if ss_tot == 0:
        # All actuals are the same
        return 1.0 if ss_res == 0 else float("-inf")

    return float(1.0 - ss_res / ss_tot)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Point forecast metrics
    "compute_mae",
    "compute_mse",
    "compute_rmse",
    "compute_mape",
    "compute_smape",
    "compute_bias",
    # Scale-invariant metrics
    "compute_naive_error",
    "compute_mase",
    "compute_mrae",
    "compute_theils_u",
    # Correlation metrics
    "compute_forecast_correlation",
    "compute_r_squared",
]
