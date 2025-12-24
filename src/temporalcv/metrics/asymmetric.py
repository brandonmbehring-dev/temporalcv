"""
Asymmetric Loss Functions Module.

Implements asymmetric loss functions that penalize over- and under-predictions
differently:

- **LinEx loss**: Linear-exponential asymmetric loss
- **Asymmetric MAPE**: Different penalties for over/under-prediction
- **Directional loss**: Custom weights for UP miss vs DOWN miss

Knowledge Tiers
---------------
[T1] LinEx loss (Varian 1975, Zellner 1986)
[T2] Asymmetric MAPE (common practice in forecasting)
[T2] Directional loss (common in trading/financial applications)

Example
-------
>>> from temporalcv.metrics.asymmetric import (
...     compute_linex_loss,
...     compute_asymmetric_mape,
...     compute_directional_loss,
... )
>>>
>>> # LinEx: penalize under-predictions exponentially
>>> loss = compute_linex_loss(predictions, actuals, a=1.0)
>>>
>>> # Asymmetric MAPE: 70% weight on under-predictions
>>> mape = compute_asymmetric_mape(predictions, actuals, alpha=0.7)
>>>
>>> # Directional: missing an UP move costs 2x vs missing DOWN
>>> loss = compute_directional_loss(predictions, actuals, up_miss_weight=2.0)

References
----------
[T1] Varian, H.R. (1975). A Bayesian approach to real estate assessment.
     Studies in Bayesian Econometrics and Statistics.
[T1] Zellner, A. (1986). Bayesian estimation and prediction using asymmetric
     loss functions. Journal of the American Statistical Association, 81(394), 446-451.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def compute_linex_loss(
    predictions: ArrayLike,
    actuals: ArrayLike,
    a: float = 1.0,
    b: float = 1.0,
) -> float:
    """
    Compute LinEx (linear-exponential) asymmetric loss.

    The LinEx loss function penalizes errors asymmetrically: one direction
    receives exponential penalties while the other receives linear penalties.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Actual observed values.
    a : float, default 1.0
        Asymmetry parameter.
        - a > 0: under-predictions (pred < actual) penalized exponentially
        - a < 0: over-predictions (pred > actual) penalized exponentially
        - |a| controls the degree of asymmetry (larger = more asymmetric)
    b : float, default 1.0
        Scaling parameter (must be > 0).

    Returns
    -------
    float
        Mean LinEx loss.

    Raises
    ------
    ValueError
        If a == 0, b <= 0, or array lengths don't match.

    Notes
    -----
    The LinEx loss is defined as:

        L(e) = b * (exp(a * e) - a * e - 1)

    where e = actual - prediction (error).

    Properties:
    - Always non-negative (L(0) = 0)
    - Convex and asymmetric around zero
    - Approximately quadratic near zero
    - Exponential growth on one side, linear on the other

    For forecasting contexts:
    - Use a > 0 when under-predictions are more costly (e.g., inventory)
    - Use a < 0 when over-predictions are more costly (e.g., overestimating sales)

    Examples
    --------
    >>> actuals = np.array([10.0, 20.0, 30.0])
    >>> predictions = np.array([12.0, 18.0, 28.0])
    >>> # Under-predictions costly (a > 0)
    >>> loss_under = compute_linex_loss(predictions, actuals, a=0.5)
    >>> # Over-predictions costly (a < 0)
    >>> loss_over = compute_linex_loss(predictions, actuals, a=-0.5)

    See Also
    --------
    compute_asymmetric_mape : Percentage-based asymmetric loss.
    compute_directional_loss : Directional prediction penalties.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if a == 0:
        raise ValueError("a cannot be 0 (would make loss symmetric)")

    if b <= 0:
        raise ValueError(f"b must be > 0, got {b}")

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predictions={len(predictions)}, actuals={len(actuals)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    # Error: actual - prediction
    # Positive error = under-prediction (pred < actual)
    # Negative error = over-prediction (pred > actual)
    errors = actuals - predictions

    # LinEx: b * (exp(a * e) - a * e - 1)
    # Clip to prevent overflow for large |a * e|
    a_e = np.clip(a * errors, -700, 700)  # exp(700) is near float max
    loss = b * (np.exp(a_e) - a * errors - 1)

    return float(np.mean(loss))


def compute_asymmetric_mape(
    predictions: ArrayLike,
    actuals: ArrayLike,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute asymmetric MAPE with different over/under penalties.

    Asymmetric MAPE allows different weights for over-predictions vs
    under-predictions, useful when the costs are asymmetric.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Actual observed values.
    alpha : float, default 0.5
        Weight for under-predictions in [0, 1].
        - alpha = 0.5: symmetric (standard MAPE behavior)
        - alpha > 0.5: under-predictions penalized more
        - alpha < 0.5: over-predictions penalized more
    epsilon : float, default 1e-8
        Small constant to prevent division by zero.

    Returns
    -------
    float
        Asymmetric MAPE as a fraction (multiply by 100 for percentage).

    Raises
    ------
    ValueError
        If alpha not in [0, 1] or array lengths don't match.

    Notes
    -----
    The asymmetric MAPE is defined as:

        AMAPE = mean( w(e) * |e| / |actual| )

    where:
    - e = actual - prediction
    - w(e) = alpha if e > 0 (under-prediction), (1-alpha) if e < 0

    This gives different penalties based on direction of error relative
    to the actual value.

    Examples
    --------
    >>> actuals = np.array([100.0, 200.0, 300.0])
    >>> predictions = np.array([110.0, 180.0, 280.0])
    >>> # Penalize under-predictions more (alpha=0.7)
    >>> amape = compute_asymmetric_mape(predictions, actuals, alpha=0.7)
    >>> print(f"Asymmetric MAPE: {amape:.2%}")

    See Also
    --------
    compute_linex_loss : Continuous asymmetric loss.
    compute_mape : Standard symmetric MAPE.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predictions={len(predictions)}, actuals={len(actuals)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    errors = actuals - predictions
    abs_errors = np.abs(errors)
    abs_actuals = np.abs(actuals) + epsilon

    # Weight based on error direction
    weights = np.where(errors > 0, alpha, 1 - alpha)

    # Weighted percentage error
    weighted_ape = weights * abs_errors / abs_actuals

    return float(np.mean(weighted_ape))


def compute_directional_loss(
    predictions: ArrayLike,
    actuals: ArrayLike,
    up_miss_weight: float = 1.0,
    down_miss_weight: float = 1.0,
    previous_actuals: ArrayLike | None = None,
) -> float:
    """
    Compute directional loss with custom weights for missing UP vs DOWN moves.

    This loss function penalizes directional prediction errors, with
    different penalties for missing an upward move vs a downward move.

    Parameters
    ----------
    predictions : array-like
        Predicted values (or predicted changes if previous_actuals not provided).
    actuals : array-like
        Actual observed values (or actual changes if previous_actuals not provided).
    up_miss_weight : float, default 1.0
        Weight for missing an UP move (predicting DOWN when actual is UP).
    down_miss_weight : float, default 1.0
        Weight for missing a DOWN move (predicting UP when actual is DOWN).
    previous_actuals : array-like, optional
        Previous actual values for computing changes. If provided, predictions
        and actuals are treated as levels and changes are computed internally.

    Returns
    -------
    float
        Mean directional loss.

    Raises
    ------
    ValueError
        If weights are negative or array lengths don't match.

    Notes
    -----
    The directional loss is computed as:

        L = up_miss_weight * I(miss_up) + down_miss_weight * I(miss_down)

    where:
    - miss_up: predicted direction is non-positive but actual is positive
    - miss_down: predicted direction is non-negative but actual is negative

    Correct predictions (same direction or either is zero) incur zero loss.

    Use cases:
    - Trading: Missing a rally may be more costly than missing a decline
    - Inventory: Under-predicting demand may be worse than over-predicting
    - Risk: Under-predicting losses may be catastrophic

    Examples
    --------
    >>> # Changes directly
    >>> pred_changes = np.array([1.0, -1.0, 1.0, -1.0])
    >>> actual_changes = np.array([0.5, 0.3, -0.2, -0.1])
    >>> # Missing UP costs 2x more than missing DOWN
    >>> loss = compute_directional_loss(
    ...     pred_changes, actual_changes,
    ...     up_miss_weight=2.0, down_miss_weight=1.0
    ... )

    >>> # From levels
    >>> previous = np.array([100, 102, 101, 103])
    >>> predictions = np.array([103, 101, 102, 105])  # Predicted levels
    >>> actuals = np.array([102, 101, 103, 102])  # Actual levels
    >>> loss = compute_directional_loss(
    ...     predictions, actuals,
    ...     up_miss_weight=2.0,
    ...     previous_actuals=previous
    ... )

    See Also
    --------
    compute_hit_rate : Simple directional accuracy (unweighted).
    compute_linex_loss : Continuous asymmetric loss on magnitude.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if up_miss_weight < 0 or down_miss_weight < 0:
        raise ValueError("Weights must be non-negative")

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predictions={len(predictions)}, actuals={len(actuals)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    # Compute changes if previous_actuals provided
    if previous_actuals is not None:
        previous = np.asarray(previous_actuals, dtype=np.float64)
        if len(previous) != len(predictions):
            raise ValueError(
                f"previous_actuals length must match. "
                f"Got {len(previous)}, expected {len(predictions)}"
            )
        pred_changes = predictions - previous
        actual_changes = actuals - previous
    else:
        pred_changes = predictions
        actual_changes = actuals

    pred_sign = np.sign(pred_changes)
    actual_sign = np.sign(actual_changes)

    # Miss UP: actual is positive, prediction is not positive
    miss_up = (actual_sign > 0) & (pred_sign <= 0)

    # Miss DOWN: actual is negative, prediction is not negative
    miss_down = (actual_sign < 0) & (pred_sign >= 0)

    # Compute weighted loss
    loss = up_miss_weight * miss_up + down_miss_weight * miss_down

    return float(np.mean(loss))


def compute_squared_log_error(
    predictions: ArrayLike,
    actuals: ArrayLike,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute mean squared logarithmic error (MSLE).

    MSLE is useful when targets span several orders of magnitude and
    relative errors are more important than absolute errors. It also
    naturally penalizes under-predictions more than over-predictions.

    Parameters
    ----------
    predictions : array-like
        Predicted values (must be non-negative).
    actuals : array-like
        Actual values (must be non-negative).
    epsilon : float, default 1e-8
        Small constant added before log to handle zeros.

    Returns
    -------
    float
        Mean squared logarithmic error.

    Raises
    ------
    ValueError
        If negative values present or array lengths don't match.

    Notes
    -----
    MSLE is defined as:

        MSLE = mean( (log(1 + actual) - log(1 + pred))^2 )

    Properties:
    - Scale-invariant (relative errors)
    - Naturally asymmetric: penalizes under-predictions more for same |error|
    - Appropriate for strictly positive targets

    The log(1 + x) transformation handles zero values gracefully.

    Examples
    --------
    >>> actuals = np.array([100, 1000, 10000])
    >>> predictions = np.array([110, 900, 11000])
    >>> msle = compute_squared_log_error(predictions, actuals)

    See Also
    --------
    compute_linex_loss : Explicit asymmetry parameter.
    compute_mape : Percentage error without log transformation.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predictions={len(predictions)}, actuals={len(actuals)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    if np.any(predictions < 0) or np.any(actuals < 0):
        raise ValueError("MSLE requires non-negative values")

    log_pred = np.log(predictions + 1 + epsilon)
    log_actual = np.log(actuals + 1 + epsilon)

    msle = np.mean((log_actual - log_pred) ** 2)

    return float(msle)


def compute_huber_loss(
    predictions: ArrayLike,
    actuals: ArrayLike,
    delta: float = 1.0,
) -> float:
    """
    Compute Huber loss (smooth approximation to MAE).

    Huber loss is quadratic for small errors and linear for large errors,
    providing robustness to outliers while maintaining differentiability.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Actual values.
    delta : float, default 1.0
        Threshold where loss transitions from quadratic to linear.

    Returns
    -------
    float
        Mean Huber loss.

    Raises
    ------
    ValueError
        If delta <= 0 or array lengths don't match.

    Notes
    -----
    The Huber loss is defined as:

        L(e) = 0.5 * e^2           if |e| <= delta
             = delta * (|e| - 0.5 * delta)  if |e| > delta

    where e = actual - prediction.

    Properties:
    - Quadratic near zero (like MSE)
    - Linear in tails (like MAE, robust to outliers)
    - Continuously differentiable (unlike MAE)
    - Symmetric (unlike LinEx)

    Examples
    --------
    >>> actuals = np.array([1.0, 2.0, 100.0])  # One outlier
    >>> predictions = np.array([1.1, 1.9, 10.0])  # Misses outlier badly
    >>> huber = compute_huber_loss(predictions, actuals, delta=1.0)
    >>> mse = np.mean((actuals - predictions) ** 2)
    >>> # Huber will be much less affected by the outlier

    See Also
    --------
    compute_linex_loss : Asymmetric loss.
    compute_mae : Standard MAE (L1 loss).
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if delta <= 0:
        raise ValueError(f"delta must be > 0, got {delta}")

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predictions={len(predictions)}, actuals={len(actuals)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    errors = actuals - predictions
    abs_errors = np.abs(errors)

    # Quadratic for |e| <= delta, linear for |e| > delta
    loss = np.where(
        abs_errors <= delta,
        0.5 * errors**2,
        delta * (abs_errors - 0.5 * delta),
    )

    return float(np.mean(loss))


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "compute_linex_loss",
    "compute_asymmetric_mape",
    "compute_directional_loss",
    "compute_squared_log_error",
    "compute_huber_loss",
]
