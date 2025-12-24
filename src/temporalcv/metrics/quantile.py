"""
Quantile and Interval Metrics Module.

Implements proper scoring rules for probabilistic forecasts:

- **Pinball loss**: Quantile regression loss (asymmetric around quantile)
- **CRPS**: Continuous Ranked Probability Score (proper scoring rule)
- **Interval score**: Proper scoring rule for prediction intervals
- **Quantile coverage**: Empirical coverage of prediction intervals

Knowledge Tiers
---------------
[T1] Pinball loss formula and properties (Koenker & Bassett 1978)
[T1] CRPS as proper scoring rule (Gneiting & Raftery 2007)
[T1] Interval score (Gneiting & Raftery 2007, equation 43)
[T2] CRPS sample approximation (standard practice when CDF unavailable)
[T3] scipy.stats CRPS availability detection (implementation detail)

Example
-------
>>> from temporalcv.metrics.quantile import (
...     compute_pinball_loss,
...     compute_crps,
...     compute_interval_score,
...     compute_quantile_coverage,
... )
>>>
>>> # Quantile regression evaluation
>>> loss = compute_pinball_loss(actuals, quantile_preds, tau=0.9)
>>>
>>> # Interval evaluation
>>> score = compute_interval_score(actuals, lower, upper, alpha=0.05)
>>> coverage = compute_quantile_coverage(actuals, lower, upper)

References
----------
[T1] Koenker, R. & Bassett, G. (1978). Regression quantiles.
     Econometrica, 46(1), 33-50.
[T1] Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
     prediction, and estimation. Journal of the American Statistical
     Association, 102(477), 359-378.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_pinball_loss(
    actuals: ArrayLike,
    quantile_preds: ArrayLike,
    tau: float,
) -> float:
    """
    Compute pinball loss (quantile loss) for quantile regression.

    The pinball loss is asymmetric around the quantile, penalizing
    under-predictions more heavily for high quantiles and over-predictions
    more heavily for low quantiles.

    Parameters
    ----------
    actuals : array-like
        Actual observed values.
    quantile_preds : array-like
        Predicted values at quantile tau.
    tau : float
        Quantile level in (0, 1). E.g., tau=0.9 for 90th percentile.

    Returns
    -------
    float
        Mean pinball loss.

    Raises
    ------
    ValueError
        If tau is not in (0, 1) or if array lengths don't match.

    Notes
    -----
    The pinball loss is defined as:

        L(y, q; τ) = τ * max(y - q, 0) + (1 - τ) * max(q - y, 0)

    Equivalently:
        L(y, q; τ) = (y - q) * (τ - I(y < q))

    where I(.) is the indicator function.

    Lower values indicate better quantile predictions. The loss is:
    - 0 when predictions exactly match actuals
    - Positive otherwise
    - Asymmetric based on tau

    Examples
    --------
    >>> actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> preds_90 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])  # 90th percentile
    >>> loss = compute_pinball_loss(actuals, preds_90, tau=0.9)

    See Also
    --------
    compute_crps : More comprehensive probabilistic forecast metric.
    compute_interval_score : Score for prediction intervals.
    """
    actuals = np.asarray(actuals, dtype=np.float64)
    quantile_preds = np.asarray(quantile_preds, dtype=np.float64)

    if not (0 < tau < 1):
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    if len(actuals) != len(quantile_preds):
        raise ValueError(
            f"Array lengths must match. "
            f"Got actuals={len(actuals)}, quantile_preds={len(quantile_preds)}"
        )

    if len(actuals) == 0:
        raise ValueError("Arrays cannot be empty")

    # Compute pinball loss
    errors = actuals - quantile_preds
    loss = np.where(
        errors >= 0,
        tau * errors,
        (tau - 1) * errors,
    )

    return float(np.mean(loss))


def compute_crps(
    actuals: ArrayLike,
    forecast_samples: ArrayLike,
) -> float:
    """
    Compute Continuous Ranked Probability Score (CRPS).

    CRPS is a proper scoring rule for probabilistic forecasts. It measures
    the compatibility between the forecast distribution and the observation.

    Parameters
    ----------
    actuals : array-like
        Actual observed values, shape (n,).
    forecast_samples : array-like
        Samples from forecast distribution, shape (n, n_samples).
        Each row contains samples for one observation.

    Returns
    -------
    float
        Mean CRPS across all observations.

    Raises
    ------
    ValueError
        If array dimensions don't match.

    Notes
    -----
    CRPS is computed as:

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X and X' are independent draws from the forecast distribution
    and y is the observation.

    For sample-based approximation:
        CRPS ≈ mean|samples - y| - 0.5 * mean|samples_i - samples_j|

    This implementation uses scipy.stats.energy_distance if available,
    otherwise falls back to a sample-based approximation.

    Lower CRPS indicates better probabilistic calibration. CRPS has the
    same units as the observations.

    Examples
    --------
    >>> # Each row: samples for one observation
    >>> forecast_samples = np.random.randn(100, 1000)  # 100 obs, 1000 samples each
    >>> actuals = np.random.randn(100)
    >>> crps = compute_crps(actuals, forecast_samples)

    See Also
    --------
    compute_pinball_loss : Point estimate at specific quantile.
    compute_interval_score : Score for prediction intervals.
    """
    actuals = np.asarray(actuals, dtype=np.float64)
    forecast_samples = np.asarray(forecast_samples, dtype=np.float64)

    if actuals.ndim != 1:
        raise ValueError(f"actuals must be 1D, got shape {actuals.shape}")

    if forecast_samples.ndim != 2:
        raise ValueError(
            f"forecast_samples must be 2D (n_obs, n_samples), got shape {forecast_samples.shape}"
        )

    n_obs = len(actuals)
    if forecast_samples.shape[0] != n_obs:
        raise ValueError(
            f"Number of observations must match. "
            f"Got {n_obs} actuals but {forecast_samples.shape[0]} sample rows"
        )

    if n_obs == 0:
        raise ValueError("Arrays cannot be empty")

    # Try to use scipy for energy distance (more efficient)
    try:
        from scipy import stats

        # Check if energy_distance is available (scipy >= 1.0)
        if hasattr(stats, "energy_distance"):
            crps_values = np.zeros(n_obs)
            for i in range(n_obs):
                # Energy distance between samples and point mass at actual
                samples = forecast_samples[i]
                actual_point = np.array([actuals[i]])
                crps_values[i] = stats.energy_distance(samples, actual_point)
            return float(np.mean(crps_values))
    except ImportError:
        pass

    # Fallback: sample-based CRPS approximation
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    crps_values = np.zeros(n_obs)

    for i in range(n_obs):
        samples = forecast_samples[i]
        y = actuals[i]

        # E|X - y|
        term1 = np.mean(np.abs(samples - y))

        # E|X - X'| using all pairs
        n_samples = len(samples)
        if n_samples > 1:
            # Efficient computation using sorted samples
            sorted_samples = np.sort(samples)
            # E|X - X'| = 2 * sum_i (2*i - n - 1) * x_{(i)} / n^2
            indices = np.arange(1, n_samples + 1)
            term2 = 2 * np.sum((2 * indices - n_samples - 1) * sorted_samples) / (
                n_samples * n_samples
            )
        else:
            term2 = 0.0

        crps_values[i] = term1 - 0.5 * abs(term2)

    return float(np.mean(crps_values))


def compute_interval_score(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
) -> float:
    """
    Compute interval score for prediction intervals.

    The interval score is a proper scoring rule for prediction intervals,
    penalizing both interval width and coverage failures.

    Parameters
    ----------
    actuals : array-like
        Actual observed values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.
    alpha : float
        Nominal non-coverage rate in (0, 1). E.g., alpha=0.05 for 95% intervals.

    Returns
    -------
    float
        Mean interval score.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1), if array lengths don't match, or if
        lower > upper for any observation.

    Notes
    -----
    The interval score is defined as (Gneiting & Raftery 2007, equation 43):

        IS(l, u; y) = (u - l) + (2/α) * (l - y) * I(y < l) + (2/α) * (y - u) * I(y > u)

    Components:
    - (u - l): Penalizes wide intervals
    - (2/α) * (l - y) * I(y < l): Penalizes under-coverage (actual below lower)
    - (2/α) * (y - u) * I(y > u): Penalizes under-coverage (actual above upper)

    Lower interval scores indicate better interval forecasts. A well-calibrated
    narrow interval scores better than a well-calibrated wide interval.

    Examples
    --------
    >>> actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> score = compute_interval_score(actuals, lower, upper, alpha=0.10)

    See Also
    --------
    compute_quantile_coverage : Empirical coverage of intervals.
    compute_pinball_loss : Related metric for quantile forecasts.
    """
    actuals = np.asarray(actuals, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if len(actuals) != len(lower) or len(actuals) != len(upper):
        raise ValueError(
            f"Array lengths must match. "
            f"Got actuals={len(actuals)}, lower={len(lower)}, upper={len(upper)}"
        )

    if len(actuals) == 0:
        raise ValueError("Arrays cannot be empty")

    # Check that lower <= upper
    if np.any(lower > upper):
        n_violations: int = int(np.sum(lower > upper))
        raise ValueError(
            f"lower must be <= upper for all observations. "
            f"Found {n_violations} violations"
        )

    # Compute interval score
    # IS = (u - l) + (2/α) * (l - y) * I(y < l) + (2/α) * (y - u) * I(y > u)
    width = upper - lower
    penalty_factor = 2.0 / alpha

    # Penalty for actuals below lower bound
    below = actuals < lower
    penalty_below = penalty_factor * (lower - actuals) * below

    # Penalty for actuals above upper bound
    above = actuals > upper
    penalty_above = penalty_factor * (actuals - upper) * above

    scores = width + penalty_below + penalty_above

    return float(np.mean(scores))


def compute_quantile_coverage(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Calculates the fraction of observations that fall within their
    prediction intervals.

    Parameters
    ----------
    actuals : array-like
        Actual observed values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    float
        Empirical coverage rate in [0, 1].

    Raises
    ------
    ValueError
        If array lengths don't match or arrays are empty.

    Notes
    -----
    Coverage is computed as:

        coverage = mean(I(lower <= actual <= upper))

    For a well-calibrated (1-α) prediction interval, empirical coverage
    should be approximately (1-α). E.g., 95% intervals should cover
    approximately 95% of observations.

    Examples
    --------
    >>> actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    >>> coverage = compute_quantile_coverage(actuals, lower, upper)
    >>> print(f"Coverage: {coverage:.1%}")  # 100%

    See Also
    --------
    compute_interval_score : Proper scoring rule for intervals.
    """
    actuals = np.asarray(actuals, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if len(actuals) != len(lower) or len(actuals) != len(upper):
        raise ValueError(
            f"Array lengths must match. "
            f"Got actuals={len(actuals)}, lower={len(lower)}, upper={len(upper)}"
        )

    if len(actuals) == 0:
        raise ValueError("Arrays cannot be empty")

    # Compute coverage
    covered = (actuals >= lower) & (actuals <= upper)

    return float(np.mean(covered))


def compute_winkler_score(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
) -> float:
    """
    Compute Winkler score for prediction intervals.

    The Winkler score is equivalent to the interval score but is historically
    the name used in forecasting literature.

    Parameters
    ----------
    actuals : array-like
        Actual observed values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.
    alpha : float
        Nominal non-coverage rate in (0, 1). E.g., alpha=0.05 for 95% intervals.

    Returns
    -------
    float
        Mean Winkler score.

    Notes
    -----
    This is an alias for compute_interval_score. The Winkler score
    (Winkler 1972) is the original formulation, while interval score
    is the term used by Gneiting & Raftery (2007).

    See Also
    --------
    compute_interval_score : Primary implementation.
    """
    return compute_interval_score(actuals, lower, upper, alpha)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "compute_pinball_loss",
    "compute_crps",
    "compute_interval_score",
    "compute_quantile_coverage",
    "compute_winkler_score",
]
