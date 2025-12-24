"""
Volatility-Weighted Metrics Module.

Metrics that account for local volatility to provide scale-invariant
evaluation across different market regimes:

- **Volatility estimation**: Rolling std, EWMA, optional GARCH
- **Volatility-normalized MAE**: Scale-invariant error metric
- **Volatility-weighted MAE**: Weight by inverse/importance of volatility
- **Volatility-stratified metrics**: MAE breakdown by volatility regime

Knowledge Tiers
---------------
[T1] Rolling standard deviation (fundamental statistics)
[T1] EWMA volatility (RiskMetrics, J.P. Morgan 1996)
[T2] GARCH volatility (Bollerslev 1986) - optional dependency
[T2] Volatility-normalized errors (common in quant finance)
[T3] Volatility tercile stratification (practical heuristic)

Example
-------
>>> from temporalcv.metrics.volatility_weighted import (
...     compute_local_volatility,
...     compute_volatility_normalized_mae,
...     compute_volatility_stratified_metrics,
... )
>>>
>>> # Estimate local volatility
>>> vol = compute_local_volatility(returns, window=13, method="rolling_std")
>>>
>>> # Scale-invariant error metric
>>> vnmae = compute_volatility_normalized_mae(predictions, actuals, vol)
>>>
>>> # Breakdown by volatility regime
>>> result = compute_volatility_stratified_metrics(predictions, actuals, vol)
>>> print(f"Low vol MAE: {result.low_vol_mae:.4f}")

References
----------
[T1] J.P. Morgan (1996). RiskMetrics Technical Document.
[T2] Bollerslev, T. (1986). Generalized autoregressive conditional
     heteroskedasticity. Journal of Econometrics, 31(3), 307-327.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


# =============================================================================
# Volatility Estimator Protocol
# =============================================================================


@runtime_checkable
class VolatilityEstimator(Protocol):
    """Protocol for volatility estimation methods.

    Allows extensibility for custom volatility estimators beyond
    the built-in rolling_std and EWMA methods.
    """

    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate local volatility.

        Parameters
        ----------
        values : ndarray
            Input values (typically returns or price changes).

        Returns
        -------
        ndarray
            Volatility estimates, same length as input.
        """
        ...


# =============================================================================
# Built-in Volatility Estimators
# =============================================================================


class RollingVolatility:
    """Rolling window standard deviation estimator.

    Parameters
    ----------
    window : int, default 13
        Rolling window size for standard deviation calculation.
    min_periods : int, optional
        Minimum observations required for a valid estimate.
        Defaults to window // 2.
    """

    def __init__(self, window: int = 13, min_periods: int | None = None):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = window
        self.min_periods = min_periods if min_periods is not None else window // 2

    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute rolling standard deviation."""
        values = np.asarray(values, dtype=np.float64)
        n = len(values)

        if n == 0:
            return np.array([], dtype=np.float64)

        volatility = np.full(n, np.nan)

        for i in range(n):
            start = max(0, i - self.window + 1)
            window_data = values[start : i + 1]

            if len(window_data) >= self.min_periods:
                volatility[i] = np.std(window_data, ddof=1)

        # Forward-fill NaNs at the start
        first_valid = np.where(~np.isnan(volatility))[0]
        if len(first_valid) > 0:
            volatility[: first_valid[0]] = volatility[first_valid[0]]

        return volatility


class EWMAVolatility:
    """Exponentially Weighted Moving Average volatility estimator.

    EWMA places more weight on recent observations, making it more
    responsive to volatility changes than rolling window methods.

    Parameters
    ----------
    span : int, default 13
        Decay span for EWMA. Lambda = 2/(span+1).
    adjust : bool, default True
        Whether to adjust for bias in early observations.
    """

    def __init__(self, span: int = 13, adjust: bool = True):
        if span < 1:
            raise ValueError(f"span must be >= 1, got {span}")
        self.span = span
        self.adjust = adjust
        self.alpha = 2.0 / (span + 1)

    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute EWMA volatility (of squared deviations from mean)."""
        values = np.asarray(values, dtype=np.float64)
        n = len(values)

        if n == 0:
            return np.array([], dtype=np.float64)

        # Center around mean for variance calculation
        mean_val = np.mean(values)
        squared_devs = (values - mean_val) ** 2

        # EWMA of squared deviations
        ewma_var = np.zeros(n)
        ewma_var[0] = squared_devs[0]

        for i in range(1, n):
            ewma_var[i] = self.alpha * squared_devs[i] + (1 - self.alpha) * ewma_var[
                i - 1
            ]

        volatility = np.sqrt(ewma_var)

        return volatility


class GARCHVolatility:
    """GARCH(1,1) volatility estimator.

    Requires the `arch` package as an optional dependency.

    Parameters
    ----------
    p : int, default 1
        GARCH lag order.
    q : int, default 1
        ARCH lag order.

    Raises
    ------
    ImportError
        If arch package is not installed.
    """

    def __init__(self, p: int = 1, q: int = 1):
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "GARCH estimation requires the 'arch' package. "
                "Install with: pip install arch"
            )
        self.p = p
        self.q = q

    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimate GARCH volatility using the arch package."""
        from arch import arch_model

        values = np.asarray(values, dtype=np.float64)
        n = len(values)

        if n < 10:
            raise ValueError("GARCH requires at least 10 observations")

        # Scale for numerical stability (arch expects percentage returns)
        scale = 100.0
        scaled_values = values * scale

        # Fit GARCH model
        model = arch_model(scaled_values, vol="Garch", p=self.p, q=self.q, rescale=False)

        # Suppress convergence warnings for short series
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp="off", show_warning=False)

        # Extract conditional volatility
        volatility = result.conditional_volatility / scale

        return volatility


# =============================================================================
# Volatility Computation Functions
# =============================================================================


def compute_local_volatility(
    values: ArrayLike,
    window: int = 13,
    method: Literal["rolling_std", "ewm", "garch"] = "rolling_std",
) -> NDArray[np.float64]:
    """
    Compute local volatility estimates.

    Parameters
    ----------
    values : array-like
        Input values (typically returns or price changes).
    window : int, default 13
        Window size for rolling methods, or span for EWMA.
    method : {"rolling_std", "ewm", "garch"}, default "rolling_std"
        Volatility estimation method:
        - "rolling_std": Rolling window standard deviation
        - "ewm": Exponentially weighted moving average
        - "garch": GARCH(1,1) model (requires arch package)

    Returns
    -------
    ndarray
        Local volatility estimates, same length as input.

    Raises
    ------
    ValueError
        If invalid method or insufficient data for GARCH.
    ImportError
        If method="garch" and arch package not installed.

    Notes
    -----
    [T3] Window of 13 (approximately one quarter for weekly data) is a
    practical default balancing responsiveness with stability.

    Examples
    --------
    >>> returns = np.random.randn(100) * 0.02
    >>> vol = compute_local_volatility(returns, window=13)

    See Also
    --------
    RollingVolatility : Class-based rolling std estimator.
    EWMAVolatility : Class-based EWMA estimator.
    GARCHVolatility : Class-based GARCH estimator.
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return np.array([], dtype=np.float64)

    estimator: VolatilityEstimator
    if method == "rolling_std":
        estimator = RollingVolatility(window=window)
    elif method == "ewm":
        estimator = EWMAVolatility(span=window)
    elif method == "garch":
        estimator = GARCHVolatility(p=1, q=1)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'rolling_std', 'ewm', or 'garch'")

    return estimator.estimate(values)


# =============================================================================
# Volatility-Adjusted Metrics
# =============================================================================


def compute_volatility_normalized_mae(
    predictions: ArrayLike,
    actuals: ArrayLike,
    volatility: ArrayLike,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute volatility-normalized MAE (scale-invariant).

    Divides errors by local volatility, making the metric comparable
    across different volatility regimes and time series.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Actual observed values.
    volatility : array-like
        Local volatility estimates.
    epsilon : float, default 1e-8
        Small constant to prevent division by zero.

    Returns
    -------
    float
        Mean volatility-normalized absolute error.

    Raises
    ------
    ValueError
        If array lengths don't match or arrays are empty.

    Notes
    -----
    Volatility-normalized MAE is computed as:

        VN-MAE = mean( |prediction - actual| / volatility )

    A value of 1.0 means errors are "typical" relative to local volatility.
    Lower is better.

    Examples
    --------
    >>> predictions = np.array([1.0, 2.0, 3.0])
    >>> actuals = np.array([1.1, 1.9, 3.2])
    >>> volatility = np.array([0.1, 0.15, 0.2])
    >>> vnmae = compute_volatility_normalized_mae(predictions, actuals, volatility)

    See Also
    --------
    compute_volatility_weighted_mae : Weight by volatility instead of normalize.
    compute_local_volatility : Compute volatility estimates.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    volatility = np.asarray(volatility, dtype=np.float64)

    if len(predictions) != len(actuals) or len(predictions) != len(volatility):
        raise ValueError(
            f"Array lengths must match. Got predictions={len(predictions)}, "
            f"actuals={len(actuals)}, volatility={len(volatility)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    abs_errors = np.abs(predictions - actuals)
    normalized_errors = abs_errors / (volatility + epsilon)

    return float(np.mean(normalized_errors))


def compute_volatility_weighted_mae(
    predictions: ArrayLike,
    actuals: ArrayLike,
    volatility: ArrayLike,
    weighting: Literal["inverse", "importance"] = "inverse",
    epsilon: float = 1e-8,
) -> float:
    """
    Compute volatility-weighted MAE.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Actual observed values.
    volatility : array-like
        Local volatility estimates.
    weighting : {"inverse", "importance"}, default "inverse"
        How to weight by volatility:
        - "inverse": Weight low-vol periods more (clearer signal)
        - "importance": Weight high-vol periods more (if those matter)
    epsilon : float, default 1e-8
        Small constant to prevent division by zero.

    Returns
    -------
    float
        Weighted mean absolute error.

    Raises
    ------
    ValueError
        If array lengths don't match, arrays are empty, or invalid weighting.

    Notes
    -----
    With inverse weighting, low-volatility periods (where predictions
    should be more precise) receive higher weight. This is useful when
    the goal is accuracy during stable periods.

    With importance weighting, high-volatility periods receive higher
    weight. This is useful when performance during turbulent periods
    matters most (e.g., risk management).

    Examples
    --------
    >>> predictions = np.array([1.0, 2.0, 3.0])
    >>> actuals = np.array([1.1, 1.9, 3.2])
    >>> volatility = np.array([0.1, 0.5, 0.2])
    >>> # Weight low-vol periods more
    >>> wmae = compute_volatility_weighted_mae(
    ...     predictions, actuals, volatility, weighting="inverse"
    ... )

    See Also
    --------
    compute_volatility_normalized_mae : Normalize instead of weight.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    volatility = np.asarray(volatility, dtype=np.float64)

    if len(predictions) != len(actuals) or len(predictions) != len(volatility):
        raise ValueError(
            f"Array lengths must match. Got predictions={len(predictions)}, "
            f"actuals={len(actuals)}, volatility={len(volatility)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    if weighting not in ("inverse", "importance"):
        raise ValueError(f"weighting must be 'inverse' or 'importance', got '{weighting}'")

    abs_errors = np.abs(predictions - actuals)

    if weighting == "inverse":
        weights = 1.0 / (volatility + epsilon)
    else:  # importance
        weights = volatility

    # Normalize weights
    weights = weights / np.sum(weights)

    weighted_mae: float = float(np.sum(weights * abs_errors))

    return weighted_mae


# =============================================================================
# Volatility-Stratified Metrics
# =============================================================================


@dataclass
class VolatilityStratifiedResult:
    """Result container for volatility-stratified metrics.

    Attributes
    ----------
    overall_mae : float
        Overall mean absolute error.
    low_vol_mae : float
        MAE for low volatility tercile.
    med_vol_mae : float
        MAE for medium volatility tercile.
    high_vol_mae : float
        MAE for high volatility tercile.
    volatility_normalized_mae : float
        Volatility-normalized MAE.
    n_low : int
        Number of observations in low tercile.
    n_med : int
        Number of observations in medium tercile.
    n_high : int
        Number of observations in high tercile.
    vol_thresholds : tuple
        (low_upper, high_lower) volatility boundaries.
    """

    overall_mae: float
    low_vol_mae: float
    med_vol_mae: float
    high_vol_mae: float
    volatility_normalized_mae: float
    n_low: int
    n_med: int
    n_high: int
    vol_thresholds: tuple

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Volatility-Stratified Metrics",
            "=" * 40,
            f"Overall MAE:       {self.overall_mae:.6f}",
            f"VN-MAE:            {self.volatility_normalized_mae:.6f}",
            "",
            "By Volatility Regime:",
            f"  Low vol (n={self.n_low}):   MAE = {self.low_vol_mae:.6f}",
            f"  Med vol (n={self.n_med}):   MAE = {self.med_vol_mae:.6f}",
            f"  High vol (n={self.n_high}):  MAE = {self.high_vol_mae:.6f}",
            "",
            f"Thresholds: low < {self.vol_thresholds[0]:.6f} < med < {self.vol_thresholds[1]:.6f} < high",
        ]
        return "\n".join(lines)


def compute_volatility_stratified_metrics(
    predictions: ArrayLike,
    actuals: ArrayLike,
    volatility: ArrayLike | None = None,
    window: int = 13,
    method: Literal["rolling_std", "ewm"] = "rolling_std",
) -> VolatilityStratifiedResult:
    """
    Compute MAE stratified by volatility terciles.

    Parameters
    ----------
    predictions : array-like
        Predicted values.
    actuals : array-like
        Actual observed values.
    volatility : array-like, optional
        Pre-computed volatility estimates. If not provided, will be
        computed from actuals using the specified method.
    window : int, default 13
        Window size for volatility estimation (if volatility not provided).
    method : {"rolling_std", "ewm"}, default "rolling_std"
        Volatility estimation method (if volatility not provided).

    Returns
    -------
    VolatilityStratifiedResult
        Dataclass with stratified metrics.

    Raises
    ------
    ValueError
        If array lengths don't match or arrays are empty.

    Notes
    -----
    [T3] Tercile stratification uses the 33rd and 67th percentiles of
    volatility to create three equally-sized groups. This is a practical
    heuristic that provides meaningful regime separation.

    Examples
    --------
    >>> predictions = np.random.randn(100)
    >>> actuals = np.random.randn(100)
    >>> result = compute_volatility_stratified_metrics(predictions, actuals)
    >>> print(result.summary())

    See Also
    --------
    compute_volatility_normalized_mae : Overall normalized metric.
    compute_local_volatility : Volatility estimation.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Array lengths must match. Got predictions={len(predictions)}, "
            f"actuals={len(actuals)}"
        )

    if len(predictions) == 0:
        raise ValueError("Arrays cannot be empty")

    # Compute volatility if not provided
    if volatility is None:
        volatility = compute_local_volatility(actuals, window=window, method=method)
    else:
        volatility = np.asarray(volatility, dtype=np.float64)
        if len(volatility) != len(predictions):
            raise ValueError(
                f"volatility length must match. Got {len(volatility)}, "
                f"expected {len(predictions)}"
            )

    # Compute tercile thresholds
    p33: float = float(np.percentile(volatility, 33.33))
    p67: float = float(np.percentile(volatility, 66.67))

    # Classify into terciles
    low_mask = volatility <= p33
    high_mask = volatility > p67
    med_mask = ~low_mask & ~high_mask

    # Compute errors
    abs_errors = np.abs(predictions - actuals)

    # Overall metrics
    overall_mae = float(np.mean(abs_errors))
    vnmae = float(np.mean(abs_errors / (volatility + 1e-8)))

    # Stratified MAE
    def safe_mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if len(arr) > 0 else np.nan

    low_vol_mae = safe_mean(abs_errors[low_mask])
    med_vol_mae = safe_mean(abs_errors[med_mask])
    high_vol_mae = safe_mean(abs_errors[high_mask])

    return VolatilityStratifiedResult(
        overall_mae=overall_mae,
        low_vol_mae=low_vol_mae,
        med_vol_mae=med_vol_mae,
        high_vol_mae=high_vol_mae,
        volatility_normalized_mae=vnmae,
        n_low=int(np.sum(low_mask)),
        n_med=int(np.sum(med_mask)),
        n_high=int(np.sum(high_mask)),
        vol_thresholds=(float(p33), float(p67)),
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Protocol
    "VolatilityEstimator",
    # Estimator classes
    "RollingVolatility",
    "EWMAVolatility",
    "GARCHVolatility",
    # Functions
    "compute_local_volatility",
    "compute_volatility_normalized_mae",
    "compute_volatility_weighted_mae",
    # Stratified metrics
    "VolatilityStratifiedResult",
    "compute_volatility_stratified_metrics",
]
