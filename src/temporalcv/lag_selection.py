"""
Lag Selection for Time Series Models.

Provides methods to select optimal lag order for AR models using:
- PACF (Partial Autocorrelation Function) significance cutoff
- AIC (Akaike Information Criterion) minimization
- BIC (Bayesian Information Criterion) minimization

These are essential for:
1. Determining AR order for forecasting models
2. Setting appropriate gap parameters in cross-validation
3. Understanding series memory/persistence

References
----------
- Box, Jenkins & Reinsel (2015). Time Series Analysis, 5th ed.
- Hyndman & Athanasopoulos (2021). Forecasting: Principles and Practice, 3rd ed.
- Schwarz (1978). "Estimating the Dimension of a Model." Annals of Statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import pacf


@dataclass(frozen=True)
class LagSelectionResult:
    """Result of lag selection procedure.

    Attributes
    ----------
    optimal_lag : int
        Selected optimal lag order.
    criterion_values : dict[int, float]
        Criterion value (AIC/BIC/PACF) for each lag tested.
        For PACF, values are the PACF coefficients.
    method : str
        Method used for selection ('aic', 'bic', 'pacf').
    all_lags_tested : list[int]
        All lag values that were evaluated.
    """

    optimal_lag: int
    criterion_values: dict[int, float]
    method: str
    all_lags_tested: list[int]


def _compute_max_lag(n: int, max_lag: int | None) -> int:
    """Compute maximum lag to test.

    Uses rule of thumb: min(10 * log10(n), n/4) if not specified.

    Parameters
    ----------
    n : int
        Series length.
    max_lag : int | None
        User-specified maximum lag.

    Returns
    -------
    int
        Maximum lag to test.
    """
    if max_lag is not None:
        return min(max_lag, n // 2 - 1)

    # Rule of thumb: min(10*log10(n), n/4)
    rule_of_thumb = min(int(10 * np.log10(n)), n // 4)
    return max(1, min(rule_of_thumb, n // 2 - 1))


def select_lag_pacf(
    series: ArrayLike,
    max_lag: int | None = None,
    alpha: float = 0.05,
) -> LagSelectionResult:
    """Select lag using PACF significance cutoff.

    Finds the last lag where PACF is significantly different from zero.
    Uses the Bartlett approximation for confidence intervals: +/- z_{alpha/2} / sqrt(n).

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    max_lag : int | None
        Maximum lag to consider. If None, uses rule of thumb.
    alpha : float
        Significance level for PACF confidence interval.

    Returns
    -------
    LagSelectionResult
        Selected lag and PACF values for all lags.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # AR(2) process
    >>> n = 200
    >>> ar2 = np.zeros(n)
    >>> for i in range(2, n):
    ...     ar2[i] = 0.5 * ar2[i-1] + 0.3 * ar2[i-2] + rng.normal(0, 1)
    >>> result = select_lag_pacf(ar2)
    >>> result.optimal_lag
    2

    Notes
    -----
    [T1] Box, Jenkins & Reinsel (2015), Chapter 3.
    PACF at lag k measures correlation between y_t and y_{t-k} after
    removing the linear dependence on y_{t-1}, ..., y_{t-k+1}.
    """
    arr = np.asarray(series).ravel()
    n = len(arr)

    if n < 10:
        raise ValueError(f"Series too short for PACF: n={n}, need >= 10")

    max_lag_to_test = _compute_max_lag(n, max_lag)

    # Compute PACF
    pacf_values = pacf(arr, nlags=max_lag_to_test, method="ywadjusted")

    # Confidence interval threshold (Bartlett approximation)
    from scipy import stats

    z_critical = stats.norm.ppf(1 - alpha / 2)
    threshold = z_critical / np.sqrt(n)

    # Find the cutoff point: first lag where PACF becomes insignificant
    # This is the standard approach for AR order selection via PACF
    criterion_values: dict[int, float] = {}
    all_lags = list(range(max_lag_to_test + 1))

    for lag in range(max_lag_to_test + 1):
        pacf_val = float(pacf_values[lag])
        criterion_values[lag] = pacf_val

    # Find optimal lag: the last consecutive significant lag starting from 1
    # (PACF at lag 0 is always 1, so we skip it)
    optimal_lag = 0
    for lag in range(1, max_lag_to_test + 1):
        if abs(criterion_values[lag]) > threshold:
            optimal_lag = lag
        else:
            # First insignificant lag - stop here
            break

    return LagSelectionResult(
        optimal_lag=optimal_lag,
        criterion_values=criterion_values,
        method="pacf",
        all_lags_tested=all_lags,
    )


def select_lag_aic(
    series: ArrayLike,
    max_lag: int | None = None,
) -> LagSelectionResult:
    """Select lag minimizing AIC (Akaike Information Criterion).

    Fits AR(p) models for p = 0, 1, ..., max_lag and selects the
    lag that minimizes AIC.

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    max_lag : int | None
        Maximum lag to consider. If None, uses rule of thumb.

    Returns
    -------
    LagSelectionResult
        Selected lag and AIC values for all lags.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ar1 = np.zeros(100)
    >>> for i in range(1, 100):
    ...     ar1[i] = 0.7 * ar1[i-1] + rng.normal(0, 1)
    >>> result = select_lag_aic(ar1)
    >>> result.optimal_lag in [1, 2]  # Usually selects 1
    True

    Notes
    -----
    [T1] Akaike (1974). "A new look at the statistical model identification."
    AIC = 2k - 2ln(L) where k is number of parameters.
    AIC tends to select larger models than BIC.
    """
    arr = np.asarray(series).ravel()
    n = len(arr)

    if n < 10:
        raise ValueError(f"Series too short for lag selection: n={n}, need >= 10")

    max_lag_to_test = _compute_max_lag(n, max_lag)

    criterion_values: dict[int, float] = {}
    # Start from lag 1 (AR(0) is trivial and has inconsistent IC calculation)
    all_lags = list(range(1, max_lag_to_test + 1))

    # Fit AR(p) for each lag
    for lag in all_lags:
        try:
            model = AutoReg(arr, lags=lag, old_names=False)
            result = model.fit()
            criterion_values[lag] = float(result.aic)
        except Exception:
            # If fitting fails, use infinity
            criterion_values[lag] = float("inf")

    # Find lag with minimum AIC
    if criterion_values:
        optimal_lag = min(criterion_values, key=lambda k: criterion_values[k])
    else:
        optimal_lag = 1

    return LagSelectionResult(
        optimal_lag=optimal_lag,
        criterion_values=criterion_values,
        method="aic",
        all_lags_tested=all_lags,
    )


def select_lag_bic(
    series: ArrayLike,
    max_lag: int | None = None,
) -> LagSelectionResult:
    """Select lag minimizing BIC (Bayesian Information Criterion).

    Fits AR(p) models for p = 0, 1, ..., max_lag and selects the
    lag that minimizes BIC. BIC penalizes complexity more than AIC,
    tending to select more parsimonious models.

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    max_lag : int | None
        Maximum lag to consider. If None, uses rule of thumb.

    Returns
    -------
    LagSelectionResult
        Selected lag and BIC values for all lags.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ar1 = np.zeros(100)
    >>> for i in range(1, 100):
    ...     ar1[i] = 0.7 * ar1[i-1] + rng.normal(0, 1)
    >>> result = select_lag_bic(ar1)
    >>> result.optimal_lag
    1

    Notes
    -----
    [T1] Schwarz (1978). "Estimating the Dimension of a Model."
    BIC = k * ln(n) - 2ln(L) where k is number of parameters.
    BIC is asymptotically consistent (selects true order as n→∞).
    """
    arr = np.asarray(series).ravel()
    n = len(arr)

    if n < 10:
        raise ValueError(f"Series too short for lag selection: n={n}, need >= 10")

    max_lag_to_test = _compute_max_lag(n, max_lag)

    criterion_values: dict[int, float] = {}
    # Start from lag 1 (AR(0) is trivial and has inconsistent IC calculation)
    all_lags = list(range(1, max_lag_to_test + 1))

    # Fit AR(p) for each lag
    for lag in all_lags:
        try:
            model = AutoReg(arr, lags=lag, old_names=False)
            result = model.fit()
            criterion_values[lag] = float(result.bic)
        except Exception:
            # If fitting fails, use infinity
            criterion_values[lag] = float("inf")

    # Find lag with minimum BIC
    if criterion_values:
        optimal_lag = min(criterion_values, key=lambda k: criterion_values[k])
    else:
        optimal_lag = 1

    return LagSelectionResult(
        optimal_lag=optimal_lag,
        criterion_values=criterion_values,
        method="bic",
        all_lags_tested=all_lags,
    )


def auto_select_lag(
    series: ArrayLike,
    method: Literal["aic", "bic", "pacf"] = "bic",
    max_lag: int | None = None,
    alpha: float = 0.05,
) -> int:
    """Convenience function returning just the optimal lag.

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    method : {'aic', 'bic', 'pacf'}
        Selection method:
        - 'bic': Bayesian Information Criterion (default, most parsimonious)
        - 'aic': Akaike Information Criterion (larger models)
        - 'pacf': Partial Autocorrelation Function significance
    max_lag : int | None
        Maximum lag to consider.
    alpha : float
        Significance level for PACF method.

    Returns
    -------
    int
        Optimal lag order.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ar2 = np.zeros(200)
    >>> for i in range(2, 200):
    ...     ar2[i] = 0.5 * ar2[i-1] + 0.3 * ar2[i-2] + rng.normal(0, 1)
    >>> auto_select_lag(ar2, method='bic')
    2
    """
    if method == "aic":
        result = select_lag_aic(series, max_lag=max_lag)
    elif method == "bic":
        result = select_lag_bic(series, max_lag=max_lag)
    elif method == "pacf":
        result = select_lag_pacf(series, max_lag=max_lag, alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'aic', 'bic', or 'pacf'.")

    return result.optimal_lag


def suggest_cv_gap(
    series: ArrayLike,
    horizon: int = 1,
    method: Literal["aic", "bic", "pacf"] = "bic",
) -> int:
    """Suggest cross-validation gap based on series autocorrelation.

    The gap should be at least as large as the forecast horizon AND
    the series memory (significant autocorrelation lags).

    Parameters
    ----------
    series : array-like
        Time series data.
    horizon : int
        Forecast horizon.
    method : {'aic', 'bic', 'pacf'}
        Method for estimating series memory.

    Returns
    -------
    int
        Suggested gap parameter for walk-forward CV.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ar5 = np.zeros(200)
    >>> for i in range(5, 200):
    ...     ar5[i] = 0.3 * ar5[i-1] + 0.2 * ar5[i-5] + rng.normal(0, 1)
    >>> gap = suggest_cv_gap(ar5, horizon=1)
    >>> gap >= 1  # At least horizon
    True

    Notes
    -----
    Rule: gap = max(horizon, optimal_lag)
    This ensures temporal separation accounts for both forecast horizon
    and series memory.
    """
    optimal_lag = auto_select_lag(series, method=method)
    return max(horizon, optimal_lag)
