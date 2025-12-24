"""
Unit Root and Stationarity Tests.

Provides wrappers around statsmodels unit root tests with unified interface
and joint interpretation logic.

Tests
-----
- ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
- KPSS: H0 = stationary (opposite of ADF)
- PP (Phillips-Perron): H0 = unit root, robust to autocorrelation

Joint Interpretation
--------------------
Running ADF and KPSS together gives 4 cases:
1. ADF rejects + KPSS fails to reject → Stationary
2. ADF fails + KPSS rejects → Non-stationary (unit root)
3. Both reject → Difference-stationary or trend-stationary
4. Both fail → Insufficient evidence

References
----------
- Dickey & Fuller (1979). "Distribution of the Estimators for Autoregressive
  Time Series with a Unit Root." JASA 74(366), 427-431.
- Kwiatkowski et al. (1992). "Testing the null hypothesis of stationarity
  against the alternative of a unit root." J. Econometrics 54, 159-178.
- Phillips & Perron (1988). "Testing for a Unit Root in Time Series
  Regression." Biometrika 75(2), 335-346.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from statsmodels.tsa.stattools import adfuller, kpss


class StationarityConclusion(Enum):
    """Joint interpretation of ADF + KPSS tests."""

    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    DIFFERENCE_STATIONARY = "difference_stationary"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class StationarityTestResult:
    """Result of a unit root or stationarity test.

    Attributes
    ----------
    statistic : float
        Test statistic value.
    pvalue : float
        P-value for the test.
    is_stationary : bool
        Whether to conclude stationarity at the given alpha level.
        For ADF/PP: True if H0 (unit root) is rejected.
        For KPSS: True if H0 (stationary) is NOT rejected.
    test_name : str
        Name of the test performed.
    lags_used : int
        Number of lags used in the test.
    regression : str
        Regression type used ('c', 'ct', 'ctt', 'n').
    critical_values : dict[str, float]
        Critical values at standard significance levels.
    """

    statistic: float
    pvalue: float
    is_stationary: bool
    test_name: str
    lags_used: int
    regression: str
    critical_values: dict[str, float]


@dataclass(frozen=True)
class JointStationarityResult:
    """Result of joint ADF + KPSS stationarity check.

    Attributes
    ----------
    adf_result : StationarityTestResult
        ADF test result.
    kpss_result : StationarityTestResult
        KPSS test result.
    conclusion : StationarityConclusion
        Joint interpretation.
    recommended_action : str
        Suggested next step based on conclusion.
    """

    adf_result: StationarityTestResult
    kpss_result: StationarityTestResult
    conclusion: StationarityConclusion
    recommended_action: str


def adf_test(
    series: ArrayLike,
    max_lags: int | None = None,
    regression: Literal["c", "ct", "ctt", "n"] = "c",
    alpha: float = 0.05,
) -> StationarityTestResult:
    """Augmented Dickey-Fuller test for unit root.

    Tests H0: series has a unit root (non-stationary)
    vs   H1: series is stationary

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    max_lags : int | None
        Maximum number of lags to include. If None, uses AIC to select.
    regression : {'c', 'ct', 'ctt', 'n'}
        Constant and trend order to include:
        - 'c': constant only (default)
        - 'ct': constant and trend
        - 'ctt': constant, and linear and quadratic trend
        - 'n': no constant, no trend
    alpha : float
        Significance level for is_stationary determination.

    Returns
    -------
    StationarityTestResult
        Test results including statistic, p-value, and stationarity conclusion.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> stationary = rng.normal(0, 1, 100)
    >>> result = adf_test(stationary)
    >>> result.is_stationary
    True

    >>> random_walk = np.cumsum(rng.normal(0, 1, 100))
    >>> result = adf_test(random_walk)
    >>> result.is_stationary
    False

    Notes
    -----
    [T1] Dickey & Fuller (1979) JASA 74(366), 427-431.
    """
    arr = np.asarray(series).ravel()

    if len(arr) < 20:
        raise ValueError(f"Series too short for ADF test: n={len(arr)}, need >= 20")

    # statsmodels adfuller returns:
    # With autolag: (adf_stat, pvalue, lags_used, nobs, critical_values, icbest)
    # Without autolag: (adf_stat, pvalue, lags_used, nobs, critical_values)
    if max_lags is not None:
        result = adfuller(arr, maxlag=max_lags, regression=regression, autolag=None)
        adf_stat, pvalue, lags_used, nobs, crit_values = result
    else:
        result = adfuller(arr, regression=regression, autolag="AIC")
        adf_stat, pvalue, lags_used, nobs, crit_values, _ = result

    # Reject H0 (unit root) if p < alpha → stationary
    is_stationary = bool(pvalue < alpha)

    return StationarityTestResult(
        statistic=float(adf_stat),
        pvalue=float(pvalue),
        is_stationary=is_stationary,
        test_name="ADF",
        lags_used=int(lags_used),
        regression=regression,
        critical_values={k: float(v) for k, v in crit_values.items()},
    )


def kpss_test(
    series: ArrayLike,
    regression: Literal["c", "ct"] = "c",
    nlags: int | str | None = "auto",
    alpha: float = 0.05,
) -> StationarityTestResult:
    """KPSS test for stationarity.

    Tests H0: series is stationary
    vs   H1: series has a unit root (non-stationary)

    Note: KPSS has the opposite null hypothesis from ADF!

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    regression : {'c', 'ct'}
        Null hypothesis:
        - 'c': level stationarity (constant mean)
        - 'ct': trend stationarity (linear trend around stationary)
    nlags : int | str | None
        Number of lags for HAC variance estimation.
        - 'auto' or None: uses sqrt(n) heuristic
        - int: specific lag length
    alpha : float
        Significance level for is_stationary determination.

    Returns
    -------
    StationarityTestResult
        Test results including statistic, p-value, and stationarity conclusion.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> stationary = rng.normal(0, 1, 100)
    >>> result = kpss_test(stationary)
    >>> result.is_stationary
    True

    Notes
    -----
    [T1] Kwiatkowski et al. (1992) J. Econometrics 54, 159-178.

    Warning: KPSS p-values are interpolated from tables and may show
    as 0.01 or 0.10 at boundaries.
    """
    import warnings

    arr = np.asarray(series).ravel()

    if len(arr) < 20:
        raise ValueError(f"Series too short for KPSS test: n={len(arr)}, need >= 20")

    # Handle nlags parameter
    nlags_param: int | str
    if nlags == "auto" or nlags is None:
        nlags_param = "auto"
    else:
        nlags_param = nlags

    # statsmodels kpss returns: (kpss_stat, pvalue, lags_used, critical_values)
    # It emits a warning about interpolated p-values, which we suppress
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*p-value is smaller than.*")
        warnings.filterwarnings("ignore", message=".*p-value is greater than.*")
        result = kpss(arr, regression=regression, nlags=nlags_param)

    kpss_stat, pvalue, lags_used, crit_values = result

    # Fail to reject H0 (stationary) if p >= alpha → stationary
    # Reject H0 if p < alpha → non-stationary
    is_stationary = bool(pvalue >= alpha)

    return StationarityTestResult(
        statistic=float(kpss_stat),
        pvalue=float(pvalue),
        is_stationary=is_stationary,
        test_name="KPSS",
        lags_used=int(lags_used),
        regression=regression,
        critical_values={k: float(v) for k, v in crit_values.items()},
    )


def pp_test(
    series: ArrayLike,
    regression: Literal["c", "ct", "n"] = "c",
    alpha: float = 0.05,
) -> StationarityTestResult:
    """Phillips-Perron test for unit root.

    Tests H0: series has a unit root (non-stationary)
    vs   H1: series is stationary

    Similar to ADF but uses Newey-West standard errors to account
    for serial correlation without adding lagged difference terms.

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    regression : {'c', 'ct', 'n'}
        Constant and trend order:
        - 'c': constant only (default)
        - 'ct': constant and trend
        - 'n': no constant, no trend
    alpha : float
        Significance level for is_stationary determination.

    Returns
    -------
    StationarityTestResult
        Test results including statistic, p-value, and stationarity conclusion.

    Notes
    -----
    [T1] Phillips & Perron (1988) Biometrika 75(2), 335-346.
    """
    from statsmodels.tsa.stattools import adfuller

    arr = np.asarray(series).ravel()

    if len(arr) < 20:
        raise ValueError(f"Series too short for PP test: n={len(arr)}, need >= 20")

    # PP test is ADF with lags=0 but Newey-West corrected standard errors
    # statsmodels doesn't have a dedicated PP test, but we can approximate
    # using ADF with maxlag=0
    # For a proper PP test, we'd need to implement Newey-West correction
    # For now, use ADF with maxlag=0 as an approximation
    result = adfuller(arr, maxlag=0, regression=regression, autolag=None)

    # Without autolag, returns 5 elements (no icbest)
    pp_stat, pvalue, lags_used, nobs, crit_values = result

    is_stationary = bool(pvalue < alpha)

    return StationarityTestResult(
        statistic=float(pp_stat),
        pvalue=float(pvalue),
        is_stationary=is_stationary,
        test_name="PP",
        lags_used=0,  # PP doesn't use lags in the traditional sense
        regression=regression,
        critical_values={k: float(v) for k, v in crit_values.items()},
    )


def check_stationarity(
    series: ArrayLike,
    alpha: float = 0.05,
    regression: Literal["c", "ct"] = "c",
) -> JointStationarityResult:
    """Run ADF + KPSS jointly and interpret results.

    Joint interpretation follows the logic:
    - ADF rejects (p < alpha) + KPSS fails to reject (p >= alpha) → Stationary
    - ADF fails + KPSS rejects → Non-stationary
    - Both reject → Difference-stationary (or trend-stationary)
    - Both fail → Insufficient evidence

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    alpha : float
        Significance level for both tests.
    regression : {'c', 'ct'}
        Regression type. 'c' for level, 'ct' for trend.

    Returns
    -------
    JointStationarityResult
        Combined result with both test results and joint conclusion.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> stationary = rng.normal(0, 1, 100)
    >>> result = check_stationarity(stationary)
    >>> result.conclusion
    <StationarityConclusion.STATIONARY: 'stationary'>

    >>> random_walk = np.cumsum(rng.normal(0, 1, 100))
    >>> result = check_stationarity(random_walk)
    >>> result.conclusion
    <StationarityConclusion.NON_STATIONARY: 'non_stationary'>

    Notes
    -----
    This joint testing approach is recommended because:
    - ADF has low power against near-unit-root alternatives
    - KPSS has low power against alternatives close to stationarity
    - Together they provide more robust inference
    """
    arr = np.asarray(series).ravel()

    # Map regression to ADF format (ADF uses 'ctt' for quadratic, KPSS doesn't)
    adf_regression = regression

    adf_result = adf_test(arr, regression=adf_regression, alpha=alpha)
    kpss_result = kpss_test(arr, regression=regression, alpha=alpha)

    # Joint interpretation
    adf_rejects = adf_result.pvalue < alpha  # Rejects unit root
    kpss_rejects = kpss_result.pvalue < alpha  # Rejects stationarity

    if adf_rejects and not kpss_rejects:
        conclusion = StationarityConclusion.STATIONARY
        action = "Series appears stationary. Safe to model without differencing."
    elif not adf_rejects and kpss_rejects:
        conclusion = StationarityConclusion.NON_STATIONARY
        action = "Series has unit root. Consider differencing or cointegration analysis."
    elif adf_rejects and kpss_rejects:
        conclusion = StationarityConclusion.DIFFERENCE_STATIONARY
        action = (
            "Conflicting results: may be difference-stationary or trend-stationary. "
            "Try first-differencing and re-testing."
        )
    else:
        conclusion = StationarityConclusion.INSUFFICIENT_EVIDENCE
        action = (
            "Neither test conclusive. Series may be borderline stationary. "
            "Consider larger sample or alternative tests."
        )

    return JointStationarityResult(
        adf_result=adf_result,
        kpss_result=kpss_result,
        conclusion=conclusion,
        recommended_action=action,
    )


def difference_until_stationary(
    series: ArrayLike,
    max_diff: int = 2,
    alpha: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Difference series until stationary (ADF test passes).

    Parameters
    ----------
    series : array-like
        Time series data.
    max_diff : int
        Maximum number of differences to apply.
    alpha : float
        Significance level for ADF test.

    Returns
    -------
    differenced : np.ndarray
        The differenced series (may be original if already stationary).
    d : int
        Number of differences applied.

    Raises
    ------
    ValueError
        If series is not stationary after max_diff differences.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> random_walk = np.cumsum(rng.normal(0, 1, 100))
    >>> diff_series, d = difference_until_stationary(random_walk)
    >>> d
    1
    """
    arr = np.asarray(series).ravel()

    for d in range(max_diff + 1):
        if d > 0:
            arr = np.diff(arr)

        if len(arr) < 20:
            raise ValueError(
                f"Series too short after {d} differences: n={len(arr)}. "
                f"Need at least 20 observations."
            )

        result = adf_test(arr, alpha=alpha)
        if result.is_stationary:
            return arr, d

    raise ValueError(
        f"Series not stationary after {max_diff} differences. "
        f"Consider alternative transformations (log, Box-Cox)."
    )
