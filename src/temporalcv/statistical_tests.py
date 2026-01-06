"""
Statistical Tests Module.

Implements statistical tests for forecast evaluation:

- **Diebold-Mariano test** (DM 1995): Compare predictive accuracy of two models
- **Giacomini-White test** (GW 2006): Test conditional predictive ability
- **Clark-West test** (CW 2007): Compare nested models (bias-corrected)
- **Pesaran-Timmermann test** (PT 1992): Test directional accuracy
- **HAC variance** (Newey-West 1987): Correct for serial correlation in h>1 forecasts
- **Self-normalized variance** (Shao 2010): Bandwidth-free variance estimation

Knowledge Tiers
---------------
[T1] DM test core methodology (Diebold & Mariano 1995)
[T1] GW test conditional predictive ability (Giacomini & White 2006)
[T1] CW test for nested models (Clark & West 2007)
[T1] Harvey small-sample adjustment (Harvey et al. 1997)
[T1] HAC variance with Bartlett kernel (Newey & West 1987)
[T1] Self-normalized variance (Shao 2010, Lobato 2001)
[T1] PT test 2-class formulas (Pesaran & Timmermann 1992)
[T1] Automatic bandwidth selection (Andrews 1991)
[T2] Self-normalized critical values (simulation-derived)
[T2] Minimum sample size n >= 30 for DM/CW, n >= 50 for GW, n >= 20 for PT
[T3] PT 3-class mode is ad-hoc extension, not published (exploratory use only)

Example
-------
>>> from temporalcv.statistical_tests import dm_test, gw_test, pt_test
>>>
>>> # Compare model to baseline (unconditional)
>>> result = dm_test(model_errors, baseline_errors, h=2)
>>> print(f"DM statistic: {result.statistic:.3f}, p-value: {result.pvalue:.4f}")
>>>
>>> # Test conditional predictive ability
>>> gw_result = gw_test(model_errors, baseline_errors, n_lags=1)
>>> print(f"GW R²: {gw_result.r_squared:.3f}, p-value: {gw_result.pvalue:.4f}")
>>>
>>> # Test directional accuracy
>>> pt_result = pt_test(actual_changes, predicted_changes, move_threshold=0.01)
>>> print(f"Direction accuracy: {pt_result.accuracy:.2%}")

References
----------
[T1] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy.
     Journal of Business & Economic Statistics, 13(3), 253-263.
[T1] Giacomini, R. & White, H. (2006). Tests of Conditional Predictive Ability.
     Econometrica, 74(6), 1545-1578.
[T1] Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality
     of prediction mean squared errors. International Journal of Forecasting,
     13(2), 281-291.
[T1] Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test
     of predictive performance. Journal of Business & Economic Statistics,
     10(4), 461-465.
[T1] Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
     heteroskedasticity and autocorrelation consistent covariance matrix.
     Econometrica, 55(3), 703-708.
[T1] Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent
     covariance matrix estimation. Econometrica, 59(3), 817-858.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy import stats


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class DMTestResult:
    """
    Result from Diebold-Mariano test.

    Attributes
    ----------
    statistic : float
        DM test statistic (asymptotically N(0,1) under H0 for HAC;
        non-standard distribution for self-normalized)
    pvalue : float
        P-value for the test
    h : int
        Forecast horizon used
    n : int
        Number of observations
    loss : str
        Loss function used ("squared" or "absolute")
    alternative : str
        Alternative hypothesis ("two-sided", "less", "greater")
    harvey_adjusted : bool
        Whether Harvey et al. (1997) small-sample adjustment was applied
    mean_loss_diff : float
        Mean loss differential (positive = model 1 has higher loss)
    variance_method : str
        Variance estimation method ("hac" or "self_normalized")
    """

    statistic: float
    pvalue: float
    h: int
    n: int
    loss: str
    alternative: str
    harvey_adjusted: bool
    mean_loss_diff: float
    variance_method: str = "hac"  # Default for backward compatibility

    def __str__(self) -> str:
        """Format result as string."""
        sig = "***" if self.pvalue < 0.01 else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.10 else ""
        return f"DM({self.h}): {self.statistic:.3f} (p={self.pvalue:.4f}){sig}"

    @property
    def significant_at_05(self) -> bool:
        """Is result significant at alpha=0.05?"""
        return self.pvalue < 0.05

    @property
    def significant_at_01(self) -> bool:
        """Is result significant at alpha=0.01?"""
        return self.pvalue < 0.01


@dataclass
class PTTestResult:
    """
    Result from Pesaran-Timmermann directional accuracy test.

    Attributes
    ----------
    statistic : float
        PT test statistic (z-score, asymptotically N(0,1) under H0)
    pvalue : float
        P-value (one-sided: testing if better than random)
    accuracy : float
        Observed directional accuracy
    expected : float
        Expected accuracy under null hypothesis (independence)
    n : int
        Number of observations
    n_classes : int
        Number of direction classes (2 or 3)
    """

    statistic: float
    pvalue: float
    accuracy: float
    expected: float
    n: int
    n_classes: int

    def __str__(self) -> str:
        """Format result as string."""
        sig = "***" if self.pvalue < 0.01 else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.10 else ""
        return f"PT: {self.accuracy:.1%} vs {self.expected:.1%} expected (z={self.statistic:.3f}, p={self.pvalue:.4f}){sig}"

    @property
    def significant_at_05(self) -> bool:
        """Is directional accuracy significantly better than random?"""
        return self.pvalue < 0.05

    @property
    def skill(self) -> float:
        """Directional skill = accuracy - expected."""
        return self.accuracy - self.expected


@dataclass
class GWTestResult:
    """
    Result from Giacomini-White test for conditional predictive ability.

    The GW test examines whether past loss differentials can predict which
    forecast will be more accurate in the future. Unlike the DM test which
    tests *unconditional* equal predictive ability (average performance),
    the GW test examines *conditional* predictive ability.

    Attributes
    ----------
    statistic : float
        GW test statistic (T × R², asymptotically χ²(q) under H0)
    pvalue : float
        P-value for the test
    r_squared : float
        R-squared from auxiliary regression (predictability of loss differential)
    n : int
        Effective sample size (after lag adjustment)
    n_lags : int
        Number of lags used in conditioning set (τ)
    q : int
        Degrees of freedom (1 + n_lags)
    loss : str
        Loss function used ("squared" or "absolute")
    alternative : str
        Alternative hypothesis ("two-sided", "less", "greater")
    mean_loss_diff : float
        Mean loss differential (positive = model 1 has higher loss)

    Notes
    -----
    **Interpretation**:

    - If GW rejects but DM does not: Models have equal *average* accuracy,
      but one is conditionally superior given recent performance.
    - If both reject: One model is unconditionally and conditionally better.
    - If neither rejects: No detectable difference in predictive ability.

    **Key insight**: R² measures how predictable the loss differential is.
    High R² means forecasters could improve by switching between models
    based on recent relative performance.

    References
    ----------
    [T1] Giacomini, R. & White, H. (2006). Tests of Conditional Predictive
         Ability. Econometrica, 74(6), 1545-1578.
    """

    statistic: float
    pvalue: float
    r_squared: float
    n: int
    n_lags: int
    q: int
    loss: str
    alternative: str
    mean_loss_diff: float

    def __str__(self) -> str:
        """Format result as string."""
        sig = "***" if self.pvalue < 0.01 else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.10 else ""
        return f"GW({self.n_lags}): {self.statistic:.3f} (p={self.pvalue:.4f}, R²={self.r_squared:.3f}){sig}"

    @property
    def significant_at_05(self) -> bool:
        """Is result significant at alpha=0.05?"""
        return self.pvalue < 0.05

    @property
    def significant_at_01(self) -> bool:
        """Is result significant at alpha=0.01?"""
        return self.pvalue < 0.01

    @property
    def conditional_predictability(self) -> bool:
        """Is there evidence of conditional predictability at 5% level?"""
        return self.significant_at_05


@dataclass
class CWTestResult:
    """
    Result from Clark-West test for nested model comparison.

    The CW test adjusts the DM test for the bias caused by estimating
    extra parameters in the unrestricted model that have true value zero.
    This bias makes the unrestricted model appear worse than it truly is.

    For nested models (where the restricted model is a special case of the
    unrestricted model), use cw_test() instead of dm_test().

    Attributes
    ----------
    statistic : float
        CW test statistic (asymptotically N(0,1) under H0)
    pvalue : float
        P-value for the test
    h : int
        Forecast horizon used
    n : int
        Number of observations
    loss : str
        Loss function used ("squared" or "absolute")
    alternative : str
        Alternative hypothesis ("two-sided", "less", "greater")
    harvey_adjusted : bool
        Whether Harvey et al. (1997) small-sample adjustment was applied
    mean_loss_diff : float
        Mean unadjusted loss differential E[d_t]
        (positive = unrestricted model has higher loss)
    mean_loss_diff_adjusted : float
        Mean adjusted loss differential E[d*_t] after CW correction
        (removes parameter estimation noise)
    adjustment_magnitude : float
        Mean of (ŷ_restricted - ŷ_unrestricted)² — the noise removed
    variance_method : str
        Variance estimation method ("hac" or "self_normalized")

    Notes
    -----
    **When to use CW vs DM**:

    - Use DM test for non-nested models (e.g., ARIMA vs Random Forest)
    - Use CW test for nested models (e.g., AR(2) vs AR(1), Full vs Reduced)

    **The adjustment**: d*_t = d_t - (ŷ_r - ŷ_u)²

    The term (ŷ_r - ŷ_u)² captures the noise cost of estimating extra
    parameters with true value zero. When the restriction is true (extra
    parameters don't help), E[d_t] > 0 spuriously, but E[d*_t] ≈ 0.

    References
    ----------
    [T1] Clark, T.E. & West, K.D. (2007). Approximately normal tests for
         equal predictive accuracy in nested models. Journal of Econometrics,
         138(1), 291-311.
    """

    statistic: float
    pvalue: float
    h: int
    n: int
    loss: str
    alternative: str
    harvey_adjusted: bool
    mean_loss_diff: float
    mean_loss_diff_adjusted: float
    adjustment_magnitude: float
    variance_method: str = "hac"

    def __str__(self) -> str:
        """Format result as string."""
        sig = "***" if self.pvalue < 0.01 else "**" if self.pvalue < 0.05 else "*" if self.pvalue < 0.10 else ""
        return f"CW({self.h}): {self.statistic:.3f} (p={self.pvalue:.4f}, adj={self.adjustment_magnitude:.4f}){sig}"

    @property
    def significant_at_05(self) -> bool:
        """Is result significant at alpha=0.05?"""
        return self.pvalue < 0.05

    @property
    def significant_at_01(self) -> bool:
        """Is result significant at alpha=0.01?"""
        return self.pvalue < 0.01

    @property
    def adjustment_ratio(self) -> float:
        """Ratio of adjustment to unadjusted loss differential magnitude."""
        if abs(self.mean_loss_diff) < 1e-10:
            return float("inf") if self.adjustment_magnitude > 0 else 0.0
        return self.adjustment_magnitude / abs(self.mean_loss_diff)


# =============================================================================
# HAC Variance Estimation
# =============================================================================


def _bartlett_kernel(j: int, bandwidth: int) -> float:
    """
    Bartlett kernel weight for lag j.

    Parameters
    ----------
    j : int
        Lag index (non-negative)
    bandwidth : int
        Kernel bandwidth

    Returns
    -------
    float
        Kernel weight in [0, 1]
    """
    if abs(j) <= bandwidth:
        return 1.0 - abs(j) / (bandwidth + 1)
    return 0.0


def compute_hac_variance(
    d: np.ndarray,
    bandwidth: Optional[int] = None,
) -> float:
    """
    Compute HAC (Heteroskedasticity and Autocorrelation Consistent) variance.

    Uses Newey-West estimator with Bartlett kernel.

    Parameters
    ----------
    d : np.ndarray
        Series (typically loss differential for DM test)
    bandwidth : int, optional
        Kernel bandwidth. If None, uses automatic selection:
        floor(4 * (n/100)^(2/9))

    Returns
    -------
    float
        HAC variance estimate

    Notes
    -----
    For h-step forecasts, errors are MA(h-1), so bandwidth = h-1 is appropriate.
    The automatic bandwidth is a general-purpose choice when h is unknown.

    Complexity: O(n × bandwidth)

    See Also
    --------
    dm_test : Primary consumer of HAC variance estimation.
    """
    n = len(d)
    d_demeaned = d - np.mean(d)

    # Automatic bandwidth: Andrews (1991) rule
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))

    # Compute autocovariances
    gamma = np.zeros(bandwidth + 1)
    for j in range(bandwidth + 1):
        if j == 0:
            gamma[j] = np.mean(d_demeaned**2)
        else:
            gamma[j] = np.mean(d_demeaned[j:] * d_demeaned[:-j])

    # Apply Bartlett kernel weights
    variance = gamma[0]
    for j in range(1, bandwidth + 1):
        weight = _bartlett_kernel(j, bandwidth)
        variance += 2 * weight * gamma[j]

    return float(variance / n)


# =============================================================================
# Self-Normalized Variance Estimation
# =============================================================================


# Critical values for self-normalized DM test [T2]
# Non-standard limiting distribution: W(1)² / ∫₀¹ W(r)² dr
# Values derived from simulation (Shao 2010, Lobato 2001)
_SN_CRITICAL_VALUES = {
    # (alternative, alpha): critical_value
    ("two-sided", 0.01): 3.24,
    ("two-sided", 0.05): 2.22,
    ("two-sided", 0.10): 1.82,
    ("one-sided", 0.01): 2.70,
    ("one-sided", 0.05): 1.95,
    ("one-sided", 0.10): 1.60,
}


def compute_self_normalized_variance(d: np.ndarray) -> float:
    """
    Compute self-normalized variance using partial sums.

    This estimator is robust to bandwidth selection and cannot produce
    negative variance, unlike HAC estimators.

    Parameters
    ----------
    d : np.ndarray
        Series (typically loss differential for DM test)

    Returns
    -------
    float
        Self-normalized variance estimate (always >= 0)

    Notes
    -----
    The self-normalized variance is computed as:

    .. math::

        V_n^{SN} = \\frac{1}{n^2} \\sum_{k=1}^{n} S_k^2

    where :math:`S_k = \\sum_{t=1}^{k} (d_t - \\bar{d})` are partial sums.

    Key advantages over HAC [T1]:
    1. No bandwidth selection required
    2. Cannot produce negative variance
    3. Better size control in small samples

    Trade-off: Slightly lower power than well-tuned HAC.

    Complexity: O(n)

    References
    ----------
    [T1] Shao, X. (2010). A self-normalized approach to confidence interval
         construction in time series. JRSSB, 72(3), 343-366.
    [T1] Lobato, I.N. (2001). Testing that a dependent process is uncorrelated.
         JASA, 96(453), 169-176.

    See Also
    --------
    compute_hac_variance : HAC variance estimator (bandwidth-dependent).
    dm_test : Primary consumer of variance estimation.
    """
    n = len(d)
    if n == 0:
        return 0.0

    d_demeaned = d - np.mean(d)
    partial_sums = np.cumsum(d_demeaned)

    # V_n^SN = (1/n²) Σ S_k²
    variance = np.sum(partial_sums**2) / (n**2)

    return float(variance)


def _sn_pvalue(statistic: float, alternative: str) -> float:
    """
    Compute p-value for self-normalized test statistic.

    Uses linear interpolation between critical values. For values
    outside the table, uses conservative bounds.

    Parameters
    ----------
    statistic : float
        Self-normalized test statistic
    alternative : str
        "two-sided", "less", or "greater"

    Returns
    -------
    float
        Approximate p-value
    """
    abs_stat = abs(statistic)

    # Determine one-sided or two-sided
    if alternative == "two-sided":
        key_prefix = "two-sided"
    else:
        key_prefix = "one-sided"

    # Get critical values
    cv_01 = _SN_CRITICAL_VALUES[(key_prefix, 0.01)]
    cv_05 = _SN_CRITICAL_VALUES[(key_prefix, 0.05)]
    cv_10 = _SN_CRITICAL_VALUES[(key_prefix, 0.10)]

    # Approximate p-value by interpolation
    if abs_stat >= cv_01:
        pvalue = 0.01  # Very significant
    elif abs_stat >= cv_05:
        # Interpolate between 0.01 and 0.05
        pvalue = 0.01 + (0.05 - 0.01) * (cv_01 - abs_stat) / (cv_01 - cv_05)
    elif abs_stat >= cv_10:
        # Interpolate between 0.05 and 0.10
        pvalue = 0.05 + (0.10 - 0.05) * (cv_05 - abs_stat) / (cv_05 - cv_10)
    else:
        # Not significant at 0.10 level
        # Use conservative linear extrapolation
        pvalue = min(1.0, 0.10 + (0.90) * (cv_10 - abs_stat) / cv_10)

    # For one-sided tests, adjust based on sign of statistic
    # The interpolated pvalue is based on |statistic| using one-sided critical values
    if alternative == "less":
        # H1: model 1 better (lower loss) => d_bar < 0 => statistic < 0
        if statistic > 0:
            # Statistic in wrong direction - not significant at all
            pvalue = 1.0
        # else: statistic <= 0, keep pvalue from one-sided critical values
    elif alternative == "greater":
        # H1: model 2 better => d_bar > 0 => statistic > 0
        if statistic < 0:
            # Statistic in wrong direction - not significant at all
            pvalue = 1.0
        # else: statistic >= 0, keep pvalue from one-sided critical values
    # else: two-sided, keep pvalue as computed from two-sided critical values

    return float(np.clip(pvalue, 0.0, 1.0))


# =============================================================================
# Diebold-Mariano Test
# =============================================================================


def dm_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    harvey_correction: bool = True,
    variance_method: Literal["hac", "self_normalized"] = "hac",
) -> DMTestResult:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t) is the loss differential.

    Parameters
    ----------
    errors_1 : array-like
        Forecast errors from model 1 (actual - prediction)
    errors_2 : array-like
        Forecast errors from model 2 (baseline)
    h : int, default=1
        Forecast horizon. Used for HAC bandwidth (h-1) and Harvey adjustment.
    loss : {"squared", "absolute"}, default="squared"
        Loss function for comparing forecasts
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Alternative hypothesis:
        - "two-sided": Models have different accuracy
        - "less": Model 1 more accurate (lower loss)
        - "greater": Model 2 more accurate (model 1 has higher loss)
    harvey_correction : bool, default=True
        Apply Harvey et al. (1997) small-sample adjustment.
        Recommended for n < 100 or h > 1. Only applies when variance_method="hac".
    variance_method : {"hac", "self_normalized"}, default="hac"
        Method for variance estimation:

        - "hac": Heteroskedasticity and autocorrelation consistent (Newey-West).
          Requires bandwidth selection, can produce negative variance in edge cases.
        - "self_normalized": Bandwidth-free, always positive variance [T1].
          Uses non-standard distribution for p-values. Slightly lower power but
          more robust size control. Recommended when bandwidth selection is uncertain.

    Returns
    -------
    DMTestResult
        Test results including statistic, p-value, and diagnostics

    Raises
    ------
    ValueError
        If inputs are invalid (different lengths, too few observations)

    Notes
    -----
    For h>1 step forecasts, errors are MA(h-1) and HAC variance is required.
    The Harvey adjustment corrects for small-sample bias in the variance estimate.

    Harvey adjustment: DM_adj = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)

    .. warning::

       **Important Limitations** (Diebold 2015 retrospective):

       1. **Designed for forecasts, not models**: The DM test compares two sets of
          *forecasts* under the assumption of a fixed data-generating process.
          It was NOT designed for comparing *models* (e.g., in nested model comparison
          or model selection contexts). See Clark & West (2007) for nested models.

       2. **Negative variance estimates**: HAC variance estimation can produce
          negative estimates with multi-step forecasts (h > 1), especially with
          strong autocorrelation. This function returns pvalue=1.0 in such cases.
          See Coroneo & Iacone (2016) for detailed analysis.

       3. **Size distortions in small samples**: Even with Harvey adjustment, the
          test may have incorrect size (reject too often or too rarely) when
          n < 50. Use bootstrap alternatives for small samples.

       4. **Low power with strong autocorrelation**: When loss differentials are
          highly persistent, the test has low power to detect real differences.

       5. **Bandwidth sensitivity**: HAC variance estimation is sensitive to
          bandwidth choice. We use h-1 (theoretically motivated for MA(h-1)
          structure), but this may be suboptimal in practice.

    References
    ----------
    Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy.
        Journal of Business & Economic Statistics, 13(3), 253-263.
    Diebold, F.X. (2015). Comparing Predictive Accuracy, Twenty Years Later:
        A Personal Perspective. Journal of Business & Economic Statistics, 33(1), 1-8.
    Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the Equality of
        Prediction Mean Squared Errors. International Journal of Forecasting, 13(2), 281-291.
    Clark, T.E. & West, K.D. (2007). Approximately Normal Tests for Equal Predictive
        Accuracy in Nested Models. Journal of Econometrics, 138(1), 291-311.
    Coroneo, L. & Iacone, F. (2016). Comparing Predictive Accuracy in Small Samples.
        Journal of Forecasting, 35(7), 608-625.
    Shao, X. (2010). A self-normalized approach to confidence interval construction
        in time series. Journal of the Royal Statistical Society: Series B, 72(3), 343-366.
    Lobato, I.N. (2001). Testing that a dependent process is uncorrelated.
        Journal of the American Statistical Association, 96(453), 169-176.

    Example
    -------
    >>> # Test if model beats persistence baseline (using HAC variance)
    >>> result = dm_test(model_errors, persistence_errors, h=2, alternative="less")
    >>> if result.significant_at_05:
    ...     print("Model significantly better than baseline")

    >>> # Use self-normalized variance for robustness
    >>> result = dm_test(model_errors, baseline_errors, variance_method="self_normalized")
    >>> print(f"DM stat: {result.statistic:.3f}, p-value: {result.pvalue:.4f}")

    See Also
    --------
    pt_test : Complementary test for directional accuracy.
    compute_hac_variance : HAC variance estimator used internally.
    compute_dm_influence : Identify high-influence observations in DM test.
    """
    errors_1 = np.asarray(errors_1, dtype=np.float64)
    errors_2 = np.asarray(errors_2, dtype=np.float64)

    # Validate no NaN values
    if np.any(np.isnan(errors_1)):
        raise ValueError(
            "errors_1 contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )
    if np.any(np.isnan(errors_2)):
        raise ValueError(
            "errors_2 contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )

    if len(errors_1) != len(errors_2):
        raise ValueError(
            f"Error arrays must have same length. "
            f"Got {len(errors_1)} and {len(errors_2)}"
        )

    n = len(errors_1)

    if n < 30:
        raise ValueError(
            f"Insufficient samples for reliable DM test. Need >= 30, got {n}. "
            f"For n < 30, consider bootstrap-based tests or qualitative comparison."
        )

    if h < 1:
        raise ValueError(f"Horizon h must be >= 1, got {h}")

    # Compute loss differential
    loss_1: np.ndarray
    loss_2: np.ndarray
    if loss == "squared":
        loss_1 = errors_1**2
        loss_2 = errors_2**2
    elif loss == "absolute":
        loss_1 = np.abs(errors_1)
        loss_2 = np.abs(errors_2)
    else:
        raise ValueError(f"Unknown loss function: {loss}. Use 'squared' or 'absolute'.")

    d = loss_1 - loss_2  # Positive = model 1 has higher loss (worse)
    d_bar = float(np.mean(d))

    # Validate variance_method
    if variance_method not in ("hac", "self_normalized"):
        raise ValueError(
            f"Unknown variance_method: {variance_method}. "
            "Use 'hac' or 'self_normalized'."
        )

    # Branch variance computation based on method
    if variance_method == "self_normalized":
        # Self-normalized variance: bandwidth-free, always positive [T1]
        var_d = compute_self_normalized_variance(d)

        # Handle degenerate case (constant loss differential)
        if var_d <= 0:
            warnings.warn(
                f"DM test self-normalized variance is zero (var_d={var_d:.2e}). "
                "Loss differences are constant. Returning pvalue=1.0. "
                "Check that predictions differ.",
                UserWarning,
                stacklevel=2,
            )
            return DMTestResult(
                statistic=float("nan"),
                pvalue=1.0,
                h=h,
                n=n,
                loss=loss,
                alternative=alternative,
                harvey_adjusted=False,  # Not applicable for self-normalized
                mean_loss_diff=d_bar,
                variance_method=variance_method,
            )

        # Self-normalized DM statistic
        dm_stat = d_bar / np.sqrt(var_d)

        # Use non-standard distribution for p-value [T1]
        # Note: Harvey correction not applicable for self-normalized
        pvalue = _sn_pvalue(dm_stat, alternative)

        return DMTestResult(
            statistic=float(dm_stat),
            pvalue=float(pvalue),
            h=h,
            n=n,
            loss=loss,
            alternative=alternative,
            harvey_adjusted=False,  # Not applicable for self-normalized
            mean_loss_diff=d_bar,
            variance_method=variance_method,
        )

    # HAC variance method (default)
    # HAC variance with h-1 bandwidth for h-step forecasts
    # For h=1, bandwidth=0 (no autocorrelation in 1-step errors)
    bandwidth = max(0, h - 1)

    # Warn if bandwidth is large relative to sample size
    # Per Coroneo & Iacone (2016), large bandwidth can cause negative variance estimates
    if bandwidth > n / 4:
        warnings.warn(
            f"DM test bandwidth ({bandwidth}) exceeds n/4 ({n/4:.0f}). "
            f"HAC variance estimation may be unreliable with long forecast horizons "
            f"relative to sample size. Consider: (1) increasing sample size, "
            f"(2) using variance_method='self_normalized', (3) reducing forecast horizon. "
            f"See Coroneo & Iacone (2016) for details on DM test limitations.",
            UserWarning,
            stacklevel=2,
        )

    var_d = compute_hac_variance(d, bandwidth=bandwidth)

    # Handle degenerate case - warn instead of failing silently
    if var_d <= 0:
        warnings.warn(
            f"DM test variance is non-positive (var_d={var_d:.2e}). "
            "This can occur when loss differences are constant or nearly constant. "
            "Returning pvalue=1.0 (cannot reject null). "
            "Consider: (1) checking for identical predictions, "
            "(2) using variance_method='self_normalized' which cannot be negative.",
            UserWarning,
            stacklevel=2,
        )
        return DMTestResult(
            statistic=float("nan"),
            pvalue=1.0,
            h=h,
            n=n,
            loss=loss,
            alternative=alternative,
            harvey_adjusted=harvey_correction,
            mean_loss_diff=d_bar,
            variance_method=variance_method,
        )

    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d)

    # Harvey et al. (1997) small-sample adjustment
    if harvey_correction:
        adjustment = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        dm_stat = dm_stat * adjustment

    # Compute p-value
    # When harvey_correction=True, use t-distribution (Harvey et al. 1997)
    # Otherwise use normal distribution (Diebold & Mariano 1995)
    if harvey_correction:
        # t-distribution with df = n - 1 for small-sample inference
        if alternative == "two-sided":
            pvalue = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
        elif alternative == "less":
            pvalue = stats.t.cdf(dm_stat, df=n - 1)
        else:  # greater
            pvalue = 1 - stats.t.cdf(dm_stat, df=n - 1)
    else:
        # Normal distribution for large-sample asymptotic inference
        if alternative == "two-sided":
            pvalue = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        elif alternative == "less":
            # H1: Model 1 better (lower loss) => d_bar < 0 => dm_stat < 0
            pvalue = stats.norm.cdf(dm_stat)
        else:  # greater
            # H1: Model 2 better => d_bar > 0 => dm_stat > 0
            pvalue = 1 - stats.norm.cdf(dm_stat)

    return DMTestResult(
        statistic=float(dm_stat),
        pvalue=float(pvalue),
        h=h,
        n=n,
        loss=loss,
        alternative=alternative,
        harvey_adjusted=harvey_correction,
        mean_loss_diff=d_bar,
        variance_method=variance_method,
    )


# =============================================================================
# Giacomini-White Test
# =============================================================================


def gw_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    n_lags: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> GWTestResult:
    """
    Giacomini-White test for conditional predictive ability.

    Tests H0: E[d_t | X_{t-1}] = 0 where d_t = L(e1_t) - L(e2_t) is the loss
    differential and X_{t-1} = [1, d_{t-1}, ..., d_{t-τ}] is the conditioning set.

    Unlike the Diebold-Mariano test which tests *unconditional* equal predictive
    accuracy (average performance), the GW test examines whether past loss
    differentials can predict future forecast superiority. Rejection indicates
    that forecasters could improve by switching between models based on recent
    relative performance.

    Knowledge Tier: [T1] - Established methodology (Giacomini & White 2006).

    Parameters
    ----------
    errors_1 : array-like
        Forecast errors from model 1 (actual - prediction)
    errors_2 : array-like
        Forecast errors from model 2 (baseline)
    n_lags : int, default=1
        Number of lags to include in conditioning set (1 <= n_lags <= 10).
        The instrument matrix is [1, d_{t-1}, ..., d_{t-n_lags}].
        Default τ=1 is the canonical choice from Giacomini & White (2006).
    loss : {"squared", "absolute"}, default="squared"
        Loss function for comparing forecasts
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Alternative hypothesis:

        - "two-sided": Conditional predictive ability differs
        - "less": Model 1 conditionally better (lower loss given past)
        - "greater": Model 2 conditionally better

    Returns
    -------
    GWTestResult
        Test results including statistic, p-value, R², and diagnostics

    Raises
    ------
    ValueError
        If inputs are invalid (different lengths, too few observations,
        invalid n_lags)

    Notes
    -----
    **Algorithm** (Giacomini & White 2006, Theorem 1):

    1. Compute loss differential: d_t = L(e1_t) - L(e2_t)
    2. Construct instrument matrix: X = [1, d_{t-1}, ..., d_{t-τ}]
    3. Demean loss differential: Z_t = d_t - d̄
    4. Regress 1 on (Z × X) via OLS
    5. Compute GW = T × R²
    6. P-value from χ²(q) where q = 1 + τ

    **Interpretation**:

    - R² measures predictability of the loss differential
    - High R² means forecasters can anticipate which model will perform better
    - Rejection of H0 implies conditional predictive ability differs

    **Relationship to DM Test**:

    The DM test is a special case testing unconditional mean: E[d_t] = 0.
    GW extends this to test: E[d_t | X_{t-1}] = 0.

    | DM Result | GW Result | Interpretation |
    |-----------|-----------|----------------|
    | Not sig   | Not sig   | No difference in predictive ability |
    | Sig       | Sig       | Model unconditionally and conditionally better |
    | Sig       | Not sig   | Better on average, but not predictably |
    | Not sig   | **Sig**   | **Equal average, but performance is predictable!** |

    **Minimum Sample Size**:

    Requires n >= 50 effective observations (after lag adjustment).
    The test involves regression with q = 1 + n_lags instruments.

    References
    ----------
    [T1] Giacomini, R. & White, H. (2006). Tests of Conditional Predictive
         Ability. Econometrica, 74(6), 1545-1578.
         DOI: 10.1111/j.1468-0262.2006.00718.x

    Example
    -------
    >>> # Test if past performance predicts future superiority
    >>> result = gw_test(model_errors, baseline_errors, n_lags=1)
    >>> if result.conditional_predictability:
    ...     print("Past loss differential predicts future performance")
    ...     print(f"R-squared: {result.r_squared:.3f}")

    >>> # Compare DM (unconditional) vs GW (conditional)
    >>> dm_result = dm_test(model_errors, baseline_errors)
    >>> gw_result = gw_test(model_errors, baseline_errors)
    >>> if not dm_result.significant_at_05 and gw_result.significant_at_05:
    ...     print("Equal average accuracy, but performance is predictable!")

    See Also
    --------
    dm_test : Unconditional predictive ability test.
    """
    # Convert to numpy arrays
    errors_1 = np.asarray(errors_1, dtype=np.float64)
    errors_2 = np.asarray(errors_2, dtype=np.float64)

    # Validate no NaN values
    if np.any(np.isnan(errors_1)):
        raise ValueError(
            "errors_1 contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )
    if np.any(np.isnan(errors_2)):
        raise ValueError(
            "errors_2 contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )

    # Validate lengths match
    if len(errors_1) != len(errors_2):
        raise ValueError(
            f"Error arrays must have same length. "
            f"Got {len(errors_1)} and {len(errors_2)}"
        )

    n_original = len(errors_1)

    # Validate n_lags
    if n_lags < 1:
        raise ValueError(f"n_lags must be >= 1, got {n_lags}")
    if n_lags > 10:
        raise ValueError(
            f"n_lags must be <= 10 to avoid overfitting, got {n_lags}. "
            "For more lags, use custom regression."
        )
    if n_lags >= n_original // 10:
        raise ValueError(
            f"n_lags={n_lags} too large for n={n_original}. "
            f"Need n >= 10 * n_lags to avoid overfitting."
        )

    # Effective sample size after lag adjustment
    n_effective = n_original - n_lags

    # Minimum sample size requirement
    if n_effective < 50:
        raise ValueError(
            f"Insufficient samples for reliable GW test after lag adjustment. "
            f"Need >= 50 effective observations, got {n_effective}. "
            f"Original n={n_original}, n_lags={n_lags}. "
            f"Consider: (1) increasing sample size, (2) reducing n_lags."
        )

    # Compute loss differential
    if loss == "squared":
        loss_1 = errors_1**2
        loss_2 = errors_2**2
    elif loss == "absolute":
        loss_1 = np.abs(errors_1)
        loss_2 = np.abs(errors_2)
    else:
        raise ValueError(f"Unknown loss function: {loss}. Use 'squared' or 'absolute'.")

    d = loss_1 - loss_2  # Positive = model 1 has higher loss (worse)
    d_bar = float(np.mean(d))

    # Degrees of freedom
    q = 1 + n_lags

    # Construct instrument matrix: [1, d_{t-1}, ..., d_{t-n_lags}]
    # Shape: (n_effective, q)
    X = np.ones((n_effective, q))
    for lag in range(1, n_lags + 1):
        X[:, lag] = d[n_lags - lag : n_original - lag]

    # Current period loss differential (aligned with instruments)
    d_current = d[n_lags:]

    # Demean loss differential
    d_mean = np.mean(d_current)
    d_demeaned = d_current - d_mean

    # Element-wise multiplication: Z_t * X_t for each observation
    # This creates the regressors for the auxiliary regression
    Z = d_demeaned[:, np.newaxis] * X  # Shape: (n_effective, q)

    # Auxiliary regression: regress 1 on Z
    # Per Giacomini & White (2006), test statistic is T × R² from this regression
    ones = np.ones(n_effective)

    # OLS using least squares (numerically stable)
    # We want to regress ones on Z, computing R²
    try:
        beta, residuals, rank, singular_values = np.linalg.lstsq(Z, ones, rcond=None)
    except np.linalg.LinAlgError:
        # Singular matrix - instruments are collinear
        warnings.warn(
            "GW test failed due to singular instrument matrix. "
            "Loss differentials may be constant or collinear. "
            "Returning pvalue=1.0.",
            UserWarning,
            stacklevel=2,
        )
        return GWTestResult(
            statistic=float("nan"),
            pvalue=1.0,
            r_squared=0.0,
            n=n_effective,
            n_lags=n_lags,
            q=q,
            loss=loss,
            alternative=alternative,
            mean_loss_diff=d_bar,
        )

    # Compute fitted values and residuals
    fitted = Z @ beta
    resid = ones - fitted

    # R-squared: 1 - SS_res / SS_tot
    # For regression on 1s, SS_tot = n (since mean(1) = 1, var(1) = 0 is degenerate)
    # Use proper centered formula: SS_tot = sum((y - y_bar)²)
    # Here y = ones, y_bar = 1, so SS_tot would be 0
    # Instead, use: R² = 1 - SS_res / n (uncentered R²)
    ss_res = np.sum(resid**2)
    r_squared = 1.0 - ss_res / n_effective

    # Ensure R² is in valid range [0, 1]
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    # GW test statistic: T × R²
    gw_stat = n_effective * r_squared

    # Compute p-value from chi-squared distribution
    if alternative == "two-sided":
        # Two-sided test: use chi-squared directly
        pvalue = 1 - stats.chi2.cdf(gw_stat, df=q)
    else:
        # One-sided tests: adjust based on direction
        base_pvalue = 1 - stats.chi2.cdf(gw_stat, df=q)

        if alternative == "less":
            # H1: Model 1 conditionally better (lower loss given past)
            # Need mean_loss_diff < 0 for this direction to make sense
            if d_bar >= 0:
                # Wrong direction for alternative
                pvalue = 1.0
            else:
                pvalue = base_pvalue / 2
        else:  # greater
            # H1: Model 2 conditionally better
            # Need mean_loss_diff > 0
            if d_bar <= 0:
                pvalue = 1.0
            else:
                pvalue = base_pvalue / 2

    pvalue = float(np.clip(pvalue, 0.0, 1.0))

    return GWTestResult(
        statistic=float(gw_stat),
        pvalue=pvalue,
        r_squared=r_squared,
        n=n_effective,
        n_lags=n_lags,
        q=q,
        loss=loss,
        alternative=alternative,
        mean_loss_diff=d_bar,
    )


# =============================================================================
# Clark-West Test
# =============================================================================


def cw_test(
    errors_unrestricted: np.ndarray,
    errors_restricted: np.ndarray,
    predictions_unrestricted: np.ndarray,
    predictions_restricted: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    harvey_correction: bool = True,
    variance_method: Literal["hac", "self_normalized"] = "hac",
) -> CWTestResult:
    """
    Clark-West test for nested model comparison.

    Tests H0: E[d*] = 0 where d* = d_t - (ŷ_r - ŷ_u)² is the adjusted
    loss differential, removing the bias from estimating parameters
    with true value zero in the unrestricted model.

    The standard DM test is biased when comparing nested models because
    estimating extra parameters (that have true value zero) adds noise
    to the forecasts, making the unrestricted model appear worse. The
    CW adjustment corrects for this bias.

    Parameters
    ----------
    errors_unrestricted : array-like
        Forecast errors from unrestricted model (more parameters).
        Convention: error = actual - prediction.
    errors_restricted : array-like
        Forecast errors from restricted model (nested, fewer parameters).
    predictions_unrestricted : array-like
        Point forecasts from unrestricted model.
    predictions_restricted : array-like
        Point forecasts from restricted model.
    h : int, default=1
        Forecast horizon. Used for HAC bandwidth (h-1) and Harvey adjustment.
    loss : {"squared", "absolute"}, default="squared"
        Loss function for computing the loss differential d_t.
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Alternative hypothesis:
        - "two-sided": Models have different accuracy
        - "less": Unrestricted model is more accurate (lower loss)
        - "greater": Restricted model is more accurate
    harvey_correction : bool, default=True
        Apply Harvey et al. (1997) small-sample adjustment.
    variance_method : {"hac", "self_normalized"}, default="hac"
        Variance estimation method.

    Returns
    -------
    CWTestResult
        Test results including adjusted loss differential.

    Raises
    ------
    ValueError
        If array lengths don't match, contain NaN, or n < 30.

    Notes
    -----
    **When to use CW vs DM**:

    - Use DM test for non-nested models (e.g., ARIMA vs Random Forest)
    - Use CW test for nested models (e.g., AR(2) vs AR(1), Full vs Reduced)

    **The adjustment formula** [T1]:

        d*_t = d_t - (ŷ_r,t - ŷ_u,t)²

    where:
    - d_t = L(e_u,t) - L(e_r,t) is the unadjusted loss differential
    - (ŷ_r - ŷ_u)² is the noise cost of estimating extra parameters

    The adjustment term is always squared regardless of the loss function
    used for the loss differential.

    **Interpretation**:

    - If CW statistic > 0: Unrestricted model has higher adjusted loss
    - If CW statistic < 0: Unrestricted model has lower adjusted loss
    - The adjustment_magnitude shows how much noise was removed

    **Comparison with DM test**:

    When predictions are identical (adjustment = 0), CW test equals DM test.
    When there's genuine improvement from extra parameters, CW test will
    show significance even when DM test is biased toward non-significance.

    References
    ----------
    [T1] Clark, T.E. & West, K.D. (2007). Approximately normal tests for
         equal predictive accuracy in nested models. Journal of Econometrics,
         138(1), 291-311.

    Example
    -------
    >>> # Compare AR(2) vs AR(1) - nested models
    >>> result = cw_test(
    ...     errors_unrestricted=ar2_errors,
    ...     errors_restricted=ar1_errors,
    ...     predictions_unrestricted=ar2_preds,
    ...     predictions_restricted=ar1_preds,
    ...     alternative="less",  # Test if AR(2) is better
    ... )
    >>> print(f"CW statistic: {result.statistic:.3f}")
    >>> print(f"P-value: {result.pvalue:.4f}")
    >>> print(f"Adjustment magnitude: {result.adjustment_magnitude:.4f}")

    >>> # Compare with DM test to see the bias
    >>> dm_result = dm_test(ar2_errors, ar1_errors, alternative="less")
    >>> print(f"DM p-value: {dm_result.pvalue:.4f} (may be biased)")
    >>> print(f"CW p-value: {result.pvalue:.4f} (corrected)")

    See Also
    --------
    dm_test : Unconditional predictive ability test (for non-nested models).
    gw_test : Conditional predictive ability test.
    """
    # Convert to numpy arrays
    errors_unrestricted = np.asarray(errors_unrestricted, dtype=np.float64)
    errors_restricted = np.asarray(errors_restricted, dtype=np.float64)
    predictions_unrestricted = np.asarray(predictions_unrestricted, dtype=np.float64)
    predictions_restricted = np.asarray(predictions_restricted, dtype=np.float64)

    # Validate no NaN values
    if np.any(np.isnan(errors_unrestricted)):
        raise ValueError(
            "errors_unrestricted contains NaN values. Clean data before processing."
        )
    if np.any(np.isnan(errors_restricted)):
        raise ValueError(
            "errors_restricted contains NaN values. Clean data before processing."
        )
    if np.any(np.isnan(predictions_unrestricted)):
        raise ValueError(
            "predictions_unrestricted contains NaN values. Clean data before processing."
        )
    if np.any(np.isnan(predictions_restricted)):
        raise ValueError(
            "predictions_restricted contains NaN values. Clean data before processing."
        )

    # Validate lengths match
    n = len(errors_unrestricted)
    if len(errors_restricted) != n:
        raise ValueError(
            f"Error arrays must have same length. "
            f"Got {n} and {len(errors_restricted)}"
        )
    if len(predictions_unrestricted) != n:
        raise ValueError(
            f"predictions_unrestricted length ({len(predictions_unrestricted)}) "
            f"must match errors length ({n})"
        )
    if len(predictions_restricted) != n:
        raise ValueError(
            f"predictions_restricted length ({len(predictions_restricted)}) "
            f"must match errors length ({n})"
        )

    # Validate horizon
    if h < 1:
        raise ValueError(f"Horizon h must be >= 1, got {h}")

    # Minimum sample size [T2]
    if n < 30:
        raise ValueError(
            f"Insufficient sample size for CW test. "
            f"Need n >= 30, got {n}. "
            f"Small samples may not satisfy asymptotic normality assumption."
        )

    # Compute loss differential (unadjusted)
    if loss == "squared":
        loss_unrestricted = errors_unrestricted**2
        loss_restricted = errors_restricted**2
    elif loss == "absolute":
        loss_unrestricted = np.abs(errors_unrestricted)
        loss_restricted = np.abs(errors_restricted)
    else:
        raise ValueError(f"Unknown loss function: {loss}. Use 'squared' or 'absolute'.")

    # Unadjusted loss differential
    # Positive = unrestricted model has higher loss (appears worse)
    d_unadjusted = loss_unrestricted - loss_restricted
    mean_d_unadjusted = float(np.mean(d_unadjusted))

    # Clark-West adjustment term: (ŷ_r - ŷ_u)²
    # This is always squared regardless of loss function
    pred_diff = predictions_restricted - predictions_unrestricted
    adjustment = pred_diff**2
    mean_adjustment = float(np.mean(adjustment))

    # Adjusted loss differential
    d_adjusted = d_unadjusted - adjustment
    mean_d_adjusted = float(np.mean(d_adjusted))

    # Compute variance of adjusted loss differential
    if variance_method == "hac":
        # HAC variance with bandwidth h-1 for MA(h-1) structure
        bandwidth = max(0, h - 1)
        variance = compute_hac_variance(d_adjusted, bandwidth=bandwidth)
    elif variance_method == "self_normalized":
        variance = compute_self_normalized_variance(d_adjusted)
    else:
        raise ValueError(
            f"Unknown variance method: {variance_method}. "
            f"Use 'hac' or 'self_normalized'."
        )

    # Handle zero variance
    if variance <= 0 or np.isnan(variance):
        warnings.warn(
            "CW test has zero or negative variance estimate. "
            "Loss differentials may be constant. Returning pvalue=1.0.",
            UserWarning,
            stacklevel=2,
        )
        return CWTestResult(
            statistic=float("nan"),
            pvalue=1.0,
            h=h,
            n=n,
            loss=loss,
            alternative=alternative,
            harvey_adjusted=harvey_correction,
            mean_loss_diff=mean_d_unadjusted,
            mean_loss_diff_adjusted=mean_d_adjusted,
            adjustment_magnitude=mean_adjustment,
            variance_method=variance_method,
        )

    # Compute test statistic
    se = np.sqrt(variance)
    cw_stat = mean_d_adjusted / se

    # Apply Harvey et al. (1997) small-sample correction
    if harvey_correction and h > 1:
        # Adjustment factor: sqrt((n + 1 - 2h + h(h-1)/n) / n)
        adjustment_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        cw_stat = cw_stat * adjustment_factor

    cw_stat = float(cw_stat)

    # Compute p-value from standard normal distribution
    # Under H0, CW statistic is asymptotically N(0,1)
    if variance_method == "hac":
        if alternative == "two-sided":
            pvalue = 2 * (1 - stats.norm.cdf(abs(cw_stat)))
        elif alternative == "less":
            # H1: Unrestricted model is better (lower loss) → mean_d_adjusted < 0
            pvalue = stats.norm.cdf(cw_stat)
        else:  # greater
            # H1: Restricted model is better → mean_d_adjusted > 0
            pvalue = 1 - stats.norm.cdf(cw_stat)
    else:  # self_normalized
        # Use self-normalized critical values
        pvalue = _sn_pvalue(cw_stat, alternative)

    pvalue = float(np.clip(pvalue, 0.0, 1.0))

    return CWTestResult(
        statistic=cw_stat,
        pvalue=pvalue,
        h=h,
        n=n,
        loss=loss,
        alternative=alternative,
        harvey_adjusted=harvey_correction and h > 1,
        mean_loss_diff=mean_d_unadjusted,
        mean_loss_diff_adjusted=mean_d_adjusted,
        adjustment_magnitude=mean_adjustment,
        variance_method=variance_method,
    )


# =============================================================================
# Pesaran-Timmermann Test
# =============================================================================


def pt_test(
    actual: np.ndarray,
    predicted: np.ndarray,
    move_threshold: Optional[float] = None,
) -> PTTestResult:
    """
    Pesaran-Timmermann test for directional accuracy.

    Tests whether the model's ability to predict direction (sign)
    is significantly better than random guessing.

    Parameters
    ----------
    actual : array-like
        Actual values (typically changes/returns)
    predicted : array-like
        Predicted values (typically changes/returns)
    move_threshold : float, optional
        If provided, uses 3-class classification (UP/DOWN/FLAT):
        - UP: value > threshold
        - DOWN: value < -threshold
        - FLAT: |value| <= threshold

        If None, uses 2-class (positive/negative sign).

        Using a threshold is recommended when comparing against persistence
        baseline (which predicts 0 = FLAT).

    Returns
    -------
    PTTestResult
        Test results including accuracy, expected, and significance

    Raises
    ------
    ValueError
        If inputs are invalid

    Notes
    -----
    H0: Direction predictions are no better than random (independence)
    H1: Direction predictions have skill (one-sided test)

    The test accounts for marginal probabilities of directions in both
    actual and predicted series, providing a proper baseline comparison.

    Warning
    -------
    For h > 1 step forecasts, forecast errors are autocorrelated (MA(h-1)).
    The current variance formula does NOT apply HAC correction, so p-values
    for h > 1 may be overly optimistic. For rigorous multi-step testing,
    consider the DM test which includes proper HAC adjustment.

    The 3-class mode (using move_threshold) employs an ad-hoc variance
    formula that has not been validated against published extensions of
    Pesaran-Timmermann (1992). Use 2-class mode for rigorous hypothesis
    testing. The 3-class mode is suitable for exploratory analysis only.

    Example
    -------
    >>> # Test with 3-class (UP/DOWN/FLAT)
    >>> result = pt_test(actual_changes, pred_changes, move_threshold=0.01)
    >>> print(f"Accuracy: {result.accuracy:.1%}, Skill: {result.skill:.1%}")

    See Also
    --------
    dm_test : Complementary test for predictive accuracy (magnitude).
    compute_direction_accuracy : Simpler direction accuracy metric.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # Validate no NaN values
    if np.any(np.isnan(actual)):
        raise ValueError(
            "actual contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )
    if np.any(np.isnan(predicted)):
        raise ValueError(
            "predicted contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            f"Arrays must have same length. "
            f"Got actual={len(actual)}, predicted={len(predicted)}"
        )

    n = len(actual)

    if n < 30:
        raise ValueError(f"Insufficient samples for PT test. Need >= 30, got {n}")

    # Classify directions
    if move_threshold is not None:
        # 3-class: UP=1, DOWN=-1, FLAT=0
        def classify(values: np.ndarray, threshold: float) -> np.ndarray:
            classes: np.ndarray = np.zeros(len(values), dtype=np.int8)
            classes[values > threshold] = 1  # UP
            classes[values < -threshold] = -1  # DOWN
            return classes

        actual_class = classify(actual, move_threshold)
        pred_class = classify(predicted, move_threshold)
        n_classes = 3
    else:
        # 2-class: sign comparison
        actual_class = np.sign(actual)
        pred_class = np.sign(predicted)
        n_classes = 2

    # Compute directional accuracy
    correct = actual_class == pred_class

    if move_threshold is not None:
        # 3-class: use all samples
        n_effective = n
        p_hat = float(np.mean(correct))

        # Marginal probabilities for each class
        p_y = {
            1: float(np.mean(actual_class == 1)),
            -1: float(np.mean(actual_class == -1)),
            0: float(np.mean(actual_class == 0)),
        }
        p_x = {
            1: float(np.mean(pred_class == 1)),
            -1: float(np.mean(pred_class == -1)),
            0: float(np.mean(pred_class == 0)),
        }

        # Expected accuracy under independence (null)
        p_star = p_y[1] * p_x[1] + p_y[-1] * p_x[-1] + p_y[0] * p_x[0]

        # Variance estimates (simplified for 3-class)
        # Note: The * 4 factor is a [T3] approximation for 3-class case
        var_p_hat = p_star * (1 - p_star) / n_effective
        var_p_star = p_star * (1 - p_star) / n_effective * 4

    else:
        # 2-class: exclude zeros (undefined direction)
        nonzero_mask = actual_class != 0
        n_effective = int(np.sum(nonzero_mask))

        if n_effective == 0:
            warnings.warn(
                "PT test has no non-zero observations for 2-class mode. "
                "All actual values may be zero. Returning pvalue=1.0. "
                "Consider using 3-class mode with move_threshold parameter.",
                UserWarning,
                stacklevel=2,
            )
            return PTTestResult(
                statistic=float("nan"),
                pvalue=1.0,
                accuracy=0.0,
                expected=0.5,
                n=n,
                n_classes=2,
            )

        p_hat = float(np.mean(correct[nonzero_mask]))

        # Marginal probabilities
        p_y_pos = float(np.mean(actual[nonzero_mask] > 0))
        p_x_pos = float(np.mean(predicted[nonzero_mask] > 0))

        # Expected accuracy under independence
        p_star = p_y_pos * p_x_pos + (1 - p_y_pos) * (1 - p_x_pos)

        # Variance estimates (2-class formula from PT 1992, equation 8) [T1]
        var_p_hat = p_star * (1 - p_star) / n_effective
        term1 = (2 * p_y_pos - 1) ** 2 * p_x_pos * (1 - p_x_pos) / n_effective
        term2 = (2 * p_x_pos - 1) ** 2 * p_y_pos * (1 - p_y_pos) / n_effective
        term3 = 4 * p_y_pos * p_x_pos * (1 - p_y_pos) * (1 - p_x_pos) / n_effective
        var_p_star = term1 + term2 + term3

    # Total variance under null
    var_total = var_p_hat + var_p_star

    if var_total <= 0:
        warnings.warn(
            f"PT test total variance is non-positive (var_total={var_total:.2e}). "
            "This can occur with degenerate probability estimates. "
            "Returning pvalue=1.0 (cannot reject null). "
            "Check that predictions have variance.",
            UserWarning,
            stacklevel=2,
        )
        return PTTestResult(
            statistic=float("nan"),
            pvalue=1.0,
            accuracy=p_hat,
            expected=p_star,
            n=n_effective,
            n_classes=n_classes,
        )

    # PT statistic (z-score)
    pt_stat = (p_hat - p_star) / np.sqrt(var_total)

    # One-sided p-value (testing if better than random)
    pvalue = 1 - stats.norm.cdf(pt_stat)

    return PTTestResult(
        statistic=float(pt_stat),
        pvalue=float(pvalue),
        accuracy=float(p_hat),
        expected=float(p_star),
        n=n_effective,
        n_classes=n_classes,
    )


# =============================================================================
# Multi-Model Comparison
# =============================================================================


from typing import Dict, List, Tuple


@dataclass
class MultiModelComparisonResult:
    """
    Result from multi-model comparison using pairwise DM tests.

    Attributes
    ----------
    pairwise_results : Dict[Tuple[str, str], DMTestResult]
        Mapping from (model_a, model_b) to DM test result.
        Tests are ordered so model_a vs model_b tests if A is better (lower loss).
    best_model : str
        Model with lowest mean loss.
    bonferroni_alpha : float
        Corrected significance level (alpha / n_comparisons).
    original_alpha : float
        Original significance level before Bonferroni correction.
    model_rankings : List[Tuple[str, float]]
        Models sorted by mean loss (ascending), with (name, mean_loss) pairs.
    significant_pairs : List[Tuple[str, str]]
        Pairs where model_a is significantly better than model_b at corrected alpha.

    Examples
    --------
    >>> result = compare_multiple_models({"A": errors_a, "B": errors_b, "C": errors_c})
    >>> print(f"Best model: {result.best_model}")
    >>> for pair in result.significant_pairs:
    ...     print(f"{pair[0]} significantly better than {pair[1]}")
    """

    pairwise_results: Dict[Tuple[str, str], DMTestResult]
    best_model: str
    bonferroni_alpha: float
    original_alpha: float
    model_rankings: List[Tuple[str, float]]
    significant_pairs: List[Tuple[str, str]]

    @property
    def n_comparisons(self) -> int:
        """Number of pairwise comparisons performed."""
        return len(self.pairwise_results)

    @property
    def n_significant(self) -> int:
        """Number of significant differences at Bonferroni-corrected level."""
        return len(self.significant_pairs)

    def summary(self) -> str:
        """Generate human-readable summary of comparison results."""
        lines = [
            f"Multi-Model Comparison ({len(self.model_rankings)} models, {self.n_comparisons} pairs)",
            f"Bonferroni-corrected α = {self.bonferroni_alpha:.4f} (original α = {self.original_alpha:.2f})",
            "",
            "Model Rankings (by mean loss):",
        ]

        for rank, (name, loss) in enumerate(self.model_rankings, 1):
            marker = " ← best" if name == self.best_model else ""
            lines.append(f"  {rank}. {name}: {loss:.6f}{marker}")

        lines.append("")

        if self.significant_pairs:
            lines.append(f"Significant differences ({self.n_significant}):")
            for model_a, model_b in self.significant_pairs:
                result = self.pairwise_results[(model_a, model_b)]
                lines.append(f"  {model_a} > {model_b}: p={result.pvalue:.4f}")
        else:
            lines.append("No significant differences at corrected α level.")

        return "\n".join(lines)

    def get_pairwise(self, model_a: str, model_b: str) -> Optional[DMTestResult]:
        """Get DM test result for specific pair (order-independent lookup)."""
        if (model_a, model_b) in self.pairwise_results:
            return self.pairwise_results[(model_a, model_b)]
        elif (model_b, model_a) in self.pairwise_results:
            return self.pairwise_results[(model_b, model_a)]
        return None


def compare_multiple_models(
    errors_dict: Dict[str, np.ndarray],
    h: int = 1,
    alpha: float = 0.05,
    loss: Literal["squared", "absolute"] = "squared",
    harvey_correction: bool = True,
) -> MultiModelComparisonResult:
    """
    Compare multiple models using pairwise DM tests with Bonferroni correction.

    Performs all pairwise comparisons and applies Bonferroni correction to
    control family-wise error rate.

    Parameters
    ----------
    errors_dict : Dict[str, np.ndarray]
        Mapping from model name to error array.
        All arrays must have the same length.
    h : int, default=1
        Forecast horizon for DM test HAC bandwidth.
    alpha : float, default=0.05
        Significance level (before Bonferroni correction).
    loss : {"squared", "absolute"}, default="squared"
        Loss function for DM test.
    harvey_correction : bool, default=True
        Apply Harvey et al. (1997) small-sample adjustment.

    Returns
    -------
    MultiModelComparisonResult
        Comprehensive comparison results including rankings and significant pairs.

    Raises
    ------
    ValueError
        If fewer than 2 models provided or arrays have mismatched lengths.

    Notes
    -----
    Bonferroni correction: α_corrected = α / n_comparisons where
    n_comparisons = n_models * (n_models - 1) / 2.

    For k models, there are k(k-1)/2 pairwise comparisons:
    - 2 models: 1 comparison
    - 3 models: 3 comparisons
    - 5 models: 10 comparisons
    - 10 models: 45 comparisons

    Alternative multiple testing corrections (e.g., Holm, FDR) could be
    more powerful but Bonferroni is most conservative and widely accepted.

    Examples
    --------
    >>> errors = {
    ...     "Ridge": model_ridge_errors,
    ...     "Lasso": model_lasso_errors,
    ...     "Persistence": baseline_errors,
    ... }
    >>> result = compare_multiple_models(errors, h=2)
    >>> print(result.summary())
    Multi-Model Comparison (3 models, 3 pairs)
    Bonferroni-corrected α = 0.0167 (original α = 0.05)

    Model Rankings (by mean loss):
      1. Ridge: 0.012345 ← best
      2. Lasso: 0.013456
      3. Persistence: 0.025678

    Significant differences (1):
      Ridge > Persistence: p=0.0034

    See Also
    --------
    dm_test : Pairwise comparison between two models.
    """
    model_names = list(errors_dict.keys())
    n_models = len(model_names)

    if n_models < 2:
        raise ValueError(
            f"Need at least 2 models to compare. Got {n_models}. "
            "Use dm_test() for single pairwise comparison."
        )

    # Validate all arrays have same length
    lengths = [len(errors_dict[name]) for name in model_names]
    if len(set(lengths)) > 1:
        length_info = ", ".join(f"{name}={lengths[i]}" for i, name in enumerate(model_names))
        raise ValueError(f"All error arrays must have same length. Got: {length_info}")

    # Compute mean loss for each model
    mean_losses: Dict[str, float] = {}
    for name, errors in errors_dict.items():
        if loss == "squared":
            mean_losses[name] = float(np.mean(errors**2))
        else:
            mean_losses[name] = float(np.mean(np.abs(errors)))

    # Rank models by mean loss (lower is better)
    model_rankings = sorted(mean_losses.items(), key=lambda x: x[1])
    best_model = model_rankings[0][0]

    # Compute number of pairwise comparisons
    n_comparisons = n_models * (n_models - 1) // 2
    bonferroni_alpha = alpha / n_comparisons

    # Run all pairwise DM tests
    pairwise_results: Dict[Tuple[str, str], DMTestResult] = {}
    significant_pairs: List[Tuple[str, str]] = []

    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1 :]:
            # Order so lower-loss model is first (tests if A is better)
            if mean_losses[name_a] < mean_losses[name_b]:
                better, worse = name_a, name_b
            else:
                better, worse = name_b, name_a

            result = dm_test(
                errors_dict[better],
                errors_dict[worse],
                h=h,
                loss=loss,
                alternative="less",  # Test if better model has lower loss
                harvey_correction=harvey_correction,
            )

            pairwise_results[(better, worse)] = result

            if result.pvalue < bonferroni_alpha:
                significant_pairs.append((better, worse))

    return MultiModelComparisonResult(
        pairwise_results=pairwise_results,
        best_model=best_model,
        bonferroni_alpha=bonferroni_alpha,
        original_alpha=alpha,
        model_rankings=model_rankings,
        significant_pairs=significant_pairs,
    )


# =============================================================================
# Multi-Horizon Comparison
# =============================================================================


from typing import Sequence


@dataclass(frozen=True)
class MultiHorizonResult:
    """
    Result from comparing two models across multiple forecast horizons.

    This dataclass aggregates DM test results for each horizon and provides
    analysis of how predictive advantage changes with forecast horizon.

    Knowledge Tier: [T1] - Extends Diebold-Mariano (1995) to horizon analysis.

    Attributes
    ----------
    horizons : Tuple[int, ...]
        Forecast horizons tested (e.g., (1, 2, 4, 8))
    dm_results : Dict[int, DMTestResult]
        Mapping from horizon to DM test result
    model_1_name : str
        Name of model 1 (the candidate model)
    model_2_name : str
        Name of model 2 (the baseline model)
    n_per_horizon : Dict[int, int]
        Sample size per horizon
    loss : str
        Loss function used ("squared" or "absolute")
    alternative : str
        Alternative hypothesis ("two-sided", "less", "greater")
    alpha : float
        Significance level used

    Properties
    ----------
    significant_horizons : List[int]
        Horizons where p < alpha
    first_insignificant_horizon : Optional[int]
        First horizon where significance is lost (predictability horizon)
    best_horizon : int
        Horizon with smallest p-value
    degradation_pattern : str
        "consistent" | "degrading" | "none" | "irregular"

    Examples
    --------
    >>> result = compare_horizons(model_errors, baseline_errors, horizons=(1, 2, 4, 8))
    >>> print(result.significant_horizons)      # [1, 2, 4]
    >>> print(result.first_insignificant_horizon)  # 8
    >>> print(result.degradation_pattern)       # "degrading"

    See Also
    --------
    compare_horizons : Function that produces this result.
    dm_test : Underlying test for each horizon.
    """

    horizons: Tuple[int, ...]
    dm_results: Dict[int, DMTestResult]
    model_1_name: str
    model_2_name: str
    n_per_horizon: Dict[int, int]
    loss: str
    alternative: str
    alpha: float

    @property
    def significant_horizons(self) -> List[int]:
        """List of horizons where p-value < alpha (model 1 significantly better)."""
        return [h for h in self.horizons if self.dm_results[h].pvalue < self.alpha]

    @property
    def first_insignificant_horizon(self) -> Optional[int]:
        """
        First horizon where significance is lost.

        Returns the first horizon h (in sorted order) where p >= alpha,
        indicating the "predictability horizon" beyond which the model's
        advantage disappears.

        Returns None if all horizons are significant (consistent advantage).
        """
        for h in sorted(self.horizons):
            if self.dm_results[h].pvalue >= self.alpha:
                return h
        return None

    @property
    def best_horizon(self) -> int:
        """Horizon with the smallest p-value (strongest evidence of difference)."""
        return min(self.horizons, key=lambda h: self.dm_results[h].pvalue)

    @property
    def degradation_pattern(self) -> str:
        """
        Classify how significance changes with horizon.

        Returns
        -------
        str
            One of:
            - "consistent": All horizons significant or all insignificant
            - "degrading": P-values increase with horizon (advantage fades)
            - "none": No significant horizons
            - "irregular": Non-monotonic pattern

        Notes
        -----
        "degrading" pattern allows one violation (noise tolerance).
        """
        sig_horizons = self.significant_horizons

        if len(sig_horizons) == 0:
            return "none"

        if len(sig_horizons) == len(self.horizons):
            return "consistent"

        # Check if pattern is monotonically degrading
        # All significant horizons should be smaller than all insignificant horizons
        sorted_horizons = sorted(self.horizons)
        pvalues = [self.dm_results[h].pvalue for h in sorted_horizons]

        # Count violations: p-value should increase with horizon
        violations = 0
        for i in range(1, len(pvalues)):
            if pvalues[i] < pvalues[i - 1]:
                violations += 1

        # Allow 1 violation for noise tolerance
        if violations <= 1:
            return "degrading"
        else:
            return "irregular"

    def get_pvalues(self) -> Dict[int, float]:
        """Get p-values for each horizon."""
        return {h: self.dm_results[h].pvalue for h in self.horizons}

    def get_statistics(self) -> Dict[int, float]:
        """Get DM statistics for each horizon."""
        return {h: self.dm_results[h].statistic for h in self.horizons}

    def summary(self) -> str:
        """Generate human-readable summary of multi-horizon comparison."""
        lines = [
            f"Multi-Horizon Comparison: {self.model_1_name} vs {self.model_2_name}",
            f"Horizons: {self.horizons}",
            f"Loss: {self.loss}, Alternative: {self.alternative}, α: {self.alpha}",
            "",
        ]

        # Results table
        lines.append("Results by Horizon:")
        for h in sorted(self.horizons):
            r = self.dm_results[h]
            sig = "***" if r.pvalue < 0.01 else "**" if r.pvalue < 0.05 else "*" if r.pvalue < 0.10 else ""
            lines.append(f"  h={h}: DM={r.statistic:7.3f}, p={r.pvalue:.4f}{sig:3s} (n={r.n})")

        lines.append("")

        # Summary statistics
        sig_h = self.significant_horizons
        lines.append(f"Significant horizons: {sig_h if sig_h else 'none'}")
        lines.append(f"Best horizon: h={self.best_horizon}")

        first_insig = self.first_insignificant_horizon
        if first_insig is not None:
            lines.append(f"First insignificant: h={first_insig}")
        else:
            lines.append("First insignificant: none (all significant)")

        lines.append(f"Pattern: {self.degradation_pattern}")

        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Generate markdown table of results."""
        lines = [
            f"## Multi-Horizon Comparison: {self.model_1_name} vs {self.model_2_name}",
            "",
            f"**Settings**: loss={self.loss}, alternative={self.alternative}, α={self.alpha}",
            "",
            "| Horizon | DM Statistic | P-value | Significant | N |",
            "|---------|-------------|---------|-------------|---|",
        ]

        for h in sorted(self.horizons):
            r = self.dm_results[h]
            sig = "✓" if r.pvalue < self.alpha else ""
            lines.append(f"| {h} | {r.statistic:.3f} | {r.pvalue:.4f} | {sig} | {r.n} |")

        lines.append("")
        lines.append(f"**Pattern**: {self.degradation_pattern}")

        first_insig = self.first_insignificant_horizon
        if first_insig is not None:
            lines.append(f"**Predictability horizon**: h={first_insig}")

        return "\n".join(lines)


@dataclass(frozen=True)
class MultiModelHorizonResult:
    """
    Result from comparing multiple models across multiple horizons.

    This dataclass provides a matrix view of multi-model comparison
    at each forecast horizon.

    Attributes
    ----------
    horizons : Tuple[int, ...]
        Forecast horizons tested
    model_names : Tuple[str, ...]
        Names of models compared
    pairwise_by_horizon : Dict[int, MultiModelComparisonResult]
        Mapping from horizon to full multi-model comparison result
    alpha : float
        Significance level used

    Properties
    ----------
    best_model_by_horizon : Dict[int, str]
        Best model at each horizon
    consistent_best : Optional[str]
        Model that wins all horizons, or None if it varies

    Examples
    --------
    >>> result = compare_models_horizons(
    ...     {"ARIMA": arima_errors, "RF": rf_errors, "Naive": naive_errors},
    ...     horizons=(1, 4, 12),
    ... )
    >>> print(result.best_model_by_horizon)  # {1: 'RF', 4: 'ARIMA', 12: 'ARIMA'}
    >>> print(result.consistent_best)        # None (varies by horizon)

    See Also
    --------
    compare_models_horizons : Function that produces this result.
    compare_multiple_models : Underlying multi-model comparison.
    """

    horizons: Tuple[int, ...]
    model_names: Tuple[str, ...]
    pairwise_by_horizon: Dict[int, MultiModelComparisonResult]
    alpha: float

    @property
    def best_model_by_horizon(self) -> Dict[int, str]:
        """Best model at each horizon (by lowest mean loss)."""
        return {h: self.pairwise_by_horizon[h].best_model for h in self.horizons}

    @property
    def consistent_best(self) -> Optional[str]:
        """
        Model that is best at all horizons.

        Returns None if the best model varies by horizon.
        """
        best_models = list(self.best_model_by_horizon.values())
        if len(set(best_models)) == 1:
            return best_models[0]
        return None

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Multi-Model × Multi-Horizon Comparison",
            f"Models: {', '.join(self.model_names)}",
            f"Horizons: {self.horizons}",
            f"α: {self.alpha}",
            "",
            "Best Model by Horizon:",
        ]

        for h in sorted(self.horizons):
            lines.append(f"  h={h}: {self.pairwise_by_horizon[h].best_model}")

        if self.consistent_best:
            lines.append(f"\nConsistent winner: {self.consistent_best}")
        else:
            lines.append("\nNo consistent winner (varies by horizon)")

        return "\n".join(lines)


def compare_horizons(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    horizons: Sequence[int] = (1, 2, 3, 4),
    *,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    alpha: float = 0.05,
    harvey_correction: bool = True,
    variance_method: Literal["hac", "self_normalized"] = "hac",
    model_1_name: str = "Model",
    model_2_name: str = "Baseline",
) -> MultiHorizonResult:
    """
    Compare two models across multiple forecast horizons using DM tests.

    For each horizon h in `horizons`, runs a DM test with appropriate
    HAC bandwidth (h-1) to account for MA(h-1) error structure.

    Knowledge Tier: [T1] - Applies Diebold-Mariano (1995) methodology
    with horizon-specific HAC adjustment.

    Parameters
    ----------
    errors_1 : np.ndarray
        Forecast errors from model 1 (candidate model)
    errors_2 : np.ndarray
        Forecast errors from model 2 (baseline model)
    horizons : Sequence[int], default=(1, 2, 3, 4)
        Forecast horizons to test. Each h uses HAC bandwidth h-1.
    loss : {"squared", "absolute"}, default="squared"
        Loss function for DM test
    alternative : {"two-sided", "less", "greater"}, default="two-sided"
        Alternative hypothesis:
        - "two-sided": Models have different accuracy
        - "less": Model 1 more accurate (lower loss)
        - "greater": Model 2 more accurate
    alpha : float, default=0.05
        Significance level for determining significant horizons
    harvey_correction : bool, default=True
        Apply Harvey et al. (1997) small-sample adjustment
    variance_method : {"hac", "self_normalized"}, default="hac"
        Variance estimation method for DM test
    model_1_name : str, default="Model"
        Name for model 1 (used in output)
    model_2_name : str, default="Baseline"
        Name for model 2 (used in output)

    Returns
    -------
    MultiHorizonResult
        Comprehensive results including per-horizon DM tests and
        pattern analysis (degradation, predictability horizon)

    Raises
    ------
    ValueError
        If horizons empty, horizons contain non-positive values,
        or errors are insufficient for DM test

    Examples
    --------
    >>> # Compare model to persistence baseline across horizons
    >>> result = compare_horizons(
    ...     model_errors, baseline_errors,
    ...     horizons=(1, 2, 4, 8, 12),
    ...     alternative="less",  # Test if model is better
    ... )
    >>> print(result.significant_horizons)      # [1, 2, 4]
    >>> print(result.first_insignificant_horizon)  # 8
    >>> print(result.degradation_pattern)       # "degrading"
    >>> print(result.to_markdown())

    Notes
    -----
    **Key use case**: Identify the "predictability horizon" — the forecast
    horizon beyond which a model's advantage over the baseline disappears.

    **Interpretation**:
    - "degrading" pattern: Model advantage fades with horizon (common)
    - "consistent" pattern: Model always better or always comparable
    - "irregular" pattern: Non-monotonic, may indicate overfitting

    See Also
    --------
    dm_test : Underlying test for each horizon.
    compare_models_horizons : Multi-model version.
    """
    errors_1 = np.asarray(errors_1, dtype=np.float64)
    errors_2 = np.asarray(errors_2, dtype=np.float64)

    # Validate horizons
    horizons_tuple = tuple(horizons)
    if len(horizons_tuple) == 0:
        raise ValueError("horizons cannot be empty")

    if any(h < 1 for h in horizons_tuple):
        raise ValueError(f"All horizons must be >= 1. Got: {horizons_tuple}")

    # Run DM test for each horizon
    dm_results: Dict[int, DMTestResult] = {}
    n_per_horizon: Dict[int, int] = {}

    for h in horizons_tuple:
        result = dm_test(
            errors_1,
            errors_2,
            h=h,
            loss=loss,
            alternative=alternative,
            harvey_correction=harvey_correction,
            variance_method=variance_method,
        )
        dm_results[h] = result
        n_per_horizon[h] = result.n

    return MultiHorizonResult(
        horizons=horizons_tuple,
        dm_results=dm_results,
        model_1_name=model_1_name,
        model_2_name=model_2_name,
        n_per_horizon=n_per_horizon,
        loss=loss,
        alternative=alternative,
        alpha=alpha,
    )


def compare_models_horizons(
    errors_dict: Dict[str, np.ndarray],
    horizons: Sequence[int] = (1, 2, 3, 4),
    *,
    loss: Literal["squared", "absolute"] = "squared",
    alpha: float = 0.05,
    harvey_correction: bool = True,
) -> MultiModelHorizonResult:
    """
    Compare multiple models across multiple horizons.

    Combines multi-model comparison (Bonferroni-corrected pairwise DM tests)
    with multi-horizon analysis. For each horizon h, runs the full
    multi-model comparison.

    Knowledge Tier: [T1] - Applies Diebold-Mariano (1995) with
    Bonferroni correction at each horizon.

    Parameters
    ----------
    errors_dict : Dict[str, np.ndarray]
        Mapping from model name to error array.
        All arrays must have the same length.
    horizons : Sequence[int], default=(1, 2, 3, 4)
        Forecast horizons to test.
    loss : {"squared", "absolute"}, default="squared"
        Loss function for DM test.
    alpha : float, default=0.05
        Significance level (before Bonferroni correction within each horizon).
    harvey_correction : bool, default=True
        Apply Harvey et al. (1997) small-sample adjustment.

    Returns
    -------
    MultiModelHorizonResult
        Matrix of multi-model comparisons at each horizon.

    Raises
    ------
    ValueError
        If fewer than 2 models or arrays have mismatched lengths.

    Examples
    --------
    >>> matrix = compare_models_horizons(
    ...     {"ARIMA": arima_errors, "RF": rf_errors, "Naive": naive_errors},
    ...     horizons=(1, 4, 12),
    ... )
    >>> print(matrix.best_model_by_horizon)  # {1: 'RF', 4: 'ARIMA', 12: 'ARIMA'}
    >>> print(matrix.consistent_best)        # None (varies by horizon)

    Notes
    -----
    **Key insight**: The best model often varies by forecast horizon.
    Short-term forecasts may favor complex models (RF, LSTM) while
    long-term forecasts may favor simpler models (ARIMA, naive).

    See Also
    --------
    compare_multiple_models : Multi-model comparison at single horizon.
    compare_horizons : Two-model comparison across horizons.
    """
    # Validate inputs
    model_names = tuple(errors_dict.keys())
    if len(model_names) < 2:
        raise ValueError(
            f"Need at least 2 models to compare. Got {len(model_names)}. "
            "Use compare_horizons() for two-model comparison."
        )

    horizons_tuple = tuple(horizons)
    if len(horizons_tuple) == 0:
        raise ValueError("horizons cannot be empty")

    if any(h < 1 for h in horizons_tuple):
        raise ValueError(f"All horizons must be >= 1. Got: {horizons_tuple}")

    # Run multi-model comparison at each horizon
    pairwise_by_horizon: Dict[int, MultiModelComparisonResult] = {}

    for h in horizons_tuple:
        result = compare_multiple_models(
            errors_dict,
            h=h,
            alpha=alpha,
            loss=loss,
            harvey_correction=harvey_correction,
        )
        pairwise_by_horizon[h] = result

    return MultiModelHorizonResult(
        horizons=horizons_tuple,
        model_names=model_names,
        pairwise_by_horizon=pairwise_by_horizon,
        alpha=alpha,
    )


# =============================================================================
# Forecast Encompassing Test (Harvey, Leybourne, Newbold 1998)
# =============================================================================


@dataclass
class EncompassingTestResult:
    """
    Result from forecast encompassing test.

    Tests whether forecast A encompasses forecast B (i.e., B contains no
    additional information beyond A).

    Attributes
    ----------
    lambda_coef : float
        Regression coefficient λ. If λ ≠ 0, B contains information not in A.
    statistic : float
        t-statistic for H₀: λ = 0
    pvalue : float
        P-value for the test
    encompasses : bool
        True if A encompasses B (fail to reject H₀ at α=0.05)
    optimal_weight_b : float
        Optimal weight for forecast B in combined forecast (equals λ)
    direction : str
        "a_encompasses_b" or "b_encompasses_a"
    n : int
        Number of observations
    h : int
        Forecast horizon used for HAC variance

    Notes
    -----
    [T1] Harvey, D.I., Leybourne, S.J., & Newbold, P. (1998).
    Tests for Forecast Encompassing. JBES, 16(2), 254-259.
    """

    lambda_coef: float
    statistic: float
    pvalue: float
    encompasses: bool
    optimal_weight_b: float
    direction: str
    n: int
    h: int


@dataclass
class BidirectionalEncompassingResult:
    """
    Result from bidirectional forecast encompassing test.

    Tests both directions: does A encompass B AND does B encompass A?

    Attributes
    ----------
    a_encompasses_b : EncompassingTestResult
        Result of testing if A encompasses B
    b_encompasses_a : EncompassingTestResult
        Result of testing if B encompasses A
    recommendation : str
        One of: "use_a", "use_b", "combine", "equivalent"
    combined_weight_b : float or None
        Optimal weight for B if recommendation is "combine"
    """

    a_encompasses_b: EncompassingTestResult
    b_encompasses_a: EncompassingTestResult
    recommendation: str
    combined_weight_b: Optional[float]


def forecast_encompassing_test(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    h: int = 1,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "a-encompasses-b"] = "two-sided",
) -> EncompassingTestResult:
    """
    Harvey-Leybourne-Newbold forecast encompassing test.

    Tests whether forecast A encompasses forecast B by regressing A's errors
    on the forecast differential:

        e_A,t = α + λ(ŷ_B,t - ŷ_A,t) + ε_t

    If λ = 0, forecast A encompasses B (B adds no information).
    If λ ≠ 0, B contains information not in A → consider combining.

    Parameters
    ----------
    actual : np.ndarray
        Actual values y_t
    forecast_a : np.ndarray
        Forecasts from model A
    forecast_b : np.ndarray
        Forecasts from model B
    h : int, default=1
        Forecast horizon (for HAC variance correction)
    alpha : float, default=0.05
        Significance level for encompassing decision
    alternative : str, default="two-sided"
        - "two-sided": λ ≠ 0 (test if forecasts differ)
        - "a-encompasses-b": λ = 0 (null is A encompasses B)

    Returns
    -------
    EncompassingTestResult
        Test results including λ coefficient and recommendation

    Notes
    -----
    [T1] The test regresses e_A on (ŷ_B - ŷ_A). Under H₀ (A encompasses B),
    λ = 0 since the forecast differential adds no explanatory power.

    The optimal combined forecast is: (1-λ)ŷ_A + λŷ_B

    For h > 1, uses HAC standard errors (Newey-West with bandwidth h-1).

    Examples
    --------
    >>> result = forecast_encompassing_test(y_test, y_pred_arima, y_pred_rf)
    >>> if result.encompasses:
    ...     print("ARIMA encompasses Random Forest - use ARIMA only")
    >>> else:
    ...     print(f"Combine with weight {result.optimal_weight_b:.2f} on RF")

    References
    ----------
    [T1] Harvey, D.I., Leybourne, S.J., & Newbold, P. (1998).
         Tests for Forecast Encompassing. Journal of Business &
         Economic Statistics, 16(2), 254-259.
    """
    actual = np.asarray(actual)
    forecast_a = np.asarray(forecast_a)
    forecast_b = np.asarray(forecast_b)

    n = len(actual)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}")
    if len(forecast_a) != n or len(forecast_b) != n:
        raise ValueError("All arrays must have same length")

    # Compute errors and forecast differential
    error_a = actual - forecast_a
    diff = forecast_b - forecast_a  # ŷ_B - ŷ_A

    # OLS regression: e_A = α + λ(ŷ_B - ŷ_A) + ε
    # Using normal equations for simplicity
    X = np.column_stack([np.ones(n), diff])
    beta = np.linalg.lstsq(X, error_a, rcond=None)[0]
    lambda_coef = beta[1]

    # Residuals
    residuals = error_a - X @ beta

    # HAC variance for λ coefficient
    # Var(β̂) = (X'X)^(-1) X' Ω X (X'X)^(-1) where Ω is HAC
    XtX_inv = np.linalg.inv(X.T @ X)

    # Use HAC for h > 1, otherwise use standard OLS variance
    if h > 1:
        bandwidth = h - 1
        # Compute "meat" of sandwich estimator using HAC
        u_X = residuals[:, np.newaxis] * X  # n x 2 matrix of u_t * X_t
        # HAC on each column
        meat = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                # Compute covariance with HAC
                series_ij = u_X[:, i] * u_X[:, j] / n
                meat[i, j] = compute_hac_variance(
                    u_X[:, i] * u_X[:, j], bandwidth=bandwidth
                ) * n
        var_beta = XtX_inv @ meat @ XtX_inv
    else:
        # Standard OLS variance: σ² (X'X)^(-1)
        sigma2 = np.sum(residuals**2) / (n - 2)
        var_beta = sigma2 * XtX_inv

    se_lambda = np.sqrt(var_beta[1, 1])
    t_stat = lambda_coef / se_lambda

    # P-value
    if alternative == "two-sided":
        pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    else:  # a-encompasses-b (one-sided)
        pvalue = 1 - stats.t.cdf(t_stat, df=n - 2)

    encompasses = pvalue >= alpha

    return EncompassingTestResult(
        lambda_coef=float(lambda_coef),
        statistic=float(t_stat),
        pvalue=float(pvalue),
        encompasses=encompasses,
        optimal_weight_b=float(lambda_coef),
        direction="a_encompasses_b",
        n=n,
        h=h,
    )


def forecast_encompassing_bidirectional(
    actual: np.ndarray,
    forecast_a: np.ndarray,
    forecast_b: np.ndarray,
    h: int = 1,
    alpha: float = 0.05,
) -> BidirectionalEncompassingResult:
    """
    Bidirectional forecast encompassing test.

    Tests both directions to determine relationship between two forecasts:
    1. Does A encompass B? (B is redundant given A)
    2. Does B encompass A? (A is redundant given B)

    Parameters
    ----------
    actual : np.ndarray
        Actual values y_t
    forecast_a : np.ndarray
        Forecasts from model A
    forecast_b : np.ndarray
        Forecasts from model B
    h : int, default=1
        Forecast horizon
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    BidirectionalEncompassingResult
        Results for both directions with recommendation

    Notes
    -----
    Recommendation logic:
    - "use_a": A encompasses B but B doesn't encompass A
    - "use_b": B encompasses A but A doesn't encompass B
    - "combine": Neither encompasses the other → combine forecasts
    - "equivalent": Both encompass each other → nearly identical forecasts

    Examples
    --------
    >>> result = forecast_encompassing_bidirectional(y, y_arima, y_rf)
    >>> print(f"Recommendation: {result.recommendation}")
    >>> if result.recommendation == "combine":
    ...     combined = (1 - result.combined_weight_b) * y_arima + result.combined_weight_b * y_rf
    """
    # Test A encompasses B
    a_enc_b = forecast_encompassing_test(actual, forecast_a, forecast_b, h, alpha)

    # Test B encompasses A (swap forecasts)
    b_enc_a_raw = forecast_encompassing_test(actual, forecast_b, forecast_a, h, alpha)
    # Adjust direction label
    b_enc_a = EncompassingTestResult(
        lambda_coef=b_enc_a_raw.lambda_coef,
        statistic=b_enc_a_raw.statistic,
        pvalue=b_enc_a_raw.pvalue,
        encompasses=b_enc_a_raw.encompasses,
        optimal_weight_b=b_enc_a_raw.optimal_weight_b,
        direction="b_encompasses_a",
        n=b_enc_a_raw.n,
        h=b_enc_a_raw.h,
    )

    # Determine recommendation
    if a_enc_b.encompasses and not b_enc_a.encompasses:
        recommendation = "use_a"
        combined_weight = None
    elif b_enc_a.encompasses and not a_enc_b.encompasses:
        recommendation = "use_b"
        combined_weight = None
    elif a_enc_b.encompasses and b_enc_a.encompasses:
        recommendation = "equivalent"
        combined_weight = None
    else:  # Neither encompasses
        recommendation = "combine"
        # Optimal weight from A→B regression
        combined_weight = float(a_enc_b.optimal_weight_b)

    return BidirectionalEncompassingResult(
        a_encompasses_b=a_enc_b,
        b_encompasses_a=b_enc_a,
        recommendation=recommendation,
        combined_weight_b=combined_weight,
    )


# =============================================================================
# Reality Check and SPA Tests (White 2000, Hansen 2005)
# =============================================================================


@dataclass
class RealityCheckResult:
    """
    Result from White's Reality Check test for data snooping.

    Tests whether ANY model has superior predictive ability over benchmark,
    properly accounting for multiple testing.

    Attributes
    ----------
    statistic : float
        V_RC = max standardized loss differential across models
    pvalue : float
        Bootstrap p-value
    best_model : str
        Name of model with lowest mean loss
    individual_statistics : dict
        Per-model standardized loss differentials
    mean_losses : dict
        Mean loss for each model
    n_bootstrap : int
        Number of bootstrap replications
    block_size : int
        Block size used for stationary bootstrap
    n : int
        Number of observations

    Notes
    -----
    [T1] White, H. (2000). A Reality Check for Data Snooping.
    Econometrica, 68(5), 1097-1126.
    """

    statistic: float
    pvalue: float
    best_model: str
    individual_statistics: dict
    mean_losses: dict
    n_bootstrap: int
    block_size: int
    n: int

    @property
    def significant_models(self) -> List[str]:
        """Return models that beat the benchmark (positive test statistics)."""
        return [
            model
            for model, stat in self.individual_statistics.items()
            if stat > 0
        ]


@dataclass
class SPATestResult:
    """
    Result from Hansen's Superior Predictive Ability test.

    Improved version of Reality Check that is less sensitive to poor models.

    Attributes
    ----------
    statistic : float
        T_SPA = max studentized loss differential
    pvalue : float
        Main SPA p-value
    pvalue_consistent : float
        Conservative p-value (Reality Check-like)
    pvalue_lower : float
        Lower bound p-value
    best_model : str
        Name of model with lowest mean loss
    individual_statistics : dict
        Per-model studentized loss differentials
    mean_losses : dict
        Mean loss for each model
    n_bootstrap : int
        Number of bootstrap replications
    block_size : int
        Block size used
    n : int
        Number of observations

    Notes
    -----
    [T1] Hansen, P.R. (2005). A Test for Superior Predictive Ability.
    Journal of Business & Economic Statistics, 23(4), 365-380.
    """

    statistic: float
    pvalue: float
    pvalue_consistent: float
    pvalue_lower: float
    best_model: str
    individual_statistics: dict
    mean_losses: dict
    n_bootstrap: int
    block_size: int
    n: int


def _stationary_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    mean_block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate stationary bootstrap indices.

    Uses geometric distribution for random block lengths (Politis & Romano 1994).

    Parameters
    ----------
    n : int
        Length of series
    n_bootstrap : int
        Number of bootstrap samples
    mean_block_size : float
        Expected block length
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray, shape (n_bootstrap, n)
        Bootstrap indices for each sample
    """
    p = 1.0 / mean_block_size  # Probability of starting new block
    indices = np.zeros((n_bootstrap, n), dtype=np.int64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n)  # Random starting point
        for t in range(n):
            indices[b, t] = idx
            if rng.random() < p:
                # Start new block
                idx = rng.integers(0, n)
            else:
                # Continue block (with wrap-around)
                idx = (idx + 1) % n

    return indices


def reality_check_test(
    benchmark_errors: np.ndarray,
    model_errors_dict: dict,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> RealityCheckResult:
    """
    White's Reality Check for data snooping.

    Tests H₀: No model has superior predictive ability over benchmark.

    When comparing k models against a benchmark, conducting k pairwise tests
    inflates Type I error. The Reality Check properly controls family-wise
    error rate by using bootstrap to account for model correlation.

    Parameters
    ----------
    benchmark_errors : np.ndarray
        Forecast errors from benchmark model (y - ŷ_benchmark)
    model_errors_dict : dict
        Dictionary mapping model names to error arrays
    h : int, default=1
        Forecast horizon (for HAC variance)
    loss : str, default="squared"
        Loss function: "squared" or "absolute"
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    block_size : int, optional
        Block size for stationary bootstrap. Default: n^(1/3)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    RealityCheckResult
        Test statistic, p-value, and best model

    Notes
    -----
    [T1] The test statistic is:
        V_RC = max_{i=1,...,k} { √n × (L̄_benchmark - L̄_i) / σ̂_i }

    P-value is computed via stationary bootstrap (Politis & Romano 1994).

    A significant result (p < α) means at least one model beats the benchmark
    after accounting for data snooping bias.

    Examples
    --------
    >>> result = reality_check_test(
    ...     persistence_errors,
    ...     {"ARIMA": arima_errors, "RF": rf_errors, "LSTM": lstm_errors}
    ... )
    >>> if result.pvalue < 0.05:
    ...     print(f"At least one model beats persistence: {result.best_model}")

    References
    ----------
    [T1] White, H. (2000). A Reality Check for Data Snooping.
         Econometrica, 68(5), 1097-1126.
    [T1] Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap.
         JASA, 89(428), 1303-1313.
    """
    benchmark_errors = np.asarray(benchmark_errors)
    n = len(benchmark_errors)

    if n < 30:
        raise ValueError(f"Need at least 30 observations for RC test, got {n}")
    if not model_errors_dict:
        raise ValueError("Need at least one model in model_errors_dict")

    rng = np.random.default_rng(random_state)

    # Compute losses
    if loss == "squared":
        benchmark_loss = benchmark_errors**2
        model_losses = {
            name: np.asarray(errors) ** 2 for name, errors in model_errors_dict.items()
        }
    else:  # absolute
        benchmark_loss = np.abs(benchmark_errors)
        model_losses = {
            name: np.abs(np.asarray(errors))
            for name, errors in model_errors_dict.items()
        }

    # Compute loss differentials: d_i = L_benchmark - L_i (positive = model i better)
    loss_diffs = {}
    for name, m_loss in model_losses.items():
        if len(m_loss) != n:
            raise ValueError(f"Model {name} has {len(m_loss)} errors, expected {n}")
        loss_diffs[name] = benchmark_loss - m_loss

    # Compute observed statistics
    mean_losses = {"benchmark": float(np.mean(benchmark_loss))}
    individual_stats = {}
    for name, diff in loss_diffs.items():
        mean_losses[name] = float(np.mean(model_losses[name]))
        mean_diff = np.mean(diff)
        # HAC standard error
        var_diff = compute_hac_variance(diff, bandwidth=h - 1 if h > 1 else None)
        se_diff = np.sqrt(var_diff) if var_diff > 0 else 1e-10
        individual_stats[name] = float(np.sqrt(n) * mean_diff / se_diff)

    # Observed test statistic: max across models
    v_rc = max(individual_stats.values())
    best_model = max(individual_stats, key=lambda k: individual_stats[k])

    # Block size for stationary bootstrap
    if block_size is None:
        block_size = max(1, int(np.floor(n ** (1 / 3))))

    # Bootstrap p-value
    bootstrap_indices = _stationary_bootstrap_indices(
        n, n_bootstrap, float(block_size), rng
    )

    bootstrap_stats = []
    for b in range(n_bootstrap):
        idx = bootstrap_indices[b]
        # Resample loss differentials
        max_stat = -np.inf
        for name, diff in loss_diffs.items():
            diff_boot = diff[idx]
            # Center at zero under H0
            diff_centered = diff_boot - np.mean(diff)
            mean_boot = np.mean(diff_centered)
            var_boot = compute_hac_variance(
                diff_centered, bandwidth=h - 1 if h > 1 else None
            )
            se_boot = np.sqrt(var_boot) if var_boot > 0 else 1e-10
            stat_boot = np.sqrt(n) * mean_boot / se_boot
            max_stat = max(max_stat, stat_boot)
        bootstrap_stats.append(max_stat)

    bootstrap_stats = np.array(bootstrap_stats)
    pvalue = float(np.mean(bootstrap_stats >= v_rc))

    return RealityCheckResult(
        statistic=v_rc,
        pvalue=pvalue,
        best_model=best_model,
        individual_statistics=individual_stats,
        mean_losses=mean_losses,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        n=n,
    )


def spa_test(
    benchmark_errors: np.ndarray,
    model_errors_dict: dict,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> SPATestResult:
    """
    Hansen's Superior Predictive Ability (SPA) test.

    An improved version of White's Reality Check that:
    1. Uses studentization (divides by model-specific standard errors)
    2. Recenters null distribution to remove influence of poor models

    Parameters
    ----------
    benchmark_errors : np.ndarray
        Forecast errors from benchmark model
    model_errors_dict : dict
        Dictionary mapping model names to error arrays
    h : int, default=1
        Forecast horizon
    loss : str, default="squared"
        Loss function: "squared" or "absolute"
    n_bootstrap : int, default=1000
        Number of bootstrap replications
    block_size : int, optional
        Block size for stationary bootstrap
    random_state : int, optional
        Random seed

    Returns
    -------
    SPATestResult
        Test statistic and three p-values (main, consistent, lower)

    Notes
    -----
    [T1] The SPA test is more powerful than Reality Check because:
    - Studentization accounts for different model variances
    - Recentering removes influence of clearly inferior models

    Three p-values allow sensitivity analysis:
    - pvalue: Main SPA p-value (recommended)
    - pvalue_consistent: Conservative, similar to Reality Check
    - pvalue_lower: Liberal lower bound

    Examples
    --------
    >>> result = spa_test(
    ...     persistence_errors,
    ...     {"ARIMA": arima_errors, "RF": rf_errors}
    ... )
    >>> print(f"SPA p-value: {result.pvalue:.4f}")
    >>> print(f"Best model: {result.best_model}")

    References
    ----------
    [T1] Hansen, P.R. (2005). A Test for Superior Predictive Ability.
         Journal of Business & Economic Statistics, 23(4), 365-380.
    """
    benchmark_errors = np.asarray(benchmark_errors)
    n = len(benchmark_errors)

    if n < 30:
        raise ValueError(f"Need at least 30 observations for SPA test, got {n}")
    if not model_errors_dict:
        raise ValueError("Need at least one model in model_errors_dict")

    rng = np.random.default_rng(random_state)

    # Compute losses
    if loss == "squared":
        benchmark_loss = benchmark_errors**2
        model_losses = {
            name: np.asarray(errors) ** 2 for name, errors in model_errors_dict.items()
        }
    else:
        benchmark_loss = np.abs(benchmark_errors)
        model_losses = {
            name: np.abs(np.asarray(errors))
            for name, errors in model_errors_dict.items()
        }

    # Loss differentials
    loss_diffs = {}
    for name, m_loss in model_losses.items():
        if len(m_loss) != n:
            raise ValueError(f"Model {name} has {len(m_loss)} errors, expected {n}")
        loss_diffs[name] = benchmark_loss - m_loss

    # Compute observed statistics with studentization
    mean_losses = {"benchmark": float(np.mean(benchmark_loss))}
    mean_diffs = {}
    se_diffs = {}
    individual_stats = {}

    for name, diff in loss_diffs.items():
        mean_losses[name] = float(np.mean(model_losses[name]))
        mean_diffs[name] = np.mean(diff)
        var_diff = compute_hac_variance(diff, bandwidth=h - 1 if h > 1 else None)
        se_diffs[name] = np.sqrt(var_diff) if var_diff > 0 else 1e-10
        individual_stats[name] = float(np.sqrt(n) * mean_diffs[name] / se_diffs[name])

    # Observed test statistic
    t_spa = max(individual_stats.values())
    best_model = max(individual_stats, key=lambda k: individual_stats[k])

    # Block size
    if block_size is None:
        block_size = max(1, int(np.floor(n ** (1 / 3))))

    # Bootstrap
    bootstrap_indices = _stationary_bootstrap_indices(
        n, n_bootstrap, float(block_size), rng
    )

    # For SPA: compute three different null distributions
    boot_stats_consistent = []  # Like RC
    boot_stats_spa = []  # Main SPA (recenter at max(0, d̄))
    boot_stats_lower = []  # Lower bound

    for b in range(n_bootstrap):
        idx = bootstrap_indices[b]

        # Resample and compute studentized statistics
        stats_b = {}
        for name, diff in loss_diffs.items():
            diff_boot = diff[idx]
            mean_boot = np.mean(diff_boot) - mean_diffs[name]  # Center at zero
            # Use original standard error for studentization
            stats_b[name] = np.sqrt(n) * mean_boot / se_diffs[name]

        # Consistent: max of raw (like RC)
        boot_stats_consistent.append(max(stats_b.values()))

        # SPA: recenter using max(0, d̄)
        spa_stats_b = {}
        for name in stats_b:
            # Recenter: subtract max(0, √n * d̄ / σ̂)
            recenter = max(0.0, np.sqrt(n) * mean_diffs[name] / se_diffs[name])
            spa_stats_b[name] = stats_b[name] + recenter
        boot_stats_spa.append(max(spa_stats_b.values()))

        # Lower: use min recentering
        min_d = min(mean_diffs.values())
        lower_stats_b = {}
        for name in stats_b:
            recenter = np.sqrt(n) * min_d / se_diffs[name]
            lower_stats_b[name] = stats_b[name] - recenter
        boot_stats_lower.append(max(lower_stats_b.values()))

    pvalue_consistent = float(np.mean(np.array(boot_stats_consistent) >= t_spa))
    pvalue = float(np.mean(np.array(boot_stats_spa) >= t_spa))
    pvalue_lower = float(np.mean(np.array(boot_stats_lower) >= t_spa))

    return SPATestResult(
        statistic=t_spa,
        pvalue=pvalue,
        pvalue_consistent=pvalue_consistent,
        pvalue_lower=pvalue_lower,
        best_model=best_model,
        individual_statistics=individual_stats,
        mean_losses=mean_losses,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        n=n,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Result classes
    "DMTestResult",
    "PTTestResult",
    "GWTestResult",
    "CWTestResult",
    "MultiModelComparisonResult",
    "MultiHorizonResult",
    "MultiModelHorizonResult",
    "EncompassingTestResult",
    "BidirectionalEncompassingResult",
    "RealityCheckResult",
    "SPATestResult",
    # Tests
    "dm_test",
    "pt_test",
    "gw_test",
    "cw_test",
    "compare_multiple_models",
    "compare_horizons",
    "compare_models_horizons",
    "forecast_encompassing_test",
    "forecast_encompassing_bidirectional",
    "reality_check_test",
    "spa_test",
    # Utilities
    "compute_hac_variance",
    "compute_self_normalized_variance",
]
