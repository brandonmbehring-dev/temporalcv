"""
Statistical Tests Module.

Implements statistical tests for forecast evaluation:

- **Diebold-Mariano test** (DM 1995): Compare predictive accuracy of two models
- **Pesaran-Timmermann test** (PT 1992): Test directional accuracy
- **HAC variance** (Newey-West 1987): Correct for serial correlation in h>1 forecasts

Knowledge Tiers
---------------
[T1] DM test core methodology (Diebold & Mariano 1995)
[T1] Harvey small-sample adjustment (Harvey et al. 1997)
[T1] HAC variance with Bartlett kernel (Newey & West 1987)
[T1] PT test 2-class formulas (Pesaran & Timmermann 1992)
[T1] Automatic bandwidth selection (Andrews 1991)
[T2] Minimum sample size n >= 30 for DM, n >= 20 for PT (standard practice)
[T3] PT 3-class mode is ad-hoc extension, not published (exploratory use only)

Example
-------
>>> from temporalcv.statistical_tests import dm_test, pt_test
>>>
>>> # Compare model to baseline
>>> result = dm_test(model_errors, baseline_errors, h=2)
>>> print(f"DM statistic: {result.statistic:.3f}, p-value: {result.pvalue:.4f}")
>>>
>>> # Test directional accuracy
>>> pt_result = pt_test(actual_changes, predicted_changes, move_threshold=0.01)
>>> print(f"Direction accuracy: {pt_result.accuracy:.2%}")

References
----------
[T1] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy.
     Journal of Business & Economic Statistics, 13(3), 253-263.
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
        DM test statistic (asymptotically N(0,1) under H0)
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
    """

    statistic: float
    pvalue: float
    h: int
    n: int
    loss: str
    alternative: str
    harvey_adjusted: bool
    mean_loss_diff: float

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
# Diebold-Mariano Test
# =============================================================================


def dm_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    harvey_correction: bool = True,
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
        Recommended for n < 100 or h > 1.

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

    Example
    -------
    >>> # Test if model beats persistence baseline
    >>> result = dm_test(model_errors, persistence_errors, h=2, alternative="less")
    >>> if result.significant_at_05:
    ...     print("Model significantly better than baseline")

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

    # HAC variance with h-1 bandwidth for h-step forecasts
    # For h=1, bandwidth=0 (no autocorrelation in 1-step errors)
    bandwidth = max(0, h - 1)
    var_d = compute_hac_variance(d, bandwidth=bandwidth)

    # Handle degenerate case - warn instead of failing silently
    if var_d <= 0:
        warnings.warn(
            f"DM test variance is non-positive (var_d={var_d:.2e}). "
            "This can occur when loss differences are constant or nearly constant. "
            "Returning pvalue=1.0 (cannot reject null). "
            "Consider: (1) checking for identical predictions, "
            "(2) using bootstrap-based tests for small samples.",
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
# Public API
# =============================================================================

__all__ = [
    # Result classes
    "DMTestResult",
    "PTTestResult",
    "MultiModelComparisonResult",
    # Tests
    "dm_test",
    "pt_test",
    "compare_multiple_models",
    # Utilities
    "compute_hac_variance",
]
