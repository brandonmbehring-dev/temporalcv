"""
Unified Guardrails Suite.

Convenience layer over validation gates for common temporal validation patterns.
Guardrails aggregate multiple gate checks into a single validation pass.

The key insight: multiple independent checks should all pass before trusting
a model's performance. This module provides:

1. **Individual guardrail functions** - Check specific conditions
2. **run_all_guardrails()** - Comprehensive validation in one call
3. **GuardrailResult** - Unified result format with actionable recommendations

Knowledge Tiers
---------------
[T1] Statistical foundations from gate implementations
[T2] Threshold values from empirical validation (myga-forecasting)
[T3] Convenience patterns for common workflows

Example
-------
>>> from temporalcv.guardrails import run_all_guardrails, GuardrailResult
>>>
>>> result = run_all_guardrails(
...     model_metric=0.15,
...     baseline_metric=0.20,
...     n_samples=100,
... )
>>> if not result.passed:
...     print(f"Guardrails failed: {result.errors}")

References
----------
See temporalcv.gates for underlying gate implementations and academic references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GuardrailResult:
    """
    Result from guardrail validation.

    Attributes
    ----------
    passed : bool
        True if all checks passed (no HALT conditions)
    warnings : List[str]
        Warning messages (non-blocking)
    errors : List[str]
        Error messages (blocking conditions)
    details : Dict[str, Any]
        Detailed results from each check
    skipped : List[str]
        Checks that were skipped due to insufficient data
    recommendations : List[str]
        Actionable recommendations based on results
    """

    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.passed

    def summary(self) -> str:
        """Return human-readable summary of guardrail results."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Guardrail Check: {status}"]

        if self.errors:
            lines.append(f"\nErrors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.skipped:
            lines.append(f"\nSkipped ({len(self.skipped)}):")
            for skip in self.skipped:
                lines.append(f"  - {skip}")

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


def check_suspicious_improvement(
    model_metric: float,
    baseline_metric: float,
    threshold: float = 0.20,
    metric_name: str = "metric",
) -> GuardrailResult:
    """
    Check if improvement over baseline is suspiciously large.

    An improvement > threshold (default 20%) triggers a HALT,
    as this often indicates data leakage.

    Parameters
    ----------
    model_metric : float
        Model's error metric (lower is better).
    baseline_metric : float
        Baseline error metric (e.g., persistence forecast).
    threshold : float, default=0.20
        Maximum acceptable improvement ratio.
    metric_name : str, default="metric"
        Name for error messages.

    Returns
    -------
    GuardrailResult
        passed=False if improvement > threshold

    Knowledge Tier: [T3] 20% threshold is empirical heuristic.

    Examples
    --------
    >>> # 10% improvement - acceptable
    >>> result = check_suspicious_improvement(model_metric=0.18, baseline_metric=0.20)
    >>> result.passed
    True

    >>> # 30% improvement - suspicious
    >>> result = check_suspicious_improvement(model_metric=0.14, baseline_metric=0.20)
    >>> result.passed
    False
    """
    if baseline_metric <= 0:
        return GuardrailResult(
            passed=True,
            skipped=["suspicious_improvement: baseline_metric <= 0"],
            details={"baseline_metric": baseline_metric},
        )

    improvement = (baseline_metric - model_metric) / baseline_metric

    details = {
        "model_metric": model_metric,
        "baseline_metric": baseline_metric,
        "improvement": improvement,
        "threshold": threshold,
    }

    if improvement > threshold:
        return GuardrailResult(
            passed=False,
            errors=[
                f"Model {metric_name} ({model_metric:.4f}) shows {improvement:.1%} "
                f"improvement over baseline ({baseline_metric:.4f}), "
                f"exceeding {threshold:.0%} threshold"
            ],
            details=details,
            recommendations=[
                "Verify feature engineering does not use future information",
                "Check train/test split respects temporal order",
                "Run gate_signal_verification() to check for signal/leakage",
            ],
        )

    # Warning if close to threshold (>15% for default 20% threshold)
    if improvement > threshold * 0.75:
        return GuardrailResult(
            passed=True,
            warnings=[
                f"Model {metric_name} ({model_metric:.4f}) shows {improvement:.1%} "
                f"improvement - approaching {threshold:.0%} threshold"
            ],
            details=details,
            recommendations=["Consider running additional validation gates"],
        )

    return GuardrailResult(passed=True, details=details)


def check_minimum_sample_size(
    n: int,
    min_n: int = 50,
    context: str = "evaluation",
) -> GuardrailResult:
    """
    Check if sample size meets minimum requirements.

    Parameters
    ----------
    n : int
        Actual sample size.
    min_n : int, default=50
        Minimum required sample size.
    context : str, default="evaluation"
        Context for error messages.

    Returns
    -------
    GuardrailResult
        passed=False if n < min_n

    Knowledge Tier: [T3] 50 is common heuristic for reliable statistics.

    Examples
    --------
    >>> result = check_minimum_sample_size(n=100, min_n=50)
    >>> result.passed
    True

    >>> result = check_minimum_sample_size(n=20, min_n=50)
    >>> result.passed
    False
    """
    details = {"n": n, "min_n": min_n, "context": context}

    if n < min_n:
        return GuardrailResult(
            passed=False,
            errors=[
                f"Insufficient sample size for {context}: "
                f"n={n} < minimum required {min_n}"
            ],
            details=details,
            recommendations=[
                f"Collect more data (need at least {min_n} samples)",
                "Consider using bootstrap confidence intervals for small samples",
            ],
        )

    # Warning if sample size is marginal (< 2x minimum)
    if n < min_n * 2:
        return GuardrailResult(
            passed=True,
            warnings=[
                f"Sample size ({n}) is marginal for {context}. "
                f"Results may have high variance."
            ],
            details=details,
        )

    return GuardrailResult(passed=True, details=details)


def check_stratified_sample_size(
    n_up: int,
    n_down: int,
    min_n: int = 10,
    context: str = "move-conditional metrics",
) -> GuardrailResult:
    """
    Check if stratified samples meet minimum requirements.

    For move-conditional or direction-stratified metrics, each stratum
    needs sufficient observations for reliable estimates.

    Parameters
    ----------
    n_up : int
        Number of observations in UP stratum.
    n_down : int
        Number of observations in DOWN stratum.
    min_n : int, default=10
        Minimum required per stratum.
    context : str, default="move-conditional metrics"
        Context for error messages.

    Returns
    -------
    GuardrailResult
        passed=False if either stratum < min_n

    Knowledge Tier: [T3] 10 is empirical minimum for stratified estimates.

    Examples
    --------
    >>> result = check_stratified_sample_size(n_up=30, n_down=25)
    >>> result.passed
    True

    >>> result = check_stratified_sample_size(n_up=5, n_down=25)
    >>> result.passed
    False
    """
    details = {
        "n_up": n_up,
        "n_down": n_down,
        "min_n": min_n,
        "context": context,
    }

    errors = []
    if n_up < min_n:
        errors.append(f"Insufficient UP samples: n_up={n_up} < minimum {min_n}")
    if n_down < min_n:
        errors.append(f"Insufficient DOWN samples: n_down={n_down} < minimum {min_n}")

    if errors:
        return GuardrailResult(
            passed=False,
            errors=errors,
            details=details,
            recommendations=[
                "Use longer evaluation period to gather more samples",
                "Report overall metrics instead of stratified",
            ],
        )

    # Warning if imbalanced (ratio > 3:1)
    ratio = max(n_up, n_down) / max(min(n_up, n_down), 1)
    if ratio > 3:
        return GuardrailResult(
            passed=True,
            warnings=[
                f"Imbalanced strata: UP={n_up}, DOWN={n_down} (ratio {ratio:.1f}:1). "
                f"Stratified metrics may be unreliable."
            ],
            details=details,
        )

    return GuardrailResult(passed=True, details=details)


def check_forecast_horizon_consistency(
    horizons: List[float],
    horizon_labels: Optional[List[str]] = None,
    max_ratio: float = 2.0,
) -> GuardrailResult:
    """
    Check if performance is consistent across forecast horizons.

    If h=1 performance is dramatically better than longer horizons,
    this can indicate temporal boundary violations.

    Parameters
    ----------
    horizons : List[float]
        Metric values for each horizon (e.g., [mae_h1, mae_h2, mae_h3]).
    horizon_labels : List[str], optional
        Labels for horizons. Defaults to ["h=1", "h=2", ...].
    max_ratio : float, default=2.0
        Maximum acceptable ratio between h=1 and other horizons.

    Returns
    -------
    GuardrailResult
        passed=False if h=1 >> other horizons

    Knowledge Tier: [T2] Horizon consistency from myga-forecasting validation.
    """
    if len(horizons) < 2:
        return GuardrailResult(
            passed=True,
            skipped=["horizon_consistency: need at least 2 horizons"],
        )

    if horizon_labels is None:
        horizon_labels = [f"h={i + 1}" for i in range(len(horizons))]

    h1_metric = horizons[0]
    other_metrics = horizons[1:]

    details = {
        "horizons": dict(zip(horizon_labels, horizons)),
        "max_ratio": max_ratio,
    }

    if h1_metric <= 0:
        return GuardrailResult(
            passed=True,
            skipped=["horizon_consistency: h=1 metric <= 0"],
            details=details,
        )

    # Check ratios
    ratios = [other / h1_metric for other in other_metrics if other > 0]

    if not ratios:
        return GuardrailResult(
            passed=True,
            skipped=["horizon_consistency: no valid ratios to compare"],
            details=details,
        )

    avg_ratio = np.mean(ratios)
    details["avg_ratio"] = avg_ratio

    if avg_ratio > max_ratio:
        return GuardrailResult(
            passed=False,
            errors=[
                f"h=1 performance ({h1_metric:.4f}) is {avg_ratio:.1f}x better than "
                f"longer horizons. This may indicate temporal boundary violations."
            ],
            details=details,
            recommendations=[
                "Verify gap >= horizon in cross-validation setup",
                "Run gate_temporal_boundary() for detailed analysis",
            ],
        )

    return GuardrailResult(passed=True, details=details)


def check_residual_autocorrelation(
    residuals: np.ndarray,
    max_lag: int = 5,
    threshold: float = 0.2,
) -> GuardrailResult:
    """
    Check if residuals exhibit significant autocorrelation.

    Significant autocorrelation in residuals suggests the model
    is missing exploitable temporal patterns.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (prediction - actual).
    max_lag : int, default=5
        Maximum lag to check.
    threshold : float, default=0.2
        Autocorrelation threshold for warning.

    Returns
    -------
    GuardrailResult
        passed=True but warns if high autocorrelation found

    Notes
    -----
    This is a warning-only check. High autocorrelation doesn't indicate
    leakage, just that the model could potentially be improved.

    Knowledge Tier: [T1] Residual diagnostics (standard time series practice).
    """
    if len(residuals) < max_lag + 10:
        return GuardrailResult(
            passed=True,
            skipped=[f"residual_autocorrelation: need at least {max_lag + 10} samples"],
        )

    # Compute autocorrelations
    residuals = np.asarray(residuals)
    residuals = residuals - np.mean(residuals)
    n = len(residuals)

    autocorrs = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        r = np.corrcoef(residuals[lag:], residuals[:-lag])[0, 1]
        autocorrs.append((lag, r))

    details = {
        "autocorrelations": {f"lag_{lag}": r for lag, r in autocorrs},
        "threshold": threshold,
    }

    # Check for significant autocorrelation
    significant = [(lag, r) for lag, r in autocorrs if abs(r) > threshold]

    if significant:
        return GuardrailResult(
            passed=True,  # Warning only, not a failure
            warnings=[
                f"Residuals show autocorrelation at lags: "
                f"{', '.join(f'{lag} (r={r:.2f})' for lag, r in significant)}. "
                f"Model may be missing temporal patterns."
            ],
            details=details,
            recommendations=[
                "Consider adding lagged features",
                "Try ARIMA-style error correction",
            ],
        )

    return GuardrailResult(passed=True, details=details)


def run_all_guardrails(
    model_metric: float,
    baseline_metric: float,
    n_samples: int,
    n_up: Optional[int] = None,
    n_down: Optional[int] = None,
    horizon_metrics: Optional[List[float]] = None,
    residuals: Optional[np.ndarray] = None,
    improvement_threshold: float = 0.20,
    min_sample_size: int = 50,
    min_stratum_size: int = 10,
) -> GuardrailResult:
    """
    Run comprehensive guardrail validation.

    Combines multiple guardrail checks into a single validation pass.
    Returns composite result with all warnings and errors.

    Parameters
    ----------
    model_metric : float
        Model's error metric (lower is better).
    baseline_metric : float
        Baseline error metric (e.g., persistence forecast).
    n_samples : int
        Total sample size for evaluation.
    n_up : int, optional
        Number of UP direction samples (for stratified checks).
    n_down : int, optional
        Number of DOWN direction samples (for stratified checks).
    horizon_metrics : List[float], optional
        Metrics for each horizon [h1, h2, h3, ...] (for horizon consistency).
    residuals : np.ndarray, optional
        Model residuals (for autocorrelation check).
    improvement_threshold : float, default=0.20
        Maximum acceptable improvement ratio.
    min_sample_size : int, default=50
        Minimum required sample size.
    min_stratum_size : int, default=10
        Minimum required per stratum.

    Returns
    -------
    GuardrailResult
        Composite result with passed=False if any check fails.

    Knowledge Tier: [T3] Unified guardrail pattern (convenience layer).

    Examples
    --------
    >>> result = run_all_guardrails(
    ...     model_metric=0.15,
    ...     baseline_metric=0.20,
    ...     n_samples=100,
    ... )
    >>> if not result.passed:
    ...     print(result.summary())
    """
    # Collect all results
    all_warnings: List[str] = []
    all_errors: List[str] = []
    all_skipped: List[str] = []
    all_recommendations: List[str] = []
    all_details: Dict[str, Any] = {}

    # 1. Check suspicious improvement
    improvement_result = check_suspicious_improvement(
        model_metric=model_metric,
        baseline_metric=baseline_metric,
        threshold=improvement_threshold,
    )
    all_warnings.extend(improvement_result.warnings)
    all_errors.extend(improvement_result.errors)
    all_skipped.extend(improvement_result.skipped)
    all_recommendations.extend(improvement_result.recommendations)
    all_details["suspicious_improvement"] = improvement_result.details

    # 2. Check minimum sample size
    sample_result = check_minimum_sample_size(
        n=n_samples,
        min_n=min_sample_size,
    )
    all_warnings.extend(sample_result.warnings)
    all_errors.extend(sample_result.errors)
    all_skipped.extend(sample_result.skipped)
    all_recommendations.extend(sample_result.recommendations)
    all_details["sample_size"] = sample_result.details

    # 3. Check stratified sample sizes (if provided)
    if n_up is not None and n_down is not None:
        stratified_result = check_stratified_sample_size(
            n_up=n_up,
            n_down=n_down,
            min_n=min_stratum_size,
        )
        all_warnings.extend(stratified_result.warnings)
        all_errors.extend(stratified_result.errors)
        all_skipped.extend(stratified_result.skipped)
        all_recommendations.extend(stratified_result.recommendations)
        all_details["stratified_sample_size"] = stratified_result.details

    # 4. Check horizon consistency (if provided)
    if horizon_metrics is not None:
        horizon_result = check_forecast_horizon_consistency(horizons=horizon_metrics)
        all_warnings.extend(horizon_result.warnings)
        all_errors.extend(horizon_result.errors)
        all_skipped.extend(horizon_result.skipped)
        all_recommendations.extend(horizon_result.recommendations)
        all_details["horizon_consistency"] = horizon_result.details

    # 5. Check residual autocorrelation (if provided)
    if residuals is not None:
        residual_result = check_residual_autocorrelation(residuals=residuals)
        all_warnings.extend(residual_result.warnings)
        all_errors.extend(residual_result.errors)
        all_skipped.extend(residual_result.skipped)
        all_recommendations.extend(residual_result.recommendations)
        all_details["residual_autocorrelation"] = residual_result.details

    # Determine overall pass/fail
    passed = len(all_errors) == 0

    # Deduplicate recommendations
    unique_recommendations = list(dict.fromkeys(all_recommendations))

    return GuardrailResult(
        passed=passed,
        warnings=all_warnings,
        errors=all_errors,
        skipped=all_skipped,
        details=all_details,
        recommendations=unique_recommendations,
    )


__all__ = [
    "GuardrailResult",
    "check_suspicious_improvement",
    "check_minimum_sample_size",
    "check_stratified_sample_size",
    "check_forecast_horizon_consistency",
    "check_residual_autocorrelation",
    "run_all_guardrails",
]
