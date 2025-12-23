"""
Validation Gates Module.

Three-stage validation framework with HALT/PASS/WARN/SKIP decisions:

1. **External validation**: Shuffled target, Synthetic AR(1)
2. **Internal validation**: Suspicious improvement detection
3. **Statistical validation**: Residual diagnostics

The key insight: if a model beats a shuffled target or significantly
outperforms theoretical bounds, it's likely learning from leakage.

Knowledge Tiers
---------------
[T1] Shuffled target test destroys temporal structure (permutation test principle)
[T1] AR(1) optimal 1-step MAE = σ√(2/π) ≈ 0.798σ (standard statistics result)
[T1] Walk-forward validation framework (Tashman 2000)
[T2] Shuffled target as definitive leakage test (myga-forecasting-v2 validation)
[T2] "External-first" validation ordering (synthetic → shuffled → internal)
[T3] 20% improvement threshold = "too good to be true" heuristic (empirical)
[T3] 5% p-value threshold for shuffled comparison (standard but arbitrary)
[T3] Tolerance factor 1.5 for AR(1) bounds (allows for finite-sample variation)

Example
-------
>>> from temporalcv.gates import run_gates, gate_shuffled_target
>>>
>>> report = run_gates(
...     model=my_model,
...     X=X, y=y,
...     gates=[
...         gate_shuffled_target(n_shuffles=5),
...         gate_suspicious_improvement(threshold=0.20),
...     ]
... )
>>> if report.status == "HALT":
...     raise ValueError(f"Validation failed: {report.failures}")

References
----------
[T1] Hewamalage, H., Bergmeir, C. & Bandara, K. (2023). Forecast evaluation
     for data scientists: Common pitfalls and best practices.
     International Journal of Forecasting, 39(3), 1238-1268.
[T1] Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy:
     An analysis and review. International Journal of Forecasting, 16(4), 437-450.
[T1] Optimal MAE for N(0,σ) = σ√(2/π): Standard result from order statistics.
     For AR(1) with innovation variance σ², the 1-step forecast error is σ·ε_t,
     hence MAE = E[|σ·ε|] = σ·√(2/π) when ε ~ N(0,1).
[T2] Three-stage validation: External validation first catches gross errors before
     trusting internal metrics. Principle established in myga-forecasting-v2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike

from temporalcv.cv import WalkForwardCV


class GateStatus(Enum):
    """Validation gate status."""

    HALT = "HALT"  # Critical failure - stop and investigate
    WARN = "WARN"  # Caution - continue but verify
    PASS = "PASS"  # Validation passed
    SKIP = "SKIP"  # Insufficient data to run gate


@dataclass
class GateResult:
    """
    Result from a validation gate.

    Attributes
    ----------
    name : str
        Gate identifier (e.g., "shuffled_target", "synthetic_ar1")
    status : GateStatus
        HALT, WARN, PASS, or SKIP
    message : str
        Human-readable description of result
    metric_value : float, optional
        Primary metric for this gate (e.g., improvement ratio)
    threshold : float, optional
        Threshold used for decision
    details : dict
        Additional metrics and diagnostics
    recommendation : str
        What to do if not PASS
    """

    name: str
    status: GateStatus
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def __str__(self) -> str:
        """Format as [STATUS] name: message."""
        return f"[{self.status.value}] {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """
    Complete validation report across all gates.

    Attributes
    ----------
    gates : list[GateResult]
        Results from all gates run
    """

    gates: List[GateResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        """
        Overall status: HALT if any HALT, WARN if any WARN, else PASS.

        Returns
        -------
        str
            "HALT", "WARN", or "PASS"
        """
        if any(g.status == GateStatus.HALT for g in self.gates):
            return "HALT"
        if any(g.status == GateStatus.WARN for g in self.gates):
            return "WARN"
        return "PASS"

    @property
    def failures(self) -> List[GateResult]:
        """Return gates that HALTed."""
        return [g for g in self.gates if g.status == GateStatus.HALT]

    @property
    def warnings(self) -> List[GateResult]:
        """Return gates that WARNed."""
        return [g for g in self.gates if g.status == GateStatus.WARN]

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            "",
        ]

        for gate in self.gates:
            lines.append(f"  {gate}")

        lines.extend([
            "",
            "=" * 60,
            f"OVERALL STATUS: {self.status}",
            "=" * 60,
        ])

        if self.failures:
            lines.append("")
            lines.append("HALTED GATES (require investigation):")
            for gate in self.failures:
                lines.append(f"  - {gate.name}: {gate.recommendation}")

        return "\n".join(lines)


# =============================================================================
# Protocol for model interface
# =============================================================================


class FitPredictModel(Protocol):
    """Protocol for models with fit/predict interface."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Any:
        """Fit model to training data."""
        ...

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Generate predictions."""
        ...


# =============================================================================
# Stage 1: External Validation Gates
# =============================================================================


def gate_shuffled_target(
    model: FitPredictModel,
    X: ArrayLike,
    y: ArrayLike,
    n_shuffles: int = 5,
    threshold: float = 0.05,
    n_cv_splits: int = 3,
    random_state: Optional[int] = None,
) -> GateResult:
    """
    Shuffled target test: definitive leakage detection.

    If a model performs better on real target than shuffled target,
    features may contain information about target ordering (leakage).

    A model should NOT beat shuffled baseline - the temporal relationship
    between X and y should be destroyed by shuffling.

    Parameters
    ----------
    model : FitPredictModel
        Model with fit(X, y) and predict(X) methods
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    n_shuffles : int, default=5
        Number of shuffled targets to average over
    threshold : float, default=0.05
        Maximum allowed improvement ratio over shuffled baseline
    n_cv_splits : int, default=3
        Number of walk-forward CV splits for out-of-sample evaluation.
        Uses expanding window CV to ensure proper temporal evaluation.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    GateResult
        HALT if model significantly beats shuffled baseline

    Notes
    -----
    This is the definitive leakage test. If your model beats a shuffled
    target, something is wrong - either features leak future information
    or there's a bug in the evaluation pipeline.

    Uses WalkForwardCV internally to compute out-of-sample MAE, avoiding
    the bias of in-sample evaluation that could mask or exaggerate leakage.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Validate shapes
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y must have same number of samples. "
            f"Got X.shape[0]={X.shape[0]}, len(y)={len(y)}"
        )

    n = len(y)
    rng = np.random.default_rng(random_state)

    # Set up walk-forward CV for out-of-sample evaluation
    cv = WalkForwardCV(
        n_splits=n_cv_splits,
        window_type="expanding",
        gap=0,  # No gap needed for leakage detection
        test_size=max(1, n // (n_cv_splits + 1)),  # Reasonable test size
    )

    def compute_cv_mae(X_data: np.ndarray, y_data: np.ndarray) -> float:
        """Compute out-of-sample MAE using walk-forward CV."""
        all_errors: List[float] = []
        for train_idx, test_idx in cv.split(X_data, y_data):
            model.fit(X_data[train_idx], y_data[train_idx])
            preds = np.asarray(model.predict(X_data[test_idx]))
            errors = np.abs(y_data[test_idx] - preds)
            all_errors.extend(errors.tolist())
        return float(np.mean(all_errors)) if all_errors else 0.0

    # Compute out-of-sample MAE on real target
    mae_real = compute_cv_mae(X, y)

    # Compute out-of-sample MAE on shuffled targets
    shuffled_maes: List[float] = []
    for _ in range(n_shuffles):
        y_shuffled = rng.permutation(y)
        mae_shuffled = compute_cv_mae(X, y_shuffled)
        shuffled_maes.append(mae_shuffled)

    mae_shuffled_avg = float(np.mean(shuffled_maes))

    # Improvement ratio: positive = model beats shuffled (suspicious)
    if mae_shuffled_avg > 0:
        improvement_ratio = 1 - (mae_real / mae_shuffled_avg)
    else:
        improvement_ratio = 0.0

    details = {
        "mae_real": mae_real,
        "mae_shuffled_avg": mae_shuffled_avg,
        "mae_shuffled_all": shuffled_maes,
        "n_shuffles": n_shuffles,
        "n_cv_splits": n_cv_splits,
        "evaluation_method": "walk_forward_cv",
    }

    if improvement_ratio > threshold:
        return GateResult(
            name="shuffled_target",
            status=GateStatus.HALT,
            message=f"Model beats shuffled by {improvement_ratio:.1%} (max: {threshold:.0%})",
            metric_value=improvement_ratio,
            threshold=threshold,
            details=details,
            recommendation="Check for data leakage. Model should NOT beat shuffled target.",
        )

    return GateResult(
        name="shuffled_target",
        status=GateStatus.PASS,
        message=f"Model improvement {improvement_ratio:.1%} is acceptable",
        metric_value=improvement_ratio,
        threshold=threshold,
        details=details,
    )


def gate_synthetic_ar1(
    model: FitPredictModel,
    phi: float = 0.95,
    sigma: float = 1.0,
    n_samples: int = 500,
    n_lags: int = 5,
    tolerance: float = 1.5,
    n_cv_splits: int = 3,
    random_state: Optional[int] = None,
) -> GateResult:
    """
    Synthetic AR(1) test: theoretical bound verification.

    Test model on synthetic AR(1) process where optimal forecast is
    phi * y_{t-1}. Model MAE should not significantly beat theoretical optimum.

    For AR(1): y_t = phi * y_{t-1} + sigma * epsilon_t

    Theoretical optimal 1-step MAE = sigma * sqrt(2/pi) ≈ 0.798 * sigma

    Parameters
    ----------
    model : FitPredictModel
        Model with fit(X, y) and predict(X) methods
    phi : float, default=0.95
        AR(1) coefficient (persistence parameter). Must be in (-1, 1) for
        stationarity.
    sigma : float, default=1.0
        Innovation standard deviation
    n_samples : int, default=500
        Number of samples to generate. Must be > n_lags.
    n_lags : int, default=5
        Number of lagged features to create
    tolerance : float, default=1.5
        How much better model can be than theoretical optimum.
        ratio < 1/tolerance triggers HALT.
    n_cv_splits : int, default=3
        Number of walk-forward CV splits for out-of-sample evaluation.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    GateResult
        HALT if model beats theoretical bound by too much

    Raises
    ------
    ValueError
        If phi is not in (-1, 1) (non-stationary) or n_samples <= n_lags.

    Notes
    -----
    If a model significantly beats the theoretical optimum on AR(1) data,
    it's likely exploiting lookahead bias or has implementation bugs.

    Uses WalkForwardCV internally to compute out-of-sample MAE, avoiding
    in-sample evaluation bias.
    """
    # Validate phi for stationarity
    if not (-1 < phi < 1):
        raise ValueError(
            f"phi must be in (-1, 1) for stationarity. Got phi={phi}. "
            f"Values outside this range produce non-stationary or explosive series."
        )

    # Validate n_samples > n_lags
    if n_samples <= n_lags:
        raise ValueError(
            f"n_samples must be > n_lags to have data for prediction. "
            f"Got n_samples={n_samples}, n_lags={n_lags}."
        )

    rng = np.random.default_rng(random_state)

    # Generate AR(1) process
    y_full = np.zeros(n_samples + n_lags)
    y_full[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2))  # Stationary initialization

    for t in range(1, len(y_full)):
        y_full[t] = phi * y_full[t - 1] + sigma * rng.normal()

    # Create lagged features (proper temporal alignment)
    y = y_full[n_lags:]  # Target: y_t
    X = np.column_stack([y_full[n_lags - lag : -lag] for lag in range(1, n_lags + 1)])

    n = len(y)

    # Set up walk-forward CV for out-of-sample evaluation
    cv = WalkForwardCV(
        n_splits=n_cv_splits,
        window_type="expanding",
        gap=0,
        test_size=max(1, n // (n_cv_splits + 1)),
    )

    # Compute out-of-sample MAE using walk-forward CV
    all_errors: List[float] = []
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        preds = np.asarray(model.predict(X[test_idx]))
        errors = np.abs(y[test_idx] - preds)
        all_errors.extend(errors.tolist())

    model_mae = float(np.mean(all_errors)) if all_errors else 0.0

    # Theoretical optimal MAE for AR(1) 1-step forecast
    # Optimal predictor is phi * y_{t-1}, error is sigma * epsilon
    # MAE of N(0, sigma) = sigma * sqrt(2/pi)
    theoretical_mae = sigma * np.sqrt(2 / np.pi)

    ratio = model_mae / theoretical_mae

    details = {
        "model_mae": model_mae,
        "theoretical_mae": theoretical_mae,
        "phi": phi,
        "sigma": sigma,
        "n_samples": n_samples,
        "n_lags": n_lags,
        "n_cv_splits": n_cv_splits,
        "evaluation_method": "walk_forward_cv",
    }

    if ratio < 1 / tolerance:
        return GateResult(
            name="synthetic_ar1",
            status=GateStatus.HALT,
            message=f"Model MAE {model_mae:.4f} << theoretical {theoretical_mae:.4f} (ratio={ratio:.2f})",
            metric_value=ratio,
            threshold=1 / tolerance,
            details=details,
            recommendation="Model beats theoretical optimum. Check for lookahead bias.",
        )

    return GateResult(
        name="synthetic_ar1",
        status=GateStatus.PASS,
        message=f"Model MAE ratio {ratio:.2f} is within bounds",
        metric_value=ratio,
        threshold=1 / tolerance,
        details=details,
    )


# =============================================================================
# Stage 2: Internal Validation Gates
# =============================================================================


def gate_suspicious_improvement(
    model_metric: float,
    baseline_metric: float,
    threshold: float = 0.20,
    warn_threshold: float = 0.10,
    metric_name: str = "MAE",
) -> GateResult:
    """
    Check for suspiciously large improvement over baseline.

    Large improvements (e.g., >20% better than persistence) in time-series
    forecasting are often indicators of data leakage rather than genuine skill.

    Parameters
    ----------
    model_metric : float
        Model's error metric (lower is better)
    baseline_metric : float
        Baseline error metric (e.g., persistence MAE)
    threshold : float, default=0.20
        Improvement ratio that triggers HALT (e.g., 0.20 = 20% better)
    warn_threshold : float, default=0.10
        Improvement ratio that triggers WARN
    metric_name : str, default="MAE"
        Name of metric for messages

    Returns
    -------
    GateResult
        HALT if improvement exceeds threshold, WARN if notable

    Notes
    -----
    Experience shows that genuine forecasting improvements are modest.
    If your model shows 40%+ improvement over persistence, verify with
    shuffled target test before trusting the results.
    """
    if baseline_metric <= 0:
        return GateResult(
            name="suspicious_improvement",
            status=GateStatus.SKIP,
            message="Baseline metric is zero or negative",
            details={"model_metric": model_metric, "baseline_metric": baseline_metric},
        )

    # Improvement ratio: higher = model is better
    improvement = 1 - (model_metric / baseline_metric)

    details = {
        f"model_{metric_name.lower()}": model_metric,
        f"baseline_{metric_name.lower()}": baseline_metric,
        "improvement_ratio": improvement,
    }

    if improvement > threshold:
        return GateResult(
            name="suspicious_improvement",
            status=GateStatus.HALT,
            message=f"Model {improvement:.1%} better than baseline (max: {threshold:.0%})",
            metric_value=improvement,
            threshold=threshold,
            details=details,
            recommendation="Run shuffled target test. This improvement is suspicious.",
        )

    if improvement > warn_threshold:
        return GateResult(
            name="suspicious_improvement",
            status=GateStatus.WARN,
            message=f"Model {improvement:.1%} better than baseline - verify carefully",
            metric_value=improvement,
            threshold=warn_threshold,
            details=details,
            recommendation="Verify with external validation before trusting.",
        )

    return GateResult(
        name="suspicious_improvement",
        status=GateStatus.PASS,
        message=f"Improvement {improvement:.1%} is reasonable",
        metric_value=improvement,
        threshold=threshold,
        details=details,
    )


def gate_temporal_boundary(
    train_end_idx: int,
    test_start_idx: int,
    horizon: int,
    gap: int = 0,
) -> GateResult:
    """
    Verify temporal boundary enforcement.

    Ensures proper gap between training end and test start for h-step forecasts.

    Parameters
    ----------
    train_end_idx : int
        Last index of training data (inclusive)
    test_start_idx : int
        First index of test data
    horizon : int
        Forecast horizon (h)
    gap : int, default=0
        Additional gap beyond horizon requirement

    Returns
    -------
    GateResult
        HALT if temporal boundary is violated

    Notes
    -----
    For h-step ahead forecasting, the last training observation should be
    at least h periods before the first test observation to prevent leakage.

    Required: test_start_idx >= train_end_idx + horizon + gap
    """
    required_gap = horizon + gap
    actual_gap = test_start_idx - train_end_idx - 1

    details = {
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "horizon": horizon,
        "gap": gap,
        "required_gap": required_gap,
        "actual_gap": actual_gap,
    }

    if actual_gap < required_gap:
        return GateResult(
            name="temporal_boundary",
            status=GateStatus.HALT,
            message=f"Gap {actual_gap} < required {required_gap} for h={horizon}",
            metric_value=actual_gap,
            threshold=required_gap,
            details=details,
            recommendation=f"Increase gap between train and test. Need {required_gap - actual_gap} more periods.",
        )

    return GateResult(
        name="temporal_boundary",
        status=GateStatus.PASS,
        message=f"Gap {actual_gap} >= required {required_gap}",
        metric_value=actual_gap,
        threshold=required_gap,
        details=details,
    )


# =============================================================================
# Residual Diagnostics Gate
# =============================================================================


def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute sample autocorrelation function (ACF) for lags 1 to max_lag.

    Parameters
    ----------
    x : np.ndarray
        Time series (residuals), assumed to be centered
    max_lag : int
        Maximum lag to compute

    Returns
    -------
    np.ndarray
        ACF values for lags 1, 2, ..., max_lag

    Knowledge Tier: [T1] Standard ACF formula (Box-Jenkins)
    """
    n = len(x)
    x = x - np.mean(x)  # Center the series
    gamma_0 = np.sum(x**2) / n  # Variance (lag 0 autocovariance)

    if gamma_0 == 0:
        # Constant residuals → no autocorrelation
        return np.zeros(max_lag)

    acf_values = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        gamma_k = np.sum(x[k:] * x[:-k]) / n
        acf_values[k - 1] = gamma_k / gamma_0

    return acf_values


def _ljung_box_test(residuals: np.ndarray, max_lag: int) -> tuple[float, float]:
    """
    Ljung-Box test for autocorrelation in residuals.

    Implements the Ljung-Box Q statistic [T1]:
        Q = n(n+2) Σ_{k=1}^{m} ρ_k² / (n-k)

    Under H₀ (no autocorrelation), Q ~ χ²(m).

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    max_lag : int
        Number of lags to test

    Returns
    -------
    tuple[float, float]
        (Q statistic, p-value)

    Knowledge Tier: [T1] Ljung & Box (1978). Biometrika 65(2), 297-303.
    """
    from scipy import stats

    n = len(residuals)
    if n <= max_lag:
        # Not enough data for test
        return 0.0, 1.0

    acf = _compute_acf(residuals, max_lag)

    # Ljung-Box Q statistic
    # Q = n(n+2) Σ ρ_k² / (n-k) for k=1 to max_lag
    Q = 0.0
    for k in range(1, max_lag + 1):
        Q += (acf[k - 1] ** 2) / (n - k)
    Q *= n * (n + 2)

    # P-value from chi-squared distribution with max_lag degrees of freedom
    p_value = 1.0 - stats.chi2.cdf(Q, df=max_lag)

    return float(Q), float(p_value)


def gate_residual_diagnostics(
    residuals: ArrayLike,
    max_lag: int = 10,
    significance: float = 0.05,
    halt_on_autocorr: bool = False,
    halt_on_normality: bool = False,
) -> GateResult:
    """
    Check residual quality via diagnostic tests.

    This gate runs three diagnostic tests on model residuals:

    1. **Ljung-Box test** [T1]: Detects residual autocorrelation
       - H₀: Residuals are white noise
       - Significant autocorrelation suggests model misspecification
       - Custom implementation to avoid statsmodels dependency

    2. **Jarque-Bera test** [T1]: Detects non-normality
       - H₀: Residuals are normally distributed
       - Non-normality may affect confidence intervals
       - Uses scipy.stats.jarque_bera

    3. **Mean-zero t-test** [T1]: Detects systematic bias
       - H₀: Mean(residuals) = 0
       - Non-zero mean indicates biased predictions
       - Uses scipy.stats.ttest_1samp

    Parameters
    ----------
    residuals : array-like
        Model residuals (actuals - predictions)
    max_lag : int, default=10
        Maximum lag for Ljung-Box test. Higher values test more lags
        but reduce power.
    significance : float, default=0.05
        Significance level for all tests [T3]
    halt_on_autocorr : bool, default=False
        If True, HALT on significant autocorrelation; otherwise WARN
    halt_on_normality : bool, default=False
        If True, HALT on non-normality; otherwise WARN

    Returns
    -------
    GateResult
        - PASS: All tests pass
        - WARN: Some tests fail but not configured to HALT
        - HALT: Tests fail and configured to HALT
        - SKIP: Insufficient data (n < 30)

    Knowledge Tiers
    ---------------
    [T1] Ljung-Box: Ljung & Box (1978). Biometrika 65(2), 297-303.
    [T1] Jarque-Bera: Jarque & Bera (1987). International Statistical Review 55(2).
    [T1] t-test: Standard hypothesis testing
    [T3] significance=0.05 default is conventional but arbitrary

    Example
    -------
    >>> residuals = y_actual - y_predicted
    >>> result = gate_residual_diagnostics(residuals, max_lag=10)
    >>> if result.status == GateStatus.WARN:
    ...     print("Check:", result.details["failing_tests"])
    """
    from scipy import stats

    residuals = np.asarray(residuals)
    n = len(residuals)

    # Minimum sample size for reliable tests
    MIN_SAMPLES = 30

    if n < MIN_SAMPLES:
        return GateResult(
            name="residual_diagnostics",
            status=GateStatus.SKIP,
            message=f"Insufficient data: n={n} < {MIN_SAMPLES} required for residual tests",
            details={"n_samples": n, "min_required": MIN_SAMPLES},
            recommendation="Collect more data before running residual diagnostics",
        )

    # Ensure max_lag is reasonable
    max_lag = min(max_lag, n // 3)  # Rule of thumb: don't test more than n/3 lags
    if max_lag < 1:
        max_lag = 1

    # === Run diagnostic tests ===
    test_results: dict[str, dict[str, Any]] = {}
    failing_tests: list[str] = []

    # 1. Ljung-Box test for autocorrelation
    lb_stat, lb_pval = _ljung_box_test(residuals, max_lag)
    test_results["ljung_box"] = {
        "statistic": lb_stat,
        "p_value": lb_pval,
        "max_lag": max_lag,
        "significant": lb_pval < significance,
    }
    if lb_pval < significance:
        failing_tests.append("ljung_box")

    # 2. Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(residuals)
    test_results["jarque_bera"] = {
        "statistic": float(jb_stat),
        "p_value": float(jb_pval),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
        "significant": jb_pval < significance,
    }
    if jb_pval < significance:
        failing_tests.append("jarque_bera")

    # 3. Mean-zero t-test
    ttest_stat, ttest_pval = stats.ttest_1samp(residuals, 0)
    test_results["mean_zero"] = {
        "statistic": float(ttest_stat),
        "p_value": float(ttest_pval),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)),
        "significant": ttest_pval < significance,
    }
    if ttest_pval < significance:
        failing_tests.append("mean_zero")

    # === Determine gate status ===
    details = {
        "n_samples": n,
        "significance": significance,
        "tests": test_results,
        "failing_tests": failing_tests,
    }

    if not failing_tests:
        return GateResult(
            name="residual_diagnostics",
            status=GateStatus.PASS,
            message="All residual diagnostics passed",
            metric_value=0.0,
            threshold=significance,
            details=details,
        )

    # Check which failures warrant HALT vs WARN
    halt_reasons = []
    if "ljung_box" in failing_tests and halt_on_autocorr:
        halt_reasons.append("autocorrelation")
    if "jarque_bera" in failing_tests and halt_on_normality:
        halt_reasons.append("non-normality")
    if "mean_zero" in failing_tests:
        # Mean-zero always warrants attention (biased predictions)
        halt_reasons.append("bias")

    if halt_reasons:
        return GateResult(
            name="residual_diagnostics",
            status=GateStatus.HALT,
            message=f"Residual diagnostics failed: {', '.join(halt_reasons)}",
            metric_value=float(len(failing_tests)),
            threshold=significance,
            details=details,
            recommendation=(
                "Investigate model specification. "
                + ("Autocorrelation suggests missing temporal structure. " if "autocorrelation" in halt_reasons else "")
                + ("Bias suggests systematic prediction error. " if "bias" in halt_reasons else "")
                + ("Non-normality may affect confidence intervals." if "non-normality" in halt_reasons else "")
            ),
        )

    # WARN for failures without halt flags
    return GateResult(
        name="residual_diagnostics",
        status=GateStatus.WARN,
        message=f"Residual diagnostics: {len(failing_tests)} test(s) failed",
        metric_value=float(len(failing_tests)),
        threshold=significance,
        details=details,
        recommendation=(
            f"Review failing tests: {', '.join(failing_tests)}. "
            "These may not be critical but warrant investigation."
        ),
    )


# =============================================================================
# Gate Runner
# =============================================================================


GateFunction = Callable[..., GateResult]


def run_gates(
    gates: List[GateResult],
) -> ValidationReport:
    """
    Aggregate gate results into a validation report.

    Parameters
    ----------
    gates : list[GateResult]
        Pre-computed gate results

    Returns
    -------
    ValidationReport
        Aggregated validation report

    Example
    -------
    >>> results = [
    ...     gate_shuffled_target(model, X, y),
    ...     gate_suspicious_improvement(model_mae, persistence_mae),
    ... ]
    >>> report = run_gates(results)
    >>> if report.status == "HALT":
    ...     print(report.summary())
    """
    return ValidationReport(gates=gates)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums and dataclasses
    "GateStatus",
    "GateResult",
    "ValidationReport",
    # Gate functions
    "gate_shuffled_target",
    "gate_synthetic_ar1",
    "gate_suspicious_improvement",
    "gate_temporal_boundary",
    "gate_residual_diagnostics",
    # Runner
    "run_gates",
]
