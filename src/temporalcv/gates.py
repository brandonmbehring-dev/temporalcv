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
[T2] Signal verification via shuffled target (myga-forecasting-v2 validation)
[T2] "External-first" validation ordering (synthetic → shuffled → internal)
[T3] 20% improvement threshold = "too good to be true" heuristic (empirical)
[T3] 5% p-value threshold for shuffled comparison (standard but arbitrary)
[T3] Tolerance factor 1.5 for AR(1) bounds (allows for finite-sample variation)

Example
-------
>>> from temporalcv.gates import run_gates, gate_signal_verification
>>>
>>> # Signal verification: does model have predictive power?
>>> signal_result = gate_signal_verification(model, X, y)
>>> if signal_result.status.name == "HALT":
...     print("Model has signal - investigate if legitimate or leakage")
>>>
>>> # Run multiple gates
>>> report = run_gates(gates=[
...     gate_signal_verification(model, X, y),
...     gate_suspicious_improvement(model_mae, baseline_mae, threshold=0.20),
... ])
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

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Literal, Optional, Protocol, Union, cast

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
# DRY Helper: Centralized CV Metric Computation
# =============================================================================


def _clone_model(model: Any) -> Any:
    """Clone model if possible, otherwise return original.

    Handles sklearn models and mock objects gracefully.
    """
    try:
        from sklearn.base import clone

        return clone(model)
    except (ImportError, TypeError):
        # ImportError: sklearn not available
        # TypeError: model doesn't support cloning (e.g., mock objects)
        return model


def _compute_cv_mae(
    model: FitPredictModel,
    X: np.ndarray,
    y: np.ndarray,
    n_cv_splits: int = 3,
    extra_gap: int = 0,
    return_errors: bool = False,
) -> Union[float, tuple[float, np.ndarray]]:
    """Compute out-of-sample MAE using walk-forward CV.

    Centralized CV computation to avoid duplication across gate functions.
    Clones model per fold to prevent state contamination from warm-start
    or incremental learning algorithms.

    Parameters
    ----------
    model : FitPredictModel
        Model with fit(X, y) and predict(X) methods.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target array.
    n_cv_splits : int, default=3
        Number of walk-forward splits.
    extra_gap : int, default=0
        Extra gap between train and test (in addition to horizon).
    return_errors : bool, default=False
        If True, return (mae, errors_array) instead of just mae.

    Returns
    -------
    mae : float
        Mean absolute error across all folds.
    errors : np.ndarray, optional
        Raw absolute errors (only if return_errors=True).
    """
    from temporalcv.cv import WalkForwardCV

    n = len(y)
    cv = WalkForwardCV(
        n_splits=n_cv_splits,
        window_type="expanding",
        extra_gap=extra_gap,
        test_size=max(1, n // (n_cv_splits + 1)),
    )

    all_errors: List[float] = []
    for train_idx, test_idx in cv.split(X, y):
        fold_model = _clone_model(model)
        fold_model.fit(X[train_idx], y[train_idx])
        preds = np.asarray(fold_model.predict(X[test_idx]))
        errors = np.abs(y[test_idx] - preds)
        all_errors.extend(errors.tolist())

    errors_arr = np.array(all_errors) if all_errors else np.array([])
    mae = float(np.mean(errors_arr)) if len(errors_arr) > 0 else 0.0

    if return_errors:
        return mae, errors_arr
    return mae


# =============================================================================
# Stage 1: External Validation Gates
# =============================================================================


def gate_signal_verification(
    model: FitPredictModel,
    X: ArrayLike,
    y: ArrayLike,
    n_shuffles: Optional[int] = None,
    threshold: float = 0.05,
    n_cv_splits: int = 3,
    permutation: Literal["iid", "block"] = "block",
    block_size: Union[int, Literal["auto"]] = "auto",
    random_state: Optional[int] = None,
    *,
    method: Literal["effect_size", "permutation"] = "permutation",
    alpha: float = 0.05,
    strict: bool = False,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 100,
    bootstrap_alpha: float = 0.05,
    bootstrap_block_length: Union[int, Literal["auto"]] = "auto",
) -> GateResult:
    """
    Signal verification test: confirm model has predictive power.

    Tests whether a model significantly outperforms a shuffled-target
    baseline. HALT indicates the model has learned signal — this could
    be legitimate temporal patterns OR data leakage.

    Interpretation:
    - HALT: Model has signal → investigate source (leakage vs legitimate)
    - PASS: Model has no signal → concerning (learned nothing)
    - WARN: Marginal signal → proceed with caution

    Parameters
    ----------
    model : FitPredictModel
        Model with fit(X, y) and predict(X) methods
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target vector
    n_shuffles : int, optional
        Number of shuffled targets to generate. Defaults depend on ``method``:

        - ``method="effect_size"``: default=5 (fast heuristic)
        - ``method="permutation"``: default=100 (statistically rigorous)

        **Power analysis** [T1]: With n permutations, the minimum achievable
        p-value is 1/(n+1). For p < 0.05, use n >= 19. For p < 0.01, use n >= 99.
        See Phipson & Smyth (2010).
    threshold : float, default=0.05
        Maximum allowed improvement ratio over shuffled baseline.
        Only used when ``method="effect_size"``.
    n_cv_splits : int, default=3
        Number of walk-forward CV splits for out-of-sample evaluation.
        Uses expanding window CV to ensure proper temporal evaluation.
    permutation : {"iid", "block"}, default="block"
        Permutation strategy for null hypothesis:

        - "iid": Standard random permutation (may produce false positives
          on persistent time series)
        - "block": Block permutation that preserves local autocorrelation
          [T1] Per Kunsch (1989), Politis & Romano (1994)
    block_size : int or "auto", default="auto"
        Block size for block permutation. "auto" uses n^(1/3) per Kunsch (1989).
        Ignored if permutation="iid".
    random_state : int, optional
        Random seed for reproducibility
    method : {"effect_size", "permutation"}, default="permutation"
        Statistical method for decision:

        - "effect_size": Compare improvement ratio to threshold (fast, heuristic).
          HALT if model_mae < shuffled_mae * (1 - threshold). Default n_shuffles=5.
        - "permutation": True permutation test with p-value (rigorous).
          HALT if p-value < alpha. Default n_shuffles=100. Per Phipson & Smyth (2010),
          p-value = (1 + count(shuffled_mae <= model_mae)) / (1 + n_shuffles).
    alpha : float, default=0.05
        Significance level for permutation test. Only used when ``method="permutation"``.
    strict : bool, default=False
        If True and method="permutation", override n_shuffles to max(n_shuffles, 199)
        for p-value resolution of 0.005. Recommended for publication.
    bootstrap_ci : bool, default=False
        If True, compute block bootstrap confidence interval for MAE.
        Results added to details dict as ci_lower, ci_upper, etc.
    n_bootstrap : int, default=100
        Number of bootstrap replications for CI.
    bootstrap_alpha : float, default=0.05
        Significance level for CI (1 - alpha CI).
    bootstrap_block_length : int or "auto", default="auto"
        Block length for bootstrap. "auto" uses n^(1/3) per Kunsch (1989).

    Returns
    -------
    GateResult
        HALT if model significantly beats shuffled baseline (has signal)

    Notes
    -----
    **Signal Verification vs Leakage Detection**:

    This gate answers: "Does my model have predictive signal?" A HALT result
    means yes — but signal can come from legitimate temporal patterns OR
    data leakage. Use this as a diagnostic:

    - **HALT → Investigate**: Confirm signal is legitimate (e.g., AR model
      with proper lagged features) or identify leakage source
    - **PASS → Concerning**: Model learned nothing from features

    For legitimate temporal models (e.g., AR with proper lagged features),
    HALT is expected and confirms the gate is working correctly.

    **Method Selection Guide**:

    - Use ``method="permutation"`` (default) for rigorous statistical testing.
      The p-value answers: "What's the probability of seeing this result by chance?"
    - Use ``method="effect_size"`` for quick sanity checks during development.
      The effect size answers: "How much better is the model than shuffled?"

    Uses WalkForwardCV internally to compute out-of-sample MAE, avoiding
    the bias of in-sample evaluation that could mask or exaggerate leakage.

    The block permutation (default) preserves local autocorrelation structure
    per Kunsch (1989), which is important for time series with persistence.
    IID permutation may produce false positives on persistent series because
    any model with legitimate predictive ability should beat a fully shuffled target.

    Models are cloned for each shuffle to prevent state leakage from
    warm-start or incremental learning algorithms.

    Complexity: O(n_shuffles × n_cv_splits × model_fit_time)

    References
    ----------
    [T1] Kunsch, H.R. (1989). The Jackknife and the Bootstrap for General
         Stationary Observations. Annals of Statistics, 17(3), 1217-1241.
    [T1] Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap.
         JASA, 89(428), 1303-1313.
    [T1] Phipson, B. & Smyth, G.K. (2010). Permutation P-values Should Never
         Be Zero: Calculating Exact P-values When Permutations Are Randomly
         Drawn. Statistical Applications in Genetics and Molecular Biology,
         9(1), Article 39. https://doi.org/10.2202/1544-6115.1585

    Examples
    --------
    Quick check during development (effect size mode):

    >>> result = gate_signal_verification(model, X, y, method="effect_size")
    >>> print(f"Improvement: {result.metric_value:.1%}")

    Rigorous testing for publication (permutation mode, default):

    >>> result = gate_signal_verification(model, X, y, method="permutation", strict=True)
    >>> print(f"p-value: {result.details['pvalue']:.4f}")

    Interpreting results:

    >>> if result.status == GateStatus.HALT:
    ...     # Model has signal - could be legitimate or leakage
    ...     print("Model has signal - investigate source")
    ... else:
    ...     # Model has NO signal - concerning
    ...     print("Model learned nothing from features")

    See Also
    --------
    gate_synthetic_ar1 : Test against theoretical AR(1) bounds.
    gate_suspicious_improvement : Check for implausible improvement ratios.
    WalkForwardCV : The CV strategy used internally.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Validate method parameter
    if method not in ("effect_size", "permutation"):
        raise ValueError(f"method must be 'effect_size' or 'permutation', got {method!r}")

    # Validate no NaN values
    if np.any(np.isnan(X)):
        raise ValueError(
            "X contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )
    if np.any(np.isnan(y)):
        raise ValueError(
            "y contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )

    # Validate shapes
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y must have same number of samples. "
            f"Got X.shape[0]={X.shape[0]}, len(y)={len(y)}"
        )

    if permutation not in ("iid", "block"):
        raise ValueError(f"permutation must be 'iid' or 'block', got {permutation!r}")

    # Set default n_shuffles based on method
    if n_shuffles is None:
        if method == "effect_size":
            n_shuffles = 5  # Fast heuristic
        else:  # permutation
            n_shuffles = 100  # Statistically rigorous

    # Handle strict mode: override n_shuffles for adequate statistical power
    effective_n_shuffles = n_shuffles
    if strict and method == "permutation":
        effective_n_shuffles = max(n_shuffles, 199)  # p-value resolution of 0.005
    elif method == "permutation" and n_shuffles < 19:
        # Warn about insufficient power with small n_shuffles in permutation mode
        # With n=19, min p-value is 1/20 = 0.05 (just at 5% level)
        warnings.warn(
            f"n_shuffles={n_shuffles} provides limited statistical power for permutation test. "
            f"Minimum achievable p-value is 1/{n_shuffles+1} ≈ {1/(n_shuffles+1):.3f}. "
            f"Use n_shuffles >= 19 for p < 0.05, or n_shuffles >= 99 for p < 0.01. "
            f"See Phipson & Smyth (2010) for permutation test power analysis.",
            UserWarning,
            stacklevel=2,
        )

    n = len(y)
    rng = np.random.default_rng(random_state)

    # Compute block size for block permutation
    if permutation == "block":
        if block_size == "auto":
            # Kunsch (1989) recommendation: n^(1/3)
            computed_block_size = max(1, int(round(n ** (1 / 3))))
        else:
            computed_block_size = int(block_size)
            if computed_block_size < 1:
                raise ValueError(f"block_size must be >= 1, got {block_size}")
    else:
        computed_block_size = 1  # Not used for IID

    def block_permute(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Block permutation: shuffle blocks to preserve local autocorrelation.

        [T1] Per Kunsch (1989): dividing data into non-overlapping blocks
        and permuting blocks preserves within-block dependence structure.
        """
        n_arr = len(arr)
        n_blocks = max(1, n_arr // computed_block_size)

        # Create block indices
        block_indices = []
        for i in range(n_blocks):
            start = i * computed_block_size
            end = min((i + 1) * computed_block_size, n_arr)
            block_indices.append(list(range(start, end)))

        # Handle remainder
        remainder_start = n_blocks * computed_block_size
        if remainder_start < n_arr:
            block_indices.append(list(range(remainder_start, n_arr)))

        # Shuffle block order
        rng.shuffle(block_indices)

        # Reconstruct permuted array
        permuted_indices = [idx for block in block_indices for idx in block]
        return arr[permuted_indices]

    # Compute out-of-sample MAE on real target using DRY helper
    result = _compute_cv_mae(model, X, y, n_cv_splits=n_cv_splits, return_errors=True)
    mae_real, errors_real = result  # type: ignore[misc]

    # Compute out-of-sample MAE on shuffled targets
    shuffled_maes: List[float] = []
    for _ in range(effective_n_shuffles):
        # Apply permutation strategy
        if permutation == "block":
            y_shuffled = block_permute(y, rng)
        else:
            y_shuffled = rng.permutation(y)

        # Use DRY helper (model cloning happens inside)
        shuffle_model = _clone_model(model)
        mae_shuffled = _compute_cv_mae(
            shuffle_model, X, y_shuffled, n_cv_splits=n_cv_splits
        )
        shuffled_maes.append(float(mae_shuffled))

    mae_shuffled_avg = float(np.mean(shuffled_maes))

    # Improvement ratio: positive = model beats shuffled (suspicious)
    if mae_shuffled_avg > 0:
        improvement_ratio = 1 - (mae_real / mae_shuffled_avg)
    else:
        warnings.warn(
            "Shuffled target MAE is zero - model achieves perfect predictions on shuffled data. "
            "This is highly unusual and indicates a degenerate case. "
            "improvement_ratio will be 0.0 (cannot compute meaningful comparison). "
            "Check: (1) constant target values, (2) data preprocessing issues, "
            "(3) model memorization.",
            UserWarning,
            stacklevel=2,
        )
        improvement_ratio = 0.0

    # Compute p-value for permutation test mode
    # Per Phipson & Smyth (2010): p = (1 + count(shuffled <= observed)) / (1 + n)
    # A lower model MAE = better, so we count shuffled MAEs that are <= model MAE
    # (i.e., how often shuffled model does at least as well as real model)
    n_shuffled_at_least_as_good = sum(1 for s_mae in shuffled_maes if s_mae <= mae_real)
    pvalue = (1 + n_shuffled_at_least_as_good) / (1 + effective_n_shuffles)

    details = {
        "mae_real": mae_real,
        "mae_shuffled_avg": mae_shuffled_avg,
        "mae_shuffled_all": shuffled_maes,
        "improvement_ratio": improvement_ratio,
        "pvalue": pvalue,
        "method": method,
        "n_shuffles": n_shuffles,
        "n_shuffles_effective": effective_n_shuffles,
        "strict": strict,
        "min_pvalue": 1 / (effective_n_shuffles + 1),
        "n_cv_splits": n_cv_splits,
        "evaluation_method": "walk_forward_cv",
        "permutation": permutation,
        "block_size": computed_block_size if permutation == "block" else None,
    }

    # Compute bootstrap CI for MAE if requested
    if bootstrap_ci and len(errors_real) >= 2:
        from temporalcv.inference.block_bootstrap_ci import bootstrap_ci_mae

        ci_result = bootstrap_ci_mae(
            errors_real,
            n_bootstrap=n_bootstrap,
            block_length=bootstrap_block_length,
            alpha=bootstrap_alpha,
            random_state=random_state,
        )
        details.update({
            "ci_lower": ci_result.ci_lower,
            "ci_upper": ci_result.ci_upper,
            "ci_alpha": ci_result.alpha,
            "bootstrap_std": ci_result.std_error,
            "n_bootstrap": ci_result.n_bootstrap,
            "bootstrap_block_length": ci_result.block_length,
        })

    # Decision logic depends on method
    if method == "permutation":
        # True permutation test: HALT if p-value < alpha
        # Low p-value means model reliably beats shuffled (has signal)
        if pvalue < alpha:
            return GateResult(
                name="signal_verification",
                status=GateStatus.HALT,
                message=f"Permutation test: p={pvalue:.4f} < α={alpha} (model has signal)",
                metric_value=pvalue,
                threshold=alpha,
                details=details,
                recommendation=(
                    "Model has predictive signal. Investigate source: "
                    "legitimate temporal patterns (expected for AR models) "
                    "OR data leakage (check feature engineering)."
                ),
            )
        return GateResult(
            name="signal_verification",
            status=GateStatus.PASS,
            message=f"Permutation test: p={pvalue:.4f} >= α={alpha} (no significant signal)",
            metric_value=pvalue,
            threshold=alpha,
            details=details,
            recommendation="Model shows no predictive signal. Check feature relevance.",
        )
    else:
        # Effect size mode: HALT if improvement_ratio > threshold
        if improvement_ratio > threshold:
            return GateResult(
                name="signal_verification",
                status=GateStatus.HALT,
                message=f"Model beats shuffled by {improvement_ratio:.1%} (has signal)",
                metric_value=improvement_ratio,
                threshold=threshold,
                details=details,
                recommendation=(
                    "Model has predictive signal. Investigate source: "
                    "legitimate temporal patterns (expected for AR models) "
                    "OR data leakage (check feature engineering)."
                ),
            )
        return GateResult(
            name="signal_verification",
            status=GateStatus.PASS,
            message=f"Model improvement {improvement_ratio:.1%} shows no significant signal",
            metric_value=improvement_ratio,
            threshold=threshold,
            details=details,
            recommendation="Model shows no predictive signal. Check feature relevance.",
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
    *,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 100,
    bootstrap_alpha: float = 0.05,
    bootstrap_block_length: Union[int, Literal["auto"]] = "auto",
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
    bootstrap_ci : bool, default=False
        If True, compute block bootstrap confidence interval for MAE.
        Results added to details dict as ci_lower, ci_upper, etc.
    n_bootstrap : int, default=100
        Number of bootstrap replications for CI.
    bootstrap_alpha : float, default=0.05
        Significance level for CI (1 - alpha CI).
    bootstrap_block_length : int or "auto", default="auto"
        Block length for bootstrap. "auto" uses n^(1/3) per Kunsch (1989).

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

    Complexity: O(n_cv_splits × model_fit_time)

    See Also
    --------
    gate_signal_verification : Signal verification via permutation test.
    gate_theoretical_bounds : Test against estimated AR(1) from real data.
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

    # Compute out-of-sample MAE using DRY helper
    result = _compute_cv_mae(model, X, y, n_cv_splits=n_cv_splits, return_errors=True)
    model_mae, errors_arr = result  # type: ignore[misc]

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

    # Compute bootstrap CI for MAE if requested
    if bootstrap_ci and len(errors_arr) >= 2:
        from temporalcv.inference.block_bootstrap_ci import bootstrap_ci_mae

        ci_result = bootstrap_ci_mae(
            errors_arr,
            n_bootstrap=n_bootstrap,
            block_length=bootstrap_block_length,
            alpha=bootstrap_alpha,
            random_state=random_state,
        )
        details.update({
            "ci_lower": ci_result.ci_lower,
            "ci_upper": ci_result.ci_upper,
            "ci_alpha": ci_result.alpha,
            "bootstrap_std": ci_result.std_error,
            "n_bootstrap": ci_result.n_bootstrap,
            "bootstrap_block_length": ci_result.block_length,
        })

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

    See Also
    --------
    gate_signal_verification : Signal verification if improvement seems too good.
    gate_theoretical_bounds : Check against AR(1) theoretical minimum.
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
    extra_gap: int = 0,
) -> GateResult:
    """
    Verify temporal boundary enforcement.

    Ensures proper separation between training end and test start for h-step forecasts.

    Parameters
    ----------
    train_end_idx : int
        Last index of training data (inclusive)
    test_start_idx : int
        First index of test data
    horizon : int
        Forecast horizon (h)
    extra_gap : int, default=0
        Additional separation beyond horizon requirement

    Returns
    -------
    GateResult
        HALT if temporal boundary is violated

    Notes
    -----
    For h-step ahead forecasting, the last training observation should be
    at least h periods before the first test observation to prevent leakage.

    Required: test_start_idx >= train_end_idx + horizon + extra_gap

    See Also
    --------
    WalkForwardCV : CV class that enforces separation automatically.
    """
    required_gap = horizon + extra_gap
    actual_gap = test_start_idx - train_end_idx - 1

    details = {
        "train_end_idx": train_end_idx,
        "test_start_idx": test_start_idx,
        "horizon": horizon,
        "extra_gap": extra_gap,
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
            recommendation=f"Increase separation between train and test. Need {required_gap - actual_gap} more periods.",
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
        return cast(np.ndarray, np.zeros(max_lag))

    acf_values = np.zeros(max_lag)
    for k in range(1, max_lag + 1):
        gamma_k = np.sum(x[k:] * x[:-k]) / n
        acf_values[k - 1] = gamma_k / gamma_0

    return cast(np.ndarray, acf_values)


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

    See Also
    --------
    gate_signal_verification : Run before residual diagnostics for signal check.
    dm_test : Uses similar autocorrelation concepts via HAC variance.
    """
    from scipy import stats

    residuals = np.asarray(residuals)

    # Validate no NaN values
    if np.any(np.isnan(residuals)):
        raise ValueError(
            "residuals contains NaN values. Clean data before processing. "
            "Use np.nan_to_num() or dropna() to handle missing values."
        )

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
# Theoretical Bounds Gate
# =============================================================================


def gate_theoretical_bounds(
    model_mae: float,
    y_train: ArrayLike,
    tolerance: float = 0.10,
) -> GateResult:
    """
    Check if model MAE violates theoretical minimum for estimated AR(1).

    This gate estimates the AR(1) autocorrelation from training data and
    computes the theoretical minimum MAE achievable by any predictor. If
    the model's out-of-sample MAE is suspiciously below this minimum, it
    indicates potential data leakage or evaluation error.

    Knowledge Tier: [T1] - Based on standard AR(1) theory and E[|Z|] = σ√(2/π)
    for Z ~ N(0, σ²).

    Parameters
    ----------
    model_mae : float
        Model's out-of-sample MAE (must be computed on held-out data)
    y_train : ArrayLike
        Training target series used for estimating AR(1) parameters.
        Requires n >= 30 samples for reliable estimation.
    tolerance : float, default=0.10
        How much below theoretical minimum triggers HALT.
        0.10 means HALT if model_mae < 0.90 * theoretical_mae.

    Returns
    -------
    GateResult
        - PASS: model_mae >= theoretical_mae * (1 - tolerance)
        - HALT: model_mae < theoretical_mae * (1 - tolerance)
        - SKIP: insufficient data (n < 30)

        Check details["ar1_assumption_warning"] if True - indicates
        AR(1) residuals show autocorrelation (higher-order dynamics).

    Notes
    -----
    Theory:
    1. Estimate φ = ACF(1) of y_train
    2. Compute AR(1) residuals: ε_t = y_t - φ * y_{t-1}
    3. σ_innovation = std(ε_t)
    4. theoretical_mae = σ_innovation * √(2/π) ≈ 0.7979 * σ_innovation

    This is the expected absolute error for optimal prediction under AR(1).

    References
    ----------
    [T1] For Z ~ N(0, σ²): E[|Z|] = σ * √(2/π) ≈ 0.7979 * σ

    Example
    -------
    >>> # After computing out-of-sample MAE
    >>> result = gate_theoretical_bounds(model_mae=0.15, y_train=y_train)
    >>> if result.status == GateStatus.HALT:
    ...     print("Model MAE beats theoretical minimum - investigate!")

    See Also
    --------
    gate_synthetic_ar1 : Test on synthetic AR(1) with known parameters.
    gate_suspicious_improvement : Complementary improvement check.
    """
    y = np.asarray(y_train, dtype=np.float64)
    n = len(y)

    # Minimum sample size for reliable ACF estimation
    if n < 30:
        return GateResult(
            name="theoretical_bounds",
            status=GateStatus.SKIP,
            message=f"Insufficient data for AR(1) estimation (n={n}, need 30)",
            metric_value=model_mae,
            threshold=np.nan,
            details={"n_samples": n, "min_required": 30},
            recommendation="Collect more training data for theoretical bounds check.",
        )

    # Estimate phi from ACF(1)
    acf = _compute_acf(y, max_lag=1)
    phi = acf[0] if len(acf) > 0 else 0.0

    # Compute AR(1) residuals (innovation terms)
    y_lagged = y[:-1]
    y_current = y[1:]
    innovations = y_current - phi * y_lagged

    # Innovation standard deviation
    sigma_innovation = float(np.std(innovations, ddof=1))

    # Theoretical minimum MAE: E[|N(0,σ²)|] = σ * sqrt(2/π)
    theoretical_mae = sigma_innovation * np.sqrt(2.0 / np.pi)

    # Threshold for HALT
    threshold = theoretical_mae * (1.0 - tolerance)

    # Check AR(1) assumption via Ljung-Box on residuals
    ar1_warning = False
    ar1_warning_message = None
    if len(innovations) >= 30:
        max_lag_for_test = min(10, len(innovations) // 3)
        if max_lag_for_test >= 1:
            _, lb_pvalue = _ljung_box_test(innovations, max_lag=max_lag_for_test)
            if lb_pvalue < 0.05:
                ar1_warning = True
                ar1_warning_message = (
                    f"AR(1) residuals show autocorrelation (Ljung-Box p={lb_pvalue:.4f}). "
                    "Series may have higher-order dynamics; theoretical bound may not apply."
                )

    details = {
        "phi_estimate": float(phi),
        "sigma_innovation": sigma_innovation,
        "theoretical_mae": theoretical_mae,
        "threshold": threshold,
        "tolerance": tolerance,
        "n_samples": n,
        "ar1_assumption_warning": ar1_warning,
    }
    if ar1_warning_message:
        details["ar1_assumption_message"] = ar1_warning_message

    # Check if model beats theoretical minimum
    if model_mae < threshold:
        return GateResult(
            name="theoretical_bounds",
            status=GateStatus.HALT,
            message=(
                f"Model MAE ({model_mae:.4f}) beats theoretical minimum "
                f"({theoretical_mae:.4f}) by more than {tolerance:.0%} tolerance"
            ),
            metric_value=model_mae,
            threshold=threshold,
            details=details,
            recommendation=(
                "Model appears to beat impossible AR(1) bounds. Investigate for: "
                "(1) Data leakage in features, (2) In-sample evaluation error, "
                "(3) Target encoding issues. Re-verify train/test split."
            ),
        )

    # PASS - model MAE is plausible
    status_msg = "Model MAE is within theoretical bounds"
    if ar1_warning:
        status_msg += " (note: AR(1) assumption may not hold)"

    return GateResult(
        name="theoretical_bounds",
        status=GateStatus.PASS,
        message=status_msg,
        metric_value=model_mae,
        threshold=threshold,
        details=details,
        recommendation="",
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
    ...     gate_signal_verification(model, X, y),
    ...     gate_suspicious_improvement(model_mae, persistence_mae),
    ... ]
    >>> report = run_gates(results)
    >>> if report.status == "HALT":
    ...     print(report.summary())
    """
    return ValidationReport(gates=gates)


# =============================================================================
# Regime-Stratified Validation
# =============================================================================


@dataclass
class StratifiedValidationReport:
    """
    Validation report with regime stratification.

    Provides both overall gate results and per-regime breakdowns,
    exposing issues that aggregate metrics might hide.

    Attributes
    ----------
    overall : ValidationReport
        Gate results on full dataset
    by_regime : Dict[str, ValidationReport]
        Gate results per regime
    regime_counts : Dict[str, int]
        Sample counts per regime
    masked_regimes : List[str]
        Regimes with n < min_n (excluded from stratification)

    Knowledge Tier: [T2] - Regime-conditional evaluation from myga-forecasting-v4

    Notes
    -----
    Only numeric gates (gate_suspicious_improvement, gate_theoretical_bounds)
    are run per-regime. Gates requiring model fitting (shuffled_target,
    synthetic_ar1) are only run overall.
    """

    overall: ValidationReport
    by_regime: dict[str, ValidationReport]
    regime_counts: dict[str, int]
    masked_regimes: List[str]

    @property
    def status(self) -> str:
        """
        Overall status: HALT if any HALT, WARN if any WARN, else PASS.

        Checks both overall and per-regime results.
        """
        # Check overall
        if self.overall.status == "HALT":
            return "HALT"

        # Check per-regime
        for report in self.by_regime.values():
            if any(g.status == GateStatus.HALT for g in report.gates):
                return "HALT"

        # Check for warnings
        if self.overall.status == "WARN":
            return "WARN"
        for report in self.by_regime.values():
            if any(g.status == GateStatus.WARN for g in report.gates):
                return "WARN"

        return "PASS"

    def summary(self) -> str:
        """Return human-readable summary with regime breakdown."""
        lines = [
            "=" * 60,
            "STRATIFIED VALIDATION REPORT",
            "=" * 60,
            "",
            "OVERALL RESULTS:",
        ]

        for gate in self.overall.gates:
            lines.append(f"  {gate}")

        if self.by_regime:
            lines.append("")
            lines.append("-" * 60)
            lines.append("PER-REGIME RESULTS:")

            for regime, report in sorted(self.by_regime.items()):
                n = self.regime_counts.get(regime, 0)
                lines.append(f"  [{regime}] (n={n}):")
                for gate in report.gates:
                    lines.append(f"    {gate}")

        if self.masked_regimes:
            lines.append("")
            lines.append(f"MASKED REGIMES (insufficient data): {self.masked_regimes}")

        lines.extend([
            "",
            "=" * 60,
            f"OVERALL STATUS: {self.status}",
            "=" * 60,
        ])

        return "\n".join(lines)


def run_gates_stratified(
    overall_gates: List[GateResult],
    actuals: ArrayLike,
    predictions: ArrayLike,
    regimes: Optional[Union[np.ndarray, Literal["auto"]]] = None,
    min_n_per_regime: int = 10,
    volatility_window: int = 13,
    improvement_threshold: float = 0.20,
    warning_threshold: float = 0.10,
) -> StratifiedValidationReport:
    """
    Run validation gates overall + stratified by regime.

    Provides regime-conditional validation to expose issues hidden
    by aggregate metrics. Only numeric gates (suspicious_improvement,
    theoretical_bounds) are run per-regime.

    Knowledge Tier: [T2] - Regime-conditional evaluation from myga-forecasting-v4

    Parameters
    ----------
    overall_gates : List[GateResult]
        Pre-computed gate results for overall dataset
    actuals : ArrayLike
        Actual values (for regime classification and per-regime metrics)
    predictions : ArrayLike
        Model predictions (for per-regime metrics)
    regimes : array | "auto" | None, default=None
        Regime labels:
        - None: No stratification (returns overall only)
        - "auto": Auto-classify volatility regimes from actuals
        - array: Use provided regime labels
    min_n_per_regime : int, default=10
        Minimum samples per regime. Below this, regime is masked.
    volatility_window : int, default=13
        Window for auto volatility classification (13 weeks ~ 1 quarter)
    improvement_threshold : float, default=0.20
        Threshold for gate_suspicious_improvement HALT
    warning_threshold : float, default=0.10
        Threshold for gate_suspicious_improvement WARN

    Returns
    -------
    StratifiedValidationReport
        Overall + per-regime gate results

    Notes
    -----
    Regime stratification exposes issues that aggregate metrics hide:
    - Model may pass overall but fail in HIGH volatility regime
    - FLAT direction regime may have artificially good metrics
    - Per-regime sample sizes affect reliability

    Example
    -------
    >>> overall = [
    ...     gate_signal_verification(model, X, y),
    ...     gate_suspicious_improvement(model_mae, persistence_mae),
    ... ]
    >>> report = run_gates_stratified(
    ...     overall, actuals, predictions, regimes="auto"
    ... )
    >>> if report.status == "HALT":
    ...     print(report.summary())
    """
    from temporalcv.regimes import (
        classify_volatility_regime,
        get_regime_counts,
        mask_low_n_regimes,
    )

    actuals = np.asarray(actuals)
    predictions = np.asarray(predictions)

    # Create overall report
    overall_report = ValidationReport(gates=overall_gates)

    # If no stratification, return overall only
    if regimes is None:
        return StratifiedValidationReport(
            overall=overall_report,
            by_regime={},
            regime_counts={},
            masked_regimes=[],
        )

    # Get regime labels
    if isinstance(regimes, str) and regimes == "auto":
        regime_labels = classify_volatility_regime(
            actuals, window=volatility_window, basis="changes"
        )
    else:
        regime_labels = np.asarray(regimes)

    # Get counts and mask low-n regimes
    regime_counts = get_regime_counts(regime_labels)
    masked_labels = mask_low_n_regimes(
        regime_labels, min_n=min_n_per_regime, mask_value="MASKED"
    )

    # Identify masked regimes
    masked_regimes = [r for r, c in regime_counts.items() if c < min_n_per_regime]

    # Run numeric gates per-regime
    by_regime: dict[str, ValidationReport] = {}
    unique_regimes = [r for r in np.unique(masked_labels) if r != "MASKED"]

    for regime in unique_regimes:
        regime_mask = masked_labels == regime
        regime_actuals = actuals[regime_mask]
        regime_preds = predictions[regime_mask]

        # Compute per-regime metrics
        regime_model_mae = float(np.mean(np.abs(regime_actuals - regime_preds)))
        regime_persistence_mae = float(np.mean(np.abs(np.diff(regime_actuals))))

        # Run numeric gates
        regime_gates: List[GateResult] = []

        # gate_suspicious_improvement
        if regime_persistence_mae > 0:
            improvement_result = gate_suspicious_improvement(
                model_metric=regime_model_mae,
                baseline_metric=regime_persistence_mae,
                threshold=improvement_threshold,
                warn_threshold=warning_threshold,
            )
            # Add regime context to message
            improvement_result = GateResult(
                name=improvement_result.name,
                status=improvement_result.status,
                message=f"[{regime}] {improvement_result.message}",
                metric_value=improvement_result.metric_value,
                threshold=improvement_result.threshold,
                details={**improvement_result.details, "regime": regime},
                recommendation=improvement_result.recommendation,
            )
            regime_gates.append(improvement_result)

        # gate_theoretical_bounds (if enough data)
        if len(regime_actuals) >= 30:
            theoretical_result = gate_theoretical_bounds(
                model_mae=regime_model_mae,
                y_train=regime_actuals,
                tolerance=0.10,
            )
            theoretical_result = GateResult(
                name=theoretical_result.name,
                status=theoretical_result.status,
                message=f"[{regime}] {theoretical_result.message}",
                metric_value=theoretical_result.metric_value,
                threshold=theoretical_result.threshold,
                details={**theoretical_result.details, "regime": regime},
                recommendation=theoretical_result.recommendation,
            )
            regime_gates.append(theoretical_result)

        by_regime[regime] = ValidationReport(gates=regime_gates)

    return StratifiedValidationReport(
        overall=overall_report,
        by_regime=by_regime,
        regime_counts=regime_counts,
        masked_regimes=masked_regimes,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enums and dataclasses
    "GateStatus",
    "GateResult",
    "ValidationReport",
    "StratifiedValidationReport",
    # Gate functions
    "gate_signal_verification",
    "gate_synthetic_ar1",
    "gate_suspicious_improvement",
    "gate_temporal_boundary",
    "gate_residual_diagnostics",
    "gate_theoretical_bounds",
    # Runners
    "run_gates",
    "run_gates_stratified",
    # Internal helpers (exposed for testing)
    "_ljung_box_test",
]
