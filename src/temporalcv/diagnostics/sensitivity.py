"""
Gap sensitivity analysis for temporal cross-validation.

Assesses how sensitive model performance is to the gap parameter,
helping detect potential data leakage and understand model robustness.

Knowledge Tier: [T2] - Gap sensitivity is empirical best practice in
time series validation; theoretical guidance on "correct" gap is domain-specific.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from temporalcv.cv import WalkForwardCV


@dataclass(frozen=True)
class GapSensitivityResult:
    """
    Result of gap sensitivity analysis.

    Attributes
    ----------
    gap_values : np.ndarray
        Tested gap values
    metrics : np.ndarray
        Metric value (e.g., MAE) at each gap
    metric_name : str
        Name of the metric used ("mae", "rmse", or "mse")
    break_even_gap : int | None
        First gap where performance degrades > degradation_threshold.
        None if no significant degradation detected.
    sensitivity_score : float
        Coefficient of variation: std(metrics) / mean(metrics).
        Higher values indicate more sensitivity to gap parameter.
    degradation_threshold : float
        Threshold used for break-even detection (for reference)
    baseline_metric : float
        Metric value at the baseline gap (typically gap=0)
    baseline_gap : int
        The gap value used as baseline

    Notes
    -----
    **Interpretation**:
    - High sensitivity_score (> 0.1): Performance varies substantially with gap
    - Low break_even_gap: Model may be exploiting near-term information
    - No break_even_gap: Model is robust to gap choices (or gap range too small)

    **Warning signs**:
    - Sharp improvement at gap=0 vs gap>0 suggests possible leakage
    - Monotonically decreasing performance as gap increases is expected
    - Non-monotonic patterns may indicate data irregularities
    """

    gap_values: np.ndarray
    metrics: np.ndarray
    metric_name: str
    break_even_gap: Optional[int]
    sensitivity_score: float
    degradation_threshold: float
    baseline_metric: float
    baseline_gap: int


def gap_sensitivity_analysis(
    model: Any,  # sklearn-compatible with fit/predict
    X: ArrayLike,
    y: ArrayLike,
    gap_range: Sequence[int] = range(0, 10),
    n_splits: int = 5,
    metric: Literal["mae", "rmse", "mse"] = "mae",
    window_type: Literal["expanding", "sliding"] = "expanding",
    window_size: Optional[int] = None,
    degradation_threshold: float = 0.10,
) -> GapSensitivityResult:
    """
    Analyze how model performance changes with gap parameter.

    Runs walk-forward cross-validation at multiple gap values to assess
    sensitivity. This helps detect potential leakage (sharp improvement
    at gap=0) and understand model robustness to temporal separation.

    Knowledge Tier: [T2] - Empirical best practice for time series validation.

    Parameters
    ----------
    model : sklearn estimator
        Model with fit(X, y) and predict(X) methods. Must be cloneable
        (support sklearn's clone or have __init__ that accepts same params).
    X : ArrayLike
        Features array of shape (n_samples, n_features)
    y : ArrayLike
        Target array of shape (n_samples,)
    gap_range : Sequence[int], default=range(0, 10)
        Gap values to test. Each gap is the number of samples between
        training end and test start.
    n_splits : int, default=5
        Number of CV splits at each gap value
    metric : {"mae", "rmse", "mse"}, default="mae"
        Evaluation metric:
        - "mae": Mean Absolute Error
        - "rmse": Root Mean Squared Error
        - "mse": Mean Squared Error
    window_type : {"expanding", "sliding"}, default="expanding"
        Type of training window for WalkForwardCV
    window_size : int, optional
        Training window size. Required for sliding window.
    degradation_threshold : float, default=0.10
        Fraction above baseline metric that triggers break-even gap.
        0.10 = 10% degradation from baseline (gap=0 or min gap).

    Returns
    -------
    GapSensitivityResult
        Contains metrics at each gap and sensitivity statistics.

    Notes
    -----
    The function uses fresh model instances at each gap to avoid state
    contamination. For models with random state, results may vary slightly.

    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> result = gap_sensitivity_analysis(
    ...     Ridge(alpha=1.0), X, y,
    ...     gap_range=range(0, 8),
    ...     metric="mae"
    ... )
    >>> print(f"Break-even gap: {result.break_even_gap}")
    >>> print(f"Sensitivity score: {result.sensitivity_score:.3f}")
    >>> # Plot metrics vs gap
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result.gap_values, result.metrics)
    >>> plt.xlabel("Gap"); plt.ylabel("MAE")

    Complexity: O(|gap_range| × n_splits × model_fit_time)

    See Also
    --------
    WalkForwardCV : CV strategy used internally.
    compute_dm_influence : Complementary influence diagnostics.
    gate_temporal_boundary : Verify gap meets requirements.
    """
    from sklearn.base import clone

    X = np.asarray(X)
    y = np.asarray(y)
    gap_values = np.array(list(gap_range))

    if len(gap_values) == 0:
        raise ValueError("gap_range must contain at least one value")

    n_samples = len(y)
    metrics_list = []

    for gap in gap_values:
        try:
            cv = WalkForwardCV(
                n_splits=n_splits,
                window_type=window_type,
                window_size=window_size,
                extra_gap=gap,
                test_size=1,
            )

            # Verify we can create valid splits
            actual_splits = cv.get_n_splits(X)
            if actual_splits == 0:
                # Not enough data for this gap
                metrics_list.append(np.nan)
                continue

            # Collect predictions across folds
            errors = []
            for train_idx, test_idx in cv.split(X):
                # Clone model for fresh state
                try:
                    model_clone = clone(model)
                except TypeError:
                    # If clone fails, use model directly (risk of state contamination)
                    model_clone = model

                model_clone.fit(X[train_idx], y[train_idx])
                preds = model_clone.predict(X[test_idx])

                fold_errors = y[test_idx] - preds
                errors.extend(fold_errors)

            errors = np.array(errors)

            # Compute metric
            if metric == "mae":
                metric_value = float(np.mean(np.abs(errors)))
            elif metric == "rmse":
                metric_value = float(np.sqrt(np.mean(errors**2)))
            else:  # mse
                metric_value = float(np.mean(errors**2))

            metrics_list.append(metric_value)

        except ValueError:
            # Invalid CV configuration for this gap
            metrics_list.append(np.nan)

    metrics = np.array(metrics_list)

    # Find baseline (first non-NaN metric)
    valid_mask = ~np.isnan(metrics)
    if not np.any(valid_mask):
        raise ValueError("Could not compute metrics for any gap value")

    first_valid_idx = np.argmax(valid_mask)
    baseline_gap = int(gap_values[first_valid_idx])
    baseline_metric = float(metrics[first_valid_idx])

    # Find break-even gap
    break_even_gap: Optional[int] = None
    threshold_metric = baseline_metric * (1 + degradation_threshold)

    for i, (gap, m) in enumerate(zip(gap_values, metrics)):
        if i <= first_valid_idx:
            continue  # Skip up to and including baseline
        if not np.isnan(m) and m > threshold_metric:
            break_even_gap = int(gap)
            break

    # Compute sensitivity score (coefficient of variation)
    valid_metrics = metrics[valid_mask]
    if len(valid_metrics) > 1 and np.mean(valid_metrics) > 0:
        sensitivity_score = float(np.std(valid_metrics) / np.mean(valid_metrics))
    else:
        sensitivity_score = 0.0

    return GapSensitivityResult(
        gap_values=gap_values,
        metrics=metrics,
        metric_name=metric,
        break_even_gap=break_even_gap,
        sensitivity_score=sensitivity_score,
        degradation_threshold=degradation_threshold,
        baseline_metric=baseline_metric,
        baseline_gap=baseline_gap,
    )
