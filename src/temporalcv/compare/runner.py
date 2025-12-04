"""
Compare Runner Module.

Provides functions to run model comparisons and benchmark suites.

Example
-------
>>> from temporalcv.compare import run_comparison, run_benchmark_suite
>>> from temporalcv.compare.adapters import NaiveAdapter, StatsforecastAdapter
>>> from temporalcv.benchmarks import create_synthetic_dataset
>>>
>>> dataset = create_synthetic_dataset()
>>> adapters = [NaiveAdapter(), StatsforecastAdapter("AutoARIMA")]
>>> result = run_comparison(dataset, adapters)
>>> print(result.best_model)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from temporalcv.compare.base import (
    ComparisonReport,
    ComparisonResult,
    ForecastAdapter,
    ModelResult,
    compute_comparison_metrics,
)


# =============================================================================
# Dataset Protocol (for type checking)
# =============================================================================


class Dataset(Protocol):
    """Protocol for benchmark datasets."""

    @property
    def metadata(self) -> Any:
        """Return dataset metadata."""
        ...

    @property
    def values(self) -> np.ndarray:
        """Return dataset values."""
        ...

    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return train/test split."""
        ...


# =============================================================================
# Comparison Functions
# =============================================================================


def run_comparison(
    dataset: Dataset,
    adapters: List[ForecastAdapter],
    primary_metric: str = "mae",
    include_dm_test: bool = True,
) -> ComparisonResult:
    """
    Compare multiple models on a single dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to evaluate on (must have train/test split)
    adapters : list[ForecastAdapter]
        Model adapters to compare
    primary_metric : str, default="mae"
        Metric to use for ranking models
    include_dm_test : bool, default=True
        Whether to run Diebold-Mariano test between best and other models

    Returns
    -------
    ComparisonResult
        Comparison result with model rankings

    Raises
    ------
    ValueError
        If adapters list is empty or dataset cannot be split

    Example
    -------
    >>> from temporalcv.compare.adapters import NaiveAdapter
    >>> result = run_comparison(dataset, [NaiveAdapter()])
    >>> print(f"Best model: {result.best_model}")
    """
    if not adapters:
        raise ValueError("adapters list cannot be empty")

    # Get train/test split
    train, test = dataset.get_train_test_split()
    test_size = len(test) if test.ndim == 1 else test.shape[-1]
    horizon = dataset.metadata.horizon

    # Run each adapter
    model_results: List[ModelResult] = []

    for adapter in adapters:
        start_time = time.perf_counter()

        try:
            predictions = adapter.fit_predict(
                train_values=train,
                test_size=test_size,
                horizon=horizon,
            )
        except Exception as e:
            # Log error but continue with other models
            print(f"Warning: {adapter.model_name} failed: {e}")
            continue

        elapsed = time.perf_counter() - start_time

        # Flatten test for comparison if multi-series
        test_flat = test.flatten() if test.ndim > 1 else test
        pred_flat = predictions.flatten() if predictions.ndim > 1 else predictions

        # Validate prediction/test alignment
        if len(test_flat) != len(pred_flat):
            raise ValueError(
                f"{adapter.model_name} returned {len(pred_flat)} predictions, "
                f"expected {len(test_flat)} (test_size). "
                f"Check adapter implementation."
            )

        # Compute metrics
        metrics = compute_comparison_metrics(pred_flat, test_flat)

        model_results.append(
            ModelResult(
                model_name=adapter.model_name,
                package=adapter.package_name,
                metrics=metrics,
                predictions=predictions,
                runtime_seconds=elapsed,
                model_params=adapter.get_params(),
            )
        )

    if not model_results:
        raise ValueError("All adapters failed to produce results")

    # Build comparison result
    result = ComparisonResult(
        dataset_name=dataset.metadata.name,
        models=model_results,
        primary_metric=primary_metric,
    )

    # Add DM test results if requested
    if include_dm_test and len(model_results) > 1:
        result.statistical_tests = _run_dm_tests(
            model_results, test, result.best_model, horizon=horizon
        )

    return result


def _run_dm_tests(
    model_results: List[ModelResult],
    test: np.ndarray,
    best_model: str,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Run Diebold-Mariano tests comparing best model to others.

    Parameters
    ----------
    model_results : list[ModelResult]
        All model results
    test : np.ndarray
        Actual test values
    best_model : str
        Name of best model
    horizon : int, default=1
        Forecast horizon (for HAC variance adjustment)

    Returns
    -------
    dict
        DM test results for each comparison
    """
    try:
        from temporalcv.statistical_tests import dm_test
    except ImportError:
        return {"error": "statistical_tests module not available"}

    # Find best model predictions
    best_preds = None
    for result in model_results:
        if result.model_name == best_model:
            best_preds = result.predictions
            break

    if best_preds is None:
        return {"error": f"Best model {best_model} not found"}

    # Flatten arrays
    test_flat = test.flatten() if test.ndim > 1 else test
    best_flat = best_preds.flatten() if best_preds.ndim > 1 else best_preds

    dm_results: Dict[str, Any] = {}

    for result in model_results:
        if result.model_name == best_model:
            continue

        other_preds = result.predictions
        other_flat = other_preds.flatten() if other_preds.ndim > 1 else other_preds

        # Align lengths
        min_len = min(len(test_flat), len(best_flat), len(other_flat))

        try:
            # Errors = actual - predicted (positive means underprediction)
            dm_result = dm_test(
                errors1=test_flat[:min_len] - best_flat[:min_len],
                errors2=test_flat[:min_len] - other_flat[:min_len],
                horizon=horizon,
                loss="absolute",
            )
            dm_results[result.model_name] = {
                "statistic": dm_result.statistic,
                "p_value": dm_result.pvalue,
                "significant": dm_result.pvalue < 0.05,
            }
        except Exception as e:
            dm_results[result.model_name] = {"error": str(e)}

    return dm_results


def run_benchmark_suite(
    datasets: List[Dataset],
    adapters: List[ForecastAdapter],
    primary_metric: str = "mae",
    include_dm_test: bool = True,
) -> ComparisonReport:
    """
    Run model comparison across multiple datasets.

    Parameters
    ----------
    datasets : list[Dataset]
        Datasets to evaluate
    adapters : list[ForecastAdapter]
        Model adapters to compare
    primary_metric : str, default="mae"
        Metric to use for ranking
    include_dm_test : bool, default=True
        Whether to run statistical tests

    Returns
    -------
    ComparisonReport
        Full comparison report with summary

    Example
    -------
    >>> from temporalcv.benchmarks import create_synthetic_dataset
    >>> datasets = [create_synthetic_dataset(seed=i) for i in range(3)]
    >>> report = run_benchmark_suite(datasets, [NaiveAdapter()])
    >>> print(report.to_markdown())
    """
    if not datasets:
        raise ValueError("datasets list cannot be empty")
    if not adapters:
        raise ValueError("adapters list cannot be empty")

    results: List[ComparisonResult] = []

    for dataset in datasets:
        try:
            result = run_comparison(
                dataset=dataset,
                adapters=adapters,
                primary_metric=primary_metric,
                include_dm_test=include_dm_test,
            )
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed on dataset {dataset.metadata.name}: {e}")
            continue

    if not results:
        raise ValueError("All datasets failed to produce results")

    return ComparisonReport(results=results)


def compare_to_baseline(
    dataset: Dataset,
    adapter: ForecastAdapter,
    baseline_adapter: Optional[ForecastAdapter] = None,
    primary_metric: str = "mae",
) -> Dict[str, Any]:
    """
    Compare a single model to a baseline.

    Parameters
    ----------
    dataset : Dataset
        Dataset to evaluate
    adapter : ForecastAdapter
        Model adapter to evaluate
    baseline_adapter : ForecastAdapter, optional
        Baseline to compare against. Default: NaiveAdapter
    primary_metric : str, default="mae"
        Metric to use for comparison

    Returns
    -------
    dict
        Comparison results with improvement percentage

    Example
    -------
    >>> result = compare_to_baseline(dataset, my_adapter)
    >>> print(f"Improvement: {result['improvement_pct']:.1f}%")
    """
    from temporalcv.compare.base import NaiveAdapter

    if baseline_adapter is None:
        baseline_adapter = NaiveAdapter()

    comparison = run_comparison(
        dataset=dataset,
        adapters=[baseline_adapter, adapter],
        primary_metric=primary_metric,
        include_dm_test=True,
    )

    # Extract metrics
    baseline_result = None
    model_result = None

    for result in comparison.models:
        if result.model_name == baseline_adapter.model_name:
            baseline_result = result
        elif result.model_name == adapter.model_name:
            model_result = result

    if baseline_result is None or model_result is None:
        raise ValueError("Could not find both baseline and model results")

    baseline_metric = baseline_result.get_metric(primary_metric)
    model_metric = model_result.get_metric(primary_metric)

    # Compute improvement (negative is better for error metrics)
    if baseline_metric != 0:
        improvement_pct = (baseline_metric - model_metric) / baseline_metric * 100
    else:
        improvement_pct = 0.0 if model_metric == 0 else float("inf")

    return {
        "model_name": adapter.model_name,
        "baseline_name": baseline_adapter.model_name,
        f"model_{primary_metric}": model_metric,
        f"baseline_{primary_metric}": baseline_metric,
        "improvement_pct": improvement_pct,
        "model_is_better": model_metric < baseline_metric,
        "statistical_tests": comparison.statistical_tests,
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "run_comparison",
    "run_benchmark_suite",
    "compare_to_baseline",
]
