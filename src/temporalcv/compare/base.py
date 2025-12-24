"""
Compare Module Base Classes.

Provides dataclasses for model comparison results and adapters.

Example
-------
>>> from temporalcv.compare import ModelResult, ComparisonResult
>>> result = ModelResult(
...     model_name="ARIMA",
...     package="statsforecast",
...     metrics={"mae": 0.5, "rmse": 0.7},
...     predictions=np.array([1.0, 2.0, 3.0]),
...     runtime_seconds=1.5
... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class ModelResult:
    """
    Result from a single model run.

    Attributes
    ----------
    model_name : str
        Name of the model (e.g., "AutoARIMA", "ETS")
    package : str
        Package that provided the model (e.g., "statsforecast", "sktime")
    metrics : dict[str, float]
        Dictionary of metric name -> value
    predictions : np.ndarray
        Predicted values
    runtime_seconds : float
        Time taken for fit + predict
    model_params : dict, optional
        Model hyperparameters used
    """

    model_name: str
    package: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    runtime_seconds: float
    model_params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate result."""
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.package:
            raise ValueError("package cannot be empty")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds cannot be negative")

    def get_metric(self, name: str) -> float:
        """
        Get metric by name.

        Parameters
        ----------
        name : str
            Metric name (case-insensitive)

        Returns
        -------
        float
            Metric value

        Raises
        ------
        KeyError
            If metric not found
        """
        # Try exact match first
        if name in self.metrics:
            return self.metrics[name]
        # Try case-insensitive
        name_lower = name.lower()
        for key, value in self.metrics.items():
            if key.lower() == name_lower:
                return value
        raise KeyError(f"Metric '{name}' not found. Available: {list(self.metrics.keys())}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "package": self.package,
            "metrics": self.metrics,
            "runtime_seconds": self.runtime_seconds,
            "model_params": self.model_params,
        }


@dataclass
class ComparisonResult:
    """
    Result from comparing multiple models on a single dataset.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset used
    models : list[ModelResult]
        Results from each model
    primary_metric : str
        Metric used for ranking (e.g., "mae")
    best_model : str
        Name of the best model by primary metric
    statistical_tests : dict, optional
        Results of statistical tests (DM test, etc.)
    """

    dataset_name: str
    models: List[ModelResult]
    primary_metric: str
    best_model: str = field(init=False)
    statistical_tests: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate and compute best model."""
        if not self.models:
            raise ValueError("models list cannot be empty")
        if not self.primary_metric:
            raise ValueError("primary_metric cannot be empty")

        # Compute best model (lowest metric value)
        best_value = float("inf")
        best_name = ""
        for model in self.models:
            try:
                value = model.get_metric(self.primary_metric)
                if value < best_value:
                    best_value = value
                    best_name = model.model_name
            except KeyError:
                continue

        if not best_name:
            raise ValueError(
                f"No model has metric '{self.primary_metric}'. "
                f"Available metrics: {self._get_all_metrics()}"
            )
        self.best_model = best_name

    def _get_all_metrics(self) -> List[str]:
        """Get all unique metric names across models."""
        metrics: set[str] = set()
        for model in self.models:
            metrics.update(model.metrics.keys())
        return sorted(metrics)

    def get_ranking(self, metric: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get models ranked by metric (ascending).

        Parameters
        ----------
        metric : str, optional
            Metric to rank by. Default: primary_metric

        Returns
        -------
        list[tuple[str, float]]
            List of (model_name, metric_value) sorted ascending
        """
        metric = metric or self.primary_metric
        results: List[Tuple[str, float]] = []
        for model in self.models:
            try:
                value = model.get_metric(metric)
                results.append((model.model_name, value))
            except KeyError:
                continue
        return sorted(results, key=lambda x: x[1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "models": [m.to_dict() for m in self.models],
            "primary_metric": self.primary_metric,
            "best_model": self.best_model,
            "statistical_tests": self.statistical_tests,
        }


@dataclass
class ComparisonReport:
    """
    Report from comparing models across multiple datasets.

    Attributes
    ----------
    results : list[ComparisonResult]
        Results from each dataset
    summary : dict
        Aggregate summary statistics
    """

    results: List[ComparisonResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Compute summary if not provided."""
        if not self.summary and self.results:
            self.summary = self._compute_summary()

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary."""
        # Count wins per model
        wins: Dict[str, int] = {}
        for result in self.results:
            wins[result.best_model] = wins.get(result.best_model, 0) + 1

        # Average metrics per model
        avg_metrics: Dict[str, Dict[str, List[float]]] = {}
        for result in self.results:
            for model in result.models:
                if model.model_name not in avg_metrics:
                    avg_metrics[model.model_name] = {}
                for metric_name, value in model.metrics.items():
                    if metric_name not in avg_metrics[model.model_name]:
                        avg_metrics[model.model_name][metric_name] = []
                    avg_metrics[model.model_name][metric_name].append(value)

        # Convert to means
        mean_metrics: Dict[str, Dict[str, float]] = {}
        for model_name, metrics in avg_metrics.items():
            mean_metrics[model_name] = {
                metric: float(np.mean(values)) for metric, values in metrics.items()
            }

        return {
            "n_datasets": len(self.results),
            "wins_by_model": wins,
            "mean_metrics_by_model": mean_metrics,
        }

    def to_markdown(self) -> str:
        """
        Generate markdown report.

        Returns
        -------
        str
            Markdown-formatted report
        """
        lines: List[str] = []
        lines.append("# Model Comparison Report\n")

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- Datasets evaluated: {self.summary.get('n_datasets', 0)}")

        if "wins_by_model" in self.summary:
            lines.append("\n### Model Wins\n")
            lines.append("| Model | Wins |")
            lines.append("|-------|------|")
            for model, wins in sorted(
                self.summary["wins_by_model"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"| {model} | {wins} |")

        # Per-dataset results
        lines.append("\n## Per-Dataset Results\n")
        for result in self.results:
            lines.append(f"### {result.dataset_name}\n")
            lines.append(f"Best model: **{result.best_model}**\n")

            # Ranking table
            ranking = result.get_ranking()
            if ranking:
                lines.append(f"| Model | {result.primary_metric.upper()} |")
                lines.append("|-------|------|")
                for model_name, value in ranking:
                    lines.append(f"| {model_name} | {value:.4f} |")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Adapter Base Class
# =============================================================================


class ForecastAdapter(ABC):
    """
    Abstract base class for model adapters.

    Adapters wrap different forecasting packages (statsforecast, sktime, etc.)
    to provide a unified interface for comparison.

    Example
    -------
    >>> class MyAdapter(ForecastAdapter):
    ...     def fit_predict(self, train, test_size, horizon):
    ...         # Your implementation
    ...         return predictions
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    @abstractmethod
    def package_name(self) -> str:
        """Return the package name."""
        ...

    @abstractmethod
    def fit_predict(
        self,
        train_values: np.ndarray,
        test_size: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Fit model and generate predictions.

        Parameters
        ----------
        train_values : np.ndarray
            Training data (1D or 2D array)
        test_size : int
            Number of test periods to predict
        horizon : int
            Forecast horizon (may differ from test_size for rolling forecasts)

        Returns
        -------
        np.ndarray
            Predictions with shape matching test_size
        """
        ...

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns
        -------
        dict
            Model hyperparameters
        """
        return {}


class NaiveAdapter(ForecastAdapter):
    """
    Naive persistence baseline adapter.

    Predicts the last observed value for all future periods.
    """

    @property
    def model_name(self) -> str:
        """Return model name."""
        return "Naive"

    @property
    def package_name(self) -> str:
        """Return package name."""
        return "temporalcv"

    def fit_predict(
        self,
        train_values: np.ndarray,
        test_size: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Generate naive forecasts.

        Parameters
        ----------
        train_values : np.ndarray
            Training data
        test_size : int
            Number of predictions needed
        horizon : int
            Not used for naive (always uses last value)

        Returns
        -------
        np.ndarray
            Array of last training value repeated
        """
        if train_values.ndim > 1:
            # Multi-series: use last value per series
            last_values = train_values[:, -1]
            return cast(np.ndarray, np.tile(last_values.reshape(-1, 1), (1, test_size)))
        else:
            last_value = train_values[-1]
            return cast(np.ndarray, np.full(test_size, last_value))


class SeasonalNaiveAdapter(ForecastAdapter):
    """
    Seasonal naive baseline adapter.

    Predicts using the value from the same period in the last season.
    """

    def __init__(self, season_length: int = 52):
        """
        Initialize seasonal naive adapter.

        Parameters
        ----------
        season_length : int, default=52
            Length of seasonal period (52 for weekly with yearly seasonality)
        """
        self.season_length = season_length

    @property
    def model_name(self) -> str:
        """Return model name."""
        return f"SeasonalNaive_{self.season_length}"

    @property
    def package_name(self) -> str:
        """Return package name."""
        return "temporalcv"

    def fit_predict(
        self,
        train_values: np.ndarray,
        test_size: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Generate seasonal naive forecasts.

        Parameters
        ----------
        train_values : np.ndarray
            Training data (1D for single series, 2D shape (n_series, n_obs) for multi)
        test_size : int
            Number of predictions needed
        horizon : int
            Not directly used

        Returns
        -------
        np.ndarray
            Predictions using seasonal lag. Shape (test_size,) for single series,
            or (n_series, test_size) for multi-series.
        """
        if train_values.ndim > 1:
            # Multi-series: apply seasonal naive to each series independently
            n_series = train_values.shape[0]
            predictions = np.zeros((n_series, test_size))
            for s in range(n_series):
                predictions[s] = self._predict_single(train_values[s], test_size)
            return cast(np.ndarray, predictions)

        return self._predict_single(train_values, test_size)

    def _predict_single(self, train: np.ndarray, test_size: int) -> np.ndarray:
        """Generate seasonal naive forecasts for a single series."""
        n_train = len(train)
        predictions = np.empty(test_size)

        for i in range(test_size):
            # Index into training data, wrapping if needed
            lag_idx = n_train - self.season_length + (i % self.season_length)
            if lag_idx < 0:
                lag_idx = n_train - 1  # Fallback to last value
            predictions[i] = train[lag_idx]

        return cast(np.ndarray, predictions)

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {"season_length": self.season_length}


# =============================================================================
# Metric Functions
# =============================================================================


def compute_comparison_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard comparison metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    actuals : np.ndarray
        Actual values

    Returns
    -------
    dict[str, float]
        Dictionary with mae, rmse, mape, direction_accuracy
    """
    if len(predictions) != len(actuals):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, actuals={len(actuals)}"
        )

    errors = predictions - actuals
    abs_errors = np.abs(errors)

    metrics: Dict[str, float] = {}

    # MAE
    metrics["mae"] = float(np.mean(abs_errors))

    # RMSE
    metrics["rmse"] = float(np.sqrt(np.mean(errors**2)))

    # MAPE (handle zeros)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_errors = np.abs(errors / actuals)
        pct_errors = pct_errors[np.isfinite(pct_errors)]
        if len(pct_errors) > 0:
            metrics["mape"] = float(np.mean(pct_errors) * 100)
        else:
            metrics["mape"] = float("nan")

    # Direction accuracy (for changes)
    if len(predictions) > 1:
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        metrics["direction_accuracy"] = float(
            np.mean(pred_direction == actual_direction)
        )
    else:
        metrics["direction_accuracy"] = float("nan")

    return metrics


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "ModelResult",
    "ComparisonResult",
    "ComparisonReport",
    "ForecastAdapter",
    "NaiveAdapter",
    "SeasonalNaiveAdapter",
    "compute_comparison_metrics",
]
