"""
Multi-Series Adapter Wrapper.

Wraps single-series adapters to handle multi-series datasets.

Example
-------
>>> from temporalcv.compare.adapters import StatsforecastAdapter
>>> from temporalcv.compare.adapters.multi_series import MultiSeriesAdapter
>>>
>>> base = StatsforecastAdapter("AutoARIMA")
>>> adapter = MultiSeriesAdapter(base, n_jobs=4)
>>> predictions = adapter.fit_predict(multi_series_train, test_size=10, horizon=2)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from temporalcv.compare.base import ForecastAdapter

logger = logging.getLogger(__name__)


class MultiSeriesAdapter(ForecastAdapter):
    """
    Wrapper that applies single-series adapters to multi-series datasets.

    For adapters that only handle single series (1D arrays), this wrapper
    iterates over each series and collects predictions.

    Parameters
    ----------
    base_adapter : ForecastAdapter
        Single-series adapter to wrap
    n_jobs : int, default=1
        Number of parallel jobs (requires joblib if n_jobs != 1)

        .. note::
            Parallel execution with n_jobs > 1 may cause issues with some
            models that are not picklable. Set n_jobs=1 for safety.

    Attributes
    ----------
    model_name : str
        Name from base adapter with "_multi" suffix
    package_name : str
        Package name from base adapter

    Example
    -------
    >>> from temporalcv.compare.adapters import StatsforecastAdapter
    >>> from temporalcv.compare.adapters.multi_series import MultiSeriesAdapter
    >>>
    >>> # Create single-series adapter
    >>> arima = StatsforecastAdapter("AutoARIMA", season_length=12)
    >>>
    >>> # Wrap for multi-series
    >>> multi_arima = MultiSeriesAdapter(arima, n_jobs=4)
    >>>
    >>> # Now works with 2D arrays
    >>> train = np.random.randn(100, 200)  # 100 series, 200 observations each
    >>> preds = multi_arima.fit_predict(train, test_size=10, horizon=2)
    >>> print(preds.shape)  # (100, 10)
    """

    def __init__(self, base_adapter: ForecastAdapter, n_jobs: int = 1):
        """
        Initialize multi-series wrapper.

        Parameters
        ----------
        base_adapter : ForecastAdapter
            Single-series adapter to wrap
        n_jobs : int, default=1
            Number of parallel jobs
        """
        self._base = base_adapter
        self._n_jobs = n_jobs

    @property
    def model_name(self) -> str:
        """Return model name with suffix."""
        return f"{self._base.model_name}_multi"

    @property
    def package_name(self) -> str:
        """Return package name from base adapter."""
        return self._base.package_name

    def fit_predict(
        self,
        train_values: np.ndarray,
        test_size: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Fit model and generate predictions for each series.

        Parameters
        ----------
        train_values : np.ndarray
            Training data. If 1D, delegates to base adapter.
            If 2D (n_series, n_obs), iterates over series.
        test_size : int
            Number of test periods to predict
        horizon : int
            Forecast horizon

        Returns
        -------
        np.ndarray
            Predictions. Shape (test_size,) for 1D input,
            or (n_series, test_size) for 2D input.
        """
        # Single series: delegate directly
        if train_values.ndim == 1:
            return self._base.fit_predict(train_values, test_size, horizon)

        n_series = train_values.shape[0]

        # Parallel execution
        if self._n_jobs != 1:
            return self._fit_predict_parallel(train_values, test_size, horizon)

        # Sequential execution
        predictions = np.zeros((n_series, test_size))

        for i in range(n_series):
            try:
                predictions[i] = self._base.fit_predict(
                    train_values[i], test_size, horizon
                )
            except Exception as e:
                logger.warning(
                    "Series %d failed for %s: %s. Using NaN.",
                    i,
                    self._base.model_name,
                    e,
                )
                predictions[i] = np.nan

        return predictions

    def _fit_predict_parallel(
        self,
        train_values: np.ndarray,
        test_size: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Fit and predict in parallel using joblib.

        Parameters
        ----------
        train_values : np.ndarray
            2D training data (n_series, n_obs)
        test_size : int
            Number of predictions per series
        horizon : int
            Forecast horizon

        Returns
        -------
        np.ndarray
            Predictions with shape (n_series, test_size)
        """
        try:
            from joblib import Parallel, delayed
        except ImportError:
            logger.warning("joblib not available, falling back to sequential")
            return self.fit_predict(train_values, test_size, horizon)

        n_series = train_values.shape[0]

        def fit_single(series_idx: int) -> np.ndarray:
            """Fit single series and return predictions."""
            try:
                return self._base.fit_predict(
                    train_values[series_idx], test_size, horizon
                )
            except Exception as e:
                logger.warning(
                    "Series %d failed for %s: %s",
                    series_idx,
                    self._base.model_name,
                    e,
                )
                return np.full(test_size, np.nan)

        results = Parallel(n_jobs=self._n_jobs)(
            delayed(fit_single)(i) for i in range(n_series)
        )

        return np.array(results)

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns
        -------
        dict
            Combined parameters from base adapter plus n_jobs
        """
        params = self._base.get_params().copy()
        params["n_jobs"] = self._n_jobs
        params["base_adapter"] = self._base.model_name
        return params


class ProgressAdapter(ForecastAdapter):
    """
    Wrapper that reports progress during multi-series iteration.

    Useful for long-running benchmarks to track completion.

    Parameters
    ----------
    base_adapter : ForecastAdapter
        Adapter to wrap
    progress_callback : callable
        Callback with signature ``callback(series_idx: int, n_series: int)``

    Example
    -------
    >>> def on_series(idx, total):
    ...     if idx % 100 == 0:
    ...         print(f"Series {idx}/{total}")
    >>> adapter = ProgressAdapter(base, progress_callback=on_series)
    """

    def __init__(
        self,
        base_adapter: ForecastAdapter,
        progress_callback: Optional[callable] = None,
    ):
        """Initialize progress wrapper."""
        self._base = base_adapter
        self._callback = progress_callback

    @property
    def model_name(self) -> str:
        """Return model name from base adapter."""
        return self._base.model_name

    @property
    def package_name(self) -> str:
        """Return package name from base adapter."""
        return self._base.package_name

    def fit_predict(
        self,
        train_values: np.ndarray,
        test_size: int,
        horizon: int,
    ) -> np.ndarray:
        """
        Fit model with progress reporting.

        For single series, delegates directly.
        For multi-series, reports progress after each series.
        """
        # Single series: delegate directly
        if train_values.ndim == 1:
            return self._base.fit_predict(train_values, test_size, horizon)

        n_series = train_values.shape[0]
        predictions = np.zeros((n_series, test_size))

        for i in range(n_series):
            try:
                predictions[i] = self._base.fit_predict(
                    train_values[i], test_size, horizon
                )
            except Exception as e:
                logger.warning(
                    "Series %d failed for %s: %s. Using NaN.",
                    i,
                    self._base.model_name,
                    e,
                )
                predictions[i] = np.nan

            # Report progress
            if self._callback:
                self._callback(i + 1, n_series)

        return predictions

    def get_params(self) -> Dict[str, Any]:
        """Get parameters from base adapter."""
        return self._base.get_params()


__all__ = [
    "MultiSeriesAdapter",
    "ProgressAdapter",
]
