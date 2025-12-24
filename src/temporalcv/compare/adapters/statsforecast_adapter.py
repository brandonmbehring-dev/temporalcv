"""
Statsforecast Package Adapter.

Provides adapter for statsforecast models (AutoARIMA, AutoETS, etc.).

Requires: statsforecast (optional dependency)

Example
-------
>>> from temporalcv.compare.adapters import StatsforecastAdapter
>>> adapter = StatsforecastAdapter(model="AutoARIMA")
>>> predictions = adapter.fit_predict(train, test_size=10, horizon=2)
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, cast

import numpy as np

from temporalcv.compare.base import ForecastAdapter


def _check_statsforecast() -> None:
    """Check statsforecast is installed."""
    try:
        from statsforecast import StatsForecast  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "statsforecast required for this adapter.\n"
            "Install with: pip install temporalcv[compare]\n"
            "Or: pip install statsforecast"
        ) from e


# Model name mapping to statsforecast classes
MODEL_MAP = {
    "AutoARIMA": "AutoARIMA",
    "AutoETS": "AutoETS",
    "Naive": "Naive",
    "SeasonalNaive": "SeasonalNaive",
    "HistoricAverage": "HistoricAverage",
    "WindowAverage": "WindowAverage",
    "AutoTheta": "AutoTheta",
    "CrostonClassic": "CrostonClassic",
    "ADIDA": "ADIDA",
    "IMAPA": "IMAPA",
}


class StatsforecastAdapter(ForecastAdapter):
    """
    Adapter for statsforecast models.

    Supports AutoARIMA, AutoETS, and baseline models.

    Parameters
    ----------
    model : str
        Model name. Options: AutoARIMA, AutoETS, Naive, SeasonalNaive,
        HistoricAverage, WindowAverage, AutoTheta
    season_length : int, optional
        Seasonal period. Default: 1 (no seasonality)
    frequency : str, default="W"
        Pandas frequency string for time series (e.g., "W" for weekly,
        "D" for daily, "M" for monthly). Must match your data.
    **model_kwargs
        Additional keyword arguments passed to the model constructor

    Example
    -------
    >>> adapter = StatsforecastAdapter("AutoARIMA", season_length=52, frequency="W")
    >>> predictions = adapter.fit_predict(train, test_size=10, horizon=2)
    """

    def __init__(
        self,
        model: Literal[
            "AutoARIMA",
            "AutoETS",
            "Naive",
            "SeasonalNaive",
            "HistoricAverage",
            "WindowAverage",
            "AutoTheta",
            "CrostonClassic",
            "ADIDA",
            "IMAPA",
        ] = "AutoARIMA",
        season_length: int = 1,
        frequency: str = "W",
        **model_kwargs: Any,
    ):
        """Initialize adapter."""
        _check_statsforecast()

        if model not in MODEL_MAP:
            raise ValueError(
                f"Unknown model '{model}'. Available: {list(MODEL_MAP.keys())}"
            )

        self._model_name = model
        self._season_length = season_length
        self._frequency = frequency
        self._model_kwargs = model_kwargs
        self._fitted_model: Optional[Any] = None

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

    @property
    def package_name(self) -> str:
        """Return package name."""
        return "statsforecast"

    def _get_model_class(self) -> Any:
        """Get the statsforecast model class."""
        from statsforecast import models as sf_models

        return getattr(sf_models, self._model_name)

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
            Training data (1D array)
        test_size : int
            Number of test periods to predict
        horizon : int
            Forecast horizon for each prediction

        Returns
        -------
        np.ndarray
            Predictions array
        """
        _check_statsforecast()

        import pandas as pd
        from statsforecast import StatsForecast

        # Reject multi-series input (silent averaging is lossy and error-prone)
        if train_values.ndim > 1:
            raise ValueError(
                f"StatsforecastAdapter expects single series. "
                f"Got shape {train_values.shape}. Use per-series loop."
            )

        # Build model instance
        model_class = self._get_model_class()

        # Models that support season_length
        seasonal_models = {"AutoARIMA", "AutoETS", "SeasonalNaive", "AutoTheta"}

        if self._model_name in seasonal_models:
            model_instance = model_class(
                season_length=self._season_length, **self._model_kwargs
            )
        else:
            model_instance = model_class(**self._model_kwargs)

        # StatsForecast requires a DataFrame with unique_id, ds, y columns
        n_train = len(train_values)
        df = pd.DataFrame(
            {
                "unique_id": ["series_0"] * n_train,
                "ds": pd.date_range("2000-01-01", periods=n_train, freq=self._frequency),
                "y": train_values,
            }
        )

        # Create StatsForecast object
        sf = StatsForecast(
            models=[model_instance],
            freq=self._frequency,
            n_jobs=1,
        )

        # Fit and predict
        sf.fit(df)

        # Single multi-step forecast (no rolling)
        # NOTE: The previous implementation used rolling forecasts but
        # incorrectly used PREDICTIONS as actuals when updating training data.
        # This is fundamentally flawed - we can't know actuals at forecast time.
        # Instead, we predict all test_size steps at once from the fitted model.
        forecast = sf.predict(h=test_size)
        predictions = forecast[self._model_name].values

        return cast(np.ndarray, np.asarray(predictions, dtype=np.float64))

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "model": self._model_name,
            "season_length": self._season_length,
            "frequency": self._frequency,
            **self._model_kwargs,
        }


# =============================================================================
# Public API
# =============================================================================

__all__ = ["StatsforecastAdapter"]
