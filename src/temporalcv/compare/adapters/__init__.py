"""
Adapters for different forecasting packages.

Provides unified interface to statsforecast, sktime, etc.

Example
-------
>>> import numpy as np
>>> from temporalcv.compare.adapters import StatsforecastAdapter
>>> rng = np.random.default_rng(0)
>>> train_data = rng.normal(0, 1, 50).cumsum() + 100.0
>>> adapter = StatsforecastAdapter(model="AutoARIMA")
>>> predictions = adapter.fit_predict(train_data, test_size=10, horizon=2)
>>> predictions.shape
(10,)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalcv.compare.adapters.multi_series import (
    MultiSeriesAdapter,
    ProgressAdapter,
)

# Always available
from temporalcv.compare.base import (
    ForecastAdapter,
    NaiveAdapter,
    SeasonalNaiveAdapter,
)

__all__ = [
    "ForecastAdapter",
    "NaiveAdapter",
    "SeasonalNaiveAdapter",
    "MultiSeriesAdapter",
    "ProgressAdapter",
]

# Conditional imports for optional dependencies
try:
    from temporalcv.compare.adapters.statsforecast_adapter import (
        StatsforecastAdapter,
    )

    __all__.append("StatsforecastAdapter")
except ImportError:
    pass

if TYPE_CHECKING:
    from temporalcv.compare.adapters.statsforecast_adapter import (
        StatsforecastAdapter as StatsforecastAdapter,
    )
