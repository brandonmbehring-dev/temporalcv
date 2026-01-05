"""
Adapters for different forecasting packages.

Provides unified interface to statsforecast, sktime, etc.

Example
-------
>>> from temporalcv.compare.adapters import StatsforecastAdapter
>>> adapter = StatsforecastAdapter(model="AutoARIMA")
>>> predictions = adapter.fit_predict(train_data, test_size=10, horizon=2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Always available
from temporalcv.compare.base import (
    ForecastAdapter,
    NaiveAdapter,
    SeasonalNaiveAdapter,
)
from temporalcv.compare.adapters.multi_series import (
    MultiSeriesAdapter,
    ProgressAdapter,
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
