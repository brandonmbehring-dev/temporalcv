"""
Benchmark Datasets Package.

Provides loaders for time series benchmark datasets with proper
licensing handling. All loaders return objects satisfying the
Dataset protocol.

Available Datasets
------------------
- **FRED rates**: Public domain (requires fredapi + API key)
- **M5 Walmart**: Kaggle TOS - cannot bundle, user must download
- **GluonTS bundle**: Open access (requires gluonts)
- **Monash repository**: Open access (requires datasetsforecast)

Example
-------
>>> from temporalcv.benchmarks import (
...     create_synthetic_dataset,
...     TimeSeriesDataset,
...     DatasetMetadata,
... )
>>>
>>> # Create synthetic dataset for testing
>>> dataset = create_synthetic_dataset(n_obs=200, ar_coef=0.95)
>>> train, test = dataset.get_train_test_split()
>>>
>>> # Load FRED data (requires fredapi)
>>> # from temporalcv.benchmarks import load_fred_rates
>>> # rates = load_fred_rates(series="DGS10")

Notes
-----
Some loaders require optional dependencies:
- FRED: pip install temporalcv[fred]
- GluonTS: pip install temporalcv[gluonts]
- Monash: pip install temporalcv[monash]
"""

from temporalcv.benchmarks.base import (
    Dataset,
    DatasetMetadata,
    DatasetNotFoundError,
    TimeSeriesDataset,
    create_synthetic_dataset,
    validate_dataset,
)

# Optional loaders - import only if dependencies available
__all__ = [
    # Core classes
    "DatasetNotFoundError",
    "DatasetMetadata",
    "Dataset",
    "TimeSeriesDataset",
    "validate_dataset",
    "create_synthetic_dataset",
]

# Try to import optional loaders
try:
    from temporalcv.benchmarks.fred import (
        FRED_RATE_SERIES,
        list_available_series,
        load_fred_rates,
    )

    __all__.extend(["load_fred_rates", "list_available_series", "FRED_RATE_SERIES"])
except ImportError:
    pass  # fredapi not installed

try:
    from temporalcv.benchmarks.m5 import (
        M5_DOWNLOAD_URL,
        M5_INSTRUCTIONS,
        load_m5,
    )

    __all__.extend(["load_m5", "M5_DOWNLOAD_URL", "M5_INSTRUCTIONS"])
except ImportError:
    pass  # pandas not installed for M5

try:
    from temporalcv.benchmarks.gluonts import load_electricity, load_traffic

    __all__.extend(["load_electricity", "load_traffic"])
except ImportError:
    pass  # gluonts not installed

try:
    from temporalcv.benchmarks.monash import (
        M3_HORIZONS,
        M4_HORIZONS,
        load_m3,
        load_m4,
    )

    __all__.extend(["load_m3", "load_m4", "M3_HORIZONS", "M4_HORIZONS"])
except ImportError:
    pass  # datasetsforecast not installed
