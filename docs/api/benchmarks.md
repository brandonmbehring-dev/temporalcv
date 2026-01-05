# Benchmarks

Dataset loaders for standard forecasting benchmarks.

## Overview

The benchmarks module provides loaders for well-known forecasting datasets:

- **FRED**: Federal Reserve Economic Data (macroeconomic indicators)
- **M5**: Walmart sales forecasting competition data
- **M3/M4**: Makridakis forecasting competitions
- **GluonTS**: Amazon's GluonTS benchmark datasets (electricity, traffic)

Each loader returns a `TimeSeriesDataset` object compatible with temporalcv evaluation.

## Installation

Each benchmark has optional dependencies:

```bash
pip install temporalcv[fred]       # FRED economic data
pip install temporalcv[monash]     # M3/M4 datasets
pip install temporalcv[gluonts]    # GluonTS datasets
pip install temporalcv[benchmarks] # All benchmark loaders
```

## Core Classes

### `TimeSeriesDataset`

Standard dataset interface for temporalcv:

```python
from temporalcv.benchmarks import TimeSeriesDataset, DatasetMetadata

dataset = TimeSeriesDataset(
    values=np.array([1.0, 2.0, 3.0, ...]),
    metadata=DatasetMetadata(
        name="my_series",
        frequency="D",
        horizon=1,
    )
)

# Get train/test split
train, test = dataset.get_train_test_split()
```

### `create_synthetic_dataset`

Create synthetic AR(1) data for testing:

```python
from temporalcv.benchmarks import create_synthetic_dataset

# Create synthetic dataset with known properties
dataset = create_synthetic_dataset(
    n_obs=200,
    ar_coef=0.95,  # High persistence
    seed=42
)
```

## Usage

### FRED Economic Data

```python
from temporalcv.benchmarks import load_fred_rates

# Load interest rate data (requires FRED API key)
rates = load_fred_rates(series="DGS10")  # 10-Year Treasury
print(f"Shape: {len(rates.values)}")
```

### M5 Competition Data

```python
from temporalcv.benchmarks import load_m5

# Load M5 data (user must download from Kaggle first)
dataset = load_m5(data_dir="/path/to/m5/data")
```

### M3/M4 Competitions

```python
from temporalcv.benchmarks import load_m3, load_m4

# Load M3 monthly series
m3_monthly = load_m3(category="monthly")

# Load M4 weekly series
m4_weekly = load_m4(category="weekly")
```

### GluonTS Datasets

```python
from temporalcv.benchmarks import load_electricity, load_traffic

# Load electricity demand data
elec = load_electricity()

# Load traffic speed data
traffic = load_traffic()
```

## Best Practices

1. **Use for benchmarking only** - These datasets help compare methods, not replace domain data
2. **Check data quality** - Some series may have missing values or anomalies
3. **Document settings** - Record which datasets and subsets you used
4. **Verify official splits** - Some loaders truncate data; check `metadata.official_split`

## Exported Symbols

```python
from temporalcv.benchmarks import (
    # Core classes
    Dataset,
    DatasetMetadata,
    TimeSeriesDataset,
    DatasetNotFoundError,

    # Functions
    create_synthetic_dataset,
    validate_dataset,

    # Optional (if dependencies installed)
    load_fred_rates,
    load_m3,
    load_m4,
    load_m5,
    load_electricity,
    load_traffic,
)
```
