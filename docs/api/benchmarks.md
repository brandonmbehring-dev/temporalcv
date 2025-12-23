# Benchmarks

Dataset loaders for standard forecasting benchmarks.

## Overview

The benchmarks module provides loaders for well-known forecasting datasets:

- **FRED**: Federal Reserve Economic Data (macroeconomic indicators)
- **M5**: Walmart sales forecasting competition data
- **Monash**: Monash Time Series Forecasting Archive
- **GluonTS**: Amazon's GluonTS benchmark datasets

Each loader returns data in a consistent format suitable for temporalcv evaluation.

## Installation

Each benchmark has optional dependencies:

```bash
pip install temporalcv[fred]       # FRED economic data
pip install temporalcv[monash]     # Monash archive
pip install temporalcv[gluonts]    # GluonTS datasets
pip install temporalcv[benchmarks] # All benchmark loaders
```

## Usage

### FRED Economic Data

```python
from temporalcv.benchmarks.fred import load_fred_series

# Load a single FRED series
data = load_fred_series("UNRATE")  # Unemployment rate
print(f"Shape: {data.shape}, Range: {data.index.min()} to {data.index.max()}")
```

### M5 Competition Data

```python
from temporalcv.benchmarks.m5 import load_m5_sample

# Load sample of M5 data
sales, calendar, prices = load_m5_sample(n_items=100)
```

### Monash Archive

```python
from temporalcv.benchmarks.monash import load_monash_dataset

# Load a Monash dataset
dataset = load_monash_dataset("tourism_monthly")
```

## Best Practices

1. **Use for benchmarking only** - These datasets help compare methods, not replace domain data
2. **Check data quality** - Some series may have missing values or anomalies
3. **Document settings** - Record which datasets and subsets you used

## API Reference

See the [API Reference](../api_reference/benchmarks.rst) for complete function signatures.
