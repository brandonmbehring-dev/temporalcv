# Model Comparison

Adapters for comparing sklearn models against statistical forecasting baselines.

## Overview

The compare module provides adapters for integrating external forecasting libraries
with temporalcv's evaluation framework:

- **StatsforecastAdapter**: Wrap statsforecast models for sklearn-compatible interface
- **ComparisonRunner**: Run systematic comparisons across multiple models

## Installation

```bash
pip install temporalcv[compare]
```

## Usage

### Basic Comparison

```python
from sklearn.linear_model import Ridge
from temporalcv.compare import StatsforecastAdapter, ComparisonRunner
from temporalcv import WalkForwardCV, dm_test

# Create models to compare
ml_model = Ridge(alpha=1.0)
stat_model = StatsforecastAdapter("naive_seasonal", season_length=12)

# Run walk-forward CV
cv = WalkForwardCV(n_splits=5, gap=1)

# Compare using DM test
result = dm_test(ml_errors, stat_errors, h=1)
print(f"DM test: {result}")
```

### Systematic Comparison

```python
from temporalcv.compare import ComparisonRunner

runner = ComparisonRunner(
    models={
        "ridge": Ridge(alpha=1.0),
        "naive": StatsforecastAdapter("naive"),
        "ses": StatsforecastAdapter("ses"),
    },
    cv=WalkForwardCV(n_splits=5, gap=1),
)

results = runner.run(X, y)
print(results.summary())
```

## Supported Models

### Statsforecast Adapters

- `naive`: Random walk (persistence)
- `naive_seasonal`: Seasonal naive
- `ses`: Simple exponential smoothing
- `ets`: Error-Trend-Seasonality

## Best Practices

1. **Always include persistence baseline** - Compare against the simplest baseline
2. **Use appropriate horizon** - Set gap >= forecast horizon
3. **Check sample sizes** - DM test needs n >= 30 for reliability

## API Reference

See the [API Reference](../api_reference/compare.rst) for complete function signatures.
