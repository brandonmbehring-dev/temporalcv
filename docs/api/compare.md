# Model Comparison

Adapters for comparing sklearn models against statistical forecasting baselines.

## Overview

The compare module provides adapters for integrating external forecasting libraries
with temporalcv's evaluation framework:

- **NaiveAdapter/SeasonalNaiveAdapter**: Built-in baseline adapters
- **StatsforecastAdapter**: Wrap statsforecast models (optional dependency)
- **run_comparison**: Run systematic comparisons across multiple models

## Installation

```bash
pip install temporalcv[compare]  # For statsforecast adapters
```

## Core Classes

### `ModelResult`

Result from a single model run:

```python
@dataclass
class ModelResult:
    model_name: str
    predictions: np.ndarray
    actuals: np.ndarray
    mae: float
    rmse: float
    mape: Optional[float]
    direction_accuracy: Optional[float]
```

### `ComparisonResult`

Result from comparing models on a single dataset:

```python
@dataclass
class ComparisonResult:
    dataset_name: str
    models: List[ModelResult]
    best_model: str  # By MAE
```

### `ForecastAdapter`

Abstract base class for model adapters:

```python
class ForecastAdapter(ABC):
    @abstractmethod
    def fit(self, values: np.ndarray) -> None: ...

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray: ...
```

## Usage

### Basic Comparison

```python
from temporalcv.compare import run_comparison, NaiveAdapter
from temporalcv.benchmarks import create_synthetic_dataset

# Create dataset
dataset = create_synthetic_dataset(n_obs=200)

# Run comparison against naive baseline
result = run_comparison(dataset, [NaiveAdapter()])

print(f"Best model: {result.best_model}")
print(f"MAE: {result.models[0].mae:.4f}")
```

### Multiple Model Comparison

```python
from temporalcv.compare import (
    run_comparison,
    NaiveAdapter,
    SeasonalNaiveAdapter,
)

# Compare multiple baselines
adapters = [
    NaiveAdapter(),
    SeasonalNaiveAdapter(season_length=12),
]

result = run_comparison(dataset, adapters)

for model in result.models:
    print(f"{model.model_name}: MAE={model.mae:.4f}")
```

### With Statsforecast (optional)

```python
from temporalcv.compare import StatsforecastAdapter

# Wrap statsforecast models
adapters = [
    StatsforecastAdapter("naive"),
    StatsforecastAdapter("ses"),
    StatsforecastAdapter("ets"),
]

result = run_comparison(dataset, adapters)
```

### Benchmark Suite

```python
from temporalcv.compare import run_benchmark_suite
from temporalcv.benchmarks import load_m3

# Run across multiple datasets
datasets = [load_m3(cat) for cat in ["monthly", "quarterly"]]

report = run_benchmark_suite(
    datasets=datasets,
    adapters=[NaiveAdapter(), SeasonalNaiveAdapter(season_length=12)],
)

print(report.summary())
```

### Compare to Baseline

```python
from temporalcv.compare import compare_to_baseline
from temporalcv import dm_test

# Quick comparison with DM test
result = compare_to_baseline(
    predictions=my_model_preds,
    actuals=actuals,
    baseline_preds=naive_preds,
    horizon=1,
)

print(f"DM statistic: {result.dm_statistic:.3f}")
print(f"p-value: {result.dm_pvalue:.4f}")
```

## Built-in Adapters

### NaiveAdapter

Random walk (persistence) baseline: ŷ[t+1] = y[t]

```python
from temporalcv.compare import NaiveAdapter

naive = NaiveAdapter()
naive.fit(train_values)
preds = naive.predict(horizon=12)
```

### SeasonalNaiveAdapter

Seasonal naive baseline: ŷ[t+s] = y[t]

```python
from temporalcv.compare import SeasonalNaiveAdapter

snaive = SeasonalNaiveAdapter(season_length=12)
snaive.fit(train_values)
preds = snaive.predict(horizon=12)
```

## Best Practices

1. **Always include persistence baseline** - Compare against the simplest baseline
2. **Use appropriate horizon** - Set gap >= forecast horizon in CV
3. **Check sample sizes** - DM test needs n >= 30 for reliability
4. **Report direction accuracy** - Especially for high-persistence data

## Exported Symbols

```python
from temporalcv.compare import (
    # Data classes
    ModelResult,
    ComparisonResult,
    ComparisonReport,

    # Adapters
    ForecastAdapter,
    NaiveAdapter,
    SeasonalNaiveAdapter,
    StatsforecastAdapter,  # Optional

    # Functions
    run_comparison,
    run_benchmark_suite,
    compare_to_baseline,
    compute_comparison_metrics,
)
```
