# Lag Selection

Optimal lag order selection for AR models.

## Overview

Provides methods to select optimal lag order using:
- **PACF**: Partial autocorrelation significance cutoff
- **AIC**: Akaike Information Criterion minimization
- **BIC**: Bayesian Information Criterion minimization

## Data Classes

### `LagSelectionResult`

```python
@dataclass(frozen=True)
class LagSelectionResult:
    optimal_lag: int                    # Selected optimal lag order
    criterion_values: dict[int, float]  # Criterion values per lag
    method: str                         # Method used ('aic', 'bic', 'pacf')
    all_lags_tested: list[int]         # All lags evaluated
```

## Usage

```python
from temporalcv.lag_selection import select_lag_order

# Select using AIC
result = select_lag_order(series, method='aic', max_lag=20)
print(f"Optimal lag: {result.optimal_lag}")

# Select using PACF significance
result = select_lag_order(series, method='pacf', alpha=0.05)

# Select using BIC (more parsimonious)
result = select_lag_order(series, method='bic', max_lag=20)
```

## Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| **PACF** | Fast, no model fitting | Requires significance cutoff |
| **AIC** | Balances fit and complexity | Can overfit |
| **BIC** | More parsimonious | May underfit |

## Complete Example: AIC vs BIC Comparison

```python
import numpy as np
from temporalcv.lag_selection import select_lag_order

# Generate AR(3) process
np.random.seed(42)
n = 500
y = np.zeros(n)
for t in range(3, n):
    y[t] = 0.5*y[t-1] + 0.2*y[t-2] + 0.1*y[t-3] + np.random.randn()

# Compare AIC and BIC
aic_result = select_lag_order(y, method='aic', max_lag=10)
bic_result = select_lag_order(y, method='bic', max_lag=10)

print(f"True lag order: 3")
print(f"AIC selected: {aic_result.optimal_lag}")
print(f"BIC selected: {bic_result.optimal_lag}")

# View criterion values
print("\nCriterion values by lag:")
print("Lag | AIC      | BIC")
print("----|----------|----------")
for lag in range(1, 6):
    aic_val = aic_result.criterion_values.get(lag, float('nan'))
    bic_val = bic_result.criterion_values.get(lag, float('nan'))
    marker = " *" if lag == aic_result.optimal_lag else ""
    print(f"{lag:3d} | {aic_val:8.2f} | {bic_val:8.2f}{marker}")
```

## Using Lag Order for CV Gap

```python
from temporalcv import WalkForwardCV
from temporalcv.lag_selection import select_lag_order

# Select lag order
result = select_lag_order(y, method='bic', max_lag=20)
optimal_lag = result.optimal_lag

# Use as gap parameter in CV
# For h-step forecasting, gap should be at least h
cv = WalkForwardCV(
    n_splits=5,
    horizon=optimal_lag,  # Minimum gap = forecast horizon
    extra_gap=0,          # Additional safety margin
)

print(f"CV configured with horizon={optimal_lag} based on BIC lag selection")
```

## Applications

1. **AR order selection** - Determine lag order for forecasting
2. **Gap parameter** - Inform gap settings in walk-forward CV
3. **Memory analysis** - Understand series persistence

## See Also

- [Walk-Forward CV](cv.md) - Configure gap based on selected lag
- [Stationarity Tests](stationarity.md) - Test before lag selection

## References

- Box, Jenkins & Reinsel (2015). Time Series Analysis, 5th ed.
- Schwarz (1978). "Estimating the Dimension of a Model." Annals of Statistics.
