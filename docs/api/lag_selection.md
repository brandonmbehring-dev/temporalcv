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

## Applications

1. **AR order selection** - Determine lag order for forecasting
2. **Gap parameter** - Inform gap settings in walk-forward CV
3. **Memory analysis** - Understand series persistence

## References

- Box, Jenkins & Reinsel (2015). Time Series Analysis, 5th ed.
- Schwarz (1978). "Estimating the Dimension of a Model." Annals of Statistics.
