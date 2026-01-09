# Changepoint Detection

Structural break detection in time series.

## Overview

Detects changepoints (structural breaks) useful for:
- Identifying regime boundaries (LOW → HIGH volatility)
- Training models on post-changepoint data only
- Creating regime indicators as features

## Algorithms

| Algorithm | Description | Dependencies |
|-----------|-------------|--------------|
| **Variance-based** | Rolling variance threshold | None (always available) |
| **PELT** | Pruned Exact Linear Time | `ruptures` (optional) |

## Data Classes

### `ChangepointResult`

```python
@dataclass(frozen=True)
class ChangepointResult:
    changepoints: list[int]      # Indices of detected changepoints
    n_segments: int              # Number of segments
    method: str                  # Algorithm used
    details: dict[str, Any]      # Algorithm-specific details
```

## Usage

### Variance-Based Detection

```python
from temporalcv.changepoint import detect_changepoints

# Simple variance-based detection (always available)
result = detect_changepoints(
    series,
    method='variance',
    window=20,
    threshold=2.0,  # Standard deviations
)

print(f"Found {len(result.changepoints)} changepoints")
```

### PELT Algorithm (requires ruptures)

```python
from temporalcv.changepoint import detect_changepoints

# More sophisticated detection
result = detect_changepoints(
    series,
    method='pelt',
    penalty='bic',
    min_size=30,
)

for cp in result.changepoints:
    print(f"Changepoint at index {cp}")
```

## Complete Example: Regime-Aware Cross-Validation

```python
import numpy as np
from temporalcv.changepoint import detect_changepoints
from temporalcv import WalkForwardCV

# Generate series with volatility regime change
np.random.seed(42)
n = 500
low_vol = np.random.randn(250) * 0.5   # Low volatility regime
high_vol = np.random.randn(250) * 2.0  # High volatility regime
series = np.concatenate([low_vol, high_vol])

# Detect the regime change
result = detect_changepoints(series, method='variance', window=30, threshold=2.0)
print(f"Changepoints: {result.changepoints}")  # Should find ~index 250

# Option 1: Train only on post-changepoint data
if result.changepoints:
    last_cp = result.changepoints[-1]
    train_data = series[last_cp:]
    print(f"Training on {len(train_data)} samples (post-regime-change)")

# Option 2: Create regime indicator as feature
regime_indicator = np.zeros(len(series))
for i, cp in enumerate(result.changepoints):
    regime_indicator[cp:] = i + 1
X = np.column_stack([series[:-1], regime_indicator[:-1]])
y = series[1:]

# Use regime-aware CV
cv = WalkForwardCV(n_splits=5, window_type='sliding')
for train_idx, test_idx in cv.split(X, y):
    # Train includes regime information
    pass
```

## Applications

1. **Regime-aware CV** - Split data at regime boundaries
2. **Feature engineering** - Create regime indicators
3. **Model retraining** - Trigger retraining on structural breaks

## See Also

- [Regimes API](regimes.md) - Volatility and direction regime classification
- [Walk-Forward CV](cv.md) - Temporal cross-validation with gap enforcement

## References

- Killick, Fearnhead & Eckley (2012). JASA 107(500), 1590-1598.
- Truong, Oudre & Vayer (2020). Signal Processing 167, 107299.
