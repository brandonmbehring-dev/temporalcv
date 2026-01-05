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

## Applications

1. **Regime-aware CV** - Split data at regime boundaries
2. **Feature engineering** - Create regime indicators
3. **Model retraining** - Trigger retraining on structural breaks

## References

- Killick, Fearnhead & Eckley (2012). JASA 107(500), 1590-1598.
- Truong, Oudre & Vayer (2020). Signal Processing 167, 107299.
