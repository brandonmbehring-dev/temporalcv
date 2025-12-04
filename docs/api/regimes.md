# API Reference: Regime Classification

Classify market regimes for conditional performance analysis.

---

## Functions

### `classify_volatility_regime`

Classify volatility regime using rolling window.

```python
def classify_volatility_regime(
    values: np.ndarray,
    window: int = 13,
    basis: Literal["changes", "levels"] = "changes",
    low_percentile: float = 33.0,
    high_percentile: float = 67.0,
) -> np.ndarray
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `np.ndarray` | required | Time series values |
| `window` | `int` | `13` | Rolling window (13 weeks ≈ 1 quarter) |
| `basis` | `str` | `"changes"` | `"changes"` (correct) or `"levels"` (legacy) |
| `low_percentile` | `float` | `33.0` | LOW volatility threshold |
| `high_percentile` | `float` | `67.0` | HIGH volatility threshold |

**Returns**: Array of regime labels (`"LOW"`, `"MED"`, `"HIGH"`)

**CRITICAL**: Use `basis="changes"` (default). Using `"levels"` mislabels steady drifts as volatile.

**Example**:

```python
from temporalcv.regimes import classify_volatility_regime

# Classify volatility on changes (correct)
vol_regimes = classify_volatility_regime(prices, window=13, basis="changes")
print(np.unique(vol_regimes, return_counts=True))
```

---

### `classify_direction_regime`

Classify direction using thresholded signs.

```python
def classify_direction_regime(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `np.ndarray` | Values to classify (changes) |
| `threshold` | `float` | Move threshold |

**Returns**: Array of direction labels (`"UP"`, `"DOWN"`, `"FLAT"`)

**Classification**:
- `|value| > threshold` and `value > 0` → `"UP"`
- `|value| > threshold` and `value < 0` → `"DOWN"`
- `|value| <= threshold` → `"FLAT"`

---

### `get_combined_regimes`

Combine volatility and direction into single label.

```python
def get_combined_regimes(
    vol_regimes: np.ndarray,
    dir_regimes: np.ndarray,
) -> np.ndarray
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vol_regimes` | `np.ndarray` | Volatility labels |
| `dir_regimes` | `np.ndarray` | Direction labels |

**Returns**: Combined labels like `"HIGH-UP"`, `"LOW-FLAT"`, etc.

**Example**:

```python
from temporalcv.regimes import (
    classify_volatility_regime,
    classify_direction_regime,
    get_combined_regimes,
)
from temporalcv import compute_move_threshold

# Classify
vol = classify_volatility_regime(prices, basis="changes")
threshold = compute_move_threshold(train_actuals)
direction = classify_direction_regime(actuals, threshold)

# Combine
combined = get_combined_regimes(vol[-len(actuals):], direction)
print(np.unique(combined, return_counts=True))
```

---

### `get_regime_counts`

Get sample counts per regime.

```python
def get_regime_counts(regimes: np.ndarray) -> Dict[str, int]
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `regimes` | `np.ndarray` | Regime labels |

**Returns**: Dict of counts, sorted by count descending

---

### `mask_low_n_regimes`

Mask regime labels with insufficient samples.

```python
def mask_low_n_regimes(
    regimes: np.ndarray,
    min_n: int = 10,
    mask_value: str = "MASKED",
) -> np.ndarray
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regimes` | `np.ndarray` | required | Regime labels |
| `min_n` | `int` | `10` | Minimum samples required |
| `mask_value` | `str` | `"MASKED"` | Value for masked regimes |

**Returns**: Regimes with low-n cells masked

**Example**:

```python
from temporalcv.regimes import mask_low_n_regimes

# Mask unreliable regimes
masked = mask_low_n_regimes(combined, min_n=10)

# Only evaluate on reliable regimes
for regime in np.unique(masked):
    if regime == "MASKED":
        continue
    mask = masked == regime
    mae = np.mean(np.abs(preds[mask] - actuals[mask]))
    print(f"{regime}: MAE = {mae:.4f} (n={mask.sum()})")
```

---

## Best Practices

1. **Use changes basis**: Always use `basis="changes"` for volatility
2. **Check sample counts**: Use `mask_low_n_regimes` before interpreting
3. **Training-only thresholds**: Compute direction threshold from training data
4. **Align arrays**: Ensure volatility and direction arrays have same length
