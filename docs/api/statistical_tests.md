# API Reference: Statistical Tests

Statistical tests for forecast evaluation with proper corrections.

---

## Data Classes

### `DMTestResult`

Result from Diebold-Mariano test.

```python
@dataclass
class DMTestResult:
    statistic: float       # DM test statistic (asymptotically N(0,1))
    pvalue: float          # P-value for the test
    h: int                 # Forecast horizon used
    n: int                 # Number of observations
    loss: str              # Loss function ("squared" or "absolute")
    alternative: str       # Alternative hypothesis
    harvey_adjusted: bool  # Whether small-sample adjustment applied
    mean_loss_diff: float  # Mean loss differential
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `significant_at_05` | `bool` | Is p-value < 0.05? |
| `significant_at_01` | `bool` | Is p-value < 0.01? |

**String representation**: `DM(h): statistic (p=pvalue)` with significance stars

---

### `PTTestResult`

Result from Pesaran-Timmermann test.

```python
@dataclass
class PTTestResult:
    statistic: float    # PT test statistic (z-score)
    pvalue: float       # P-value (one-sided)
    accuracy: float     # Observed directional accuracy
    expected: float     # Expected accuracy under null
    n: int              # Number of observations
    n_classes: int      # Number of direction classes (2 or 3)
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `significant_at_05` | `bool` | Is p-value < 0.05? |
| `skill` | `float` | accuracy - expected |

**String representation**: `PT: accuracy vs expected (z=statistic, p=pvalue)`

---

## Functions

### `dm_test`

Diebold-Mariano test for equal predictive accuracy.

```python
def dm_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    harvey_correction: bool = True,
) -> DMTestResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `errors_1` | `np.ndarray` | required | Errors from model 1 (`actual - prediction`) |
| `errors_2` | `np.ndarray` | required | Errors from model 2 (baseline) |
| `h` | `int` | `1` | Forecast horizon |
| `loss` | `str` | `"squared"` | Loss function: `"squared"` or `"absolute"` |
| `alternative` | `str` | `"two-sided"` | `"two-sided"`, `"less"`, or `"greater"` |
| `harvey_correction` | `bool` | `True` | Apply small-sample adjustment |

**Returns**: `DMTestResult`

**Raises**: `ValueError` if fewer than 30 samples

**Alternative hypotheses**:
- `"two-sided"`: Models have different accuracy
- `"less"`: Model 1 more accurate (lower loss)
- `"greater"`: Model 2 more accurate

**Example**:

```python
from temporalcv import dm_test

# Compute errors: actual - predicted
model_errors = actuals - model_preds
persistence_errors = actuals - 0  # Persistence predicts 0

result = dm_test(
    errors_1=model_errors,
    errors_2=persistence_errors,
    h=2,
    loss="absolute",
    alternative="less"  # Test if model 1 is better
)

print(f"DM statistic: {result.statistic:.3f}")
print(f"p-value: {result.pvalue:.4f}")
print(f"Significant: {result.significant_at_05}")
```

---

### `pt_test`

Pesaran-Timmermann test for directional accuracy.

```python
def pt_test(
    actual: np.ndarray,
    predicted: np.ndarray,
    move_threshold: Optional[float] = None,
) -> PTTestResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `actual` | `np.ndarray` | required | Actual values (changes) |
| `predicted` | `np.ndarray` | required | Predicted values (changes) |
| `move_threshold` | `float` | `None` | Threshold for 3-class mode |

**Returns**: `PTTestResult`

**Raises**: `ValueError` if fewer than 20 samples

**Modes**:

| Mode | Condition | Classes |
|------|-----------|---------|
| 2-class | `move_threshold=None` | Positive/Negative sign |
| 3-class | `move_threshold` provided | UP/DOWN/FLAT |

**Example**:

```python
from temporalcv import pt_test, compute_move_threshold

# Compute threshold from training data
threshold = compute_move_threshold(train_actuals, percentile=70)

# Test directional accuracy with 3-class
result = pt_test(
    actual=test_actuals,
    predicted=test_preds,
    move_threshold=threshold
)

print(f"Direction accuracy: {result.accuracy:.1%}")
print(f"Expected (random): {result.expected:.1%}")
print(f"Skill: {result.skill:.1%}")
print(f"Significant: {result.significant_at_05}")
```

**Warning**: For h>1 step forecasts, p-values may be optimistic (no HAC correction). Use DM test for rigorous multi-step testing.

---

### `compute_hac_variance`

Compute HAC (Heteroskedasticity and Autocorrelation Consistent) variance.

```python
def compute_hac_variance(
    d: np.ndarray,
    bandwidth: Optional[int] = None,
) -> float
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d` | `np.ndarray` | required | Series (typically loss differential) |
| `bandwidth` | `int` | `None` | Kernel bandwidth. If None, auto-selected. |

**Returns**: HAC variance estimate

**Notes**:
- Uses Newey-West estimator with Bartlett kernel
- For h-step forecasts: `bandwidth = h - 1` is appropriate
- Automatic bandwidth: `floor(4 * (n/100)^(2/9))`
