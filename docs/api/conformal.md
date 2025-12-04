# API Reference: Conformal Prediction

Distribution-free prediction intervals with coverage guarantees.

---

## Data Classes

### `PredictionInterval`

Container for prediction intervals.

```python
@dataclass
class PredictionInterval:
    point: np.ndarray       # Point predictions
    lower: np.ndarray       # Lower bounds
    upper: np.ndarray       # Upper bounds
    confidence: float       # Nominal confidence (1 - alpha)
    method: str             # Method used
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `width` | `np.ndarray` | Interval width at each point |
| `mean_width` | `float` | Mean interval width |

**Methods**:

- `coverage(actuals) -> float`: Empirical coverage
- `to_dict() -> dict`: Convert to dictionary

---

## Classes

### `SplitConformalPredictor`

Split Conformal Prediction for regression.

```python
class SplitConformalPredictor:
    def __init__(self, alpha: float = 0.05)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Miscoverage rate (0.05 = 95% intervals) |

**Attributes**:
- `quantile_`: Calibrated quantile (after `calibrate()`)

**Warning**: Assumes exchangeability. For time series, consider `AdaptiveConformalPredictor`.

#### Methods

##### `calibrate(predictions, actuals)`

Calibrate on held-out data.

```python
def calibrate(
    self,
    predictions: np.ndarray,
    actuals: np.ndarray,
) -> SplitConformalPredictor
```

**Requires**: At least 10 calibration samples

**Returns**: self (for chaining)

##### `predict_interval(predictions)`

Construct prediction intervals.

```python
def predict_interval(
    self,
    predictions: np.ndarray,
) -> PredictionInterval
```

**Returns**: `PredictionInterval` with coverage guarantee

**Example**:

```python
from temporalcv import SplitConformalPredictor

conformal = SplitConformalPredictor(alpha=0.10)  # 90% intervals
conformal.calibrate(cal_preds, cal_actuals)

intervals = conformal.predict_interval(test_preds)
print(f"Coverage: {intervals.coverage(test_actuals):.1%}")
print(f"Mean width: {intervals.mean_width:.4f}")
```

---

### `AdaptiveConformalPredictor`

Adaptive Conformal Inference for time series with distribution shift.

```python
class AdaptiveConformalPredictor:
    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.1,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | `float` | `0.05` | Target miscoverage rate |
| `gamma` | `float` | `0.1` | Adaptation rate (higher = faster) |

**Attributes**:
- `quantile_history`: List of adaptive quantiles
- `current_quantile`: Current quantile

#### Methods

##### `initialize(initial_predictions, initial_actuals)`

Initialize with calibration data.

##### `update(prediction, actual)`

Update quantile based on coverage feedback.

```python
def update(self, prediction: float, actual: float) -> float
```

**Returns**: Updated quantile

##### `predict_interval(prediction)`

Construct interval for single prediction.

```python
def predict_interval(self, prediction: float) -> Tuple[float, float]
```

**Returns**: `(lower, upper)` tuple

**Example**:

```python
from temporalcv import AdaptiveConformalPredictor

adaptive = AdaptiveConformalPredictor(alpha=0.10, gamma=0.05)
adaptive.initialize(cal_preds, cal_actuals)

for pred, actual in zip(test_preds, test_actuals):
    lower, upper = adaptive.predict_interval(pred)
    adaptive.update(pred, actual)
    print(f"Interval: [{lower:.3f}, {upper:.3f}]")
```

---

### `BootstrapUncertainty`

Bootstrap-based prediction intervals.

```python
class BootstrapUncertainty:
    def __init__(
        self,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        random_state: int = 42,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_bootstrap` | `int` | `100` | Bootstrap samples |
| `alpha` | `float` | `0.05` | Miscoverage rate |
| `random_state` | `int` | `42` | Random seed |

#### Methods

##### `fit(predictions, actuals)`

Fit bootstrap estimator on residuals.

##### `predict_interval(predictions)`

Construct bootstrap prediction intervals.

---

## Functions

### `evaluate_interval_quality`

Evaluate prediction interval quality.

```python
def evaluate_interval_quality(
    intervals: PredictionInterval,
    actuals: np.ndarray,
) -> dict[str, object]
```

**Returns** dict with:

| Key | Description |
|-----|-------------|
| `coverage` | Empirical coverage |
| `target_coverage` | Nominal coverage (1 - α) |
| `coverage_gap` | coverage - target |
| `mean_width` | Average interval width |
| `interval_score` | Proper scoring rule (lower = better) |
| `conditional_gap` | Coverage difference by prediction magnitude |

**Interval Score**:
```
IS = width + (2/α) × (lower - y) × I(y < lower) + (2/α) × (y - upper) × I(y > upper)
```

---

### `walk_forward_conformal`

Apply conformal prediction to walk-forward results.

```python
def walk_forward_conformal(
    predictions: np.ndarray,
    actuals: np.ndarray,
    calibration_fraction: float = 0.3,
    alpha: float = 0.05,
) -> Tuple[PredictionInterval, dict[str, object]]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `np.ndarray` | required | Walk-forward predictions |
| `actuals` | `np.ndarray` | required | Corresponding actuals |
| `calibration_fraction` | `float` | `0.3` | Fraction for calibration |
| `alpha` | `float` | `0.05` | Miscoverage rate |

**Returns**: `(intervals, quality_metrics)`

**CRITICAL**: Coverage computed ONLY on post-calibration holdout.

**Example**:

```python
from temporalcv import walk_forward_conformal

intervals, quality = walk_forward_conformal(
    predictions=all_preds,
    actuals=all_actuals,
    calibration_fraction=0.3,
    alpha=0.10
)

print(f"Calibration samples: {quality['calibration_size']}")
print(f"Holdout coverage: {quality['coverage']:.1%}")
print(f"Interval score: {quality['interval_score']:.4f}")
```

---

## Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| Split Conformal | Coverage guarantee, simple | Needs separate calibration set |
| Adaptive Conformal | Handles drift | No finite-sample guarantee |
| Bootstrap | No assumptions | Computationally expensive |
