# API Reference: Conformal Prediction

Distribution-free prediction intervals with coverage guarantees.

**⚠️ Time Series Caveat**: Conformal prediction provides *exact* coverage guarantees only under exchangeability (i.i.d. data). For time series, coverage is *approximate* because temporal dependence violates exchangeability. Use `AdaptiveConformalPredictor` for distribution shift, and expect coverage to be within ±5% of nominal for well-behaved series.

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

---

## Coverage Diagnostics

### `CoverageDiagnostics`

Detailed coverage diagnostics for conformal prediction intervals.

```python
@dataclass
class CoverageDiagnostics:
    overall_coverage: float           # Empirical coverage
    target_coverage: float            # Nominal coverage (1 - α)
    coverage_gap: float               # target - empirical
    undercoverage_warning: bool       # True if gap > threshold
    coverage_by_window: Dict[str, float]  # Window-based coverage
    coverage_by_regime: Optional[Dict[str, float]]  # Per-regime coverage
    n_observations: int               # Total observations
```

---

### `compute_coverage_diagnostics`

Compute detailed coverage diagnostics for prediction intervals.

```python
def compute_coverage_diagnostics(
    intervals: PredictionInterval,
    actuals: np.ndarray,
    *,
    target_coverage: Optional[float] = None,
    window_size: int = 50,
    regimes: Optional[np.ndarray] = None,
    undercoverage_threshold: float = 0.05,
) -> CoverageDiagnostics
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intervals` | `PredictionInterval` | required | Intervals to evaluate |
| `actuals` | `np.ndarray` | required | Actual values |
| `target_coverage` | `float` | `None` | Target level (uses interval.confidence if None) |
| `window_size` | `int` | `50` | Rolling window size for time-based analysis |
| `regimes` | `np.ndarray` | `None` | Regime labels for stratified coverage |
| `undercoverage_threshold` | `float` | `0.05` | Warning threshold for undercoverage |

**Returns**: `CoverageDiagnostics` with detailed coverage analysis

**Example**:

```python
from temporalcv import (
    SplitConformalPredictor,
    compute_coverage_diagnostics,
)

conformal = SplitConformalPredictor(alpha=0.05)
conformal.calibrate(cal_preds, cal_actuals)
intervals = conformal.predict_interval(test_preds)

diag = compute_coverage_diagnostics(
    intervals,
    test_actuals,
    regimes=volatility_regime,  # Optional regime stratification
)

print(f"Coverage: {diag.overall_coverage:.1%}")
print(f"Target: {diag.target_coverage:.1%}")
print(f"Gap: {diag.coverage_gap:+.1%}")

if diag.undercoverage_warning:
    print("WARNING: Coverage significantly below target!")

if diag.coverage_by_regime:
    for regime, cov in diag.coverage_by_regime.items():
        print(f"  {regime}: {cov:.1%}")
```

**Use Cases**:
1. Production monitoring for coverage degradation
2. Identifying time periods with poor coverage
3. Regime-specific performance analysis
