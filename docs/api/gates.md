# API Reference: Validation Gates

Three-stage validation framework with HALT/PASS/WARN/SKIP decisions for leakage detection.

---

## Enums

### `GateStatus`

Validation gate status enumeration.

| Value | Meaning |
|-------|---------|
| `HALT` | Critical failure - stop and investigate |
| `WARN` | Caution - continue but verify |
| `PASS` | Validation passed |
| `SKIP` | Insufficient data to run gate |

---

## Data Classes

### `GateResult`

Result from a validation gate.

```python
@dataclass
class GateResult:
    name: str                          # Gate identifier
    status: GateStatus                 # HALT, WARN, PASS, or SKIP
    message: str                       # Human-readable description
    metric_value: Optional[float]      # Primary metric for this gate
    threshold: Optional[float]         # Threshold used for decision
    details: dict[str, Any]            # Additional metrics and diagnostics
    recommendation: str                # What to do if not PASS
```

**String representation**: `[STATUS] name: message`

---

### `ValidationReport`

Complete validation report across all gates.

```python
@dataclass
class ValidationReport:
    gates: List[GateResult]
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `status` | `str` | Overall status: "HALT" if any HALT, "WARN" if any WARN, else "PASS" |
| `failures` | `List[GateResult]` | Gates that HALTed |
| `warnings` | `List[GateResult]` | Gates that WARNed |

**Methods**:

- `summary() -> str`: Human-readable report summary

---

## Gate Functions

### `gate_shuffled_target`

**Definitive leakage detection.** If a model beats a shuffled target, features contain temporal information that shouldn't exist.

```python
def gate_shuffled_target(
    model: FitPredictModel,
    X: ArrayLike,
    y: ArrayLike,
    n_shuffles: int = 5,
    threshold: float = 0.05,
    random_state: Optional[int] = None,
) -> GateResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `FitPredictModel` | required | Model with `fit(X, y)` and `predict(X)` methods |
| `X` | `ArrayLike` | required | Feature matrix (n_samples, n_features) |
| `y` | `ArrayLike` | required | Target vector (n_samples,) |
| `n_shuffles` | `int` | `5` | Number of shuffled targets to average over |
| `threshold` | `float` | `0.05` | Maximum allowed improvement ratio (5%) |
| `random_state` | `int` | `None` | Random seed for reproducibility |

**Returns**: `GateResult` with status HALT if model beats shuffled baseline

**Details dict**:
- `mae_real`: MAE on real target
- `mae_shuffled_avg`: Mean MAE on shuffled targets
- `mae_shuffled_all`: List of all shuffled MAEs
- `n_shuffles`: Number of shuffles performed

---

### `gate_synthetic_ar1`

Test model on synthetic AR(1) where theoretical optimum is known.

```python
def gate_synthetic_ar1(
    model: FitPredictModel,
    phi: float = 0.95,
    sigma: float = 1.0,
    n_samples: int = 500,
    n_lags: int = 5,
    tolerance: float = 1.5,
    random_state: Optional[int] = None,
) -> GateResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `FitPredictModel` | required | Model to test |
| `phi` | `float` | `0.95` | AR(1) coefficient |
| `sigma` | `float` | `1.0` | Innovation standard deviation |
| `n_samples` | `int` | `500` | Samples to generate |
| `n_lags` | `int` | `5` | Lag features to create |
| `tolerance` | `float` | `1.5` | How much better model can be than theoretical |
| `random_state` | `int` | `None` | Random seed |

**Returns**: `GateResult` with status HALT if model beats theoretical by too much

**Theoretical bound**: MAE_optimal = σ × √(2/π) ≈ 0.798 × σ

---

### `gate_suspicious_improvement`

Flag too-good-to-be-true improvements over baseline.

```python
def gate_suspicious_improvement(
    model_metric: float,
    baseline_metric: float,
    threshold: float = 0.20,
    warn_threshold: float = 0.10,
    metric_name: str = "MAE",
) -> GateResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_metric` | `float` | required | Model's error metric (lower = better) |
| `baseline_metric` | `float` | required | Baseline error metric |
| `threshold` | `float` | `0.20` | Improvement that triggers HALT (20%) |
| `warn_threshold` | `float` | `0.10` | Improvement that triggers WARN (10%) |
| `metric_name` | `str` | `"MAE"` | Metric name for messages |

**Returns**: `GateResult` with appropriate status based on improvement

---

### `gate_temporal_boundary`

Verify proper gap between training and test for h-step forecasts.

```python
def gate_temporal_boundary(
    train_end_idx: int,
    test_start_idx: int,
    horizon: int,
    gap: int = 0,
) -> GateResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_end_idx` | `int` | required | Last training index (inclusive) |
| `test_start_idx` | `int` | required | First test index |
| `horizon` | `int` | required | Forecast horizon (h) |
| `gap` | `int` | `0` | Additional gap beyond horizon |

**Returns**: `GateResult` with status HALT if boundary violated

**Requirement**: `test_start_idx >= train_end_idx + horizon + gap`

---

## Runner

### `run_gates`

Aggregate gate results into a validation report.

```python
def run_gates(gates: List[GateResult]) -> ValidationReport
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `gates` | `List[GateResult]` | Pre-computed gate results |

**Returns**: `ValidationReport` with overall status and summary

**Example**:

```python
from temporalcv.gates import (
    run_gates,
    gate_shuffled_target,
    gate_suspicious_improvement,
)

results = [
    gate_shuffled_target(model, X, y, random_state=42),
    gate_suspicious_improvement(model_mae, persistence_mae),
]

report = run_gates(results)

if report.status == "HALT":
    print(report.summary())
    for failure in report.failures:
        print(f"  - {failure.name}: {failure.recommendation}")
```
