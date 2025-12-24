# Tutorial: Leakage Detection with Validation Gates

Data leakage is the #1 cause of inflated ML results. This tutorial shows how to catch it before it corrupts your work.

---

## The Problem

Time-series leakage is insidious:

```python
# WRONG: Lag features from full series
X = np.column_stack([np.roll(y, i) for i in range(1, 6)])  # Leaks future!

# WRONG: Thresholds from full data
threshold = np.percentile(np.abs(y), 70)  # Future info in classification!

# WRONG: No gap between train and test
train, test = y[:100], y[100:]  # y[99] leaks into test via features!
```

**If your model shows >20% improvement over persistence baseline, be suspicious.**

---

## Gate 1: Shuffled Target Test (Definitive)

The shuffled target test is the **definitive** leakage detector:

```python
from temporalcv.gates import gate_shuffled_target
from sklearn.linear_model import Ridge

model = Ridge()

result = gate_shuffled_target(
    model=model,
    X=X,
    y=y,
    n_shuffles=5,      # Average over 5 shuffles
    threshold=0.05,    # Max 5% improvement allowed
    random_state=42
)

print(result)
```

### How It Works

1. Fit model on real `(X, y)`, compute MAE
2. Shuffle `y` randomly (destroys temporal relationship)
3. Fit model on `(X, y_shuffled)`, compute MAE
4. If real MAE < shuffled MAE * (1 - threshold), something is wrong

**If your model beats a shuffled target, features contain temporal information that shouldn't exist.**

### Interpreting Results

| Status | Meaning | Action |
|--------|---------|--------|
| `PASS` | Model doesn't beat shuffled baseline | Safe to proceed |
| `HALT` | Model significantly beats shuffled | **Investigate immediately** |

```python
if result.status.name == "HALT":
    print(f"LEAKAGE DETECTED!")
    print(f"Real MAE: {result.details['mae_real']:.4f}")
    print(f"Shuffled MAE: {result.details['mae_shuffled_avg']:.4f}")
    print(f"Improvement: {result.details['improvement_ratio']:.1%}")
```

---

## Gate 2: Synthetic AR(1) Bounds

Verify your model on data with known theoretical properties:

```python
from temporalcv.gates import gate_synthetic_ar1

result = gate_synthetic_ar1(
    model=model,
    phi=0.95,          # AR(1) coefficient
    sigma=1.0,         # Innovation std
    n_samples=500,
    n_lags=5,          # Number of lag features
    tolerance=1.5,     # Allow 50% above theoretical MAE
    random_state=42
)

print(result)
```

### How It Works

For AR(1) with coefficient φ, the theoretical optimal 1-step MAE is:

```
MAE_optimal = σ * sqrt(2/π) ≈ 0.798 * σ
```

If your model achieves MAE significantly below this, something is wrong.

### When to Use

- Model development sanity check
- Validating new feature engineering
- Regression testing after code changes

---

## Gate 3: Suspicious Improvement

Automatically flag too-good results:

```python
from temporalcv.gates import gate_suspicious_improvement

# After computing metrics
model_mae = 0.05
baseline_mae = 0.10  # 50% improvement - suspicious!

result = gate_suspicious_improvement(
    model_metric=model_mae,
    baseline_metric=baseline_mae,
    threshold=0.20,      # >20% improvement = HALT
    warn_threshold=0.10, # >10% improvement = WARN
    metric_name="MAE"
)

print(result)
```

### Why 20%?

From empirical studies on financial time series:
- 5-10% improvement: Plausible with good features
- 10-20%: Unusual, warrants investigation
- >20%: Almost certainly a bug or leakage

---

## Gate 4: Temporal Boundary

Verify proper separation between train and test:

```python
from temporalcv.gates import gate_temporal_boundary

result = gate_temporal_boundary(
    train_end_idx=99,
    test_start_idx=100,
    horizon=2,
    gap=0  # Current gap between train and test
)

if result.status.name == "HALT":
    print(f"Gap too small! Need {result.details['required_gap']}, have {gap}")
```

### Gap Calculation

For h-step forecasting, the last training observation must be at least `h` steps before the first test observation:

```
train_end_idx + gap >= test_start_idx - horizon + 1
```

---

## Combining Gates

Run multiple gates and aggregate results:

```python
from temporalcv import run_gates
from temporalcv.gates import (
    gate_shuffled_target,
    gate_suspicious_improvement,
    gate_temporal_boundary,
)

# Collect gate results
gates = [
    gate_shuffled_target(model, X, y, random_state=42),
    gate_suspicious_improvement(model_mae, baseline_mae),
    gate_temporal_boundary(train_end, test_start, horizon=2),
]

# Aggregate
report = run_gates(gates)

print(report.summary())

# Check overall status
if report.status == "HALT":
    print("\n❌ VALIDATION FAILED")
    for failure in report.failures:
        print(f"  - {failure.name}: {failure.message}")
elif report.status == "WARN":
    print("\n⚠️ WARNINGS PRESENT")
    for warning in report.warnings:
        print(f"  - {warning.name}: {warning.message}")
else:
    print("\n✓ All gates passed")
```

---

## Real-World Example: Detecting Lag Leakage

```python
import numpy as np
from sklearn.linear_model import Ridge
from temporalcv.gates import gate_shuffled_target

# Generate high-persistence series
np.random.seed(42)
n = 300
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.98 * y[t-1] + np.random.randn() * 0.05

# WRONG: Create lag features from full series (LEAKAGE!)
X_leaky = np.column_stack([np.roll(y, i) for i in range(1, 4)])[3:]
y_leaky = y[3:]

# Test for leakage
model = Ridge(alpha=1.0)
result = gate_shuffled_target(model, X_leaky, y_leaky, random_state=42)

print(f"Status: {result.status.name}")
print(f"Real MAE: {result.details['mae_real']:.4f}")
print(f"Shuffled MAE: {result.details['mae_shuffled_avg']:.4f}")
# Status: HALT (leakage detected!)
```

### The Fix

Compute lag features within each CV split:

```python
from temporalcv import WalkForwardCV

cv = WalkForwardCV(n_splits=5, window_type="sliding", window_size=100, gap=2)

for train_idx, test_idx in cv.split(y):
    # Create features ONLY from training portion
    y_train = y[train_idx]

    # Now use these features...
```

---

## Best Practices

1. **Always run shuffled target test** before trusting results
2. **Set threshold to 20%** for suspicious improvement (lower for financial data)
3. **Use AR(1) synthetic test** during development
4. **Enforce gap >= horizon** in all CV splits
5. **Compute thresholds from training data only**

---

## API Reference

- [`gate_shuffled_target`](../api/gates.md#gate_shuffled_target)
- [`gate_synthetic_ar1`](../api/gates.md#gate_synthetic_ar1)
- [`gate_suspicious_improvement`](../api/gates.md#gate_suspicious_improvement)
- [`gate_temporal_boundary`](../api/gates.md#gate_temporal_boundary)
- [`run_gates`](../api/gates.md#run_gates)
