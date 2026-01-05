# Quickstart Guide

Get started with temporalcv in 5 minutes.

---

## Installation

```bash
pip install temporalcv
```

---

## Step 1: Create Sample Data

```python
import numpy as np
from sklearn.linear_model import Ridge

# Generate AR(1) time series (high persistence)
np.random.seed(42)
n = 200
phi = 0.95  # High autocorrelation
y = np.zeros(n)
for t in range(1, n):
    y[t] = phi * y[t-1] + np.random.randn() * 0.1

# Create lag features (CORRECT way: will be done per-split)
# For now, simple features
X = np.column_stack([
    np.roll(y, 1),  # lag-1
    np.roll(y, 2),  # lag-2
])[2:]  # Remove first 2 rows (NaN from rolling)
y = y[2:]
```

---

## Step 2: Validate with Gates

**Before training, check for leakage:**

```python
from temporalcv import run_gates
from temporalcv.gates import gate_shuffled_target, gate_suspicious_improvement

# Create a simple model
model = Ridge(alpha=1.0)

# Run the shuffled target test (permutation mode - default)
# n_shuffles>=100 required for statistical power in permutation mode
shuffled_result = gate_shuffled_target(
    model=model,
    X=X,
    y=y,
    n_shuffles=100,
    random_state=42
)

print(shuffled_result)
# GateResult(name='shuffled_target', status=PASS, ...)

# If your model beats shuffled targets, something is WRONG
if shuffled_result.status.name == "HALT":
    raise ValueError("Leakage detected! Model beats shuffled baseline.")
```

---

## Step 3: Walk-Forward Cross-Validation

**Use temporal CV that respects time ordering:**

```python
from temporalcv import WalkForwardCV

# Sliding window with gap enforcement
cv = WalkForwardCV(
    n_splits=5,
    window_type="sliding",
    window_size=100,  # 100 observations per training window
    horizon=2,        # Minimum separation for 2-step forecasts
    extra_gap=0,      # Optional: additional safety margin (default: 0)
    test_size=1       # 1 test period per split
)

# Evaluate
from sklearn.metrics import mean_absolute_error

maes = []
for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    mae = mean_absolute_error(y[test_idx], preds)
    maes.append(mae)

print(f"Mean MAE: {np.mean(maes):.4f} (+/- {np.std(maes):.4f})")
```

---

## Step 4: Statistical Significance

**Is your model actually better than persistence?**

```python
from temporalcv import dm_test

# Get predictions from both models
model.fit(X[:150], y[:150])
model_preds = model.predict(X[150:])
persistence_preds = np.zeros(len(y[150:]))  # Persistence predicts 0 change

# Actual values
actuals = y[150:]

# Compute errors
model_errors = actuals - model_preds
persistence_errors = actuals - persistence_preds

# Diebold-Mariano test
result = dm_test(
    errors_1=model_errors,
    errors_2=persistence_errors,
    h=1,  # 1-step forecast
    loss="absolute"
)

print(f"DM statistic: {result.statistic:.3f}")
print(f"p-value: {result.pvalue:.4f}")
print(f"Significant at 5%: {result.significant_at_05}")
```

---

## Step 5: Uncertainty Quantification

**Add prediction intervals:**

```python
from temporalcv import SplitConformalPredictor

# Split data: train, calibration, test
train_end = 120
cal_end = 160

# Fit model
model.fit(X[:train_end], y[:train_end])

# Calibrate conformal predictor
cal_preds = model.predict(X[train_end:cal_end])
cal_actuals = y[train_end:cal_end]

conformal = SplitConformalPredictor(alpha=0.10)  # 90% intervals
conformal.calibrate(cal_preds, cal_actuals)

# Generate intervals for test set
test_preds = model.predict(X[cal_end:])
intervals = conformal.predict_interval(test_preds)

print(f"Mean interval width: {intervals.mean_width:.4f}")
print(f"Coverage on test: {intervals.coverage(y[cal_end:]):.1%}")
```

---

## Step 6: High-Persistence Metrics

**Standard MAE misleads for sticky series. Use move-conditional metrics:**

```python
from temporalcv import compute_move_threshold, compute_move_conditional_metrics

# Compute threshold from TRAINING CHANGES only (critical!)
train_changes = np.diff(y[:train_end])
threshold = compute_move_threshold(train_changes, percentile=70)

# IMPORTANT: MC-SS works on CHANGES, not levels
# Convert predictions and actuals to changes
pred_changes = np.diff(test_preds)
actual_changes = np.diff(y[cal_end:])

mc_result = compute_move_conditional_metrics(
    predictions=pred_changes,
    actuals=actual_changes,
    threshold=threshold
)

print(f"MC-SS (skill score): {mc_result.skill_score:.3f}")
print(f"MAE on UP moves: {mc_result.mae_up:.4f}")
print(f"MAE on DOWN moves: {mc_result.mae_down:.4f}")
print(f"MAE on FLAT: {mc_result.mae_flat:.4f}")
```

---

## Complete Workflow

Here's everything together:

```python
import numpy as np
from sklearn.linear_model import Ridge
from temporalcv import (
    WalkForwardCV,
    run_gates,
    dm_test,
    SplitConformalPredictor,
    compute_move_threshold,
    compute_move_conditional_metrics,
)
from temporalcv.gates import gate_shuffled_target

# 1. Generate data
np.random.seed(42)
n = 300
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.95 * y[t-1] + np.random.randn() * 0.1

X = np.column_stack([np.roll(y, i) for i in range(1, 6)])[5:]
y = y[5:]

# 2. Validate - no leakage
model = Ridge(alpha=1.0)
gate_result = gate_shuffled_target(model, X, y, random_state=42)
assert gate_result.status.name != "HALT", "Leakage detected!"

# 3. Walk-forward evaluation
cv = WalkForwardCV(n_splits=5, window_type="sliding", window_size=150, horizon=2, extra_gap=0)
predictions_all, actuals_all = [], []

for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    predictions_all.extend(preds)
    actuals_all.extend(y[test_idx])

predictions_all = np.array(predictions_all)
actuals_all = np.array(actuals_all)

# 4. Statistical test vs persistence
model_errors = actuals_all - predictions_all
persistence_errors = actuals_all  # Persistence predicts 0

dm_result = dm_test(model_errors, persistence_errors, h=2, loss="absolute")
print(f"DM test p-value: {dm_result.pvalue:.4f}")

# 5. Move-conditional metrics
threshold = compute_move_threshold(y[:200], percentile=70)
mc = compute_move_conditional_metrics(predictions_all, actuals_all, threshold)
print(f"MC-SS: {mc.skill_score:.3f}")

print("\n✓ Validation complete!")
```

---

## Next Steps

- **[Leakage Detection Tutorial](tutorials/leakage_detection.md)** — Deep dive into validation gates
- **[Walk-Forward CV Tutorial](tutorials/walk_forward_cv.md)** — Advanced CV configurations
- **[API Reference](api/gates.md)** — Complete function documentation
