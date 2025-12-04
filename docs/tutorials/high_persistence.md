# Tutorial: High-Persistence Time Series

Standard metrics mislead when your series barely moves. Learn to evaluate models properly.

---

## The Problem

High-persistence series have strong autocorrelation (ACF(1) > 0.9):

```python
# Interest rates, unemployment, many financial series
# These move slowly - most observations are "flat"

import numpy as np
np.random.seed(42)

# Simulate high-persistence series
n = 200
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.98 * y[t-1] + np.random.randn() * 0.02  # ACF(1) ≈ 0.98
```

### Why Standard Metrics Fail

**Persistence baseline (predict no change) achieves excellent MAE:**

```python
changes = np.diff(y)
persistence_mae = np.mean(np.abs(changes))
print(f"Persistence MAE: {persistence_mae:.4f}")  # Very low!

# A model that predicts 0 for everything looks "good"
# But it's completely useless for detecting actual moves
```

---

## Move-Conditional Metrics

Evaluate performance separately for UP, DOWN, and FLAT periods:

```python
from temporalcv import (
    compute_move_threshold,
    compute_move_conditional_metrics,
    classify_moves
)

# Step 1: Compute threshold from TRAINING data only
train_changes = changes[:150]
threshold = compute_move_threshold(train_changes, percentile=70)
print(f"Move threshold: {threshold:.4f}")

# Step 2: Classify test observations
test_changes = changes[150:]
moves = classify_moves(test_changes, threshold)
print(f"Moves: {np.unique(moves, return_counts=True)}")
```

---

## MC-SS: Move-Conditional Skill Score

The key metric for high-persistence series:

```python
# Get model predictions
predictions = model.predict(X_test)
actuals = y_test

# Compute move-conditional metrics
mc_result = compute_move_conditional_metrics(
    predictions=predictions,
    actuals=actuals,
    threshold=threshold
)

print(f"MC-SS: {mc_result.skill_score:.3f}")
print(f"MAE on UP moves: {mc_result.mae_up:.4f}")
print(f"MAE on DOWN moves: {mc_result.mae_down:.4f}")
print(f"MAE on FLAT: {mc_result.mae_flat:.4f}")
print(f"Reliable: {mc_result.is_reliable}")
```

### Interpreting MC-SS

| MC-SS | Meaning |
|-------|---------|
| < 0 | Worse than persistence on moves |
| 0 | Same as persistence (no skill) |
| 0.1-0.2 | Modest skill on moves |
| > 0.2 | Strong skill (verify not leakage!) |

---

## Direction Accuracy

Did you predict the right direction?

```python
from temporalcv import compute_direction_accuracy

# 2-class (sign-based)
acc_2class = compute_direction_accuracy(predictions, actuals)
print(f"Direction accuracy (2-class): {acc_2class:.1%}")

# 3-class (with move threshold)
acc_3class = compute_direction_accuracy(predictions, actuals, move_threshold=threshold)
print(f"Direction accuracy (3-class): {acc_3class:.1%}")
```

### 2-Class vs 3-Class

- **2-class**: UP (positive) vs DOWN (negative)
  - Fails when most changes are near zero
  - Small noise determines "direction"

- **3-class**: UP, DOWN, FLAT
  - More realistic for high-persistence
  - Matches what you actually care about

---

## Persistence Baseline

Always compare against persistence:

```python
from temporalcv import compute_persistence_mae

# Overall persistence MAE
persistence_mae = compute_persistence_mae(actuals)
print(f"Persistence MAE: {persistence_mae:.4f}")

# Persistence MAE on moves only
persistence_mae_moves = compute_persistence_mae(actuals, threshold=threshold)
print(f"Persistence MAE (moves only): {persistence_mae_moves:.4f}")
```

---

## Regime-Aware Evaluation

Performance often varies by volatility regime:

```python
from temporalcv import classify_volatility_regime, get_combined_regimes

# Classify volatility (use changes, not levels!)
vol_regimes = classify_volatility_regime(
    values=y,
    window=13,        # ~1 quarter
    basis="changes",  # CRITICAL: use changes
)

# Classify direction
dir_regimes = classify_direction_regime(actuals, threshold)

# Combine
combined = get_combined_regimes(vol_regimes[-len(actuals):], dir_regimes)

# Evaluate per regime
for regime in np.unique(combined):
    mask = combined == regime
    if mask.sum() < 10:  # Skip small regimes
        continue

    regime_mae = np.mean(np.abs(predictions[mask] - actuals[mask]))
    print(f"{regime}: MAE = {regime_mae:.4f} (n={mask.sum()})")
```

---

## Complete Workflow

```python
import numpy as np
from sklearn.linear_model import Ridge
from temporalcv import (
    WalkForwardCV,
    compute_move_threshold,
    compute_move_conditional_metrics,
    compute_persistence_mae,
)

# Generate high-persistence data
np.random.seed(42)
n = 400
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.97 * y[t-1] + np.random.randn() * 0.03

# Create features
X = np.column_stack([np.roll(y, i) for i in range(1, 6)])[5:]
y = y[5:]
changes = np.diff(y)
y_changes = changes
X_changes = X[1:]

# Split points
train_end = 250
cal_end = 300

# Compute threshold from TRAINING ONLY
train_changes = y_changes[:train_end]
threshold = compute_move_threshold(train_changes, percentile=70)
print(f"Move threshold: {threshold:.4f}")

# Train model
model = Ridge(alpha=1.0)
model.fit(X_changes[:train_end], y_changes[:train_end])

# Predict on test
test_predictions = model.predict(X_changes[cal_end:])
test_actuals = y_changes[cal_end:]

# Standard MAE (misleading)
standard_mae = np.mean(np.abs(test_predictions - test_actuals))
persistence_mae = compute_persistence_mae(test_actuals)

print(f"\nStandard MAE: {standard_mae:.4f}")
print(f"Persistence MAE: {persistence_mae:.4f}")
print(f"Improvement: {(persistence_mae - standard_mae) / persistence_mae:.1%}")

# Move-conditional (informative)
mc = compute_move_conditional_metrics(
    predictions=test_predictions,
    actuals=test_actuals,
    threshold=threshold
)

print(f"\n--- Move-Conditional Metrics ---")
print(f"MC-SS: {mc.skill_score:.3f}")
print(f"MAE on UP: {mc.mae_up:.4f} (n={mc.n_up})")
print(f"MAE on DOWN: {mc.mae_down:.4f} (n={mc.n_down})")
print(f"MAE on FLAT: {mc.mae_flat:.4f} (n={mc.n_flat})")
print(f"Reliable: {mc.is_reliable}")
```

---

## Statistical Testing for High-Persistence

Use Pesaran-Timmermann test for direction accuracy:

```python
from temporalcv import pt_test

result = pt_test(
    actual=test_actuals,
    predicted=test_predictions,
    move_threshold=threshold  # 3-class mode
)

print(f"Direction accuracy: {result.accuracy:.1%}")
print(f"Expected (random): {result.expected:.1%}")
print(f"Skill: {result.skill:.1%}")
print(f"p-value: {result.pvalue:.4f}")
print(f"Significant: {result.significant_at_05}")
```

---

## Best Practices

### 1. Always Use Training-Only Threshold

```python
# WRONG
threshold = compute_move_threshold(all_data)

# RIGHT
threshold = compute_move_threshold(training_data_only)
```

### 2. Report Both Standard and Move-Conditional

```python
print(f"Standard MAE: {standard_mae:.4f}")
print(f"MC-SS: {mc.skill_score:.3f}")
# Reader can see both perspectives
```

### 3. Check Regime Reliability

```python
if mc.n_up < 10 or mc.n_down < 10:
    print("Warning: Insufficient samples for reliable move-conditional metrics")
```

### 4. Use Changes, Not Levels

```python
# WRONG
vol_regimes = classify_volatility_regime(prices, basis="levels")

# RIGHT
vol_regimes = classify_volatility_regime(prices, basis="changes")
```

---

## API Reference

- [`compute_move_threshold`](../api/persistence.md#compute_move_threshold)
- [`compute_move_conditional_metrics`](../api/persistence.md#compute_move_conditional_metrics)
- [`classify_moves`](../api/persistence.md#classify_moves)
- [`compute_direction_accuracy`](../api/persistence.md#compute_direction_accuracy)
- [`classify_volatility_regime`](../api/regimes.md#classify_volatility_regime)
- [`pt_test`](../api/statistical_tests.md#pt_test)
