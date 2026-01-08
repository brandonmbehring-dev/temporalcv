# Failure Cases Guide

Learn from common time-series ML mistakes. Each case study shows what goes wrong, why it happens, and how to fix it.

---

## Why Study Failures?

> "Good judgment comes from experience. Experience comes from bad judgment."

These five examples demonstrate the most common ways time-series ML pipelines fail. Understanding these patterns helps you:

1. **Recognize** leakage before it corrupts results
2. **Use** the right validation gates to catch issues
3. **Apply** correct fixes without over-engineering

---

## The Five Failure Modes

| # | Failure | Root Cause | Gate That Catches It |
|---|---------|------------|---------------------|
| 16 | Rolling Stats | `.rolling()` without `.shift()` | `gate_shuffled_target` |
| 17 | Threshold Leak | Quantile on full series | Regime validation |
| 18 | Nested DM Test | DM bias for nested models | Statistical knowledge |
| 19 | Missing Gap | No gap for h-step | `gate_temporal_boundary` |
| 20 | KFold Trap | Random CV on time series | `gate_shuffled_target` |

---

## Failure 16: Rolling Stats Without Shift

**File**: [`examples/16_failure_rolling_stats.py`](../../examples/16_failure_rolling_stats.py)

### The Problem

Rolling statistics are the most common source of leakage in time-series ML:

```python
# WRONG - Leaks future information
df['ma_20'] = df['price'].rolling(20).mean()
df['volatility_20'] = df['price'].rolling(20).std()
```

At time `t`, the rolling mean includes `price[t]` in the calculation. If you're predicting `price[t+1]` based on information at `t`, this seems fine. But if your target is defined using `price[t]` (e.g., returns, classification), you've leaked.

### Why It's Dangerous

- The leakage is subtle and often passes manual review
- Cross-validation metrics look great (because the leak helps prediction)
- Real-world performance crashes when the leak disappears

### The Fix

Always shift before rolling:

```python
# CORRECT - Uses only past data
df['ma_20'] = df['price'].shift(1).rolling(20).mean()
df['volatility_20'] = df['price'].shift(1).rolling(20).std()
```

### Detection

```python
from temporalcv.gates import gate_shuffled_target

result = gate_shuffled_target(model, X, y, n_shuffles=100)
if result.status == "HALT":
    print("LEAKAGE DETECTED: Features encode target position")
```

---

## Failure 17: Threshold Computed on Full Data

**File**: [`examples/17_failure_threshold_leak.py`](../../examples/17_failure_threshold_leak.py)

### The Problem

Regime thresholds or classification boundaries computed on the full dataset leak future information:

```python
# WRONG - Uses future to define "high volatility"
threshold = df['volatility'].quantile(0.8)
df['high_vol'] = df['volatility'] > threshold
```

At any point in time, this threshold uses the full history—including future values. The model "knows" what constitutes high volatility in hindsight.

### Why It's Dangerous

- Regime transitions become artificially predictable
- Backtests show false alpha from regime-timing
- The threshold shifts in live trading, breaking the model

### The Fix

Use expanding windows for thresholds:

```python
# CORRECT - Only uses past data
expanding_threshold = df['volatility'].expanding().quantile(0.8).shift(1)
df['high_vol'] = df['volatility'] > expanding_threshold
```

### Detection

Look for suspiciously high performance during regime transitions. If your model perfectly times regime shifts, the threshold is probably leaked.

---

## Failure 18: DM Test for Nested Models

**File**: [`examples/18_failure_nested_dm.py`](../../examples/18_failure_nested_dm.py)

### The Problem

The Diebold-Mariano test is biased when comparing nested models:

```python
# Model A: y_t = c + e_t (random walk with drift)
# Model B: y_t = c + beta*x_t + e_t (adds predictor)

# WRONG - DM test is biased under H0: beta=0
dm_result = dm_test(errors_A, errors_B)  # Inflated rejection rate!
```

Under the null hypothesis (B adds nothing), the DM test over-rejects. It says B is "significantly better" when it's actually just capturing noise.

### Why It Happens

The loss differential variance estimator is inconsistent when one model nests the other. Clark & West (2007) proved this bias and provided a correction.

### The Fix

Use the Clark-West test for nested model comparisons:

```python
from temporalcv import cw_test

# CORRECT - CW test adjusts for nesting bias
cw_result = cw_test(errors_A, errors_B, nested=True)
```

### When to Use Each Test

| Situation | Test |
|-----------|------|
| Non-nested models (RF vs XGBoost) | Diebold-Mariano |
| Nested models (AR vs AR+X) | Clark-West |
| Unsure | Clark-West (conservative) |

---

## Failure 19: Missing Gap for H-Step Forecasting

**File**: [`examples/19_failure_missing_gap.py`](../../examples/19_failure_missing_gap.py)

### The Problem

For h-step ahead forecasting, you need a gap of at least h between training and test data:

```python
# WRONG - No gap for 5-step ahead forecast
cv = WalkForwardCV(n_splits=5)
# Train: [0, 1, ..., 99], Test: [100]
# But we're predicting y[100] using info from t=95!
```

If you're forecasting 5 steps ahead, the last 5 training observations contain information about the target.

### Why It's Dangerous

- Model learns the transition from `train[-5:]` to `test[0]`
- Backtest shows skill that evaporates in live prediction
- Particularly severe with trending or autoregressive data

### The Fix

Set `horizon` parameter to enforce the gap:

```python
# CORRECT - Gap enforces h-step separation
cv = WalkForwardCV(n_splits=5, horizon=5)
# Train: [0, 1, ..., 94], GAP: [95-99], Test: [100]
```

### Detection

```python
from temporalcv.gates import gate_temporal_boundary

result = gate_temporal_boundary(cv, horizon=5)
if result.status == "HALT":
    print("GAP VIOLATION: Insufficient separation for h-step forecast")
```

---

## Failure 20: KFold on Time Series (47.8% Fake Improvement)

**File**: [`examples/20_failure_kfold.py`](../../examples/20_failure_kfold.py)

### The Problem

Using sklearn's `KFold` on time series destroys temporal order:

```python
from sklearn.model_selection import KFold

# WRONG - Random splits leak future into training
cv = KFold(n_splits=5, shuffle=True)
score = cross_val_score(model, X, y, cv=cv)  # Optimistically biased!
```

### Why It's Catastrophic

In the example, a simple autoregressive model shows:
- **KFold**: MAE = 0.52
- **WalkForwardCV**: MAE = 0.77

That's a **47.8% fake improvement** from the validation bug alone!

### Why It Happens

When you shuffle:
1. Future observations end up in training data
2. The model learns temporal patterns that include future info
3. Test performance looks great because it's trained on the answer

### The Fix

Always use temporal cross-validation:

```python
from temporalcv import WalkForwardCV

# CORRECT - Temporal order preserved
cv = WalkForwardCV(n_splits=5, gap=1)
score = cross_val_score(model, X, y, cv=cv)  # Realistic estimate
```

### Detection

```python
from temporalcv.gates import gate_shuffled_target

result = gate_shuffled_target(model, X, y, n_shuffles=100)
# If the model beats shuffled targets, something is leaking
```

---

## Summary: The Detection Arsenal

| Gate | What It Catches | When to Use |
|------|-----------------|-------------|
| `gate_shuffled_target` | Feature leakage, KFold trap | Always (first check) |
| `gate_temporal_boundary` | Insufficient gap | h-step forecasting |
| `gate_suspicious_improvement` | Unrealistic performance | Any time results seem too good |

### The Golden Rule

> If your model beats a baseline by more than 20% on first try, something is probably wrong.

---

## See Also

- [Examples Index](examples_index.md) - All 21 examples
- [Common Pitfalls](../guide/common_pitfalls.md) - 8 anti-patterns
- [Why Time Series Is Different](../guide/why_time_series_is_different.md) - Foundational concepts
