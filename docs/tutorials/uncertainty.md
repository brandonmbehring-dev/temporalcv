# Tutorial: Uncertainty Quantification

Point predictions aren't enough. Learn to generate prediction intervals with coverage guarantees.

---

## Why Uncertainty Matters

A point prediction of "0.05" is meaningless without context:
- Is the model confident? (interval: 0.04 - 0.06)
- Or highly uncertain? (interval: -0.10 - 0.20)

**Decisions require knowing what you don't know.**

---

## Method 1: Split Conformal Prediction

Distribution-free intervals with finite-sample coverage guarantee.

```python
from temporalcv import SplitConformalPredictor
import numpy as np
from sklearn.linear_model import Ridge

# Split data: train, calibration, test
train_end = 150
cal_end = 200

# Fit model
model = Ridge(alpha=1.0)
model.fit(X[:train_end], y[:train_end])

# Calibrate conformal predictor
cal_predictions = model.predict(X[train_end:cal_end])
cal_actuals = y[train_end:cal_end]

conformal = SplitConformalPredictor(alpha=0.10)  # 90% intervals
conformal.calibrate(cal_predictions, cal_actuals)

# Generate intervals for test set
test_predictions = model.predict(X[cal_end:])
intervals = conformal.predict_interval(test_predictions)

print(f"Interval width: {intervals.mean_width:.4f}")
print(f"Coverage: {intervals.coverage(y[cal_end:]):.1%}")
```

### How It Works

1. Compute residuals on calibration set: `|y - ŷ|`
2. Find quantile `q` such that `(1-α)` of residuals are below `q`
3. Interval: `[ŷ - q, ŷ + q]`

### Coverage Guarantee

For exchangeable data with `n` calibration points:

```
P(Y_new ∈ interval) ≥ 1 - α
```

**Note**: Time series violates exchangeability. Coverage may be approximate.

---

## Method 2: Adaptive Conformal Prediction

Adjusts intervals online as new data arrives:

```python
from temporalcv import AdaptiveConformalPredictor

# Initialize with historical errors
adaptive = AdaptiveConformalPredictor(alpha=0.10, gamma=0.05)
adaptive.initialize(cal_predictions, cal_actuals)

# Update as new observations arrive
for i, (pred, actual) in enumerate(zip(test_predictions, y[cal_end:])):
    # Get interval BEFORE seeing actual
    lower, upper = adaptive.predict_interval(pred)

    # Update with actual (for next prediction)
    new_quantile = adaptive.update(pred, actual)

    if i < 5:
        print(f"t={i}: pred={pred:.3f}, interval=[{lower:.3f}, {upper:.3f}]")
```

### When to Use

- Distribution shift expected
- Online learning scenarios
- Long forecast horizons where calibration may drift

---

## Method 3: Bootstrap Uncertainty

Non-parametric intervals from ensemble predictions:

```python
from temporalcv import BootstrapUncertainty

bootstrap = BootstrapUncertainty(
    n_bootstrap=100,
    alpha=0.10,
    random_state=42
)

# Fit on calibration residuals
bootstrap.fit(cal_predictions, cal_actuals)

# Get intervals
intervals = bootstrap.predict_interval(test_predictions)

print(f"Bootstrap interval width: {intervals.mean_width:.4f}")
print(f"Coverage: {intervals.coverage(y[cal_end:]):.1%}")
```

---

## Method 4: Bagging with Uncertainty

Time-series-aware bagging provides natural uncertainty estimates:

```python
from temporalcv import create_block_bagger
from sklearn.linear_model import Ridge

# Create bagger with block bootstrap
bagger = create_block_bagger(
    base_model=Ridge(alpha=1.0),
    n_estimators=50,
    block_length=10,  # Preserve autocorrelation
    random_state=42
)

# Fit
bagger.fit(X[:train_end], y[:train_end])

# Get predictions with uncertainty
mean_pred, std_pred = bagger.predict_with_uncertainty(X[cal_end:])

# Or get intervals directly
mean_pred, lower, upper = bagger.predict_interval(X[cal_end:], alpha=0.10)

# Check coverage
coverage = np.mean((y[cal_end:] >= lower) & (y[cal_end:] <= upper))
print(f"Bagging coverage: {coverage:.1%}")
```

### Bootstrap Strategies

| Strategy | Use When |
|----------|----------|
| `MovingBlockBootstrap` | Standard time series |
| `StationaryBootstrap` | Varying block lengths needed |
| `FeatureBagging` | High-dimensional features |

---

## Evaluating Interval Quality

Coverage isn't everything. Use multiple metrics:

```python
from temporalcv import evaluate_interval_quality

quality = evaluate_interval_quality(intervals, y[cal_end:])

print(f"Coverage: {quality['coverage']:.1%}")
print(f"Coverage gap: {quality['coverage_gap']:.1%}")  # vs nominal
print(f"Mean width: {quality['mean_width']:.4f}")
print(f"Interval score: {quality['interval_score']:.4f}")
```

### Interval Score

Combines coverage and sharpness:

```
IS = (upper - lower) + (2/α) * (lower - y) * I(y < lower) + (2/α) * (y - upper) * I(y > upper)
```

Lower is better. Penalizes:
- Wide intervals (first term)
- Miscoverage (second and third terms)

---

## Walk-Forward Conformal

Apply conformal prediction within walk-forward evaluation:

```python
from temporalcv import walk_forward_conformal

# After walk-forward CV, you have predictions and actuals
# Split into calibration and holdout

intervals, metadata = walk_forward_conformal(
    predictions=all_predictions,
    actuals=all_actuals,
    calibration_fraction=0.3,  # First 30% for calibration
    alpha=0.10
)

print(f"Calibration samples: {metadata['n_calibration']}")
print(f"Holdout samples: {metadata['n_holdout']}")
print(f"Holdout coverage: {metadata['holdout_coverage']:.1%}")
```

---

## Complete Example

```python
import numpy as np
from sklearn.linear_model import Ridge
from temporalcv import (
    WalkForwardCV,
    SplitConformalPredictor,
    AdaptiveConformalPredictor,
    create_block_bagger,
    evaluate_interval_quality,
)

# Generate data
np.random.seed(42)
n = 400
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.9 * y[t-1] + np.random.randn() * 0.1

X = np.column_stack([np.roll(y, i) for i in range(1, 6)])[5:]
y = y[5:]

# Split points
train_end = 200
cal_end = 300

# === Method 1: Split Conformal ===
model = Ridge(alpha=1.0)
model.fit(X[:train_end], y[:train_end])

conformal = SplitConformalPredictor(alpha=0.10)
cal_preds = model.predict(X[train_end:cal_end])
conformal.calibrate(cal_preds, y[train_end:cal_end])

test_preds = model.predict(X[cal_end:])
conf_intervals = conformal.predict_interval(test_preds)

# === Method 2: Bagging ===
bagger = create_block_bagger(
    base_model=Ridge(alpha=1.0),
    n_estimators=30,
    block_length=15,
    random_state=42
)
bagger.fit(X[:train_end], y[:train_end])
bag_mean, bag_lower, bag_upper = bagger.predict_interval(X[cal_end:], alpha=0.10)

# === Compare ===
test_actuals = y[cal_end:]

print("=== Interval Comparison ===")
print(f"\nSplit Conformal:")
print(f"  Coverage: {conf_intervals.coverage(test_actuals):.1%}")
print(f"  Mean width: {conf_intervals.mean_width:.4f}")

bag_coverage = np.mean((test_actuals >= bag_lower) & (test_actuals <= bag_upper))
bag_width = np.mean(bag_upper - bag_lower)
print(f"\nBlock Bootstrap Bagging:")
print(f"  Coverage: {bag_coverage:.1%}")
print(f"  Mean width: {bag_width:.4f}")

# Evaluate conformal quality
quality = evaluate_interval_quality(conf_intervals, test_actuals)
print(f"\nConformal Interval Score: {quality['interval_score']:.4f}")
```

---

## Best Practices

### 1. Separate Calibration from Test

```python
# WRONG: Calibrate on same data you test
conformal.calibrate(test_preds, test_actuals)

# RIGHT: Use separate calibration set
conformal.calibrate(cal_preds, cal_actuals)
```

### 2. Use Enough Calibration Samples

- Minimum: 30-50 samples
- Recommended: 100+ samples
- Finite-sample guarantee improves with n

### 3. Consider Time Series Structure

For time series, use adaptive conformal or bagging with block bootstrap:

```python
# Standard conformal assumes exchangeability
# This is violated by autocorrelation

# Better options:
# 1. Adaptive conformal
adaptive = AdaptiveConformalPredictor(alpha=0.10, gamma=0.05)

# 2. Block bootstrap
bagger = create_block_bagger(model, n_estimators=50)
```

### 4. Report Multiple Metrics

```python
# Don't just report coverage
print(f"Coverage: {coverage:.1%}")
print(f"Mean width: {mean_width:.4f}")
print(f"Interval score: {interval_score:.4f}")
# Reader gets full picture
```

---

## Method Comparison

| Method | Pros | Cons |
|--------|------|------|
| Split Conformal | Coverage guarantee, simple | Needs separate calibration set |
| Adaptive Conformal | Handles drift | No finite-sample guarantee |
| Bootstrap | No assumptions | Computationally expensive |
| Bagging | Built-in, time-aware | May undercover |

---

## API Reference

- [`SplitConformalPredictor`](../api/conformal.md#splitconformalpredictor)
- [`AdaptiveConformalPredictor`](../api/conformal.md#adaptiveconformalpredictor)
- [`BootstrapUncertainty`](../api/conformal.md#bootstrapuncertainty)
- [`evaluate_interval_quality`](../api/conformal.md#evaluate_interval_quality)
- [`TimeSeriesBagger`](../api/bagging.md#timeseriesbagger)
- [`create_block_bagger`](../api/bagging.md#create_block_bagger)
