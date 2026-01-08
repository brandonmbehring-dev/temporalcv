# Why Time Series Is Different

```{admonition} TL;DR
:class: tip

Time series data violates the **independent and identically distributed (IID)** assumption
that standard ML tools rely on. Using `sklearn.model_selection.KFold` on time series
data causes **data leakage** and produces **optimistically biased** performance estimates.
```

## The Problem You Didn't Know You Had

If you've trained models using scikit-learn, you're familiar with this pattern:

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

# Standard cross-validation
cv = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)
print(f"CV Score: {scores.mean():.3f}")  # Looks great!
```

**This works perfectly for tabular data.** Each row is independent—shuffling doesn't
change anything fundamental about the data distribution.

**For time series, this code is silently broken.** The model will appear to perform
well in cross-validation but fail in production.

## Why Shuffling Breaks Everything

Consider predicting tomorrow's stock price. With `KFold(shuffle=True)`:

```
Timeline: [Jan 1] [Jan 2] [Jan 3] [Jan 4] [Jan 5]

Fold 1 Train: [Jan 1] [Jan 3] [Jan 5]  <- Knows the FUTURE!
Fold 1 Test:  [Jan 2] [Jan 4]           <- Testing on the PAST!
```

The model learns from January 3rd to predict January 2nd. In production, you
never have tomorrow's data to predict today. This is called **temporal leakage**.

```{admonition} The Leakage Effect
:class: warning

Models trained with temporal leakage typically show:
- Cross-validation error 30-70% **lower** than production error
- Confidence intervals that exclude the true value
- Features that look predictive but aren't

You won't catch this with standard metrics. The numbers look good until deployment.
```

## The Solution: Walk-Forward Validation

Time series cross-validation must respect the **arrow of time**:

```
Timeline: [Jan 1] [Jan 2] [Jan 3] [Jan 4] [Jan 5]

Fold 1: Train [Jan 1]         → Test [Jan 2]
Fold 2: Train [Jan 1, Jan 2]  → Test [Jan 3]
Fold 3: Train [Jan 1-3]       → Test [Jan 4]
Fold 4: Train [Jan 1-4]       → Test [Jan 5]
```

Every training set contains only data that was **available before** the test period.

```python
from temporalcv import WalkForwardCV

# Time-aware cross-validation
cv = WalkForwardCV(
    n_splits=4,
    min_train_periods=10,  # Minimum training history
    gap=1,                 # Prediction horizon (h-step ahead)
)

for train_idx, test_idx in cv.split(X, y):
    assert train_idx.max() < test_idx.min(), "Time order preserved!"
```

## The Gap Parameter: Your Horizon Matters

If you're forecasting 5 days ahead, you need a **gap** between training and test:

```
Without gap (h=5 forecast):
  Train: [Day 1...Day 100]
  Test:  [Day 101]  <- But you don't have Day 96-100 features at prediction time!

With gap=5:
  Train: [Day 1...Day 95]
  Test:  [Day 101]  <- Features only use data through Day 95
```

```python
cv = WalkForwardCV(
    n_splits=4,
    gap=5,  # 5-day forecast horizon
)
```

```{admonition} Rule of Thumb
:class: note

Set `gap` equal to your forecast horizon. If predicting 1 week ahead, use `gap=7`
(for daily data). This ensures your cross-validation matches production conditions.
```

## Beyond Cross-Validation: Validation Gates

Even with proper CV, subtle leakages can occur:

1. **Feature engineering leakage**: Computing rolling means on the full dataset
2. **Target leakage**: Using future information in feature calculations
3. **Regime leakage**: Defining market regimes using future data

temporalcv provides **validation gates** that catch these issues:

```python
from temporalcv.gates import run_gates

# Automatic leakage detection
result = run_gates(X_train, y_train, X_test, y_test)

if result.status == "HALT":
    print(f"LEAKAGE DETECTED: {result.reason}")
    # Fix the pipeline before proceeding
elif result.status == "WARN":
    print(f"Potential issue: {result.reason}")
```

### Gate Types

| Gate | What It Catches | When It Fires |
|------|-----------------|---------------|
| `temporal_boundary` | Train/test time overlap | `max(train_time) >= min(test_time)` |
| `shuffled_target` | Random target patterns | `r²(y, shuffle(y)) > 0.1` |
| `regime_leakage` | Future regime info in features | Regime-conditioned r² too high |
| `feature_correlation` | Features that "know" test data | Suspicious train/test correlations |

## Quick Comparison: Standard ML vs Time Series

| Aspect | Standard ML (IID) | Time Series |
|--------|-------------------|-------------|
| **CV Method** | `KFold(shuffle=True)` | `WalkForwardCV` |
| **Gap Parameter** | Not needed | Required (= forecast horizon) |
| **Feature Engineering** | Any aggregations | Must use `.shift()` |
| **Normalization** | Fit on all data | Fit on train only, per fold |
| **Performance Metric** | Any standard metric | MASE, directional accuracy |
| **Statistical Test** | t-test (IID errors) | DM test (serial correlation robust) |

## Common Mistakes (and How temporalcv Catches Them)

### Mistake 1: Rolling Features Without Shift

```python
# WRONG: Uses future data
df['rolling_mean'] = df['price'].rolling(5).mean()

# RIGHT: Shift to use only past data
df['rolling_mean'] = df['price'].shift(1).rolling(5).mean()
```

temporalcv's `gate_shuffled_target` detects when features contain future information.

### Mistake 2: Normalizing on Full Dataset

```python
# WRONG: Test data statistics leak into training
scaler.fit(X_all)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# RIGHT: Fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Uses train statistics
```

This must be done **inside each CV fold**, not once globally.

### Mistake 3: Computing Thresholds on Full Data

```python
# WRONG: Threshold computed using future data
threshold = df['returns'].quantile(0.90)  # Uses ALL data
df['high_volatility'] = df['returns'] > threshold

# RIGHT: Expanding threshold
df['high_volatility'] = df['returns'] > df['returns'].expanding().quantile(0.90).shift(1)
```

temporalcv's `gate_regime_leakage` catches regime-based features computed incorrectly.

## Next Steps

1. **[Common Pitfalls](common_pitfalls.md)**: Detailed guide to 8 anti-patterns with Don't/Do examples
2. **[Algorithm Decision Tree](algorithm_decision_tree.md)**: Which CV method? Which metric? When to HALT?
3. **[Quickstart](../quickstart.md)**: Get started with temporalcv in 5 minutes
4. **[API Reference](../api/cv.md)**: Full documentation for all CV classes

## References

- Bergmeir, C. & Benítez, J.M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437-450.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. OTexts.
