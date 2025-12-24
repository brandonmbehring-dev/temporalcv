# Tutorial: Walk-Forward Cross-Validation

Proper temporal CV that respects time ordering and prevents leakage.

---

## Why Walk-Forward?

Standard k-fold CV shuffles data, destroying temporal relationships:

```python
# WRONG: sklearn's KFold ignores time
from sklearn.model_selection import KFold
for train_idx, test_idx in KFold(5).split(X):
    # test_idx might contain observations BEFORE train_idx!
    pass
```

**Walk-forward CV always trains on past, tests on future.**

---

## Basic Usage

```python
from temporalcv import WalkForwardCV
import numpy as np

# Sample data
X = np.random.randn(200, 5)
y = np.random.randn(200)

# Create splitter
cv = WalkForwardCV(
    n_splits=5,
    window_type="expanding",  # Training window grows
    gap=0,                    # No gap (adjust for multi-step forecasts)
    test_size=1               # 1 observation per test fold
)

# Use like sklearn
for train_idx, test_idx in cv.split(X):
    print(f"Train: {train_idx[0]}-{train_idx[-1]}, Test: {test_idx[0]}-{test_idx[-1]}")
```

Output:
```
Train: 0-159, Test: 160-160
Train: 0-167, Test: 168-168
Train: 0-175, Test: 176-176
Train: 0-183, Test: 184-184
Train: 0-191, Test: 192-192
```

---

## Window Types

### Expanding Window

Training window grows with each split. Good when more data is always better.

```python
cv = WalkForwardCV(
    n_splits=5,
    window_type="expanding",
    test_size=10
)
```

```
Split 1: Train [0, 150), Test [150, 160)
Split 2: Train [0, 160), Test [160, 170)
Split 3: Train [0, 170), Test [170, 180)
...
```

### Sliding Window

Fixed-size training window. Good when recent data is more relevant.

```python
cv = WalkForwardCV(
    n_splits=5,
    window_type="sliding",
    window_size=100,  # Required for sliding
    test_size=10
)
```

```
Split 1: Train [50, 150), Test [150, 160)
Split 2: Train [60, 160), Test [160, 170)
Split 3: Train [70, 170), Test [170, 180)
...
```

---

## Gap Enforcement

**Critical for multi-step forecasting.** If you predict h steps ahead, you need a gap of at least h-1:

```python
# For 2-step ahead forecasts
cv = WalkForwardCV(
    n_splits=5,
    window_type="sliding",
    window_size=100,
    gap=2,  # 2-period gap
    test_size=1
)

for train_idx, test_idx in cv.split(X):
    # Guaranteed: train_idx[-1] + gap < test_idx[0]
    assert train_idx[-1] + cv.gap < test_idx[0]
```

### Why Gap Matters

Without gap, the last training observation can leak into test features:

```
h=2 forecast: y[t+2] = f(y[t], y[t-1], ...)

If train ends at t=99 and test starts at t=100:
- Test prediction uses y[99] (last training observation)
- This is fine for h=1, but for h=2 it's LEAKAGE
```

---

## sklearn Compatibility

Works with `cross_val_score` and `GridSearchCV`:

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

cv = WalkForwardCV(n_splits=5, window_type="expanding")
model = Ridge(alpha=1.0)

scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
print(f"MAE: {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

---

## Split Inspection

Get detailed information about each split:

```python
cv = WalkForwardCV(n_splits=5, window_type="sliding", window_size=100, gap=2)

for split_info in cv.get_split_info(X):
    print(f"Split {split_info.split_idx}:")
    print(f"  Train: [{split_info.train_start}, {split_info.train_end})")
    print(f"  Test:  [{split_info.test_start}, {split_info.test_end})")
    print(f"  Train size: {split_info.train_size}")
    print(f"  Gap: {split_info.gap}")
```

---

## Complete Example: Walk-Forward Evaluation

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from temporalcv import WalkForwardCV

# Generate AR(1) data
np.random.seed(42)
n = 300
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.9 * y[t-1] + np.random.randn() * 0.1

# Helper: Create lag features WITHIN a fold (prevents leakage)
def create_lag_features(data, n_lags=5):
    """Create lag features from data - use only within CV folds."""
    X = np.column_stack([np.roll(data, i) for i in range(1, n_lags + 1)])
    return X[n_lags:], data[n_lags:]  # Remove rows with NaN

# Walk-forward CV
cv = WalkForwardCV(
    n_splits=10,
    window_type="sliding",
    window_size=150,
    gap=2,  # For 2-step forecasts
    test_size=5
)

# Evaluate - compute features INSIDE each fold to prevent leakage
results = []
model = Ridge(alpha=1.0)
n_lags = 5

for fold, (train_idx, test_idx) in enumerate(cv.split(y)):
    # Extract data for this fold
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Create features INSIDE the fold (correct approach)
    X_train, y_train_clean = create_lag_features(y_train, n_lags=n_lags)

    # For test: need context from training for first n_lags predictions
    # Use last n_lags values from training as context
    y_context = np.concatenate([y_train[-n_lags:], y_test])
    X_test, _ = create_lag_features(y_context, n_lags=n_lags)
    # X_test now has correct features for y_test (first n_lags rows are for context)

    # Fit on training
    model.fit(X_train, y_train_clean)

    # Predict on test
    preds = model.predict(X_test)
    actuals = y_test

    # Compute metrics
    mae = mean_absolute_error(actuals, preds)
    results.append({
        'fold': fold,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'mae': mae
    })

# Summary
import pandas as pd
df = pd.DataFrame(results)
print(df.to_string(index=False))
print(f"\nMean MAE: {df['mae'].mean():.4f}")
print(f"Std MAE:  {df['mae'].std():.4f}")
```

---

## Best Practices

### 1. Choose Window Type Based on Data

| Scenario | Window Type | Why |
|----------|-------------|-----|
| Stationary process | Sliding | Old data equally relevant |
| Trending/seasonal | Expanding | More data improves estimates |
| Regime changes | Sliding (short) | Recent data more relevant |
| Limited data | Expanding | Maximize training size |

### 2. Set Appropriate Gap

```python
# Rule of thumb: gap >= horizon - 1
horizon = 2  # 2-step forecast
gap = horizon - 1  # Minimum safe gap

# Conservative: gap = horizon
gap = horizon  # Extra safety margin
```

### 3. Use Enough Splits

- Minimum: 5 splits (statistical validity)
- Typical: 10-20 splits
- Maximum: Limited by min_train_size and test_size

### 4. Validate Split Boundaries

```python
from temporalcv.gates import gate_temporal_boundary

for train_idx, test_idx in cv.split(X):
    result = gate_temporal_boundary(
        train_end_idx=train_idx[-1],
        test_start_idx=test_idx[0],
        horizon=2,
        gap=cv.gap
    )
    assert result.status.name == "PASS"
```

---

## Common Pitfalls

### Pitfall 1: Features Computed Before Split

```python
# WRONG - features computed on full series before split
X = create_lag_features(y)  # Uses future data for lags!
for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])

# WRONG - test features have no context for first n_lags
for train_idx, test_idx in cv.split(y):
    X_train = create_lag_features(y[train_idx])
    X_test = create_lag_features(y[test_idx])  # First n_lags rows invalid!
    model.fit(X_train, y[train_idx])

# RIGHT - use training context for test features
for train_idx, test_idx in cv.split(y):
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, y_train_clean = create_lag_features(y_train, n_lags)

    # Test features need context from end of training
    y_context = np.concatenate([y_train[-n_lags:], y_test])
    X_test, _ = create_lag_features(y_context, n_lags)

    model.fit(X_train, y_train_clean)
    preds = model.predict(X_test)
```

> **Note**: The "Complete Example" above shows this pattern in detail.

### Pitfall 2: Insufficient Gap

```python
# WRONG for h=3 forecasts
cv = WalkForwardCV(gap=0)  # No gap!

# RIGHT
cv = WalkForwardCV(gap=2)  # At least h-1
```

### Pitfall 3: Too Few Test Observations

```python
# WRONG
cv = WalkForwardCV(n_splits=50, test_size=1)  # 50 single-observation tests

# BETTER
cv = WalkForwardCV(n_splits=10, test_size=5)  # 10 tests with 5 obs each
```

---

## API Reference

- [`WalkForwardCV`](../api/cv.md#walkforwardcv)
- [`SplitInfo`](../api/cv.md#splitinfo)
