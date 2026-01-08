# Lag Selection Guide

How to choose the right lag structure for your time-series features and forecasting horizon.

---

## The Two Lag Decisions

1. **Feature lags**: How many lags to include as features (e.g., `y[t-1], y[t-2], ..., y[t-p]`)
2. **Forecast horizon**: How far ahead to predict (e.g., 1-step, 5-step, 20-step)

These decisions are interconnected: your feature lags should inform your horizon, and your horizon constrains your feature lags.

---

## Autocorrelation Analysis

The starting point for lag selection is understanding your data's autocorrelation structure.

### ACF Interpretation

```python
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Compute and visualize ACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y, lags=40, ax=axes[0], title='ACF')
plot_pacf(y, lags=40, ax=axes[1], title='PACF')
plt.tight_layout()
```

**What the ACF tells you**:

| ACF Pattern | Meaning | Implication |
|-------------|---------|-------------|
| Slow decay | High persistence | Difficult to beat persistence baseline |
| Quick dropoff | Low persistence | Model can add value |
| Seasonal spikes | Seasonality present | Include seasonal lags |
| Cutoff at lag k | MA(k) structure | Consider k lags |

**What the PACF tells you**:

| PACF Pattern | Meaning | Implication |
|--------------|---------|-------------|
| Cutoff at lag p | AR(p) structure | Use p lags as features |
| Slow decay | ARMA structure | More complex lag selection |
| Seasonal spikes | Seasonal AR | Include seasonal AR terms |

---

## Feature Lag Selection

### Method 1: PACF-Based Selection

Use the PACF cutoff to determine the number of AR lags:

```python
from statsmodels.tsa.stattools import pacf

# Compute PACF with confidence intervals
pacf_values, confint = pacf(y, nlags=20, alpha=0.05)

# Find significant lags (outside confidence interval)
significant_lags = []
for i, (p, (low, high)) in enumerate(zip(pacf_values[1:], confint[1:])):
    if p < low or p > high:
        significant_lags.append(i + 1)

print(f"Significant lags: {significant_lags}")
```

### Method 2: Information Criteria

Use AIC/BIC to select optimal lag order:

```python
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# Automatic lag selection using BIC
selection = ar_select_order(y, maxlag=20, ic='bic')
optimal_lag = selection.ar_lags[-1] if selection.ar_lags else 1

print(f"Optimal lag order (BIC): {optimal_lag}")
```

### Method 3: Cross-Validation

Empirically test different lag structures:

```python
from temporalcv import WalkForwardCV
from sklearn.linear_model import Ridge
import numpy as np

def create_lag_features(y, n_lags):
    """Create lagged features from series."""
    X = np.column_stack([
        np.roll(y, shift=lag)[n_lags:]
        for lag in range(1, n_lags + 1)
    ])
    return X, y[n_lags:]

# Test different lag structures
cv = WalkForwardCV(n_splits=5, horizon=1)
results = {}

for n_lags in [1, 3, 5, 10, 20]:
    X, y_aligned = create_lag_features(y, n_lags)

    scores = []
    for train_idx, test_idx in cv.split(X):
        model = Ridge().fit(X[train_idx], y_aligned[train_idx])
        pred = model.predict(X[test_idx])
        mae = np.abs(pred - y_aligned[test_idx]).mean()
        scores.append(mae)

    results[n_lags] = np.mean(scores)

print("MAE by lag count:", results)
```

---

## Forecast Horizon Selection

### Predictability Decay

Predictability typically decays with horizon. Use `compare_horizons()` to find the optimal horizon:

```python
from temporalcv import compare_horizons

# Compare horizons 1 through 20
horizon_results = compare_horizons(
    model=my_model,
    X=X,
    y=y,
    horizons=range(1, 21),
    cv=WalkForwardCV(n_splits=5)
)

# Find where model stops beating baseline
for h, result in horizon_results.items():
    skill_score = 1 - result['model_mae'] / result['baseline_mae']
    print(f"Horizon {h}: Skill Score = {skill_score:.3f}")
    if skill_score < 0.05:
        print(f"  → Consider stopping at horizon {h-1}")
        break
```

### The Horizon-Gap Rule

For h-step forecasting, your CV must have `gap >= h`:

```python
from temporalcv import WalkForwardCV

# For 5-step ahead forecasting
cv = WalkForwardCV(
    n_splits=5,
    horizon=5,  # Enforces gap of 5
    extra_gap=0  # Optional additional safety margin
)
```

**Why this matters**:
- At time t, you're predicting y[t+h]
- Training data includes y[t-1], y[t-2], ...
- If gap < h, training includes observations too close to test

---

## Lag-Horizon Interaction

### Rule of Thumb

```
feature_lags <= horizon
```

If you're forecasting 5 steps ahead, including lags beyond 5 may leak information:

```python
# For h=5 forecasting
# SAFE: y[t-1], ..., y[t-5] as features
# RISKY: y[t-6], y[t-7]... might be fine, but need careful validation
```

### Multi-Step Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Direct** | Train separate model for each horizon | Small number of horizons |
| **Recursive** | Predict h=1, use as input for h=2, etc. | Autoregressive models |
| **Multi-output** | Single model predicts all horizons | Deep learning, multi-output regression |

```python
# Direct multi-step: train model for h=5 specifically
X, y_h5 = create_features_for_horizon(data, horizon=5)
model_h5 = Ridge().fit(X_train, y_h5_train)

# Recursive multi-step: iterate predictions
predictions = []
current_input = X_test[0]
for step in range(horizon):
    pred = model.predict(current_input.reshape(1, -1))
    predictions.append(pred[0])
    current_input = np.roll(current_input, -1)
    current_input[-1] = pred[0]
```

---

## Seasonal Lag Selection

For seasonal data, include both:
1. Recent lags (AR structure)
2. Seasonal lags (e.g., lag 7 for daily data with weekly pattern)

```python
def create_seasonal_features(y, ar_lags, seasonal_period, n_seasonal_lags):
    """Create AR + seasonal lag features."""
    features = []
    max_lag = max(ar_lags[-1], seasonal_period * n_seasonal_lags)

    # AR lags
    for lag in ar_lags:
        features.append(np.roll(y, lag)[max_lag:])

    # Seasonal lags
    for i in range(1, n_seasonal_lags + 1):
        features.append(np.roll(y, seasonal_period * i)[max_lag:])

    return np.column_stack(features), y[max_lag:]

# Daily data with weekly seasonality
X, y_aligned = create_seasonal_features(
    y,
    ar_lags=[1, 2, 3],  # Recent 3 days
    seasonal_period=7,   # Weekly
    n_seasonal_lags=2    # 1 and 2 weeks ago
)
```

---

## Decision Flow

```mermaid
graph TD
    A[Start: New Time Series] --> B[Compute ACF/PACF]
    B --> C{High persistence?<br>ACF(1) > 0.8}

    C -->|Yes| D[Use MC-SS metrics<br>Persistence baseline]
    C -->|No| E[Standard metrics OK]

    D --> F{PACF cutoff?}
    E --> F

    F -->|Clear cutoff at p| G[Use p lags]
    F -->|No clear cutoff| H[Use IC selection<br>or CV]

    G --> I[Determine horizon h]
    H --> I

    I --> J{Seasonality?}

    J -->|Yes| K[Add seasonal lags]
    J -->|No| L[AR lags only]

    K --> M[Validate with<br>compare_horizons]
    L --> M

    M --> N[Set CV gap >= h]
```

---

## See Also

- [Example 09: Multi-Horizon](examples_index.md#09-multi-horizon) — `compare_horizons()` in action
- [High Persistence Tutorial](high_persistence.md) — Handling sticky series
- [Walk-Forward CV](walk_forward_cv.md) — Gap enforcement details
- [Feature Engineering Safety](feature_engineering_safety.md) — Safe lag feature patterns
