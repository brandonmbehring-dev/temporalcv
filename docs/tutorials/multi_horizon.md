# Multi-Horizon Forecasting Tutorial

How to systematically evaluate predictability across multiple forecast horizons and find your model's effective range.

---

## The Multi-Horizon Question

> "How far ahead can my model usefully predict?"

Predictability decays with horizon. A model that beats persistence at h=1 may be worthless at h=20. This tutorial shows how to:

1. Compare performance across horizons
2. Find the "predictability horizon" where skill vanishes
3. Choose the right horizon for your use case

---

## The `compare_horizons()` Function

temporalcv provides `compare_horizons()` for systematic horizon analysis:

```python
from temporalcv import compare_horizons, WalkForwardCV
from sklearn.linear_model import Ridge

# Compare horizons 1 through 20
results = compare_horizons(
    model=Ridge(),
    X=X,
    y=y,
    horizons=range(1, 21),
    cv=WalkForwardCV(n_splits=5),
    metric='mae'
)

# Analyze results
for h, metrics in results.items():
    skill = 1 - metrics['model_mae'] / metrics['baseline_mae']
    print(f"Horizon {h:2d}: MAE={metrics['model_mae']:.4f}, Skill={skill:+.2%}")
```

---

## Skill Score Interpretation

The **skill score** measures improvement over baseline:

```
Skill Score = 1 - (Model Error / Baseline Error)
```

| Skill Score | Interpretation |
|-------------|----------------|
| > 0.20 | Strong skill |
| 0.10 - 0.20 | Moderate skill |
| 0.05 - 0.10 | Marginal skill |
| 0.00 - 0.05 | No practical skill |
| < 0.00 | Worse than baseline |

### Visualizing Skill Decay

```python
import matplotlib.pyplot as plt
import numpy as np

horizons = list(results.keys())
skill_scores = [
    1 - results[h]['model_mae'] / results[h]['baseline_mae']
    for h in horizons
]

plt.figure(figsize=(10, 6))
plt.plot(horizons, skill_scores, 'bo-', markersize=8)
plt.axhline(y=0.05, color='r', linestyle='--', label='Practical skill threshold')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.fill_between(horizons, 0, skill_scores, alpha=0.3, where=np.array(skill_scores) > 0.05)
plt.xlabel('Forecast Horizon')
plt.ylabel('Skill Score')
plt.title('Predictability Decay with Horizon')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Finding the Predictability Horizon

The **predictability horizon** is where skill drops below a practical threshold:

```python
def find_predictability_horizon(results, threshold=0.05):
    """Find the horizon where skill drops below threshold."""
    for h in sorted(results.keys()):
        skill = 1 - results[h]['model_mae'] / results[h]['baseline_mae']
        if skill < threshold:
            return h - 1 if h > 1 else 0
    return max(results.keys())  # Skill never drops below threshold

pred_horizon = find_predictability_horizon(results, threshold=0.05)
print(f"Predictability horizon: {pred_horizon} steps")
```

---

## Multi-Step Forecasting Strategies

### Strategy 1: Direct (Independent Models)

Train a separate model for each horizon:

```python
from temporalcv import WalkForwardCV
from sklearn.linear_model import Ridge

direct_models = {}
for h in range(1, pred_horizon + 1):
    # Create features and targets for horizon h
    X_h, y_h = create_features_for_horizon(data, horizon=h)

    # Train with proper gap
    cv = WalkForwardCV(n_splits=5, horizon=h)
    model = Ridge()

    scores = cross_val_score(model, X_h, y_h, cv=cv, scoring='neg_mae')
    direct_models[h] = model.fit(X_h, y_h)  # Final fit on all data

    print(f"Horizon {h}: MAE = {-scores.mean():.4f}")
```

**Pros**: Each model optimized for its horizon
**Cons**: More models to maintain, no information sharing

### Strategy 2: Recursive (Iterative Prediction)

Use h=1 model iteratively:

```python
def recursive_forecast(model, initial_features, horizon):
    """Generate multi-step forecast recursively."""
    predictions = []
    current_features = initial_features.copy()

    for step in range(horizon):
        # Predict next step
        pred = model.predict(current_features.reshape(1, -1))[0]
        predictions.append(pred)

        # Update features with prediction
        current_features = np.roll(current_features, -1)
        current_features[-1] = pred

    return np.array(predictions)

# Use h=1 model recursively
forecasts = recursive_forecast(model_h1, X_test[0], horizon=10)
```

**Pros**: Single model, captures dynamics
**Cons**: Error accumulation, sensitive to model bias

### Strategy 3: Multi-Output (Single Model)

Train one model to predict all horizons simultaneously:

```python
from sklearn.multioutput import MultiOutputRegressor

# Create multi-horizon targets
def create_multi_horizon_data(y, max_horizon):
    """Create features and multi-horizon targets."""
    n = len(y) - max_horizon
    X = y[:n].reshape(-1, 1)  # Simplified: just lag-1

    Y = np.column_stack([
        y[h:n+h] for h in range(1, max_horizon + 1)
    ])
    return X, Y

X_multi, Y_multi = create_multi_horizon_data(y, max_horizon=pred_horizon)

# Train multi-output model
model = MultiOutputRegressor(Ridge())
model.fit(X_train, Y_train)

# Predict all horizons at once
all_forecasts = model.predict(X_test)  # Shape: (n_samples, n_horizons)
```

**Pros**: Captures cross-horizon correlations
**Cons**: May underperform at specific horizons

---

## Gap Enforcement for Multi-Horizon

**Critical**: CV gap must match your forecast horizon.

```python
from temporalcv import WalkForwardCV

# For h=5 forecasting
cv_h5 = WalkForwardCV(
    n_splits=5,
    horizon=5,  # Gap = 5
)

# For h=20 forecasting
cv_h20 = WalkForwardCV(
    n_splits=5,
    horizon=20,  # Gap = 20
)

# WRONG: Single CV for all horizons
# cv_shared = WalkForwardCV(n_splits=5)  # No gap!
```

---

## Statistical Significance Across Horizons

Use the Diebold-Mariano test to check if skill is significant at each horizon:

```python
from temporalcv import dm_test

for h in range(1, pred_horizon + 1):
    model_errors = results[h]['model_errors']
    baseline_errors = results[h]['baseline_errors']

    dm_result = dm_test(
        errors1=baseline_errors,
        errors2=model_errors,
        h=h  # Horizon for HAC variance
    )

    skill = 1 - results[h]['model_mae'] / results[h]['baseline_mae']
    sig = "***" if dm_result.pvalue < 0.01 else "**" if dm_result.pvalue < 0.05 else ""

    print(f"Horizon {h:2d}: Skill={skill:+.2%}, p={dm_result.pvalue:.3f} {sig}")
```

---

## Complete Example

Based on [Example 09: Multi-Horizon](../../examples/09_multi_horizon.py):

```python
import numpy as np
from temporalcv import WalkForwardCV, compare_horizons, dm_test
from sklearn.linear_model import Ridge

# Generate synthetic data with decaying predictability
np.random.seed(42)
n = 500
y = np.cumsum(np.random.randn(n) * 0.1)  # Random walk
X = np.column_stack([
    np.roll(y, 1),  # Lag 1
    np.roll(y, 2),  # Lag 2
    np.roll(y, 7),  # Seasonal lag
])
X = X[10:]  # Remove edge effects
y = y[10:]

# Compare horizons
print("=" * 50)
print("MULTI-HORIZON ANALYSIS")
print("=" * 50)

for h in [1, 5, 10, 20]:
    # Proper CV with gap
    cv = WalkForwardCV(n_splits=5, horizon=h)

    model_errors = []
    baseline_errors = []

    for train_idx, test_idx in cv.split(X, y):
        model = Ridge().fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        model_errors.extend(np.abs(pred - y[test_idx]))
        baseline_errors.extend(np.abs(y[train_idx][-1] - y[test_idx]))  # Persistence

    model_mae = np.mean(model_errors)
    baseline_mae = np.mean(baseline_errors)
    skill = 1 - model_mae / baseline_mae

    # Statistical test
    dm = dm_test(np.array(baseline_errors), np.array(model_errors), h=h)

    print(f"\nHorizon h={h}:")
    print(f"  Model MAE:    {model_mae:.4f}")
    print(f"  Baseline MAE: {baseline_mae:.4f}")
    print(f"  Skill Score:  {skill:+.2%}")
    print(f"  DM p-value:   {dm.pvalue:.4f}")
    print(f"  Significant:  {'Yes' if dm.pvalue < 0.05 else 'No'}")
```

---

## Key Takeaways

1. **Predictability decays**: Always test multiple horizons
2. **Match gap to horizon**: CV gap >= forecast horizon
3. **Statistical significance**: Use DM test at each horizon
4. **Choose strategy**: Direct for accuracy, recursive for simplicity
5. **Document the horizon**: Report the practical limit of your model

---

## See Also

- [Example 09: Multi-Horizon](examples_index.md#09-multi-horizon) — Complete code example
- [Lag Selection Guide](lag_selection.md) — Choosing feature lags
- [Statistical Tests](../api/statistical_tests.md) — DM test for significance
- [Walk-Forward CV](walk_forward_cv.md) — Gap enforcement
