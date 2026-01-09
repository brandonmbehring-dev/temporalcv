# Common Pitfalls and Best Practices

```{admonition} About This Guide
:class: note

This page documents the most common mistakes when applying machine learning to
time series data. Each pitfall includes:
- **What goes wrong** and why
- **Don't/Do** code examples
- **Which validation gate catches it**
```

## Pitfall #1: Using KFold on Time Series

### The Problem

Standard k-fold cross-validation shuffles data randomly, mixing future observations
into the training set. This creates **temporal leakage**—the model learns patterns
it won't have access to in production.

### Don't

```python
from sklearn.model_selection import KFold, cross_val_score

# WRONG: Shuffles time series data
cv = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)
```

### Do

```python
from temporalcv import WalkForwardCV

# RIGHT: Respects temporal order
cv = WalkForwardCV(n_splits=5, gap=horizon)
scores = cross_val_score(model, X, y, cv=cv)
```

### Gate Detection

```python
from temporalcv.gates import gate_signal_verification

result = gate_signal_verification(y_train, y_test)
# Returns HALT if target appears randomly shuffled (r² with shuffle > 0.1)
```

---

## Pitfall #2: Rolling Features Without Shift

### The Problem

Rolling statistics (mean, std, max) include the current observation by default.
When predicting `y[t]`, using `rolling_mean[t]` includes `y[t]` in the calculation—
this is **look-ahead bias**.

### Don't

```python
# WRONG: rolling_mean[t] includes price[t]
df['rolling_mean'] = df['price'].rolling(5).mean()
df['rolling_std'] = df['price'].rolling(5).std()
```

### Do

```python
# RIGHT: Shift to exclude current observation
df['rolling_mean'] = df['price'].shift(1).rolling(5).mean()
df['rolling_std'] = df['price'].shift(1).rolling(5).std()
```

### Gate Detection

```python
from temporalcv.gates import gate_signal_verification

# If features contain target information, shuffled correlation will be high
result = gate_signal_verification(y_train, y_test, features=X_train)
```

---

## Pitfall #3: Normalizing on Full Dataset

### The Problem

Fitting a scaler on the entire dataset (train + test) leaks test set statistics
into training. The model learns the scale of future data it shouldn't know about.

### Don't

```python
from sklearn.preprocessing import StandardScaler

# WRONG: Scaler sees test data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
X_train, X_test = X_scaled[:split], X_scaled[split:]
```

### Do

```python
from sklearn.preprocessing import StandardScaler

# RIGHT: Fit only on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X[:split])  # Fit on train only
X_test = scaler.transform(X[split:])       # Transform test with train params
```

### Important: Do This Per Fold

```python
from temporalcv import WalkForwardCV
from sklearn.pipeline import Pipeline

# Use a pipeline to ensure scaler fits per fold
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor())
])

cv = WalkForwardCV(n_splits=5)
scores = cross_val_score(pipeline, X, y, cv=cv)  # Scaler refits each fold
```

---

## Pitfall #4: Computing Thresholds on All Data

### The Problem

Regime indicators (high volatility, bull/bear market, recession) are often defined
using quantiles of the full dataset. This means future extreme values inform
historical regime classifications.

### Don't

```python
# WRONG: Threshold computed from ALL data (including future)
vol_threshold = df['volatility'].quantile(0.90)
df['high_vol'] = df['volatility'] > vol_threshold
```

### Do

```python
# RIGHT: Expanding threshold uses only past data
df['high_vol'] = df['volatility'] > (
    df['volatility'].expanding().quantile(0.90).shift(1)
)
```

### Gate Detection

```python
from temporalcv.gates import gate_regime_leakage

result = gate_regime_leakage(
    regime_indicator=df['high_vol'],
    target=df['returns'],
    train_mask=train_mask
)
# Returns HALT if regime indicator shows impossible predictive power
```

---

## Pitfall #5: Insufficient Gap for Multi-Step Forecasting

### The Problem

When forecasting `h` steps ahead, features at time `t` should only use data through
time `t-h`. Without a gap, features implicitly contain information about the
prediction target.

### Don't

```python
# WRONG: No gap for 5-day forecast
cv = WalkForwardCV(n_splits=5)  # gap defaults to 0

# Features computed at t might use data through t
# But predicting y[t+5] shouldn't see data after t
```

### Do

```python
# RIGHT: Gap matches forecast horizon
cv = WalkForwardCV(n_splits=5, gap=5)  # 5-day gap

# Or explicitly with SlidingWindowCV
cv = SlidingWindowCV(
    train_size=252,
    test_size=5,
    gap=5  # Matches h-step ahead forecast
)
```

### Gate Detection

```python
from temporalcv.gates import gate_temporal_boundary

result = gate_temporal_boundary(
    train_times=train_dates,
    test_times=test_dates,
    gap_required=5
)
# Returns HALT if gap is insufficient
```

---

## Pitfall #6: Trusting MAE on High-Persistence Series

### The Problem

For highly autocorrelated series (e.g., stock prices, GDP), a naive "predict last value"
model achieves low MAE. Your sophisticated model may look good but add no value.

### Don't

```python
# WRONG: Raw MAE without baseline comparison
mae = mean_absolute_error(y_test, predictions)
print(f"MAE: {mae:.2f}")  # Looks good, but is it better than naive?
```

### Do

```python
from temporalcv.persistence import compute_persistence_metrics, compare_to_naive

# RIGHT: Compare to naive baseline
results = compare_to_naive(
    y_test,
    predictions,
    naive_predictions=y_test_shifted,  # Previous value
    metric='mase'  # Mean Absolute Scaled Error
)

print(f"MASE: {results['mase']:.3f}")  # <1 means better than naive
print(f"Skill Score: {results['skill_score']:.1%}")  # % improvement over naive
```

### Statistical Test

```python
from temporalcv.statistical_tests import dm_test

# Is the improvement statistically significant?
result = dm_test(
    errors_naive,
    errors_model,
    h=forecast_horizon
)
print(f"p-value: {result.pvalue:.4f}")
```

---

## Pitfall #7: Using center=True for Rolling Windows

### The Problem

Pandas `rolling(..., center=True)` centers the window, using both past AND future
values. This is useful for smoothing visualizations but creates look-ahead bias
in features.

### Don't

```python
# WRONG: center=True uses future values
df['smooth_price'] = df['price'].rolling(5, center=True).mean()
```

### Do

```python
# RIGHT: Default center=False uses only past values
df['smooth_price'] = df['price'].rolling(5).mean()  # center=False is default

# Explicitly for clarity
df['smooth_price'] = df['price'].rolling(5, center=False).mean()
```

---

## Pitfall #8: GroupBy Transform Including Test Data

### The Problem

When computing group statistics (e.g., sector mean returns), using `.transform()`
on the full DataFrame includes test data in the calculation.

### Don't

```python
# WRONG: Group mean computed on ALL data
df['sector_mean'] = df.groupby('sector')['returns'].transform('mean')

# Then split
train = df[:split]
test = df[split:]  # test sector_mean includes test data!
```

### Do

```python
# RIGHT: Compute group statistics on training data only
train = df[:split]
test = df[split:]

sector_means = train.groupby('sector')['returns'].mean()
train['sector_mean'] = train['sector'].map(sector_means)
test['sector_mean'] = test['sector'].map(sector_means)
```

### For Expanding Windows

```python
# RIGHT: Expanding group mean (uses only past data)
df['sector_mean'] = (
    df.groupby('sector')['returns']
    .transform(lambda x: x.expanding().mean().shift(1))
)
```

---

## Summary: Validation Gate Coverage

| Pitfall | Gate | Status When Violated |
|---------|------|---------------------|
| #1 KFold on time series | `gate_signal_verification` | HALT |
| #2 Rolling without shift | `gate_signal_verification` | HALT |
| #3 Normalizing on full data | `gate_feature_correlation` | WARN |
| #4 Thresholds on all data | `gate_regime_leakage` | HALT |
| #5 Insufficient gap | `gate_temporal_boundary` | HALT |
| #6 Trusting raw MAE | (use `compare_to_naive`) | — |
| #7 center=True rolling | `gate_signal_verification` | HALT |
| #8 GroupBy with test data | `gate_feature_correlation` | WARN |

## Running All Gates

```python
from temporalcv.gates import run_gates

result = run_gates(
    X_train, y_train,
    X_test, y_test,
    dates_train=train_dates,
    dates_test=test_dates
)

if result.status == "HALT":
    raise ValueError(f"Critical leakage: {result.reason}")
elif result.status == "WARN":
    print(f"Warning: {result.reason}")
else:
    print("All gates passed!")
```

## Next Steps

- **[Why Time Series Is Different](why_time_series_is_different.md)**: Conceptual foundation
- **[Algorithm Decision Tree](algorithm_decision_tree.md)**: Choose the right CV and metrics
- **[API: Validation Gates](../api/gates.md)**: Full gate reference documentation
