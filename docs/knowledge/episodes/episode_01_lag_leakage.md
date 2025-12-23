# Episode 01: Lag Feature Leakage

**Category**: Bug Category 2 (Future Data in Lags)
**Discovered**: myga-forecasting-v3, 2024-Q2
**Impact**: Overstated model skill by ~25%

---

## The Bug

Rolling statistics (mean, std, EMA) were computed on the full time series before the train/test split.

```python
# BUGGY CODE
df['rolling_mean'] = df['price'].rolling(10).mean()
df['rolling_std'] = df['price'].rolling(10).std()

# Split AFTER computing features
train = df[:split_point]
test = df[split_point:]
```

**Problem**: The rolling mean at time `t` in the test set used values from times `t-9` through `t`, some of which may be in the test period. This means future information leaked into the features.

---

## How It Was Discovered

Suspiciously good results triggered investigation:
- Model beat persistence by 28% (threshold: 20%)
- Improvement held across all horizons (unusual)
- Model beat shuffled target at p < 0.001

Running `gate_shuffled_target()` revealed the bug:

```python
result = gate_shuffled_target(model, X, y, n_shuffles=100)
print(result.status)  # HALT
print(result.message)  # "Model beats shuffled by 32% (max: 5%)"
```

---

## Root Cause Analysis

The bug is subtle because rolling operations are "causal" (they only look backwards). The issue is:

1. When you compute `rolling(10)` on the full series at index `t`, you use values `t-9, t-8, ..., t`
2. If `t` is in the test set but `t-9` is in the training set, no problem
3. But if both `t` and `t-5` are in the test set, the rolling mean at `t` used the "future" value at `t-5` that shouldn't be available during training

**The fix**: Compute rolling features WITHIN each walk-forward split:

```python
# CORRECT CODE
for train_idx, test_idx in cv.split(X, y):
    # Compute features using only training data
    train_data = df.iloc[train_idx]
    train_rolling_mean = train_data['price'].rolling(10).mean()

    # For test, extend the rolling window properly
    extended_data = df.iloc[:test_idx[-1]+1]
    all_rolling_mean = extended_data['price'].rolling(10).mean()
    test_rolling_mean = all_rolling_mean.iloc[test_idx]
```

---

## Why gate_shuffled_target Catches This

When targets are shuffled randomly:
- The temporal alignment between features and targets is broken
- Rolling features no longer have any relationship to the target
- A legitimate model should NOT beat this random baseline

If the model beats shuffled targets, it means the features encode the target's *position* in the series, not genuine predictive signal.

---

## Prevention Checklist

- [ ] Compute all rolling/lagged features WITHIN walk-forward splits
- [ ] Verify features don't use information from test indices
- [ ] Run `gate_shuffled_target()` on every model before deployment
- [ ] Document all feature computation in the pipeline

---

## Test Case

```python
def test_lag_leakage_detection():
    """Gate should catch rolling features computed on full series."""
    # Create data with leaky features
    n = 200
    y = np.random.randn(n)
    X_leaky = np.column_stack([
        y.cumsum(),  # Leaky: uses all data
        np.roll(y, 1),  # Safe: proper lag
    ])

    # Train simple model
    model = LinearRegression()
    cv = WalkForwardCV(window_size=100)

    # Gate should HALT
    result = gate_shuffled_target(model, X_leaky, y)
    assert result.status == GateStatus.HALT
```

---

## Related

- [Leakage Audit Trail](../leakage_audit_trail.md) - Full category list
- [Episode 02: Boundary Violations](episode_02_boundary_violations.md) - Similar temporal issue
- SPECIFICATION.md Section 1.2 - Shuffled target threshold
