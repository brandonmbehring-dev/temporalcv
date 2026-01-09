# Feature Engineering Safety Guide

**Audience**: ML practitioners who know sklearn but are new to time-series feature engineering.

**Purpose**: Teach you to identify safe vs dangerous features **without memorization** — understand the principle, apply it anywhere.

---

## The One Rule

> **Every feature at time `t` must be computable using ONLY data from times `≤ t-gap`.**

That's it. If you internalize this rule, you'll never accidentally leak information.

**Why gap?** If you're predicting `h` steps ahead, your features at time `t` cannot use any information from `[t, t+h)`. The gap ensures temporal separation.

---

## Quick Decision Tree

```
Question 1: Does this feature use future data (y[t+1], x[t+1], ...)?
    YES → 🚫 LEAKAGE - Never use
    NO  → Continue

Question 2: Does this feature use the target variable y?
    YES → Question 2a: Is it computed on training data only?
        YES → ⚠️ DANGEROUS - Requires careful handling
        NO  → 🚫 LEAKAGE - Target-derived features on full series
    NO  → Continue

Question 3: Does this feature use centered windows (data from both sides of t)?
    YES → 🚫 LEAKAGE - Bidirectional operations
    NO  → ✅ SAFE - Backward-looking only
```

---

## Feature Categories

### ✅ SAFE: Backward-Looking Only

These features use only past data and are always safe when properly lagged.

| Feature Type | Example | Why Safe |
|-------------|---------|----------|
| Lag features | `y[t-1]`, `y[t-5]` | Explicitly past |
| Expanding statistics | `mean(y[0:t])` | Only uses [0, t) |
| Rolling (left-aligned) | `y[t-window:t].mean()` | Only uses past window |
| Cumulative sums | `sum(y[0:t])` | Only uses [0, t) |
| Calendar features | `day_of_week(t)` | Deterministic, no data needed |
| External regressors | `x[t-1]` (lagged) | Past values only |

**Code Example — Safe Rolling Mean**:
```python
# ✅ SAFE: Rolling mean using only past values
def safe_rolling_mean(series, window=5):
    """Compute rolling mean using only past values (t-window to t-1)."""
    result = np.full_like(series, np.nan)
    for t in range(window, len(series)):
        result[t] = series[t-window:t].mean()  # excludes t!
    return result

# Equivalent pandas (note: shift BEFORE rolling!)
df['safe_rolling'] = df['y'].shift(1).rolling(window=5).mean()
```

---

### ⚠️ DANGEROUS: Requires Careful Handling

These features are legitimate but easy to implement incorrectly.

| Feature Type | Danger | Safe Implementation |
|-------------|--------|---------------------|
| Rolling statistics | Default `center=True` | Always `center=False` + `shift(1)` |
| Percentile ranks | Full-series percentiles | Compute on training only, apply to test |
| Standardization | Fit on all data | Fit on training only |
| Target encoding | Include test targets | Use only training targets |
| Regime indicators | Computed on full series | Use changepoint detection with lag |

**Code Example — Dangerous vs Safe Percentile**:
```python
# 🚫 WRONG: Percentile uses full series (future information!)
def leaky_percentile_rank(series, value):
    return (series < value).sum() / len(series)

# ✅ SAFE: Percentile uses only training data
def safe_percentile_rank(training_series, value):
    return (training_series < value).sum() / len(training_series)

# In practice:
train_percentiles = np.percentile(y_train, [25, 50, 75])
# Apply these thresholds to test data — never recompute on test
```

**Code Example — Dangerous vs Safe Rolling**:
```python
# 🚫 WRONG: Center=True uses future values
df['leaky'] = df['y'].rolling(window=5, center=True).mean()

# 🚫 WRONG: No shift means y[t] is used to predict y[t]
df['also_leaky'] = df['y'].rolling(window=5, center=False).mean()

# ✅ SAFE: Shift first, then roll
df['safe'] = df['y'].shift(1).rolling(window=5, center=False).mean()
```

---

### 🚫 LEAKAGE: Never Use These

These features inherently use future information and cannot be fixed.

| Feature Type | Why It Leaks | What Happens |
|-------------|--------------|--------------|
| Centered rolling | Uses `y[t+1], y[t+2]...` | Model "sees" future |
| Full-series normalization | Mean/std include test | Test distribution leaked |
| Target encoding (full) | Test targets in encoding | Direct target leakage |
| Cross-validation leakage | KFold shuffles time | Future in training |
| Forward-looking indicators | Any `y[t+k]` for k>0 | Crystal ball |

**How Leakage Manifests**:
```python
# Generate high-persistence AR(1) data
np.random.seed(42)
y = np.zeros(500)
for t in range(1, 500):
    y[t] = 0.95 * y[t-1] + np.random.normal()

# 🚫 LEAKY FEATURE: Rolling mean includes current value
X_leaky = pd.DataFrame({
    'rolling_mean': pd.Series(y).rolling(5).mean()  # Uses y[t]!
})

# Train/test split
X_train, y_train = X_leaky.iloc[:400], y[:400]
X_test, y_test = X_leaky.iloc[400:], y[400:]

# Model will appear to perform impossibly well on test set
# because X_test features contain y_test information
```

---

## Common Mistakes by Category

### 1. Pandas Rolling Window Traps

```python
# Trap 1: Default includes current value
df['bad1'] = df['y'].rolling(5).mean()  # y[t-4:t+1].mean(), includes t

# Trap 2: center=True uses both sides
df['bad2'] = df['y'].rolling(5, center=True).mean()  # y[t-2:t+3]

# Trap 3: min_periods allows partial windows at start (ok), but still includes t
df['bad3'] = df['y'].rolling(5, min_periods=1).mean()  # still includes t

# ✅ CORRECT: shift(1) BEFORE rolling
df['good'] = df['y'].shift(1).rolling(5, min_periods=1).mean()
```

### 2. Sklearn Transformer Pitfalls

```python
from sklearn.preprocessing import StandardScaler

# 🚫 WRONG: Fit on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses mean/std of entire dataset
X_train_scaled = X_scaled[:split_idx]  # Test info leaked into training!

# ✅ CORRECT: Fit on training only
scaler = StandardScaler()
scaler.fit(X_train)  # Only training statistics
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Apply training statistics
```

### 3. GroupBy Aggregation Leakage

```python
# 🚫 WRONG: Group statistics include all data
df['sector_mean'] = df.groupby('sector')['returns'].transform('mean')
# If sector appears in both train and test, test values contaminate training

# ✅ CORRECT: Compute on training, merge to full
train_sector_means = df.loc[train_idx].groupby('sector')['returns'].mean()
df['sector_mean'] = df['sector'].map(train_sector_means)
```

### 4. Technical Indicators Leakage

```python
# Many TA-Lib indicators are fine, but some use future data

# ⚠️ CHECK: Does the indicator use future values?
# - MACD, RSI, Bollinger: Safe (backward-looking)
# - Pivot Points: Often calculated for "today" using "today's" HLOC
#   → Safe if using previous day's HLOC

# ✅ SAFE RSI (standard implementation is backward-looking)
import talib
df['RSI'] = talib.RSI(df['close'].shift(1), timeperiod=14)  # shift for safety
```

---

## Validation: How to Detect Leakage

### Method 1: Shuffled Target Test

The gold standard. If your model beats a shuffled target, you have leakage.

```python
from temporalcv import gate_signal_verification

result = gate_signal_verification(
    model=your_model,
    X=X_train,
    y=y_train,
    n_shuffles=5,
    random_state=42
)

if result.status == "HALT":
    print("LEAKAGE DETECTED!")
    print(f"Real MAE: {result.real_mae:.4f}")
    print(f"Shuffled MAE: {result.shuffled_mae:.4f}")
    # Model performs nearly as well on shuffled target
    # → Features encode target position, not predictive signal
```

### Method 2: Too-Good-to-Be-True Check

If you're beating persistence by >20% on high-persistence data, investigate.

```python
from temporalcv import compute_mase, compute_acf

# Check persistence level
acf1 = compute_acf(y_train, max_lag=1)[1]  # ACF at lag 1
if acf1 > 0.9:
    print(f"High persistence series (ACF(1)={acf1:.2f})")

# If MASE < 0.8 on high-persistence data, be suspicious
mase = compute_mase(predictions, actuals, y_train)
if mase < 0.8 and acf1 > 0.9:
    print("WARNING: Suspiciously good performance")
    print("Run gate_signal_verification() before proceeding")
```

### Method 3: Horizon Consistency Check

If h=1 is dramatically better than h=2,3,4, you may have gap issues.

```python
# If gap=0, the h=1 prediction can "see" y[t] in features
# Compare performance across horizons
mase_h1 = compute_mase(preds_h1, actuals_h1, y_train)
mase_h4 = compute_mase(preds_h4, actuals_h4, y_train)

if mase_h1 < 0.5 * mase_h4:
    print("WARNING: h=1 >> h=4 suggests gap enforcement issue")
```

---

## Safe Feature Engineering Checklist

Before using any feature, verify:

- [ ] **No future data**: Feature at time `t` uses only data from `[0, t-gap]`
- [ ] **Shift before rolling**: `df['x'].shift(1).rolling(...)` not `df['x'].rolling(...)`
- [ ] **Training-only statistics**: Percentiles, means, encodings fit on training only
- [ ] **Explicit lag**: If using lag, is it `y[t-lag]` or accidentally `y[t]`?
- [ ] **No centered windows**: `center=False` for all rolling operations
- [ ] **Gap respected**: If predicting `h` steps ahead, gap `>= h` in all features

---

## Quick Reference Card

| Operation | Leaky Version | Safe Version |
|-----------|---------------|--------------|
| Rolling mean | `df['y'].rolling(5).mean()` | `df['y'].shift(1).rolling(5).mean()` |
| Percentile | `np.percentile(full_series, 75)` | `np.percentile(train_only, 75)` |
| Standardization | `scaler.fit_transform(X)` | `scaler.fit(X_train).transform(X)` |
| Group encoding | `df.groupby('g')['y'].transform('mean')` | Fit on train, map to full |
| Expanding mean | `df['y'].expanding().mean()` | `df['y'].shift(1).expanding().mean()` |
| Diff | `df['y'].diff()` | `df['y'].diff()` (safe, uses t and t-1) |

---

## Summary

1. **The One Rule**: Features at `t` use only data from `≤ t-gap`
2. **Shift before rolling**: The most common pandas mistake
3. **Training-only statistics**: Percentiles, normalization, encoding
4. **Validate with shuffled target**: The definitive leakage test
5. **Suspicious improvement = investigate**: >20% over persistence = verify

**When in doubt, ask**: "Could I compute this feature in real-time production, knowing only the past?"

If the answer is "no," it's leakage.

---

## See Also

- [Notebook 00: Time Series Fundamentals](../../notebooks/00_time_series_fundamentals.ipynb) — Why time series differs from regular ML
- [Walk-Forward CV Tutorial](walk_forward_cv.md) — Proper temporal validation
- [Diagnostic Flowchart](diagnostic_flowchart.md) — What to do when validation fails
- [Metric Selection Guide](metric_selection.md) — Which metric for which problem
