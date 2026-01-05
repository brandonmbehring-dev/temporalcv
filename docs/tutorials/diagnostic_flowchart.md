# Validation Diagnostic Flowchart

**Audience**: ML practitioners who need to debug unexpected validation results.

**Purpose**: Step-by-step decision tree for common validation failures and suspicious results.

---

## Master Flowchart

```
                    START
                      │
                      ▼
         ┌──────────────────────┐
         │  Model beats baseline │
         │      by >20%?         │
         └──────────────────────┘
                      │
           ┌──────────┴──────────┐
           │                     │
          YES                    NO
           │                     │
           ▼                     ▼
    ┌─────────────┐     ┌───────────────────┐
    │ SUSPICIOUS! │     │ MASE > 1?         │
    │ Go to       │     │ (Worse than naive)│
    │ Section A   │     └───────────────────┘
    └─────────────┘              │
                      ┌──────────┴──────────┐
                     YES                    NO
                      │                     │
                      ▼                     ▼
               ┌─────────────┐      ┌─────────────┐
               │ Go to       │      │ PASS:       │
               │ Section B   │      │ Valid model │
               └─────────────┘      └─────────────┘
```

---

## Section A: Suspiciously Good Results (>20% Improvement)

### Step A1: Run Shuffled Target Test FIRST

This is the definitive leakage detector.

```python
from temporalcv import gate_shuffled_target
from sklearn.linear_model import Ridge

result = gate_shuffled_target(
    model=your_model,
    X=X_train, y=y_train,
    n_shuffles=100,
    method="permutation",
    random_state=42,
)

if result.status.value == "HALT":
    print("LEAKAGE DETECTED — Go to A1a")
else:
    print("No leakage evidence — Go to A2")
```

### Step A1a: Shuffled Target HALT → Investigate Feature Leakage

**Symptoms**:
- Model performs nearly as well on shuffled targets
- p-value ≥ 0.05 in shuffled target test

**Common Causes**:

| Cause | Check | Fix |
|-------|-------|-----|
| Centered rolling windows | `df['x'].rolling(n, center=True)` | Use `center=False` + `shift(1)` |
| Target in features | Any feature derived from y | Remove or properly lag |
| Full-series normalization | `scaler.fit(X_all)` | Fit on training only |
| Percentiles on full data | `np.percentile(all_data, q)` | Compute on training only |
| GroupBy includes test | `df.groupby().transform()` | Aggregate training only |

**Diagnostic Code**:
```python
# Check each feature individually
for i in range(X.shape[1]):
    single_feature = X[:, [i]]
    result = gate_shuffled_target(
        model=Ridge(alpha=1.0),
        X=single_feature, y=y,
        method="effect_size",  # Fast check per feature
        random_state=42,
    )
    if result.status.value == "HALT":
        print(f"Feature {i}: LEAKY")
    else:
        print(f"Feature {i}: OK")
```

**Resolution**: Remove or fix leaky features, then re-run validation.

---

### Step A2: Check Gap Enforcement

If shuffled target passed, check temporal boundaries.

```python
from temporalcv import gate_temporal_boundary

# For each CV split
for split_info in cv.get_split_info(X):
    result = gate_temporal_boundary(
        train_end_idx=split_info.train_end,
        test_start_idx=split_info.test_start,
        horizon=your_horizon,
    )

    if result.status.value == "HALT":
        print(f"Split {split_info.split_idx}: GAP VIOLATION")
        print(f"  Gap: {split_info.gap}, Required: {your_horizon}")
```

**Common Causes**:

| Cause | Check | Fix |
|-------|-------|-----|
| gap=0 for h>1 forecasting | `WalkForwardCV(horizon=0, extra_gap=0)` | Set `gap >= horizon` |
| Target includes future | `y[t] = f(y[t+h])` | Ensure y uses only past |
| Features computed wrong | Rolling windows include y[t] | Add `shift(1)` |

**Resolution**: Set `gap >= horizon` in `WalkForwardCV`.

---

### Step A3: Check Threshold/Regime Computation

```python
# WRONG: Threshold computed on all data
threshold_leaky = np.percentile(all_data, 70)  # Includes test!

# CORRECT: Threshold from training only
threshold_safe = np.percentile(train_data, 70)
```

**Common Causes**:

| Cause | Check | Fix |
|-------|-------|-----|
| Percentile on full series | Look for `np.percentile(full_series)` | Compute on training only |
| Regime labels use future | Changepoint detection on all data | Detect on training, apply to test |
| Volatility from full data | Rolling vol includes test period | Estimate from training only |

**Resolution**: All statistics must be computed on training data only.

---

### Step A4: Still Suspicious? Deep Investigation

If all gates pass but results still seem too good:

1. **Compare horizon performance**:
```python
# If h=1 >> h=2,3,4, you may have subtle leakage
for h in [1, 2, 3, 4]:
    mase_h = evaluate_at_horizon(model, X, y, horizon=h)
    print(f"h={h}: MASE={mase_h:.3f}")

# Suspicious if h=1 is dramatically better
```

2. **Check feature correlations**:
```python
# High correlation with target = possible leakage
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    if abs(corr) > 0.9:
        print(f"Feature {i}: correlation={corr:.3f} — INVESTIGATE")
```

3. **Out-of-sample validation**:
```python
# Hold out completely unseen data
train, val, test = temporal_split(data, [0.6, 0.2, 0.2])
# Train on train, tune on val, final eval on test
# If test performance drops sharply, you're overfitting to val
```

---

## Section B: Model Worse Than Naive (MASE > 1)

### Step B1: Check Persistence Level

```python
acf1 = np.corrcoef(series[1:], series[:-1])[0, 1]
print(f"ACF(1) = {acf1:.3f}")
```

| ACF(1) | Interpretation |
|--------|----------------|
| > 0.95 | MASE > 1 is **normal**. Beating persistence is very hard. |
| 0.90-0.95 | MASE > 1 is **common**. Consider move-conditional metrics. |
| < 0.90 | MASE > 1 suggests **model issues**. Continue to B2. |

**For high persistence (ACF > 0.9)**: This is expected behavior. Consider:
- Move-conditional metrics instead of MASE
- Whether prediction is even the right task
- Ensemble with persistence baseline

---

### Step B2: Check Feature Quality

```python
# Are features properly lagged?
for i in range(X.shape[1]):
    # Check if feature is just the target
    corr = np.corrcoef(X[1:, i], y[:-1])[0, 1]  # Lag correlation
    if abs(corr) > 0.95:
        print(f"Feature {i} may be improperly lagged target")
```

**Common Issues**:

| Issue | Symptom | Fix |
|-------|---------|-----|
| Using y[t] to predict y[t] | Perfect training, bad test | Use y[t-1], y[t-2], ... |
| Features not lagged | Features at t use info from t | Add `shift(1)` |
| Too few lags | Model can't capture dynamics | Increase n_lags |
| Wrong lag order | Ignoring important lags | Check PACF for significant lags |

---

### Step B3: Check Model Capacity

```python
# Is the model too simple?
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

models = {
    "Ridge": Ridge(alpha=1.0),
    "RF": RandomForestRegressor(n_estimators=100, max_depth=5),
}

for name, model in models.items():
    mase = evaluate_model(model, X, y)
    print(f"{name}: MASE={mase:.3f}")
```

**If no model beats persistence**: The data may simply not be predictable beyond persistence. This is information, not failure.

---

### Step B4: Check Sample Size

```python
# Minimum samples per fold
n_samples = len(X)
n_splits = 5
test_size = 50

min_train = n_samples - (n_splits * test_size)
print(f"Minimum training samples: {min_train}")

# Rule of thumb
if min_train < 50:
    print("WARNING: May have insufficient training data")
if min_train < 100 and X.shape[1] > 10:
    print("WARNING: High-dimensional with few samples")
```

**Guidelines**:
- Minimum 50 observations per CV fold
- n_train > 10 × n_features for reliable estimates
- High persistence needs even more data

---

## Section C: Common Error Messages

### "Gate HALT: Features beat shuffled target"

**Meaning**: Model performs equally well on shuffled targets → features encode target position.

**Fix**: Check for centered rolling windows, full-series normalization, or target leakage.

---

### "Gate HALT: Gap < horizon"

**Meaning**: Temporal gap between train and test is insufficient.

**Fix**: Set `gap >= horizon` in `WalkForwardCV`.

---

### "Gate WARN: Suspicious improvement (10-20%)"

**Meaning**: Results are good but not alarming. Proceed with verification.

**Fix**: Double-check features, run on held-out data, document assumptions.

---

### "Gate HALT: Improvement > 20%"

**Meaning**: Results are too good. Almost always indicates leakage.

**Fix**: Run through Section A flowchart.

---

## Quick Debugging Checklist

Before trusting any time-series model:

- [ ] **Shuffled target test PASS** — Features don't encode position
- [ ] **Gap >= horizon** — No temporal overlap
- [ ] **Thresholds from training only** — Percentiles, normalization, regimes
- [ ] **Features lagged properly** — `shift(1)` before rolling operations
- [ ] **MASE reported** — Not just raw MAE
- [ ] **Multiple horizons checked** — h=1 shouldn't be dramatically better
- [ ] **Improvement reasonable** — <20% for high-persistence data

---

## Code Template: Full Validation

```python
def full_validation_check(model, X, y, horizon=1, verbose=True):
    """
    Complete validation pipeline with diagnostics.

    Returns: (is_valid, report_dict)
    """
    from temporalcv import (
        gate_shuffled_target,
        gate_temporal_boundary,
        gate_suspicious_improvement,
        WalkForwardCV,
        compute_mase,
    )
    from sklearn.metrics import mean_absolute_error
    from sklearn.base import clone

    report = {"checks": [], "status": "PASS"}

    # 1. Shuffled target test
    shuffle_result = gate_shuffled_target(
        model=model, X=X, y=y,
        n_shuffles=100, method="permutation", random_state=42
    )
    report["checks"].append({
        "name": "shuffled_target",
        "status": shuffle_result.status.value,
        "details": shuffle_result.message
    })
    if shuffle_result.status.value == "HALT":
        report["status"] = "HALT"
        report["primary_issue"] = "Feature leakage detected"

    # 2. Walk-forward with gap
    cv = WalkForwardCV(n_splits=5, gap=horizon, test_size=50)
    all_preds, all_actuals = [], []

    for train_idx, test_idx in cv.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[test_idx])
        all_preds.extend(preds)
        all_actuals.extend(y[test_idx])

    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    # 3. Compute MASE
    mase = compute_mase(all_preds, all_actuals, y[:len(y)//2])  # Training half
    report["mase"] = mase

    # 4. Check improvement
    model_mae = mean_absolute_error(all_actuals, all_preds)
    persistence_mae = mean_absolute_error(all_actuals[1:], all_actuals[:-1])
    improvement = (persistence_mae - model_mae) / persistence_mae

    improvement_result = gate_suspicious_improvement(
        model_metric=model_mae,
        baseline_metric=persistence_mae,
        threshold=0.20
    )
    report["checks"].append({
        "name": "suspicious_improvement",
        "status": improvement_result.status.value,
        "improvement": improvement
    })
    if improvement_result.status.value == "HALT":
        report["status"] = "HALT"
        report["primary_issue"] = "Suspiciously large improvement"

    if verbose:
        print(f"Validation Status: {report['status']}")
        print(f"MASE: {mase:.3f}")
        print(f"Improvement: {improvement:.1%}")
        for check in report["checks"]:
            print(f"  {check['name']}: {check['status']}")

    return report["status"] == "PASS", report

# Usage
is_valid, report = full_validation_check(Ridge(alpha=1.0), X, y, horizon=1)
```

---

## See Also

- [Feature Engineering Safety Guide](feature_engineering_safety.md) — Fix leaky features
- [Metric Selection Guide](metric_selection.md) — Choose the right metric
- [Walk-Forward CV Tutorial](walk_forward_cv.md) — Proper temporal validation
- [Notebook 01: Why Temporal CV](../../notebooks/01_why_temporal_cv.ipynb) — Visual explanations
