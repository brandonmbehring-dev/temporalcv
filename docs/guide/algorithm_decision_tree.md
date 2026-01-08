# Algorithm Decision Tree

```{admonition} Quick Reference
:class: tip

This guide helps you choose:
1. **Which cross-validation method** for your data
2. **Which performance metric** for your problem
3. **Which statistical test** to compare models
4. **What to do** when a gate returns HALT
```

## Part 1: Choosing Your Cross-Validation Method

### Decision Flowchart

```
START: Do you have time series data?
│
├─ NO → Use standard sklearn KFold
│
└─ YES → Is your training data limited?
         │
         ├─ YES (small data) → Use WalkForwardCV (expanding window)
         │   • Training set grows with each fold
         │   • Maximizes training data usage
         │   • Best for: < 500 observations
         │
         └─ NO (large data) → Is the data generating process stable?
              │
              ├─ YES (stable) → Use SlidingWindowCV (fixed window)
              │   • Fixed-size training window
              │   • Downweights old data
              │   • Best for: stable relationships
              │
              └─ NO (regime changes) → Does overlap matter?
                   │
                   ├─ YES (financial) → Use PurgedKFold + embargo
                   │   • Removes label overlap
                   │   • Gap prevents info leakage
                   │   • Best for: finance, trading
                   │
                   └─ NO → Use BlockingTimeSeriesSplit
                        • No overlap between folds
                        • Simpler than purging
```

### Quick Selection Table

| Scenario | CV Method | Key Parameters |
|----------|-----------|----------------|
| Standard forecasting | `WalkForwardCV` | `min_train_periods`, `gap` |
| Non-stationary data | `SlidingWindowCV` | `train_size`, `test_size`, `gap` |
| Financial returns | `PurgedKFold` | `n_splits`, `embargo`, `pct_embargo` |
| Irregular timestamps | `GapCrossValidator` | `gap_before`, `gap_after` |
| Multi-horizon | `MultiHorizonCV` | `horizons=[1, 5, 20]` |

### Code Examples

```python
from temporalcv import WalkForwardCV, SlidingWindowCV
from temporalcv.cv_financial import PurgedKFold

# Expanding window (default choice)
cv = WalkForwardCV(
    n_splits=5,
    min_train_periods=100,
    gap=1  # 1-step ahead forecast
)

# Fixed window (for regime-changing data)
cv = SlidingWindowCV(
    train_size=252,  # 1 year of daily data
    test_size=21,    # 1 month test
    gap=5            # 5-day forecast horizon
)

# Financial data (label overlap concerns)
cv = PurgedKFold(
    n_splits=5,
    embargo_days=5,
    pct_embargo=0.01
)
```

---

## Part 2: Choosing Your Performance Metric

### Decision Flowchart

```
START: What are you predicting?
│
├─ Point forecast → Is the series highly autocorrelated?
│   │
│   ├─ YES (persistence > 0.9) → Use MASE or Skill Score
│   │   • Compares to naive (random walk) baseline
│   │   • MASE < 1 means better than naive
│   │
│   └─ NO → Use RMSE or MAE
│       • Consider asymmetric loss if errors have different costs
│       • RMSE penalizes large errors more
│
├─ Direction/Sign → Use Directional Accuracy
│   │
│   └─ Also run PT test to verify statistical significance
│
├─ Quantile/Interval → Use Pinball Loss or Coverage
│   │
│   ├─ Point-in-interval: Use coverage (should be ~95% for 95% CI)
│   └─ Quantile accuracy: Use pinball loss (asymmetric)
│
└─ Probability → Use Log Loss or Brier Score
```

### Metric Selection Table

| Prediction Type | Primary Metric | Why |
|-----------------|----------------|-----|
| Price level | **MASE** | Naive-adjusted, scale-free |
| Returns | **Sharpe-weighted MSE** | Penalizes volatility-adjusted errors |
| Direction (up/down) | **Directional Accuracy** | + PT test for significance |
| Volatility | **QLIKE** | Penalizes underestimation |
| Quantiles | **Pinball Loss** | Proper scoring rule |
| Intervals | **Coverage + Width** | Both calibration and precision |

### Code Examples

```python
from temporalcv.persistence import compute_persistence_metrics
from temporalcv.metrics import mase, directional_accuracy
from temporalcv.statistical_tests import pt_test

# For persistent series (stock prices, etc.)
metrics = compute_persistence_metrics(y_test, predictions)
print(f"MASE: {metrics['mase']:.3f}")  # < 1 is better than naive

# For direction prediction
da = directional_accuracy(y_test, predictions)
pt_result = pt_test(y_test, predictions)
print(f"Directional Accuracy: {da:.1%}")
print(f"PT test p-value: {pt_result.pvalue:.4f}")

# For quantile forecasts
from temporalcv.metrics import pinball_loss
loss = pinball_loss(y_test, quantile_preds, tau=0.95)
```

---

## Part 3: Choosing Your Statistical Test

### Decision Flowchart

```
START: Comparing forecast accuracy?
│
├─ Equal complexity models → Use DM test
│   │
│   └─ Is h > 1 (multi-step)? → Apply Harvey adjustment
│       cv = dm_test(e1, e2, h=h, harvey_correction=True)
│
├─ Nested models (Model B = Model A + features)
│   │
│   └─ Use Clark-West test (CW)
│       • DM test is biased for nested models
│       • CW adjusts for "noise estimation" under null
│
├─ Multiple models (>2) → Use Model Confidence Set (MCS)
│   │
│   └─ Returns set of "best" models (statistically equivalent)
│
└─ Conditional accuracy → Use Giacomini-White test
    │
    └─ Tests if accuracy varies by conditioning variable
        • E.g., "Is model A better during recessions?"
```

### Test Selection Table

| Comparison | Test | When to Use |
|------------|------|-------------|
| Model A vs B (equal) | **DM test** | Standard pairwise comparison |
| Model A vs B (nested) | **CW test** | B = A + extra features |
| Models A, B, C, ... | **MCS** | Multiple comparison |
| Conditional | **GW test** | Accuracy varies by state |
| Direction only | **PT test** | Sign prediction |

### Code Examples

```python
from temporalcv.statistical_tests import dm_test, cw_test, pt_test

# Standard comparison (non-nested models)
result = dm_test(
    errors_model_a,
    errors_model_b,
    h=5,  # 5-step ahead
    harvey_correction=True
)
print(f"DM statistic: {result.statistic:.3f}")
print(f"p-value: {result.pvalue:.4f}")

# Nested models (Model B adds features to Model A)
result = cw_test(
    y_true,
    predictions_model_a,
    predictions_model_b,
    h=1
)
print(f"CW statistic: {result.statistic:.3f}")

# Directional accuracy
result = pt_test(y_true, predictions)
print(f"PT statistic: {result.statistic:.3f}")
```

---

## Part 4: When Gates Return HALT

### HALT Decision Tree

```
Gate returns HALT
│
├─ gate_temporal_boundary → Train/test overlap
│   │
│   └─ FIX: Ensure max(train_time) < min(test_time) - gap
│       • Check your CV split logic
│       • Verify date indexing is correct
│       • Add gap parameter if h-step forecast
│
├─ gate_shuffled_target → Target looks randomly shuffled
│   │
│   └─ FIX: Your features contain the target
│       • Check rolling calculations (need .shift(1))
│       • Verify no direct target leakage
│       • Review feature engineering pipeline
│
├─ gate_regime_leakage → Future regime info in features
│   │
│   └─ FIX: Regime computed using future data
│       • Use expanding quantiles with .shift()
│       • Don't use full-sample thresholds
│       • Compute regime on train only
│
└─ gate_feature_correlation → Suspicious train/test correlation
    │
    └─ INVESTIGATE: May be legitimate or leakage
        • Check for normalizing on full data
        • Verify group statistics computed on train only
        • Review any feature transformations
```

### HALT Response Protocol

```python
from temporalcv.gates import run_gates

result = run_gates(X_train, y_train, X_test, y_test)

if result.status == "HALT":
    # 1. Identify the failing gate
    failing_gate = result.failed_gates[0]
    print(f"HALT triggered by: {failing_gate.name}")
    print(f"Reason: {failing_gate.reason}")

    # 2. Do NOT proceed with model training
    raise ValueError(f"Data leakage detected: {failing_gate.reason}")

elif result.status == "WARN":
    # Investigate but may proceed with caution
    print(f"Warning: {result.warnings}")
    # Log warning for review

else:  # PASS
    # Safe to proceed
    model.fit(X_train, y_train)
```

### Common HALT Causes and Fixes

| Gate | Common Cause | Fix |
|------|--------------|-----|
| `temporal_boundary` | Dates shuffled during preprocessing | Sort by date before splitting |
| `shuffled_target` | `.rolling().mean()` without `.shift()` | Add `.shift(1)` to all rolling features |
| `regime_leakage` | `df['vol'].quantile(0.9)` on full data | Use `expanding().quantile().shift(1)` |
| `feature_correlation` | `StandardScaler().fit(X_all)` | Fit scaler on `X_train` only |

---

## Part 5: Complete Decision Workflow

```
1. PREPARE DATA
   │
   └─ Run gates early → Fix any HALT issues

2. CHOOSE CV METHOD
   │
   ├─ Small data? → WalkForwardCV (expanding)
   ├─ Regime changes? → SlidingWindowCV (fixed)
   └─ Financial? → PurgedKFold (with embargo)

3. CHOOSE METRIC
   │
   ├─ Persistent series? → MASE (vs naive)
   ├─ Direction matters? → Directional Accuracy
   └─ Standard? → MAE/RMSE

4. EVALUATE
   │
   └─ Run CV → Get fold-wise scores

5. COMPARE MODELS
   │
   ├─ Two equal models? → DM test
   ├─ Nested models? → CW test
   └─ Many models? → MCS

6. DEPLOY
   │
   └─ Monitor for regime changes → Consider adaptive retraining
```

## Quick Reference Card

```{list-table} Algorithm Selection Summary
:header-rows: 1

* - Decision
  - Default Choice
  - Alternative When
* - CV Method
  - `WalkForwardCV`
  - `PurgedKFold` for finance
* - Gap Parameter
  - `gap = forecast_horizon`
  - `gap = 0` only for h=1
* - Primary Metric
  - MASE
  - Directional Accuracy for sign prediction
* - Statistical Test
  - DM test + Harvey
  - CW test for nested models
* - On HALT
  - Stop, investigate, fix
  - Never proceed with HALT
```

## Next Steps

- **[Why Time Series Is Different](why_time_series_is_different.md)**: Conceptual foundation
- **[Common Pitfalls](common_pitfalls.md)**: Avoid these 8 mistakes
- **[API Reference](../api/cv.md)**: Full class documentation
