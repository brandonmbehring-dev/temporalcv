# Metric Selection Guide

**Audience**: ML practitioners who need to choose the right evaluation metric for their time-series problem.

**Purpose**: Quick decision matrix + deeper understanding of when each metric matters.

---

## Quick Decision Matrix

| Your Data Type | Recommended Metric | Why | Alternative |
|----------------|-------------------|-----|-------------|
| High persistence (ACF > 0.9) | **MASE** | Compares to naive baseline | Improvement % |
| Levels (prices, rates) | **MAE** | Same units as target | MASE if persistent |
| Returns/changes | **RMSE** | Penalizes large errors | MAE if outliers dominate |
| Direction matters | **Direction Accuracy** | 50% = random guess | Pesaran-Timmermann test |
| Prediction intervals | **Coverage + Width** | Coverage ≥ 1-α required | Interval Score |
| Trading decisions | **Sharpe Ratio** | Risk-adjusted returns | Max Drawdown |
| Probabilistic forecasts | **CRPS** | Proper scoring rule | Pinball Loss (quantiles) |

---

## The Persistence Problem

### Why Standard Metrics Mislead

On high-persistence data (ACF(1) > 0.9), the naive forecast `y_pred[t] = y[t-1]` achieves excellent MAE:

```
Treasury rates (φ ≈ 0.99):
  - Naive MAE ≈ 0.001 (incredibly small!)
  - Your model MAE = 0.0009
  - "10% improvement" → But did you learn anything?
```

**The problem**: Both metrics are tiny, making comparison meaningless.

### MASE: The Solution [T1]

Mean Absolute Scaled Error normalizes by the naive forecast error:

```
MASE = Model MAE / Naive MAE (from training)
```

| MASE Value | Interpretation |
|------------|----------------|
| MASE < 1.0 | Model beats naive — genuine skill |
| MASE = 1.0 | Model equals naive — no skill |
| MASE > 1.0 | Model worse than naive — negative skill |

**Code Example**:
```python
from temporalcv import compute_mase

# CRITICAL: Compute naive errors from TRAINING data only
mase = compute_mase(predictions, actuals, training_data)

if mase < 1:
    print(f"Model beats persistence by {(1-mase)*100:.1f}%")
else:
    print(f"Model is {(mase-1)*100:.1f}% worse than persistence")
```

**When to Use**: Always use MASE (or report improvement %) on time-series data. Raw MAE without context is meaningless.

---

## Point Forecast Metrics

### MAE (Mean Absolute Error)

```
MAE = mean(|y - ŷ|)
```

| Pros | Cons |
|------|------|
| Same units as target | Doesn't penalize large errors much |
| Interpretable | Misleading on persistent data |
| Robust to outliers | |

**Use when**: You care about typical error magnitude and outliers shouldn't dominate.

### RMSE (Root Mean Squared Error)

```
RMSE = sqrt(mean((y - ŷ)²))
```

| Pros | Cons |
|------|------|
| Penalizes large errors | Sensitive to outliers |
| Differentiable | Different units (squared, then rooted) |
| Standard in optimization | |

**Use when**: Large errors are disproportionately costly (e.g., extreme weather events).

### MAPE (Mean Absolute Percentage Error)

```
MAPE = mean(|y - ŷ| / |y|) × 100%
```

| Pros | Cons |
|------|------|
| Scale-independent | Undefined when y = 0 |
| Easy to communicate | Asymmetric (underprediction penalized more) |

**Use when**: You need a percentage error and target is always positive (e.g., sales).

**Avoid when**: Target can be zero or negative (use SMAPE instead).

### Theil's U

```
U = RMSE(model) / RMSE(naive)
```

| Value | Interpretation |
|-------|----------------|
| U < 1 | Model beats naive |
| U = 1 | Model equals naive |
| U > 1 | Model worse than naive |

**Use when**: Quick comparison to persistence, similar to MASE but uses RMSE.

---

## Directional Metrics

### Direction Accuracy

```
DA = proportion(sign(y - y_prev) == sign(ŷ - y_prev))
```

| Value | Interpretation |
|-------|----------------|
| DA > 0.5 | Better than random |
| DA = 0.5 | Random guessing |
| DA < 0.5 | Worse than random (contrarian signal?) |

**Use when**: You care about predicting direction (up/down) more than magnitude.

**Code Example**:
```python
from temporalcv import compute_direction_accuracy

da = compute_direction_accuracy(predictions, actuals, previous_values)
print(f"Direction accuracy: {da:.1%}")

# Statistical significance via Pesaran-Timmermann test
from temporalcv import pt_test
result = pt_test(predictions, actuals)
print(f"PT p-value: {result.pvalue:.4f}")
```

### Pesaran-Timmermann Test [T1]

Tests whether direction forecasts have predictive ability beyond chance.

```python
from temporalcv import pt_test

result = pt_test(predictions, actuals)
if result.pvalue < 0.05:
    print("Significant directional predictive ability")
else:
    print("No evidence of directional skill")
```

**Use when**: You need statistical evidence that direction predictions are meaningful.

---

## Interval Metrics

### Coverage

```
Coverage = proportion(lower ≤ y ≤ upper)
```

| Target | Interpretation |
|--------|----------------|
| Coverage ≥ 1-α | Intervals are valid |
| Coverage < 1-α | Undercoverage (intervals too narrow) |
| Coverage >> 1-α | Overcoverage (intervals too wide) |

**Requirement**: Coverage must be ≥ 1-α (e.g., ≥ 95% for 95% intervals).

### Mean Width

```
Width = mean(upper - lower)
```

**Goal**: Minimize width while maintaining coverage. Narrow + valid intervals = sharp predictions.

### Interval Score [T1]

Proper scoring rule that penalizes both miscoverage and excessive width:

```
IS = width + (2/α) × (lower - y) × I(y < lower) + (2/α) × (y - upper) × I(y > upper)
```

**Use when**: You want a single number combining coverage and width.

**Code Example**:
```python
from temporalcv import evaluate_interval_quality

quality = evaluate_interval_quality(intervals, actuals)
print(f"Coverage: {quality['coverage']:.1%} (target: {quality['target_coverage']:.1%})")
print(f"Mean width: {quality['mean_width']:.4f}")
print(f"Interval score: {quality['interval_score']:.4f}")
```

---

## Statistical Test Selection

### Comparing Two Models

| Situation | Test | Null Hypothesis |
|-----------|------|-----------------|
| General comparison | **DM test** | Equal predictive accuracy |
| One model nests the other | **Clark-West test** | Simpler model is better |
| Directional forecasts | **PT test** | No directional ability |
| Conditional ability | **Giacomini-White test** | Equal conditional accuracy |

### Diebold-Mariano Test [T1]

The standard for comparing forecast accuracy.

```python
from temporalcv import dm_test

result = dm_test(errors_model_a, errors_model_b, horizon=1)
print(f"DM statistic: {result.statistic:.3f}")
print(f"p-value: {result.pvalue:.4f}")

if result.pvalue < 0.05:
    if result.statistic > 0:
        print("Model B significantly better")
    else:
        print("Model A significantly better")
```

**Key considerations**:
- Use HAC variance for autocorrelated errors
- For h > 1, set `horizon=h` to adjust variance
- See [DM Test Limitations](../api/statistical_tests.md#dm-test-limitations)

### Clark-West Test [T1]

For nested model comparison (e.g., AR(1) vs AR(1) + feature).

```python
from temporalcv import cw_test

# Model A is nested in Model B
result = cw_test(predictions_a, predictions_b, actuals)
if result.pvalue < 0.05:
    print("Larger model significantly better")
```

**Use when**: Testing whether additional features improve over a baseline.

### Multiple Model Comparison

When comparing 3+ models, use p-value correction:

```python
from temporalcv import compare_multiple_models

result = compare_multiple_models(
    predictions_dict={"AR": preds_ar, "Ridge": preds_ridge, "RF": preds_rf},
    actuals=actuals,
    correction="holm"  # Holm-Bonferroni correction
)

for comparison in result.pairwise_results:
    print(f"{comparison['model_a']} vs {comparison['model_b']}: "
          f"p={comparison['adjusted_pvalue']:.4f}")
```

---

## High-Persistence Special Cases

### Move-Conditional Metrics

When persistence is very high (ACF > 0.95), most periods show "no significant change." Evaluate only on "move" periods:

```python
from temporalcv import (
    compute_move_threshold,
    compute_move_conditional_metrics,
)

# Compute threshold from training data ONLY
threshold = compute_move_threshold(y_train)

# Evaluate on test data
mc_metrics = compute_move_conditional_metrics(
    predictions, actuals, threshold=threshold
)

print(f"Move-Conditional MAE: {mc_metrics.mc_mae:.4f}")
print(f"Move-Conditional Skill Score: {mc_metrics.skill_score:.3f}")
```

**Use when**:
- ACF(1) > 0.95
- Most periods are "no change"
- You care about predicting actual movements

### When to NOT Try Beating Persistence

| ACF(1) | Guidance |
|--------|----------|
| > 0.99 | Extremely difficult. Consider: Is prediction even the right task? |
| 0.95-0.99 | Very difficult. Use move-conditional metrics. |
| 0.90-0.95 | Difficult but possible. MASE essential. |
| 0.70-0.90 | Moderate difficulty. Standard metrics work. |
| < 0.70 | Standard ML metrics are meaningful. |

---

## Trading/Financial Metrics

### Sharpe Ratio

```
Sharpe = mean(returns) / std(returns) × sqrt(252)  # Annualized
```

**Use when**: Evaluating trading strategies. Higher = better risk-adjusted returns.

### Max Drawdown

```
MaxDD = max(peak - trough) / peak
```

**Use when**: Risk management. How bad can it get?

### Hit Rate

```
Hit Rate = proportion(sign(predicted_return) == sign(actual_return))
```

**Use when**: Binary trading decisions (long/short).

---

## Decision Flowchart

```
START: What kind of prediction?
  |
  ├─> Point forecast
  |     |
  |     ├─> High persistence (ACF > 0.9)?
  |     |     |
  |     |     YES ──> MASE + Move-Conditional
  |     |     |
  |     |     NO ──> MAE or RMSE
  |     |
  |     └─> Direction important?
  |           |
  |           YES ──> Add Direction Accuracy + PT test
  |
  ├─> Intervals/uncertainty
  |     |
  |     └─> Coverage + Width + Interval Score
  |
  ├─> Probabilistic
  |     |
  |     └─> CRPS or Pinball Loss
  |
  └─> Trading strategy
        |
        └─> Sharpe + Max Drawdown + Hit Rate
```

---

## Quick Reference

### Point Forecasts
| Metric | When to Use | Code |
|--------|-------------|------|
| MASE | Always for time series | `compute_mase(preds, actuals, train)` |
| MAE | Interpretable units | `compute_mae(preds, actuals)` |
| RMSE | Large errors costly | `compute_rmse(preds, actuals)` |
| Direction Accuracy | Up/down matters | `compute_direction_accuracy(...)` |

### Intervals
| Metric | When to Use | Code |
|--------|-------------|------|
| Coverage | Always check first | `intervals.coverage(actuals)` |
| Mean Width | After coverage valid | `intervals.mean_width` |
| Interval Score | Single summary | `evaluate_interval_quality(...)` |

### Statistical Tests
| Test | When to Use | Code |
|------|-------------|------|
| DM | Compare 2 models | `dm_test(errors_a, errors_b)` |
| PT | Direction significance | `pt_test(preds, actuals)` |
| CW | Nested models | `cw_test(preds_a, preds_b, actuals)` |

---

## See Also

- [Feature Engineering Safety Guide](feature_engineering_safety.md) — Safe vs dangerous features
- [Diagnostic Flowchart](diagnostic_flowchart.md) — What to do when validation fails
- [High Persistence Tutorial](high_persistence.md) — Deep dive on sticky data
- [Statistical Tests API](../api/statistical_tests.md) — Full API reference
