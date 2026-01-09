# Leakage Audit Trail

**Purpose**: Document data leakage patterns, how they were discovered, and how temporalcv's gates prevent them.

**Source**: Compiled from lever_of_archimedes patterns and myga-forecasting postmortems.

---

## The 10 Bug Categories

These categories were identified through multiple production ML projects. Each bug was discovered the hard way—through suspiciously good results that couldn't replicate.

| # | Category | What Goes Wrong | Gate That Catches It |
|---|----------|-----------------|---------------------|
| 1 | Target Alignment | Train/test date boundaries violated | `gate_temporal_boundary()` |
| 2 | Future Data in Lags | Lag features use future information | `gate_signal_verification()` |
| 3 | Persistence Implementation | Predicted levels instead of changes | Manual verification |
| 4 | Feature Selection on Target | Feature selection used test target | `gate_signal_verification()` |
| 5 | Regime Computation | Thresholds computed from full series | `gate_signal_verification()` |
| 6 | Weights Computation | Look-ahead in sample weighting | `gate_signal_verification()` |
| 7 | Walk-Forward Splits | Incorrect gap calculation | `gate_temporal_boundary()` |
| 8 | Multiple Sources of Truth | Conflicting documentation | SPECIFICATION.md |
| 9 | Internal-Only Validation | No external verification | `gate_synthetic_ar1()` |
| 10 | Architecture Mismatch | Model incompatible with data frequency | Manual verification |

---

## Bug Category Details

### Category 1: Target Alignment

**Symptom**: Model accuracy far exceeds theoretical bounds.

**Root Cause**: Train/test boundary is off-by-one, allowing test target into training.

**Example (myga-forecasting-v1)**:
```python
# WRONG: test_idx includes target observation
train_idx = range(0, split_point)
test_idx = range(split_point, n)  # First test obs used for training

# RIGHT: explicit gap enforcement
train_idx = range(0, split_point - gap)
test_idx = range(split_point, n)
```

**Prevention**: `gate_temporal_boundary(horizon=h)` verifies `train_end + gap < test_start`.

---

### Category 2: Future Data in Lags

**Symptom**: Lag features have predictive power on shuffled target.

**Root Cause**: Rolling statistics (mean, std, EMA) computed on full series before train/test split.

**Example**:
```python
# WRONG: computes on full series, leaks future
df['rolling_mean'] = df['target'].rolling(10).mean()
train, test = df[:split], df[split:]

# RIGHT: compute within each training window
for train_idx, test_idx in cv.split(X, y):
    train_mean = y[train_idx].rolling(10).mean()
```

**Prevention**: `gate_signal_verification()` detects when features encode target position.

---

### Category 3: Persistence Implementation

**Symptom**: "Persistence" baseline shows implausible skill.

**Root Cause**: Persistence model predicts levels instead of changes.

**Example (myga-forecasting-v2)**:
```python
# WRONG: predicting levels (trivially correlated with target)
persistence_pred = y[:-1]  # Predict level from yesterday

# RIGHT: predicting changes (zero change = persistence)
persistence_pred = np.zeros_like(y_test)  # Predict no change
```

**Prevention**: Manual verification against known-answer calculations. Compare MAE against theoretical σ×√(2/π).

---

### Category 4: Feature Selection on Target

**Symptom**: Feature importance scores suspiciously high.

**Root Cause**: Feature selection (e.g., mutual information, correlation) computed using test target.

**Example**:
```python
# WRONG: feature selection on full data including test
selected = SelectKBest(k=5).fit(X_all, y_all)

# RIGHT: feature selection on training data only
selected = SelectKBest(k=5).fit(X_train, y_train)
```

**Prevention**: `gate_signal_verification()` catches this—selected features will encode target position.

---

### Category 5: Regime Computation

**Symptom**: Regime-conditional metrics show impossible stratification.

**Root Cause**: Regime thresholds (e.g., volatility percentiles) computed from full series including test.

**Example (myga-forecasting-v2 BUG-003)**:
```python
# WRONG: threshold from full series
threshold = np.percentile(np.abs(y_all), 70)

# RIGHT: threshold from training only
threshold = compute_move_threshold(y_train, percentile=70)
```

**Prevention**:
- `gate_signal_verification()` catches this indirectly
- Explicit training-only computation in `compute_move_threshold()`
- Documentation warning in docstrings

---

### Category 6: Weights Computation

**Symptom**: Weighted metrics show unusual patterns.

**Root Cause**: Sample weights computed using future information (e.g., importance sampling on full distribution).

**Example**:
```python
# WRONG: weights use future distribution
weights = compute_importance_weights(y_all)

# RIGHT: weights from training distribution only
weights = compute_importance_weights(y_train)
```

**Prevention**: `gate_signal_verification()` can detect when weights encode future information.

---

### Category 7: Walk-Forward Splits

**Symptom**: h=1 performance vastly exceeds h=2,3,4.

**Root Cause**: Gap between train and test is less than forecast horizon.

**Example**:
```python
# WRONG: no gap for 2-step forecast
train = data[:100]
test = data[100:105]  # First test uses data from train!

# RIGHT: gap = horizon
train = data[:100]
test = data[102:107]  # Gap of 2 for h=2
```

**Prevention**:
- `WalkForwardCV(gap=horizon)` enforces correct gaps
- `gate_temporal_boundary()` validates all splits

---

### Category 8: Multiple Sources of Truth

**Symptom**: Results differ between runs with identical data.

**Root Cause**: Threshold values defined in multiple places with inconsistencies.

**Example**:
```python
# In config.py: THRESHOLD = 0.05
# In model.py:  threshold = 0.10  # Different!
```

**Prevention**:
- Single authoritative `SPECIFICATION.md`
- All thresholds frozen with Amendment Process
- No magic numbers in code—reference SPECIFICATION

---

### Category 9: Internal-Only Validation

**Symptom**: Results don't replicate on new data.

**Root Cause**: Model validated only on internal cross-validation, not external data.

**Solution Order**:
1. `gate_signal_verification()` — Does model beat random alignment?
2. `gate_synthetic_ar1()` — Does model beat theoretical bounds?
3. Then internal validation

**Prevention**: Always run external gates BEFORE reporting internal metrics.

---

### Category 10: Architecture Mismatch

**Symptom**: Model produces NaN or extreme values.

**Root Cause**: Model architecture incompatible with data characteristics.

**Example (myga-forecasting-v1)**:
```
MIDAS model expects mixed frequencies (daily → weekly)
Data was already weekly → model misinterpreted
```

**Prevention**: Manual verification. No automated gate—requires domain knowledge.

---

## Gate Coverage Matrix

| Gate | Categories Caught | Detection Mechanism |
|------|------------------|---------------------|
| `gate_signal_verification()` | 2, 4, 5, 6 | Model shouldn't beat random target alignment |
| `gate_synthetic_ar1()` | 9 | Model shouldn't beat theoretical optimum |
| `gate_suspicious_improvement()` | 1, 2, 3, 4, 5 | >20% improvement = likely bug |
| `gate_temporal_boundary()` | 1, 7 | Explicit gap enforcement |

---

## Verification Protocol

For any new model, run gates in this order:

```python
from temporalcv import run_gates, gate_signal_verification, gate_synthetic_ar1
from temporalcv import gate_suspicious_improvement, gate_temporal_boundary

# 1. External verification FIRST
gates = [
    gate_signal_verification(model, X, y),
    gate_synthetic_ar1(model),
]

# 2. Then internal validation
gates += [
    gate_suspicious_improvement(model_mae, baseline_mae),
    gate_temporal_boundary(cv, horizon=h),
]

# 3. Aggregate
report = run_gates(gates)
if report.status == "HALT":
    raise ValueError(f"Validation failed: {report.summary()}")
```

---

## Lessons Learned

### From myga-forecasting-v1 (2023-Q3)
- **Bug**: MIDAS architecture mismatch with data frequency
- **Impact**: 2 weeks debugging
- **Lesson**: Verify architecture assumptions before implementation

### From myga-forecasting-v2 (2024-Q1)
- **Bug**: Regime threshold computed from full series (BUG-003)
- **Impact**: Overstated MC-SS by ~15%
- **Lesson**: ALL thresholds must come from training data

### From myga-forecasting-v3 (2024-Q2)
- **Bug**: Rolling mean computed on full series before split
- **Impact**: Lag features encoded future information
- **Lesson**: Compute features WITHIN walk-forward splits

---

## Adding New Bugs

When a new leakage bug is discovered:

1. **Document the symptom**: What made you suspicious?
2. **Identify the root cause**: What code allowed future information?
3. **Map to category**: Does it fit existing categories or is it new?
4. **Verify gate detection**: Does an existing gate catch it?
5. **Add test case**: Create regression test to prevent recurrence
6. **Update this document**: Add to Lessons Learned

Follow the Amendment Process in AI_CONTEXT.md for any changes.
