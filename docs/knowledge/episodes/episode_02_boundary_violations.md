# Episode 02: Temporal Boundary Violations

**Category**: Bug Category 7 (Walk-Forward Splits)
**Discovered**: myga-forecasting-v1, 2023-Q3
**Impact**: h=1 appeared 3x better than h=2,3,4

---

## The Bug

Walk-forward cross-validation splits had no gap between training and test sets when forecasting multiple steps ahead.

```python
# BUGGY CODE
def walk_forward_split(data, train_size, test_size):
    for i in range(train_size, len(data) - test_size):
        train = data[:i]
        test = data[i:i+test_size]  # NO GAP!
        yield train, test
```

**Problem**: For h=2 (2-step ahead forecast), the model trains on data up to time `t`, then predicts for times `t+1` and `t+2`. But the test set starts at `t+1`, meaning the target at `t+1` is only 1 step ahead, not 2.

---

## How It Was Discovered

The symptom was bizarre performance patterns:

| Horizon | MAE | Improvement over Persistence |
|---------|-----|------------------------------|
| h=1 | 0.02 | 45% |
| h=2 | 0.05 | 12% |
| h=3 | 0.06 | 8% |
| h=4 | 0.07 | 5% |

h=1 was suspiciously better than longer horizons. Investigation revealed:
- h=1 was effectively h=0 (nowcasting, not forecasting)
- The "gap" was negative for longer horizons

---

## Root Cause Analysis

The fundamental confusion was between:
- **Forecast horizon**: How far ahead we're predicting
- **Gap**: Minimum distance between train end and test start

For h-step forecasting:
- Train on data up to time `T`
- First valid test observation is at time `T + h`
- Gap MUST be at least `h`

```
WRONG (gap=0):
Train: [0, 1, 2, ..., T]
Test:  [T+1, T+2, ...]  ← T+1 is only 1 step ahead!

RIGHT (gap=h):
Train: [0, 1, 2, ..., T]
Test:  [T+h, T+h+1, ...]  ← T+h is h steps ahead
```

---

## The Fix

temporalcv enforces gap >= horizon:

```python
# CORRECT CODE
cv = WalkForwardCV(
    window_size=100,
    gap=2,      # Gap MUST equal forecast horizon
    test_size=1
)

for train_idx, test_idx in cv.split(X, y):
    # Guaranteed: train_idx[-1] + gap < test_idx[0]
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

---

## Why gate_temporal_boundary Catches This

The gate explicitly validates split geometry:

```python
result = gate_temporal_boundary(
    cv=cv,
    X=X,
    y=y,
    horizon=2
)

if result.status == GateStatus.HALT:
    print(result.message)
    # "Gap (0) less than horizon (2) in 15 splits"
```

---

## Prevention Checklist

- [ ] Always set `gap=horizon` in `WalkForwardCV`
- [ ] Run `gate_temporal_boundary()` with correct horizon
- [ ] Verify that `train_idx[-1] + gap < test_idx[0]` for all splits
- [ ] Compare h=1 vs h>1 performance—if h=1 >> h>1, investigate

---

## Test Case

```python
def test_boundary_violation_detection():
    """Gate should catch insufficient gap between train and test."""
    X = np.random.randn(200, 5)
    y = np.random.randn(200)

    # CV with gap=0 but horizon=2
    cv = WalkForwardCV(window_size=100, gap=0, test_size=1)

    # Gate should HALT
    result = gate_temporal_boundary(cv, X, y, horizon=2)
    assert result.status == GateStatus.HALT
    assert "gap" in result.message.lower()
```

---

## The "h=1 >> h=2,3,4" Heuristic

When h=1 performance is more than 2x better than h=2,3,4:

```python
if mae_h1 < mae_h2 / 2:
    print("WARNING: Likely boundary violation")
    print("h=1 should not be 2x better than h=2")
```

This pattern almost always indicates gap issues.

---

## Related

- [Leakage Audit Trail](../leakage_audit_trail.md) - Full category list
- [Episode 01: Lag Leakage](episode_01_lag_leakage.md) - Feature-level temporal issue
- SPECIFICATION.md Section 1.4 - Temporal boundary threshold
