# WalkForwardCV Model Card

**Version**: 1.0.0
**Module**: `temporalcv.cv`
**Type**: Cross-validator (sklearn-compatible)
**License**: MIT
**Knowledge Tier**: [T1] Walk-forward validation (Tashman 2000, Bergmeir & Benitez 2012)

---

## Component Details

WalkForwardCV provides sklearn-compatible temporal cross-validation with leakage prevention. It implements walk-forward validation per Tashman (2000) with gap enforcement per Bergmeir & Benitez (2012).

**Key Innovation**: Unlike sklearn's `TimeSeriesSplit`, WalkForwardCV enforces `gap >= horizon` to prevent target leakage in multi-step forecasting scenarios.

---

## Intended Use

### Primary Use Cases

- Time-series model evaluation with leakage prevention
- sklearn `cross_val_score` integration for temporal data
- Walk-forward backtesting for forecasting models
- Multi-step forecast evaluation with gap enforcement

### Out-of-Scope Uses

- **Cross-sectional data**: Use standard `KFold` instead
- **Panel data with entities**: Consider grouped splits or entity-aware CV
- **Real-time streaming**: This is batch-oriented
- **Nested hyperparameter tuning**: Use outer/inner CV loops to avoid nested leakage

### Target Users

- ML practitioners with time-series forecasting experience
- Quantitative researchers requiring rigorous backtesting
- **Prerequisites**: Understanding of temporal ordering and lookahead bias

---

## Parameters

| Parameter | Type | Default | Description | Tier |
|-----------|------|---------|-------------|------|
| `n_splits` | int | 5 | Number of CV folds | [T2] |
| `horizon` | int | None | Forecast horizon h; validates `gap >= horizon` | [T1] |
| `window_type` | str | "expanding" | "expanding" or "sliding" | [T1] |
| `window_size` | int | None | Training window size; required for sliding | [T2] |
| `gap` | int | 0 | Samples excluded between train/test | [T1] |
| `test_size` | int | 1 | Samples per test fold | [T2] |

### Parameter Constraints

- `gap >= horizon` **enforced** when horizon is set (raises `ValueError` otherwise)
- `window_size` required if `window_type == "sliding"`
- `n_splits >= 1`, `gap >= 0`, `test_size >= 1`

### Window Types

| Type | Behavior | Use Case |
|------|----------|----------|
| **expanding** | Training set grows from initial size | Stationary data, maximize training |
| **sliding** | Fixed-size window moves forward | Non-stationary data, regime changes |

---

## Assumptions

| Assumption | Required For | Violation Consequence | Validation Method |
|------------|--------------|----------------------|-------------------|
| Temporal ordering | Valid train/test split | Training on future data | `assert np.all(np.diff(timestamps) > 0)` |
| `gap >= horizon` | Prevent lookahead | Target leakage | Enforced at construction |
| Sufficient samples | Minimum splits | `ValueError` raised | `n > window_size + n_splits * test_size` |
| Features computed before split | No feature leakage | Information leakage | Manual verification |
| Targets computed before split | No target leakage | Information leakage | Manual verification |

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `split()` | O(n_splits) | Index computation only |
| `get_n_splits()` | O(1) | Returns stored value |

### Space Complexity

- O(n) for storing indices per split
- No data copying during iteration

### Sample Size Requirements

| Context | Minimum | Recommended | Justification |
|---------|---------|-------------|---------------|
| Basic usage | window_size + gap + test_size | 100+ samples | Reasonable evaluation variance |
| Statistical inference | 50+ test samples total | 100+ test samples | CLT requirements for DM test |
| Sliding window | 2 × window_size | 5 × window_size | Multiple independent folds |

---

## Limitations and Caveats

### Known Limitations

1. **No entity grouping**: Cannot handle panel data with multiple entities sharing a time axis
2. **Fixed test size**: All folds have same `test_size` (unlike some implementations)
3. **No blocked cross-validation**: For that, consider specialized libraries
4. **Stationarity assumption**: Expanding window accumulates data from different regimes

### When NOT to Use

- Data is not temporally ordered
- Cross-sectional or i.i.d. data (use `KFold`)
- Real-time/streaming applications
- When you need nested CV for hyperparameter tuning without an outer loop

### Common Misconfigurations

| Mistake | Problem | Fix |
|---------|---------|-----|
| Forgetting gap for h-step forecasts | Target leakage | Set `gap >= h` or use `horizon` parameter |
| Computing features on full data | Feature leakage | Split data BEFORE feature engineering |
| Using expanding window with regime changes | Training on irrelevant data | Use `window_type="sliding"` |
| Small `n_splits` with large test_size | Poor variance estimation | Ensure `n_splits >= 5` |

---

## Examples

### Basic Usage

```python
from temporalcv.cv import WalkForwardCV
from sklearn.linear_model import Ridge

cv = WalkForwardCV(n_splits=5, horizon=2, extra_gap=0)
for train_idx, test_idx in cv.split(X):
    model = Ridge()
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

### With sklearn Integration

```python
from sklearn.model_selection import cross_val_score

cv = WalkForwardCV(n_splits=5, horizon=3, extra_gap=0)
scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
```

### Multi-Step Forecasting with Gap

```python
# For 4-week ahead forecasting with weekly data
# y[t] = target[t+4] - target[t], so horizon=4 required (minimum safe separation)
cv = WalkForwardCV(n_splits=5, horizon=4, extra_gap=0)
```

---

## References

### [T1] Academic Sources

- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. *International Journal of Forecasting*, 16(4), 437-450.
- Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

### [T2] Empirical Sources

- sklearn `TimeSeriesSplit`: Extended with gap and sliding window support
- temporalcv validation: Verified against known leakage scenarios

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-Q1 | Initial release with gap enforcement |
| 0.4.0 | 2024-12 | Added `horizon` parameter for automatic gap validation |
| 0.3.0 | 2024-11 | Added sliding window support |

---

## See Also

- `gate_temporal_boundary`: Verify gap enforcement meets requirements
- `gate_signal_verification`: First-stage leakage detection
- `sklearn.model_selection.TimeSeriesSplit`: sklearn's built-in (no gap support)
