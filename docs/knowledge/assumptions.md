# Assumptions by Module

This document explicitly lists the assumptions required by each temporalcv module, what happens when they're violated, and how to validate them.

---

## Global Assumptions

These assumptions apply across all modules:

| Assumption | Description | Validation |
|------------|-------------|------------|
| **Temporal ordering** | Data is sorted by time, oldest first | `assert np.all(np.diff(timestamps) > 0)` |
| **No missing values** | Arrays contain no NaN/None | `assert not np.any(np.isnan(data))` |
| **Consistent dtype** | Arrays are float64 for precision | `data.astype(np.float64)` |

---

## statistical_tests.py

### dm_test()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| n ≥ 30 | Asymptotic normality | Unreliable p-values | Raises `ValueError` |
| h ≥ 1 | Valid horizon | Invalid bandwidth | Raises `ValueError` |
| Same-length errors | Paired comparison | Cannot compute | Raises `ValueError` |
| Stationary errors | Asymptotic theory | Biased variance | Check ADF test |

**Critical**: For h > 1, errors are MA(h-1). HAC variance handles this, but stationarity is still assumed.

### pt_test()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| n ≥ 20 | Asymptotic normality | Unreliable p-values | Raises `ValueError` |
| Independence under H₀ | Valid null distribution | p-values too optimistic | Check residual ACF |
| Same-length arrays | Paired comparison | Cannot compute | Raises `ValueError` |

**Warning**: For h > 1, HAC correction is NOT applied to PT test. P-values may be overly optimistic. Use DM test for rigorous multi-step comparison.

### compute_hac_variance()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| Weak stationarity | Consistent estimation | Biased variance | KPSS test |
| Sufficient bandwidth | Capture autocorrelation | Underestimated variance | bandwidth ≥ h - 1 |

---

## conformal.py

### SplitConformalPredictor

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| **Exchangeability** | Coverage guarantee | Coverage < (1-α) | Shuffle test |
| Calibration ≥ 30 | Stable quantile | High variance | n_calibration check |
| i.i.d. residuals | Valid quantile | Systematic under/over-coverage | Residual diagnostics |

**Critical**: Time series data is NOT exchangeable. Split conformal provides approximate, not exact, coverage for temporal data.

**Mitigation**: Use `AdaptiveConformalPredictor` for distribution shift, or accept approximate coverage.

### AdaptiveConformalPredictor

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| Gradual shift | Adaptation can track | Lag in coverage | Monitor coverage rate |
| γ well-tuned | Convergence | Oscillation or slow adaptation | Grid search |

---

## cv.py

### WalkForwardCV

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| gap ≥ horizon | Prevent lookahead | Information leakage | Asserted in split |
| Sufficient data | Minimum splits | Too few splits for inference | n > window_size + n_splits × test_size |
| Temporal ordering | Valid train/test split | Training on future | Check timestamps |

**Critical**: If computing CHANGE targets (y[t+h] - y[t]), ensure target is computed BEFORE train/test split to avoid leakage through target construction.

---

## persistence.py

### compute_move_threshold()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| Training data only | Prevent regime leakage | Optimistic threshold | Strict data partitioning |
| n ≥ 30 | Stable percentile | High variance | Sample size check |

**Critical [BUG-003]**: Computing threshold from full data (train + test) leaks future regime information. Always use training data only.

### compute_move_conditional_metrics()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| n_up ≥ 10, n_down ≥ 10 | Reliable estimates | High variance | Check `is_reliable` property |
| Threshold from training | No leakage | Optimistic MC-SS | Pass explicit threshold |
| Same-length arrays | Paired comparison | Cannot compute | Raises `ValueError` |

---

## regimes.py

### classify_volatility_regime()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| basis='changes' | Correct methodology | Mislabeled regimes | Use default |
| window < n | Valid rolling | All MED labels | Check data length |
| Sufficient variance | Meaningful thresholds | Degenerate classification | Check std > 1e-8 |

**Critical [BUG-005]**: Using `basis='levels'` mislabels steady drifts as "high volatility". Always use `basis='changes'` (default).

### classify_direction_regime()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| Threshold from training | No leakage | Biased regime labels | Strict partitioning |
| threshold ≥ 0 | Valid classification | Raises ValueError | Asserted |

---

## gates.py

### gate_shuffled_target()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| Temporal relationship | Shuffling destroys signal | False HALT | Not applicable |
| Model is fit per call | Fresh fit on shuffled | Stale model state | Provided model |
| n_shuffles ≥ 3 | Stable average | High variance | Default is 5 |

**Interpretation**: If model beats shuffled target, the features contain temporal structure correlated with target ordering — likely leakage.

### gate_synthetic_ar1()

| Assumption | Required For | Violation Consequence | Validation |
|------------|--------------|----------------------|------------|
| phi < 1 | Stationarity | Explosive process | Validated range |
| n_samples ≥ 200 | Stable estimates | High variance | Default is 500 |

---

## Validation Checklist

Before using temporalcv, verify:

### Data Quality
- [ ] Data is sorted by time (oldest first)
- [ ] No missing values
- [ ] Sufficient sample size (n ≥ 50 for most applications)

### Train/Test Split
- [ ] Gap ≥ forecast horizon
- [ ] Thresholds computed from training only
- [ ] No feature engineering on full data

### Statistical Tests
- [ ] n ≥ 30 for DM test
- [ ] n ≥ 20 for PT test
- [ ] Residuals approximately stationary

### Conformal Prediction
- [ ] Understand that coverage is approximate for time series
- [ ] Calibration set is large enough (≥ 30)
- [ ] Consider adaptive conformal for distribution shift

### Regime Classification
- [ ] Using basis='changes' for volatility (not levels)
- [ ] Thresholds from training data only
- [ ] Check sample counts per regime

---

## When Assumptions Fail

| Assumption | Fallback |
|------------|----------|
| n too small | Use bootstrap or qualitative comparison |
| Non-stationarity | Apply differencing or detrending first |
| Exchangeability violated | Accept approximate conformal coverage |
| Too few samples per regime | Mask low-n regimes or aggregate |
| Temporal order unknown | Cannot use temporal validation — stop |

---

## Knowledge Tiers for Assumptions

| Assumption | Tier | Justification |
|------------|------|---------------|
| n ≥ 30 for DM | T2 | Standard practice, some debate |
| Gap ≥ horizon | T1 | Theoretically required |
| Training-only thresholds | T2 | BUG-003 fix in v2 |
| basis='changes' | T2 | BUG-005 fix in v2 |
| n_up, n_down ≥ 10 | T3 | Rule of thumb |
| Exchangeability | T1 | Required for exact conformal |
