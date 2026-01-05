# SPECIFICATION.md - temporalcv

**Version**: 0.1.0 | **Last Updated**: 2025-12-23
**Status**: AUTHORITATIVE (changes require Amendment Process)

This document freezes all parameters, thresholds, and mathematical definitions. Changes must follow the Amendment Process in CLAUDE.md.

---

## 1. Gate Thresholds

### 1.1 Suspicious Improvement Gate [T3]

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `threshold` | 0.20 (20%) | >20% improvement over baseline = "too good to be true" |
| `warn_threshold` | 0.10 (10%) | 10-20% improvement = proceed with caution |

**Source**: Empirical heuristic from myga-forecasting-v2 postmortem.

**Formula**:
```
improvement = (baseline_error - model_error) / baseline_error
HALT if: improvement > 0.20
WARN if: 0.10 < improvement <= 0.20
PASS if: improvement <= 0.10
```

### 1.2 Shuffled Target Gate [T1/T3]

The shuffled target gate supports two statistical methods:

#### Method: `"permutation"` (default, recommended) [T1]

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `method` | "permutation" | True permutation test with p-value [Phipson & Smyth 2010] |
| `alpha` | 0.05 | Standard significance level |
| `n_shuffles` | 100 (default) | Min p-value of 0.0099, sufficient for α=0.05 |
| `strict` | False | If True, uses n_shuffles=199 for p-value resolution of 0.005 |

**Interpretation**: If model beats shuffled targets at p < α, features encode target position.

**Formula** (per Phipson & Smyth 2010):
```
p-value = (1 + count(shuffled_mae <= model_mae)) / (1 + n_shuffles)
HALT if: p-value < alpha
```

#### Method: `"effect_size"` (fast, heuristic) [T3]

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `method` | "effect_size" | Compare improvement ratio to threshold |
| `threshold` | 0.05 (5%) | Maximum acceptable improvement over shuffled |
| `n_shuffles` | 5 (default) | Balance between accuracy and runtime |

**Interpretation**: If model improves by more than threshold over shuffled, likely leakage.

**Formula**:
```
improvement_ratio = 1 - (model_mae / mean_shuffled_mae)
HALT if: improvement_ratio > threshold
```

**Method Selection Guide**:
- Use `"permutation"` (default) for rigorous statistical testing and publication
- Use `"effect_size"` for quick sanity checks during development

### 1.3 Synthetic AR(1) Gate [T1]

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `phi` | 0.95 | High persistence coefficient (typical of financial data) |
| `n_samples` | 500 | Sufficient for reliable AR(1) estimation |
| `tolerance` | 1.5 | Allow 1.5x margin for finite-sample variation |

**Theoretical Bound** (for AR(1) process):
```
Optimal MAE = sigma * sqrt(2/pi) ≈ 0.798 * sigma

HALT if: model_MAE < theoretical_MAE / tolerance
       = model_MAE < 0.798 * sigma / 1.5
       ≈ model_MAE < 0.532 * sigma
```

### 1.4 Temporal Boundary Gate [T2]

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `required_gap` | horizon | Gap must equal forecast horizon to prevent leakage |

**Formula**:
```
HALT if: gap < horizon
PASS if: gap >= horizon

For all splits: train_idx[-1] + gap < test_idx[0]
```

---

## 2. Move Threshold Parameters [T2]

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `percentile` | 70.0 | 70th percentile of |actuals| defines "significant" move |

**Move Classification**:
```
UP:   actual > threshold
DOWN: actual < -threshold
FLAT: |actual| <= threshold
```

**Source**: myga-forecasting-v2 Phase 11 analysis. Provides ~30% moves, ~70% flat periods.

**Critical**: Threshold MUST be computed from training data only to prevent regime threshold leakage.

---

## 3. Statistical Test Parameters

### 3.1 Diebold-Mariano Test [T1]

| Parameter | Default | Reference |
|-----------|---------|-----------|
| `alpha` | 0.05 | Standard significance level |
| `alternative` | "two-sided" | Test for any difference |
| `loss_type` | "squared" | Squared loss (MSE comparison) |

**Bandwidth Selection** (Andrews 1991):
```
bandwidth = floor(4 * (n/100)^(2/9))
```

For h-step forecasts: `bandwidth = h - 1` (MA(h-1) structure).

**Reference**: Diebold & Mariano (1995). *Comparing predictive accuracy*. JBES 13(3).

### 3.2 Harvey Adjustment [T1]

For small samples, apply Harvey et al. (1997) correction:
```
DM_adj = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)
```

**Reference**: Harvey, Leybourne & Newbold (1997). *Testing equality of prediction MSE*. IJF 13(2).

### 3.3 HAC Variance Estimation [T1]

**Bartlett Kernel**:
```
w(j) = 1 - |j| / (bandwidth + 1)    if |j| <= bandwidth
w(j) = 0                             otherwise
```

**HAC Variance Formula**:
```
V_hat(d_bar) = (1/n) * [gamma_0 + 2 * sum_{j=1}^{bandwidth} w(j) * gamma_j]
```

**Reference**: Newey & West (1987). *HAC covariance matrix*. Econometrica 55(3).

### 3.4 Pesaran-Timmermann Test [T1]

**2-Class Mode** (academically validated):
```
p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
PT = (p_hat - p_star) / sqrt(V(p_hat) + V(p_star))
```

**3-Class Mode** [T3] (ad-hoc extension, not published):
```
p_star = sum_k(p_y^k * p_x^k) for k in {UP, DOWN, FLAT}
```

**Warning**: Use 2-class mode for rigorous statistical testing.

**Reference**: Pesaran & Timmermann (1992). *Simple nonparametric test*. JBES 10(4).

---

## 4. Conformal Prediction Parameters [T1]

### 4.1 Split Conformal

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `alpha` | 0.05 | Target 95% coverage |
| `calibration_fraction` | 0.3 | Fraction of data for calibration |

**Quantile Formula**:
```
q = ceil((n + 1) * (1 - alpha)) / n
```

**Reference**: Romano, Patterson & Candès (2019). *Conformalized quantile regression*. NeurIPS.

### 4.2 Adaptive Conformal [T1]

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `gamma` | 0.1 | Learning rate for online adaptation |

**Update Rule**:
```
q_{t+1} = q_t - gamma * alpha        if y_t in C_t(x_t)   (covered)
q_{t+1} = q_t + gamma * (1 - alpha)  if y_t not in C_t(x_t) (not covered)
```

**Reference**: Gibbs & Candès (2021). *Adaptive conformal inference*. NeurIPS.

---

## 5. Cross-Validation Parameters [T2]

### 5.1 Walk-Forward CV

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `window_type` | "sliding" | Sliding window for stationarity |
| `gap` | 0 | Gap between train and test (set to horizon!) |
| `test_size` | 1 | Single observation per fold |

**Gap Enforcement** (CRITICAL):
```
train_end = train_idx[-1]
test_start = test_idx[0]

REQUIRED: train_end + gap < test_start
```

### 5.2 Minimum Observations

| Context | Enforced Min | Recommended Min | Justification |
|---------|--------------|-----------------|---------------|
| DM test | 30 | 50 | CLT requirement for asymptotic normality + Harvey adjustment |
| PT test | 20 | 30 | Variance estimation stability |
| Conformal | 10 | 30-50 | Quantile estimation; 10 allows use, 50 for reliable inference |

---

## 6. Bagging Parameters [T2]

### 6.1 Block Bootstrap

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `block_size` | "auto" | n^(1/3) heuristic for dependent data (Politis & Romano 1994) |

### 6.2 Stationary Bootstrap

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `mean_block_size` | "auto" | Expected block length = n^(1/3) (Politis & Romano 1994) |

### 6.3 Feature Bagging

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `max_features` | 0.7 | Sample 70% of features per estimator |

---

## 7. Regime Classification [T2]

### 7.1 Volatility Regime

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `window` | 13 | ~13 weeks for quarterly volatility |
| `low_percentile` | 33.0 | Bottom third = LOW volatility |
| `high_percentile` | 67.0 | Top third = HIGH volatility |

**Classification**:
```
LOW:    rolling_vol <= percentile_33
MEDIUM: percentile_33 < rolling_vol < percentile_67
HIGH:   rolling_vol >= percentile_67
```

### 7.2 Direction Regime

Uses move threshold (Section 2):
```
UP:   value > threshold
DOWN: value < -threshold
FLAT: |value| <= threshold
```

---

## 8. Exit Codes (CLI) [T1]

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | EXIT_OK | All gates passed |
| 1 | EXIT_HALT | Critical failure - stop pipeline |
| 2 | EXIT_WARN | Warning - proceed with caution |
| 3 | EXIT_SKIP | Insufficient data - gate skipped |
| 4 | EXIT_ERROR | Unexpected error |

---

## Amendment History

| Date | Section | Change | Justification |
|------|---------|--------|---------------|
| 2025-12-23 | All | Initial specification | v1.0 preparation |
| 2025-12-23 | 1.2, 5.2, 6 | Sync spec to match code: n_shuffles=5, DM min=30, block_len=n^(1/3) | Codex audit resolution - spec drift |

---

## References

1. Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
2. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
3. Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
4. Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.
5. Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.
6. Romano, Y., Patterson, E. & Candès, E.J. (2019). Conformalized quantile regression. *NeurIPS*.
7. Gibbs, I. & Candès, E.J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
8. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
