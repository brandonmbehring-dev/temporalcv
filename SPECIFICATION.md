# SPECIFICATION.md - temporalcv

**Version**: 1.0.0-rc1 | **Last Updated**: 2025-01-05
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
| `required_gap` | horizon + extra_gap | Total separation must equal or exceed forecast horizon |

**Semantics** (v1.0.0+):
```
total_separation = horizon + extra_gap

Where:
- horizon: Minimum required separation for h-step forecasting
- extra_gap: Additional safety margin (default: 0)

Examples:
- horizon=5, extra_gap=0  → total_separation=5 (minimum safe)
- horizon=5, extra_gap=2  → total_separation=7 (with safety margin)
```

**Formula**:
```
HALT if: actual_gap < (horizon + extra_gap)
PASS if: actual_gap >= (horizon + extra_gap)

For all splits: train_idx[-1] + (horizon + extra_gap) < test_idx[0]
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

### 3.4 Self-Normalized Variance [T1]

Alternative to HAC that eliminates bandwidth selection and cannot produce negative variance.

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `variance_method` | "hac" | Backward compatible default |

**Formula** (partial sums approach):
```
d_demeaned = d_t - d_bar
S_k = sum_{t=1}^{k} d_demeaned_t    (partial sums, k = 1,...,n)
V_n^SN = (1/n²) * sum_{k=1}^{n} S_k²
```

**Critical Values** [T2] (non-standard limiting distribution):
| α | Two-sided | One-sided |
|---|-----------|-----------|
| 0.01 | 3.24 | 2.70 |
| 0.05 | 2.22 | 1.95 |
| 0.10 | 1.82 | 1.60 |

**Note**: Critical values are NOT from N(0,1). The limiting distribution is a functional of Brownian motion: W(1)² / ∫₀¹ W(r)² dr.

**Trade-offs vs HAC**:
- ✓ No bandwidth selection required
- ✓ Cannot produce negative variance
- ✓ Better size control in small samples
- ✗ Slightly lower power than well-tuned HAC

**When to Use**:
- Uncertain about bandwidth selection
- Small samples (n < 50)
- Long forecast horizons where HAC may be unreliable

**Reference**: Shao (2010). *Self-normalized CI construction*. JRSSB 72(3). Lobato (2001). JASA 96(453).

### 3.5 Pesaran-Timmermann Test [T1]

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

**When to Use 3-Class Mode**:
Use 3-class ONLY when ALL of these conditions are met:
1. Your model explicitly predicts 3 categories (UP, DOWN, FLAT) — not thresholded probabilities
2. You care about distinguishing FLAT from incorrect direction (not just UP vs DOWN)
3. You accept the approximate variance formula (no formal asymptotic theory)
4. Sample size is large (n ≥ 100) to mitigate approximation error

**Otherwise**: Use 2-class mode (threshold predictions at 0, classify as UP if > 0, DOWN if ≤ 0).

**Reference**: Pesaran & Timmermann (1992). *Simple nonparametric test*. JBES 10(4).

### 3.6 Giacomini-White Test [T1]

Tests *conditional* predictive ability: can past loss differentials predict which model will be better in the future?

| Parameter | Default | Reference |
|-----------|---------|-----------|
| `n_lags` | 1 | Number of lags in conditioning set (τ) |
| `min_n` | 50 | Minimum effective sample size (after lag adjustment) |
| `loss` | "squared" | Loss function for comparison |
| `alternative` | "two-sided" | Hypothesis direction |

**Algorithm** (Giacomini & White 2006, Theorem 1):
```
1. Compute loss differential: d_t = L(e1_t) - L(e2_t)
2. Construct instrument matrix: X = [1, d_{t-1}, ..., d_{t-τ}]
3. Demean loss differential: Z_t = d_t - d̄
4. Regress 1 on (Z × X) via OLS
5. Compute GW = T × R²
6. P-value from χ²(q) where q = 1 + τ
```

**Interpretation Table**:
| DM Result | GW Result | Interpretation |
|-----------|-----------|----------------|
| Not sig   | Not sig   | No difference in predictive ability |
| Sig       | Sig       | Model unconditionally and conditionally better |
| Sig       | Not sig   | Better on average, but not predictably |
| Not sig   | **Sig**   | **Equal average, but performance is predictable!** |

**Key Insight**: R² measures predictability of the loss differential. High R² means forecasters could improve by switching between models based on recent relative performance.

**n_lags Selection**:
- Default τ=1 is canonical (Giacomini & White 2006)
- Support τ ∈ {1,...,10} to avoid overfitting
- Guard: n_lags < n // 10 to prevent degrees-of-freedom exhaustion

**Reference**: Giacomini, R. & White, H. (2006). *Tests of Conditional Predictive Ability*. Econometrica 74(6), 1545-1578.

### 3.7 Multi-Horizon Comparison [T1]

Compare models across multiple forecast horizons to identify the "predictability horizon" — the forecast horizon beyond which a model's advantage disappears.

| Parameter | Default | Reference |
|-----------|---------|-----------|
| `horizons` | (1, 2, 3, 4) | Default horizon range |
| `alpha` | 0.05 | Significance level |
| `harvey_correction` | True | Small-sample adjustment |
| `variance_method` | "hac" | HAC or self-normalized |

**Functions**:
- `compare_horizons()`: Two-model comparison across horizons
- `compare_models_horizons()`: Multi-model comparison across horizons

**Degradation Patterns** (from `MultiHorizonResult.degradation_pattern`):
| Pattern | Definition |
|---------|------------|
| `consistent` | All horizons significant or all insignificant |
| `degrading` | P-values increase with horizon (advantage fades) |
| `none` | No significant horizons |
| `irregular` | Non-monotonic pattern (>1 violation) |

**Key Metrics**:
- `significant_horizons`: List of h where p < alpha
- `first_insignificant_horizon`: First h where significance is lost
- `best_horizon`: h with smallest p-value
- `best_model_by_horizon`: (multi-model) Best model at each horizon

**Reference**: Extends Diebold, F.X. & Mariano, R.S. (1995). *Comparing Predictive Accuracy*. JBES 13(3), 253-263.

---

### 3.8 Clark-West Test [T1]

Test for comparing **nested** forecasting models. The standard DM test is biased when comparing nested models because estimating extra parameters with true value zero adds noise that makes the unrestricted model appear worse.

| Parameter | Default | Reference |
|-----------|---------|-----------|
| `h` | 1 | Forecast horizon |
| `loss` | "squared" | "squared" or "absolute" |
| `alternative` | "two-sided" | "two-sided", "less", "greater" |
| `harvey_correction` | True | Harvey et al. (1997) |
| `variance_method` | "hac" | "hac" or "self_normalized" |
| Minimum n | 30 | Asymptotic normality requirement |

**The CW Adjustment**:
```
d*_t = d_t - (ŷ_restricted - ŷ_unrestricted)²
```

Where:
- `d_t = L(e_unrestricted) - L(e_restricted)` is the unadjusted loss differential
- `(ŷ_r - ŷ_u)²` removes the noise cost of estimating unnecessary parameters

**CWTestResult Properties**:
- `mean_loss_diff`: Unadjusted mean loss differential
- `mean_loss_diff_adjusted`: Adjusted mean (after CW correction)
- `adjustment_magnitude`: Mean of (ŷ_r - ŷ_u)² — the noise removed
- `adjustment_ratio`: Ratio of adjustment to unadjusted loss differential

**When to Use CW vs DM**:
| Situation | Test |
|-----------|------|
| Non-nested models (ARIMA vs Random Forest) | `dm_test()` |
| Nested models (AR(2) vs AR(1), Full vs Reduced) | `cw_test()` |

**Reference**: Clark, T.E. & West, K.D. (2007). *Approximately normal tests for equal predictive accuracy in nested models*. Journal of Econometrics 138(1), 291-311.

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
| `extra_gap` | 0 | Additional separation beyond horizon (total = horizon + extra_gap) |
| `test_size` | 1 | Single observation per fold |

**Gap Enforcement** (CRITICAL):
```
train_end = train_idx[-1]
test_start = test_idx[0]

REQUIRED: train_end + (horizon + extra_gap) < test_start
```

### 5.2 Minimum Observations

| Context | Enforced Min | Recommended Min | Justification |
|---------|--------------|-----------------|---------------|
| DM test | 30 | 50 | CLT requirement for asymptotic normality + Harvey adjustment |
| PT test | 20 | 30 | Variance estimation stability |
| Conformal | 10 | 30-50 | Quantile estimation; 10 allows use, 50 for reliable inference |

### 5.3 Nested Cross-Validation [T1]

Nested CV for unbiased hyperparameter selection per Bergmeir & Benítez (2012), Varma & Simon (2006).

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `n_outer_splits` | 3 | Outer folds for unbiased performance estimation |
| `n_inner_splits` | 5 | Inner folds for hyperparameter selection |
| `horizon` | 1 | Forecast horizon (minimum required separation) |
| `extra_gap` | 0 | Additional separation (total = horizon + extra_gap) |
| `window_type` | "expanding" | Training window type for both loops |
| `refit` | True | Refit best model on all data after CV |
| Min samples/inner fold | 30 | Bergmeir (2012) recommendation |

**Temporal Safety**:
```
Outer loop: Unbiased performance estimation
  └─ Inner loop: Hyperparameter selection on OUTER TRAINING DATA ONLY
      └─ Both loops: total_separation = horizon + extra_gap (WalkForwardCV in both)

Key invariant: Inner validation NEVER sees outer test data
```

**Best Params Selection**:
- Uses voting across outer folds
- Most frequently selected parameters win
- `params_stability` measures consistency (1.0 = all folds agree)

**When to Use**:
- Use nested CV when: Hyperparameters significantly affect predictions
- Skip nested CV when: Hyperparameters are fixed or have little effect

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

## 9. Block Bootstrap CI Parameters [T1]

Confidence intervals for gate metrics using Moving Block Bootstrap (MBB). Preserves temporal dependence while providing uncertainty quantification.

### 9.1 Default Parameters

| Parameter | Default | Justification |
|-----------|---------|---------------|
| `bootstrap_ci` | `False` | Off by default for backward compatibility |
| `n_bootstrap` | `100` | Sufficient for 95% CI estimation |
| `bootstrap_alpha` | `0.05` | Standard 95% confidence level |
| `bootstrap_block_length` | `"auto"` | Uses n^(1/3) rule (Politis & Romano 1994) |

### 9.2 Block Length Selection [T1]

**Formula** (asymptotically optimal for variance estimation):
```
block_length = max(1, floor(n^(1/3)))
```

**Examples**:
| n | Block Length |
|---|--------------|
| 30 | 3 |
| 100 | 4 |
| 500 | 7 |
| 1000 | 10 |

**When to override**:
- Multi-step forecasting: Use `block_length = max(horizon, n^(1/3))`
- Known autocorrelation lag: Match block length to decorrelation time
- Very short series (n < 30): Consider `block_length = 2`

### 9.3 Supported Gates

| Gate | CI Support | Rationale |
|------|------------|-----------|
| `gate_shuffled_target` | ✓ | Resamples (X, y) blocks, refits model |
| `gate_synthetic_ar1` | ✓ | Resamples synthetic series |
| `gate_suspicious_improvement` | ✗ | Takes pre-computed metrics only |
| `gate_temporal_boundary` | ✗ | Structural check, no metric |

### 9.4 Output Format

When `bootstrap_ci=True`, the `details` dict gains these fields:

```python
details = {
    # ... existing fields ...
    "ci_lower": float,              # Lower bound of CI
    "ci_upper": float,              # Upper bound of CI
    "ci_alpha": float,              # Significance level used
    "bootstrap_std": float,         # Bootstrap standard error
    "n_bootstrap": int,             # Number of bootstrap samples
    "bootstrap_block_length": int,  # Block length used
}
```

### 9.5 Mathematical Foundation [T1]

**Moving Block Bootstrap Algorithm** (Kunsch 1989):

1. Given series of length n, choose block length l = floor(n^(1/3))
2. Create overlapping blocks: B_i = (X_i, ..., X_{i+l-1}) for i = 1, ..., n-l+1
3. Sample k = ceil(n/l) blocks with replacement
4. Concatenate to form bootstrap sample of length ≈ n
5. Compute statistic on bootstrap sample
6. Repeat B times to get bootstrap distribution

**Percentile CI**:
```
ci_lower = percentile(bootstrap_metrics, alpha/2 * 100)
ci_upper = percentile(bootstrap_metrics, (1 - alpha/2) * 100)
```

**Reference**: Kunsch (1989), Politis & Romano (1994). See Section 10 References.

---

## Amendment History

| Date | Section | Change | Justification |
|------|---------|--------|---------------|
| 2025-12-23 | All | Initial specification | v1.0 preparation |
| 2025-12-23 | 1.2, 5.2, 6 | Sync spec to match code: n_shuffles=5, DM min=30, block_len=n^(1/3) | Codex audit resolution - spec drift |
| 2025-12-31 | 9 | Add Block Bootstrap CI Parameters section | ROADMAP v1.1.0 feature |

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
9. Kunsch, H.R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217-1241.
10. Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303-1313.
11. Lahiri, S.N. (2003). *Resampling Methods for Dependent Data*. Springer.
