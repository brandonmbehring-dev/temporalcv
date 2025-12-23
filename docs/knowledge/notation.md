# Notation Guide

Standard notation used throughout temporalcv documentation and code.

---

## Time Series Variables

| Symbol | Python Variable | Meaning |
|--------|-----------------|---------|
| y_t | `y[t]` | Target value at time t |
| ŷ_t | `predictions[t]` | Predicted value at time t |
| e_t | `errors[t]` | Forecast error: y_t - ŷ_t |
| Δy_t | `changes[t]` | First difference: y_t - y_{t-1} |

---

## Forecast Parameters

| Symbol | Python Parameter | Meaning | Typical Values |
|--------|------------------|---------|----------------|
| h | `horizon` | Forecast horizon (steps ahead) | 1, 2, 4, 8 |
| n | `n` or `len(y)` | Sample size | 50-500 |
| k | `n_lags` | Number of lagged features | 3-10 |
| w | `window_size` | Training window size | 52-104 (weeks) |

---

## Statistical Parameters

| Symbol | Python Parameter | Meaning | Typical Values |
|--------|------------------|---------|----------------|
| α | `alpha` | Significance level (Type I error) | 0.05, 0.10 |
| 1-α | `coverage` | Target coverage probability | 0.90, 0.95 |
| γ | `gamma` | Learning rate (adaptive conformal) | 0.01-0.1 |

---

## AR(1) Process Parameters

| Symbol | Python Variable | Meaning | Typical Values |
|--------|-----------------|---------|----------------|
| φ | `phi` | Persistence coefficient | 0.9-0.99 |
| σ | `sigma` | Innovation standard deviation | Data-dependent |
| ε_t | `epsilon[t]` | Innovation (white noise) | N(0, 1) |

**AR(1) Process**: y_t = φ × y_{t-1} + σ × ε_t

---

## Test Statistics

| Symbol | Python Variable | Meaning |
|--------|-----------------|---------|
| d_t | `d[t]` | Loss differential: L(e₁,ₜ) - L(e₂,ₜ) |
| d̄ | `d_bar` | Mean loss differential |
| DM | `dm_stat` | Diebold-Mariano test statistic |
| PT | `pt_stat` | Pesaran-Timmermann test statistic |
| p̂ | `p_hat` | Observed directional accuracy |
| p* | `p_star` | Expected accuracy under null |

---

## Variance Estimators

| Symbol | Python Variable | Meaning |
|--------|-----------------|---------|
| γ_j | `gamma[j]` | Autocovariance at lag j |
| V̂(·) | `var_hat` | Estimated variance |
| w(j) | `weight` | Bartlett kernel weight |

**HAC Variance**: V̂ = (1/n) × [γ₀ + 2 Σⱼ w(j) × γⱼ]

---

## Regime Classification

| Symbol | Python Value | Meaning |
|--------|--------------|---------|
| τ | `threshold` | Move threshold |
| UP | `'UP'` or `MoveDirection.UP` | Upward move: value > τ |
| DOWN | `'DOWN'` or `MoveDirection.DOWN` | Downward move: value < -τ |
| FLAT | `'FLAT'` or `MoveDirection.FLAT` | No significant move: \|value\| ≤ τ |
| LOW | `'LOW'` | Low volatility regime |
| MED | `'MED'` | Medium volatility regime |
| HIGH | `'HIGH'` | High volatility regime |

---

## Skill Scores

| Symbol | Python Variable | Formula | Range |
|--------|-----------------|---------|-------|
| SS | `skill_score` | 1 - (model_error / baseline_error) | (-∞, 1] |
| MC-SS | `skill_score` | 1 - (model_MAE_moves / persistence_MAE_moves) | (-∞, 1] |

**Interpretation**:
- SS > 0: Model beats baseline
- SS = 0: Model equals baseline
- SS < 0: Model worse than baseline
- SS = 1: Perfect forecast

---

## Conformal Prediction

| Symbol | Python Variable | Meaning |
|--------|-----------------|---------|
| s_i | `scores[i]` | Nonconformity score |
| q̂ | `quantile` | Empirical quantile of scores |
| Ĉ(x) | `prediction_interval` | Conformal prediction set |

**Prediction Interval**: Ĉ(x) = [ŷ(x) - q̂, ŷ(x) + q̂]

---

## Cross-Validation

| Symbol | Python Variable | Meaning |
|--------|-----------------|---------|
| K | `n_splits` | Number of CV splits |
| g | `gap` | Gap between train and test |

**Gap Requirement**: train_end + gap < test_start

---

## Gate Results

| Symbol | Python Value | Meaning | Action |
|--------|--------------|---------|--------|
| HALT | `GateStatus.HALT` | Critical failure | Stop and investigate |
| WARN | `GateStatus.WARN` | Caution | Continue with verification |
| PASS | `GateStatus.PASS` | Validation passed | Proceed |
| SKIP | `GateStatus.SKIP` | Insufficient data | Cannot validate |

---

## Common Constants

| Value | Meaning | Source |
|-------|---------|--------|
| √(2/π) ≈ 0.798 | E[\|Z\|] for Z ~ N(0,1) | Standard result |
| 0.70 | Default move threshold percentile | v2 empirical |
| 0.20 | Suspicious improvement threshold | v2 empirical |
| 13 | Default volatility window (weeks) | Quarterly assumption |
| 30 | Minimum n for DM test | Convention |
| 20 | Minimum n for PT test | Convention |

---

## Variable Naming Conventions

| Pattern | Meaning | Example |
|---------|---------|---------|
| `*_train` | Training data only | `X_train`, `threshold_train` |
| `*_test` | Test data only | `y_test` |
| `*_moves` | Computed on moves only | `mae_moves` |
| `*_hat` | Estimated quantity | `p_hat`, `var_hat` |
| `*_star` | Expected under null | `p_star` |
| `n_*` | Count of | `n_up`, `n_splits` |

---

## Code-to-Math Mapping

```python
# DM Test
d_bar = np.mean(d)           # d̄
var_d = compute_hac_variance(d)  # V̂(d̄)
dm_stat = d_bar / np.sqrt(var_d)  # DM

# PT Test
p_hat = np.mean(correct)     # p̂
p_star = p_y * p_x + (1-p_y) * (1-p_x)  # p*

# MC-SS
mc_ss = 1 - (model_mae_moves / persistence_mae_moves)  # MC-SS

# AR(1) Optimal MAE
optimal_mae = sigma * np.sqrt(2 / np.pi)  # σ × √(2/π)
```
