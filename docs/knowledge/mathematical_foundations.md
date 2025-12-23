# Mathematical Foundations

This document provides the mathematical derivations underlying temporalcv's statistical tests and metrics. Each section is tagged with its knowledge tier.

---

## 1. Diebold-Mariano Test [T1]

**Purpose**: Compare predictive accuracy of two forecasting models.

**Reference**: Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

### 1.1 Loss Differential

Given two forecast error series e₁,ₜ and e₂,ₜ, define the loss differential:

```
d_t = L(e₁,ₜ) - L(e₂,ₜ)
```

Where L(·) is a loss function:
- **Squared loss**: L(e) = e²
- **Absolute loss**: L(e) = |e|

### 1.2 Null Hypothesis

```
H₀: E[d_t] = 0  (equal predictive accuracy)
H₁: E[d_t] ≠ 0  (different predictive accuracy)
```

### 1.3 Test Statistic

```
DM = d̄ / √(V̂(d̄))
```

Where:
- d̄ = (1/n) Σ d_t is the sample mean of loss differentials
- V̂(d̄) is the HAC variance estimator (see Section 2)

Under H₀, DM → N(0,1) asymptotically.

### 1.4 Harvey Adjustment [T1]

For small samples, Harvey et al. (1997) proposed:

```
DM_adj = DM × √((n + 1 - 2h + h(h-1)/n) / n)
```

This corrects for small-sample bias in the variance estimate.

**Reference**: Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.

---

## 2. HAC Variance Estimation [T1]

**Purpose**: Correct for serial correlation in forecast errors for h > 1.

**Reference**: Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.

### 2.1 Problem

For h-step forecasts, forecast errors follow an MA(h-1) process, inducing autocorrelation. Standard variance estimators are biased.

### 2.2 Bartlett Kernel

The Bartlett kernel weight for lag j:

```
w(j) = 1 - |j| / (bandwidth + 1)    if |j| ≤ bandwidth
w(j) = 0                             otherwise
```

### 2.3 HAC Variance Formula

```
V̂(d̄) = (1/n) × [γ₀ + 2 Σⱼ₌₁^bandwidth w(j) × γⱼ]
```

Where γⱼ is the sample autocovariance at lag j:

```
γⱼ = (1/n) Σₜ (d_t - d̄)(d_{t-j} - d̄)
```

### 2.4 Automatic Bandwidth Selection [T1]

Andrews (1991) rule:

```
bandwidth = floor(4 × (n/100)^(2/9))
```

For h-step forecasts, setting bandwidth = h - 1 is theoretically motivated (MA(h-1) structure).

**Reference**: Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.

---

## 3. Pesaran-Timmermann Test [T1]

**Purpose**: Test whether directional forecasts are better than random.

**Reference**: Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. *Journal of Business & Economic Statistics*, 10(4), 461-465.

### 3.1 Observed Accuracy

```
p̂ = (number of correct directions) / n
```

### 3.2 Expected Accuracy Under Independence

Under the null hypothesis that predictions are independent of actuals:

```
p* = p_y × p_x + (1 - p_y) × (1 - p_x)
```

Where:
- p_y = P(actual > 0) = fraction of positive actuals
- p_x = P(prediction > 0) = fraction of positive predictions

### 3.3 Variance Components

```
V(p̂) = p* × (1 - p*) / n

V(p*) = [(2p_y - 1)² × p_x(1-p_x) + (2p_x - 1)² × p_y(1-p_y)
         + 4 × p_y × p_x × (1-p_y) × (1-p_x) / n] / n
```

### 3.4 Test Statistic

```
PT = (p̂ - p*) / √(V(p̂) + V(p*))
```

Under H₀, PT → N(0,1) asymptotically. One-sided test: reject if PT > z_α.

### 3.5 Three-Class Extension [T3]

**Warning**: The 3-class mode (UP/DOWN/FLAT) is an ad-hoc extension not published in the academic literature.

For 3 classes with marginal probabilities p_y^k and p_x^k for k ∈ {UP, DOWN, FLAT}:

```
p* = Σₖ p_y^k × p_x^k
```

The variance formulas are approximations. Use 2-class mode for rigorous testing.

---

## 4. Conformal Prediction [T1]

**Purpose**: Distribution-free prediction intervals with coverage guarantee.

**Reference**: Romano, Y., Patterson, E. & Candès, E.J. (2019). Conformalized quantile regression. *NeurIPS*.

### 4.1 Finite-Sample Coverage Guarantee

For calibration set of size n and miscoverage rate α:

```
P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 - α
```

This holds for any distribution (no parametric assumptions needed).

### 4.2 Quantile Formula

The critical step uses the ceiling function:

```
q = ceil((n + 1) × (1 - α)) / n
```

**Why ceiling?**

The (n+1)(1-α) quantile of n nonconformity scores gives exact coverage. The ceiling ensures we round up, guaranteeing at least (1-α) coverage.

### 4.3 Nonconformity Score

For regression with residual-based scores:

```
s_i = |y_i - ŷ_i|
```

The prediction interval is:

```
Ĉ(x) = [ŷ(x) - q̂, ŷ(x) + q̂]
```

Where q̂ is the empirical quantile of calibration scores.

### 4.4 Adaptive Conformal Inference [T1]

For distribution shift, Gibbs & Candès (2021) proposed:

```
q_{t+1} = q_t - γα        if y_t ∈ Ĉ_t(x_t)  (covered)
q_{t+1} = q_t + γ(1-α)    if y_t ∉ Ĉ_t(x_t)  (not covered)
```

This adapts the quantile online to maintain target coverage.

**Reference**: Gibbs, I. & Candès, E.J. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.

---

## 5. Move-Conditional Skill Score [T2]

**Purpose**: Measure forecasting skill on significant moves, excluding flat periods.

### 5.1 Motivation

For high-persistence series (ACF(1) > 0.9), the persistence baseline (predict no change) achieves trivially low overall MAE because most periods are "flat."

Conditioning on moves isolates genuine forecasting skill.

### 5.2 Move Classification

Given threshold τ (typically 70th percentile of |actuals| from training):

```
UP:   actual > τ
DOWN: actual < -τ
FLAT: |actual| ≤ τ
```

### 5.3 MC-SS Formula

```
MC-SS = 1 - (model_MAE_moves / persistence_MAE_moves)
```

Where:
- model_MAE_moves = MAE of model predictions on UP and DOWN periods only
- persistence_MAE_moves = mean(|actual|) on moves (since persistence predicts 0)

### 5.4 Interpretation

| MC-SS | Meaning |
|-------|---------|
| > 0 | Model beats persistence on moves |
| = 0 | Model equals persistence on moves |
| < 0 | Model worse than persistence on moves |

### 5.5 Threshold Selection [T2]

The 70th percentile was chosen empirically:
- ~30% of periods are "moves" (UP or DOWN)
- ~70% are "flat"

This provides meaningful signal while maintaining sufficient sample size.

**Source**: myga-forecasting-v2 Phase 11 analysis.

---

## 6. AR(1) Theoretical Bounds [T1]

**Purpose**: Establish optimal forecast error for AR(1) process.

### 6.1 AR(1) Process

```
y_t = φ × y_{t-1} + σ × ε_t
```

Where:
- φ = persistence coefficient (typically 0.9 < φ < 1 for financial data)
- σ = innovation standard deviation
- ε_t ~ N(0, 1) i.i.d.

### 6.2 Optimal 1-Step Predictor

```
ŷ_t|t-1 = φ × y_{t-1}
```

The forecast error is:

```
e_t = y_t - ŷ_t|t-1 = σ × ε_t
```

### 6.3 Optimal MAE

Since ε_t ~ N(0, 1), the expected absolute error is:

```
E[|ε|] = √(2/π) ≈ 0.798
```

Therefore:

```
Optimal MAE = σ × √(2/π) ≈ 0.798 × σ
```

### 6.4 Validation Gate Application [T2]

If a model achieves MAE significantly below σ × √(2/π) on synthetic AR(1) data, it indicates lookahead bias (the model is "seeing" future ε values).

The tolerance factor of 1.5 allows for finite-sample variation:

```
HALT if: model_MAE < (1/1.5) × theoretical_MAE
```

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| h | Forecast horizon (steps ahead) |
| n | Sample size |
| α | Significance level or miscoverage rate |
| φ | AR(1) persistence coefficient |
| σ | Innovation standard deviation |
| d_t | Loss differential at time t |
| L(·) | Loss function |
| γⱼ | Autocovariance at lag j |
| p̂ | Observed accuracy |
| p* | Expected accuracy under null |

---

## Knowledge Tier Summary

| Section | Tier | Confidence |
|---------|------|------------|
| DM Test | T1 | Academically validated |
| HAC Variance | T1 | Academically validated |
| PT Test (2-class) | T1 | Academically validated |
| PT Test (3-class) | T3 | Ad-hoc extension |
| Conformal | T1 | Academically validated |
| MC-SS | T2 | Empirical (v2) |
| AR(1) Bounds | T1 | Standard statistics |
