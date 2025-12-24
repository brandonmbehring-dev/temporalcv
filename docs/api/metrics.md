# API Reference: Metrics

Comprehensive metrics for time series forecast evaluation.

---

## Core Metrics

Foundational point forecast and scale-invariant metrics.

### `compute_mae()`

```python
def compute_mae(predictions: ArrayLike, actuals: ArrayLike) -> float
```

Compute Mean Absolute Error.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Predicted values |
| `actuals` | `ArrayLike` | Actual observed values |

**Returns**: `float` - Mean absolute error

**Notes**: MAE = mean(|y_hat - y|). Scale-dependent; compare only within same series.

**Example**:

```python
from temporalcv.metrics import compute_mae

mae = compute_mae(predictions, actuals)
print(f"MAE: {mae:.4f}")
```

---

### `compute_mse()`

```python
def compute_mse(predictions: ArrayLike, actuals: ArrayLike) -> float
```

Compute Mean Squared Error.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Predicted values |
| `actuals` | `ArrayLike` | Actual observed values |

**Returns**: `float` - Mean squared error

**Notes**: MSE = mean((y_hat - y)²). Penalizes large errors more than MAE.

---

### `compute_rmse()`

```python
def compute_rmse(predictions: ArrayLike, actuals: ArrayLike) -> float
```

Compute Root Mean Squared Error.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Predicted values |
| `actuals` | `ArrayLike` | Actual observed values |

**Returns**: `float` - Root mean squared error

**Notes**: RMSE = sqrt(MSE). Same units as the target variable.

**Example**:

```python
from temporalcv.metrics import compute_rmse

rmse = compute_rmse(predictions, actuals)
print(f"RMSE: {rmse:.4f}")
```

---

### `compute_mape()`

```python
def compute_mape(
    predictions: ArrayLike,
    actuals: ArrayLike,
    epsilon: float = 1e-8,
) -> float
```

Compute Mean Absolute Percentage Error.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual observed values |
| `epsilon` | `float` | `1e-8` | Prevents division by zero |

**Returns**: `float` - MAPE as percentage (0-100+)

**Notes**: MAPE = 100 * mean(|y_hat - y| / |y|). Asymmetric and undefined when actuals = 0. Consider SMAPE for bounded alternative.

**Example**:

```python
from temporalcv.metrics import compute_mape

mape = compute_mape(predictions, actuals)
print(f"MAPE: {mape:.1f}%")
```

---

### `compute_smape()`

```python
def compute_smape(predictions: ArrayLike, actuals: ArrayLike) -> float
```

Compute Symmetric Mean Absolute Percentage Error.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Predicted values |
| `actuals` | `ArrayLike` | Actual observed values |

**Returns**: `float` - SMAPE as percentage (bounded 0-200%)

**Notes**: SMAPE = 100 * mean(2|y_hat - y| / (|y_hat| + |y|)). Symmetric and bounded, addressing MAPE limitations. Reference: Armstrong (1985).

---

### `compute_bias()`

```python
def compute_bias(predictions: ArrayLike, actuals: ArrayLike) -> float
```

Compute mean signed error (bias).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Predicted values |
| `actuals` | `ArrayLike` | Actual observed values |

**Returns**: `float` - Mean error (positive = over-prediction)

**Notes**: Bias = mean(y_hat - y). Positive indicates systematic over-prediction; negative indicates under-prediction.

---

### `compute_naive_error()`

```python
def compute_naive_error(
    values: ArrayLike,
    method: Literal["persistence", "mean"] = "persistence",
) -> float
```

Compute naive forecast MAE for scale normalization.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `ArrayLike` | - | Training series values |
| `method` | `str` | `"persistence"` | `"persistence"` (y[t-1]) or `"mean"` |

**Returns**: `float` - MAE of naive forecast on training data

**Notes**: Used as denominator for MASE. For persistence: MAE = mean(|y[t] - y[t-1]|). Reference: Hyndman & Koehler (2006).

**Example**:

```python
from temporalcv.metrics import compute_naive_error, compute_mase

naive_mae = compute_naive_error(train_values)
mase = compute_mase(predictions, actuals, naive_mae)
```

---

### `compute_mase()`

```python
def compute_mase(
    predictions: ArrayLike,
    actuals: ArrayLike,
    naive_mae: float,
) -> float
```

Compute Mean Absolute Scaled Error.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Model predictions |
| `actuals` | `ArrayLike` | Actual values |
| `naive_mae` | `float` | MAE of naive forecast (from `compute_naive_error`) |

**Returns**: `float` - MASE value

**Interpretation**:

| MASE | Meaning |
|------|---------|
| < 1 | Better than naive forecast |
| = 1 | Equal to naive forecast |
| > 1 | Worse than naive forecast |

**Notes**: Scale-free metric for comparing across different time series. Reference: Hyndman & Koehler (2006).

---

### `compute_mrae()`

```python
def compute_mrae(
    predictions: ArrayLike,
    actuals: ArrayLike,
    naive_predictions: ArrayLike,
) -> float
```

Compute Mean Relative Absolute Error.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Model predictions |
| `actuals` | `ArrayLike` | Actual values |
| `naive_predictions` | `ArrayLike` | Naive baseline predictions |

**Returns**: `float` - MRAE value (< 1 = better than naive)

**Notes**: MRAE = mean(|y_hat - y| / |y_naive - y|). Compares each error to naive error at that point.

---

### `compute_theils_u()`

```python
def compute_theils_u(
    predictions: ArrayLike,
    actuals: ArrayLike,
    naive_predictions: ArrayLike | None = None,
) -> float
```

Compute Theil's U statistic.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Model predictions |
| `actuals` | `ArrayLike` | - | Actual values |
| `naive_predictions` | `ArrayLike` | `None` | Naive predictions (uses persistence if None) |

**Returns**: `float` - Theil's U (< 1 = better than naive)

**Notes**: U = RMSE(model) / RMSE(naive). Reference: Theil (1966).

---

### `compute_forecast_correlation()`

```python
def compute_forecast_correlation(
    predictions: ArrayLike,
    actuals: ArrayLike,
    method: Literal["pearson", "spearman"] = "pearson",
) -> float
```

Compute correlation between predictions and actuals.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `method` | `str` | `"pearson"` | Correlation method |

**Returns**: `float` - Correlation coefficient [-1, 1]

**Notes**: Correlation measures association, not accuracy. A model can have high correlation but large errors (wrong scale/offset).

---

### `compute_r_squared()`

```python
def compute_r_squared(predictions: ArrayLike, actuals: ArrayLike) -> float
```

Compute R² (coefficient of determination).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predictions` | `ArrayLike` | Predicted values |
| `actuals` | `ArrayLike` | Actual values |

**Returns**: `float` - R² value (can be negative)

**Interpretation**:

| R² | Meaning |
|----|---------|
| 1 | Perfect predictions |
| 0 | Equal to mean forecast |
| < 0 | Worse than mean forecast |

---

## Event & Direction Metrics

Novel metrics for direction prediction with proper calibration.

### Data Classes

#### `BrierScoreResult`

Result from Brier score computation.

```python
@dataclass
class BrierScoreResult:
    brier_score: float    # Mean squared error (0 = perfect, 1 = worst)
    reliability: float    # Calibration component (lower = better)
    resolution: float     # Refinement component (higher = better)
    uncertainty: float    # Base rate uncertainty
    n_samples: int        # Number of samples
    n_classes: int        # Number of classes (2 or 3)
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `skill_score` | `float` | BSS = 1 - (BS / uncertainty) |

**Decomposition** (Murphy 1973):
```
BS = Reliability - Resolution + Uncertainty
```

---

#### `PRAUCResult`

Result from PR-AUC computation.

```python
@dataclass
class PRAUCResult:
    pr_auc: float                  # Area under PR curve
    baseline: float                # Random classifier PR-AUC
    precision_at_50_recall: float  # Precision at 50% recall
    n_positive: int                # Positive samples
    n_negative: int                # Negative samples
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `lift_over_baseline` | `float` | PR-AUC / baseline |
| `n_total` | `int` | Total samples |
| `imbalance_ratio` | `float` | Majority / minority class ratio |

---

### `compute_direction_brier()`

Compute Brier score for direction prediction.

```python
def compute_direction_brier(
    pred_probs: np.ndarray,
    actual_directions: np.ndarray,
    n_classes: Literal[2, 3] = 2,
) -> BrierScoreResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_probs` | `np.ndarray` | required | Predicted probabilities |
| `actual_directions` | `np.ndarray` | required | Actual directions as integers |
| `n_classes` | `int` | `2` | Number of classes (2 or 3) |

**For 2-class**:
- `pred_probs`: 1D array, P(positive)
- `actual_directions`: 0 = negative, 1 = positive

**For 3-class**:
- `pred_probs`: (n_samples, 3), probabilities for [DOWN, FLAT, UP]
- `actual_directions`: 0 = DOWN, 1 = FLAT, 2 = UP

**Example**:

```python
from temporalcv.metrics.event import compute_direction_brier

# 2-class
probs = np.array([0.7, 0.3, 0.8, 0.2])
actuals = np.array([1, 0, 1, 0])
result = compute_direction_brier(probs, actuals, n_classes=2)
print(f"Brier: {result.brier_score:.4f}")
print(f"Skill: {result.skill_score:.3f}")
```

---

### `compute_pr_auc()`

Compute Area Under Precision-Recall Curve.

```python
def compute_pr_auc(
    pred_probs: np.ndarray,
    actual_binary: np.ndarray,
) -> PRAUCResult
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pred_probs` | `np.ndarray` | Predicted probabilities of positive class |
| `actual_binary` | `np.ndarray` | Binary labels (0 or 1) |

**Returns**: `PRAUCResult`

**Notes**:
- Preferred over ROC-AUC for imbalanced classification
- Baseline equals positive class rate (random classifier)
- Uses **trapezoidal integration**, which differs from sklearn's `average_precision_score` (step integration). Differences can be a few percentage points for jagged curves.

> **sklearn compatibility**: For sklearn-equivalent results, use:
> ```python
> from sklearn.metrics import average_precision_score
> ap = average_precision_score(actual_binary, pred_probs)
> ```

**Example**:

```python
from temporalcv.metrics.event import compute_pr_auc

probs = np.array([0.9, 0.8, 0.3, 0.1, 0.7])
actuals = np.array([1, 1, 0, 0, 1])

result = compute_pr_auc(probs, actuals)
print(f"PR-AUC: {result.pr_auc:.3f}")
print(f"Baseline: {result.baseline:.3f}")
print(f"Lift: {result.lift_over_baseline:.2f}x")
```

---

### `compute_calibrated_direction_brier()`

Compute Brier score with reliability diagram data.

```python
def compute_calibrated_direction_brier(
    pred_probs: np.ndarray,
    actual_directions: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_probs` | `np.ndarray` | required | Predicted probabilities (1D) |
| `actual_directions` | `np.ndarray` | required | Binary outcomes |
| `n_bins` | `int` | `10` | Number of calibration bins |

**Returns**: `(brier_score, bin_means, bin_true_fractions)`

**Example** (plotting reliability diagram):

```python
brier, bin_means, bin_fracs = compute_calibrated_direction_brier(probs, actuals)

import matplotlib.pyplot as plt
plt.plot(bin_means, bin_fracs, 'o-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.legend()
```

---

### `convert_predictions_to_direction_probs()`

Convert point predictions with uncertainty to direction probabilities.

```python
def convert_predictions_to_direction_probs(
    point_predictions: np.ndarray,
    prediction_std: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `point_predictions` | `np.ndarray` | required | Point predictions |
| `prediction_std` | `np.ndarray` | required | Prediction standard deviation |
| `threshold` | `float` | `0.0` | UP/DOWN threshold |

**Returns**: P(UP) = P(X > threshold)

**Assumes**: Gaussian prediction distribution

**Example**:

```python
from temporalcv.bagging import create_block_bagger
from temporalcv.metrics.event import (
    convert_predictions_to_direction_probs,
    compute_direction_brier,
)

# Get predictions with uncertainty
mean, std = bagger.predict_with_uncertainty(X_test)

# Convert to direction probabilities
p_up = convert_predictions_to_direction_probs(mean, std, threshold=0.01)

# Compute Brier score
actual_up = (actuals > 0.01).astype(int)
result = compute_direction_brier(p_up, actual_up, n_classes=2)
```

---

### Metric Interpretation

#### Brier Score

| Score | Interpretation |
|-------|----------------|
| 0.0 | Perfect |
| 0.25 | Random guessing (50% base rate) |
| 1.0 | Worst possible |

#### Brier Skill Score (BSS)

| BSS | Interpretation |
|-----|----------------|
| < 0 | Worse than climatology |
| 0 | Same as climatology |
| > 0 | Skill over climatology |
| 1.0 | Perfect |

#### PR-AUC

| Context | Interpretation |
|---------|----------------|
| = baseline | Random classifier |
| > baseline | Some skill |
| = 1.0 | Perfect separation |

---

## Quantile & Interval Metrics

Proper scoring rules for probabilistic forecasts. Reference: Gneiting & Raftery (2007).

### `compute_pinball_loss()`

```python
def compute_pinball_loss(
    actuals: ArrayLike,
    quantile_preds: ArrayLike,
    tau: float,
) -> float
```

Compute pinball loss (quantile loss) for quantile regression.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `actuals` | `ArrayLike` | Actual observed values |
| `quantile_preds` | `ArrayLike` | Predicted values at quantile tau |
| `tau` | `float` | Quantile level in (0, 1), e.g., 0.9 for 90th percentile |

**Returns**: `float` - Mean pinball loss (lower is better)

**Notes**: The pinball loss is asymmetric around the quantile:
- L(y, q; τ) = τ * max(y - q, 0) + (1 - τ) * max(q - y, 0)
- Penalizes under-predictions more for high quantiles
- Reference: Koenker & Bassett (1978)

**Example**:

```python
from temporalcv.metrics import compute_pinball_loss

# Evaluate 90th percentile predictions
loss = compute_pinball_loss(actuals, preds_90, tau=0.9)
```

---

### `compute_crps()`

```python
def compute_crps(
    actuals: ArrayLike,
    forecast_samples: ArrayLike,
) -> float
```

Compute Continuous Ranked Probability Score.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `actuals` | `ArrayLike` | Actual values, shape (n,) |
| `forecast_samples` | `ArrayLike` | Samples from forecast distribution, shape (n, n_samples) |

**Returns**: `float` - Mean CRPS (same units as observations, lower is better)

**Notes**:
- CRPS = E|X - y| - 0.5 * E|X - X'| where X, X' are forecast samples
- Uses `scipy.stats.energy_distance` if available, otherwise sample approximation
- Proper scoring rule for probabilistic forecasts
- Reference: Gneiting & Raftery (2007)

**Example**:

```python
from temporalcv.metrics import compute_crps

# Each row: samples for one observation
forecast_samples = ensemble_predictions  # shape (100, 50)
actuals = y_test  # shape (100,)
crps = compute_crps(actuals, forecast_samples)
```

---

### `compute_interval_score()`

```python
def compute_interval_score(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
) -> float
```

Compute interval score for prediction intervals.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `actuals` | `ArrayLike` | Actual observed values |
| `lower` | `ArrayLike` | Lower bounds of prediction intervals |
| `upper` | `ArrayLike` | Upper bounds of prediction intervals |
| `alpha` | `float` | Nominal non-coverage rate, e.g., 0.05 for 95% intervals |

**Returns**: `float` - Mean interval score (lower is better)

**Notes**: The interval score penalizes both width and coverage failures:
- IS = (u - l) + (2/α)(l - y)I(y < l) + (2/α)(y - u)I(y > u)
- A well-calibrated narrow interval scores better than a wide one
- Reference: Gneiting & Raftery (2007, equation 43)

**Example**:

```python
from temporalcv.metrics import compute_interval_score

# 95% prediction intervals
score = compute_interval_score(actuals, lower, upper, alpha=0.05)
```

---

### `compute_quantile_coverage()`

```python
def compute_quantile_coverage(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> float
```

Compute empirical coverage of prediction intervals.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `actuals` | `ArrayLike` | Actual observed values |
| `lower` | `ArrayLike` | Lower bounds of prediction intervals |
| `upper` | `ArrayLike` | Upper bounds of prediction intervals |

**Returns**: `float` - Empirical coverage rate in [0, 1]

**Notes**: For well-calibrated (1-α) intervals, coverage should be approximately (1-α).

**Example**:

```python
from temporalcv.metrics import compute_quantile_coverage

coverage = compute_quantile_coverage(actuals, lower, upper)
print(f"Coverage: {coverage:.1%}")  # Should be ~95% for 95% intervals
```

---

### `compute_winkler_score()`

```python
def compute_winkler_score(
    actuals: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    alpha: float,
) -> float
```

Compute Winkler score for prediction intervals.

**Notes**: Alias for `compute_interval_score()`. Winkler (1972) is the original formulation; interval score is the term used by Gneiting & Raftery (2007).

---

## Financial & Trading Metrics

Risk-adjusted and trading performance metrics. Reference: Sharpe (1994), Goodwin (1998).

### `compute_sharpe_ratio()`

```python
def compute_sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> float
```

Compute annualized Sharpe ratio.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns` | `ArrayLike` | - | Period returns |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate per period |
| `annualization` | `float` | `252.0` | Periods per year (252=daily, 52=weekly, 12=monthly) |

**Returns**: `float` - Annualized Sharpe ratio

**Interpretation**:

| Sharpe | Interpretation |
|--------|----------------|
| < 0 | Negative risk-adjusted return |
| 0-1 | Acceptable |
| 1-2 | Good |
| > 2 | Excellent |

**Example**:

```python
from temporalcv.metrics import compute_sharpe_ratio

# Daily returns with 2% annual risk-free rate
sharpe = compute_sharpe_ratio(daily_returns, risk_free_rate=0.02/252)
```

---

### `compute_max_drawdown()`

```python
def compute_max_drawdown(
    cumulative_returns: Optional[ArrayLike] = None,
    returns: Optional[ArrayLike] = None,
) -> float
```

Compute maximum drawdown from peak to trough.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `cumulative_returns` | `ArrayLike` | Cumulative returns (or equity curve) |
| `returns` | `ArrayLike` | Period returns (if cumulative not provided) |

**Returns**: `float` - Maximum drawdown as positive fraction (0.20 = 20%)

**Example**:

```python
from temporalcv.metrics import compute_max_drawdown

mdd = compute_max_drawdown(returns=daily_returns)
print(f"Max drawdown: {mdd:.1%}")
```

---

### `compute_cumulative_return()`

```python
def compute_cumulative_return(
    returns: ArrayLike,
    method: str = "geometric",
) -> float
```

Compute cumulative return over the period.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns` | `ArrayLike` | - | Period returns |
| `method` | `str` | `"geometric"` | `"geometric"` (compounding) or `"arithmetic"` |

**Returns**: `float` - Cumulative return as fraction (0.25 = 25%)

---

### `compute_information_ratio()`

```python
def compute_information_ratio(
    portfolio_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    annualization: float = 252.0,
) -> float
```

Compute information ratio (active return per unit tracking error).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `portfolio_returns` | `ArrayLike` | - | Portfolio period returns |
| `benchmark_returns` | `ArrayLike` | - | Benchmark period returns |
| `annualization` | `float` | `252.0` | Periods per year |

**Returns**: `float` - Annualized information ratio

**Interpretation**:

| IR | Interpretation |
|----|----------------|
| < 0.5 | Low skill |
| 0.5-1.0 | Good |
| > 1.0 | Excellent |

Reference: Goodwin (1998).

---

### `compute_hit_rate()`

```python
def compute_hit_rate(
    predicted_changes: ArrayLike,
    actual_changes: ArrayLike,
) -> float
```

Compute directional hit rate.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predicted_changes` | `ArrayLike` | Predicted changes (sign = direction) |
| `actual_changes` | `ArrayLike` | Actual changes |

**Returns**: `float` - Hit rate in [0, 1]

**Notes**: Hit rate = fraction where sign(pred) == sign(actual). Above 0.5 indicates directional skill.

**Example**:

```python
from temporalcv.metrics import compute_hit_rate

hr = compute_hit_rate(predicted_changes, actual_changes)
print(f"Hit rate: {hr:.1%}")
```

---

### `compute_profit_factor()`

```python
def compute_profit_factor(
    predicted_changes: ArrayLike,
    actual_changes: ArrayLike,
    returns: Optional[ArrayLike] = None,
) -> float
```

Compute profit factor (gross profit / gross loss).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `predicted_changes` | `ArrayLike` | Predicted changes (sign = trade direction) |
| `actual_changes` | `ArrayLike` | Actual changes |
| `returns` | `ArrayLike` | Actual returns (uses `actual_changes` if not provided) |

**Returns**: `float` - Profit factor (> 1.0 = profitable)

**Interpretation**:

| PF | Interpretation |
|----|----------------|
| < 1 | Losing strategy |
| 1-1.5 | Marginal |
| 1.5-2 | Good |
| > 2 | Excellent |

---

### `compute_calmar_ratio()`

```python
def compute_calmar_ratio(
    returns: ArrayLike,
    annualization: float = 252.0,
) -> float
```

Compute Calmar ratio (annualized return / max drawdown).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `returns` | `ArrayLike` | - | Period returns |
| `annualization` | `float` | `252.0` | Periods per year |

**Returns**: `float` - Calmar ratio (higher is better)

**Notes**: Measures return relative to worst-case decline. Useful when drawdowns are a key concern.

---

## Asymmetric Loss Functions

Loss functions that penalize over- and under-predictions differently.

### `compute_linex_loss()`

```python
def compute_linex_loss(
    predictions: ArrayLike,
    actuals: ArrayLike,
    a: float = 1.0,
    b: float = 1.0,
) -> float
```

Compute LinEx (linear-exponential) asymmetric loss.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `a` | `float` | `1.0` | Asymmetry: a > 0 penalizes under-prediction exponentially |
| `b` | `float` | `1.0` | Scaling parameter (> 0) |

**Returns**: `float` - Mean LinEx loss

**Notes**:
- L(e) = b * (exp(a * e) - a * e - 1) where e = actual - prediction
- a > 0: under-predictions penalized exponentially (e.g., inventory)
- a < 0: over-predictions penalized exponentially (e.g., overestimating sales)
- Reference: Varian (1975), Zellner (1986)

**Example**:

```python
from temporalcv.metrics import compute_linex_loss

# Under-predictions are costly
loss = compute_linex_loss(predictions, actuals, a=0.5)
```

---

### `compute_asymmetric_mape()`

```python
def compute_asymmetric_mape(
    predictions: ArrayLike,
    actuals: ArrayLike,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
) -> float
```

Compute asymmetric MAPE with different over/under penalties.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `alpha` | `float` | `0.5` | Weight for under-predictions (0.5 = symmetric) |
| `epsilon` | `float` | `1e-8` | Prevents division by zero |

**Returns**: `float` - Asymmetric MAPE as fraction

**Notes**: alpha > 0.5 penalizes under-predictions more; alpha < 0.5 penalizes over-predictions more.

---

### `compute_directional_loss()`

```python
def compute_directional_loss(
    predictions: ArrayLike,
    actuals: ArrayLike,
    up_miss_weight: float = 1.0,
    down_miss_weight: float = 1.0,
    previous_actuals: ArrayLike | None = None,
) -> float
```

Compute directional loss with custom weights for missing UP vs DOWN moves.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values or changes |
| `actuals` | `ArrayLike` | - | Actual values or changes |
| `up_miss_weight` | `float` | `1.0` | Weight for missing UP moves |
| `down_miss_weight` | `float` | `1.0` | Weight for missing DOWN moves |
| `previous_actuals` | `ArrayLike` | `None` | If provided, computes changes internally |

**Returns**: `float` - Mean directional loss

**Example**:

```python
from temporalcv.metrics import compute_directional_loss

# Missing UP costs 2x more than missing DOWN
loss = compute_directional_loss(
    predicted_changes, actual_changes,
    up_miss_weight=2.0, down_miss_weight=1.0
)
```

---

### `compute_squared_log_error()`

```python
def compute_squared_log_error(
    predictions: ArrayLike,
    actuals: ArrayLike,
    epsilon: float = 1e-8,
) -> float
```

Compute mean squared logarithmic error (MSLE).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values (non-negative) |
| `actuals` | `ArrayLike` | - | Actual values (non-negative) |
| `epsilon` | `float` | `1e-8` | Added before log for zeros |

**Returns**: `float` - Mean squared log error

**Notes**: MSLE = mean((log(1 + actual) - log(1 + pred))²). Scale-invariant and naturally penalizes under-predictions more.

---

### `compute_huber_loss()`

```python
def compute_huber_loss(
    predictions: ArrayLike,
    actuals: ArrayLike,
    delta: float = 1.0,
) -> float
```

Compute Huber loss (smooth approximation to MAE).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `delta` | `float` | `1.0` | Transition threshold |

**Returns**: `float` - Mean Huber loss

**Notes**: Quadratic for |error| ≤ delta, linear for |error| > delta. Robust to outliers while remaining differentiable.

---

## Volatility-Weighted Metrics

Metrics that account for local volatility for scale-invariant evaluation.

### Classes

#### `VolatilityEstimator` (Protocol)

Protocol for custom volatility estimators.

```python
class VolatilityEstimator(Protocol):
    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

---

#### `RollingVolatility`

Rolling window standard deviation estimator.

```python
class RollingVolatility:
    def __init__(self, window: int = 13, min_periods: int | None = None): ...
    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

---

#### `EWMAVolatility`

Exponentially Weighted Moving Average volatility estimator.

```python
class EWMAVolatility:
    def __init__(self, span: int = 13, adjust: bool = True): ...
    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

Reference: J.P. Morgan RiskMetrics (1996).

---

#### `GARCHVolatility`

GARCH(1,1) volatility estimator. Requires optional `arch` package.

```python
class GARCHVolatility:
    def __init__(self, p: int = 1, q: int = 1): ...
    def estimate(self, values: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

Reference: Bollerslev (1986).

---

### `compute_local_volatility()`

```python
def compute_local_volatility(
    values: ArrayLike,
    window: int = 13,
    method: Literal["rolling_std", "ewm", "garch"] = "rolling_std",
) -> NDArray[np.float64]
```

Compute local volatility estimates.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | `ArrayLike` | - | Input values (returns or changes) |
| `window` | `int` | `13` | Window size or EWMA span |
| `method` | `str` | `"rolling_std"` | Estimation method |

**Returns**: `ndarray` - Volatility estimates (same length as input)

**Example**:

```python
from temporalcv.metrics import compute_local_volatility

vol = compute_local_volatility(returns, window=13, method="ewm")
```

---

### `compute_volatility_normalized_mae()`

```python
def compute_volatility_normalized_mae(
    predictions: ArrayLike,
    actuals: ArrayLike,
    volatility: ArrayLike,
    epsilon: float = 1e-8,
) -> float
```

Compute volatility-normalized MAE (scale-invariant).

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `volatility` | `ArrayLike` | - | Local volatility estimates |
| `epsilon` | `float` | `1e-8` | Prevents division by zero |

**Returns**: `float` - Mean volatility-normalized absolute error

**Notes**: VN-MAE = mean(|pred - actual| / volatility). A value of 1.0 means errors are "typical" relative to local volatility.

---

### `compute_volatility_weighted_mae()`

```python
def compute_volatility_weighted_mae(
    predictions: ArrayLike,
    actuals: ArrayLike,
    volatility: ArrayLike,
    weighting: Literal["inverse", "importance"] = "inverse",
    epsilon: float = 1e-8,
) -> float
```

Compute volatility-weighted MAE.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `volatility` | `ArrayLike` | - | Local volatility estimates |
| `weighting` | `str` | `"inverse"` | `"inverse"` (low-vol more) or `"importance"` (high-vol more) |
| `epsilon` | `float` | `1e-8` | Prevents division by zero |

**Returns**: `float` - Weighted MAE

**Notes**:
- `"inverse"`: Low-volatility periods weighted more (clearer signal)
- `"importance"`: High-volatility periods weighted more (if turbulent periods matter)

---

### `VolatilityStratifiedResult`

```python
@dataclass
class VolatilityStratifiedResult:
    overall_mae: float           # Overall MAE
    low_vol_mae: float           # MAE in low volatility tercile
    med_vol_mae: float           # MAE in medium volatility tercile
    high_vol_mae: float          # MAE in high volatility tercile
    volatility_normalized_mae: float
    n_low: int                   # Observations in low tercile
    n_med: int                   # Observations in medium tercile
    n_high: int                  # Observations in high tercile
    vol_thresholds: tuple        # (low_upper, high_lower) boundaries
```

**Methods**:
- `summary() -> str`: Human-readable summary

---

### `compute_volatility_stratified_metrics()`

```python
def compute_volatility_stratified_metrics(
    predictions: ArrayLike,
    actuals: ArrayLike,
    volatility: ArrayLike | None = None,
    window: int = 13,
    method: Literal["rolling_std", "ewm"] = "rolling_std",
) -> VolatilityStratifiedResult
```

Compute MAE stratified by volatility terciles.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `ArrayLike` | - | Predicted values |
| `actuals` | `ArrayLike` | - | Actual values |
| `volatility` | `ArrayLike` | `None` | Pre-computed volatility (computed if not provided) |
| `window` | `int` | `13` | Window for volatility estimation |
| `method` | `str` | `"rolling_std"` | Volatility estimation method |

**Returns**: `VolatilityStratifiedResult`

**Example**:

```python
from temporalcv.metrics import compute_volatility_stratified_metrics

result = compute_volatility_stratified_metrics(predictions, actuals)
print(result.summary())
# Output:
# Volatility-Stratified Metrics
# ========================================
# Overall MAE:       0.023456
# VN-MAE:            1.234567
#
# By Volatility Regime:
#   Low vol (n=33):   MAE = 0.012345
#   Med vol (n=34):   MAE = 0.023456
#   High vol (n=33):  MAE = 0.034567
```

---

## References

**Core Metrics**:
- Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Armstrong, J.S. (1985). *Long-Range Forecasting: From Crystal Ball to Computer*. Wiley.
- Theil, H. (1966). *Applied Economic Forecasting*. North-Holland Publishing.

**Event Metrics**:
- Brier, G.W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3.
- Murphy, A.H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12(4), 595-600.
- Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. *ICML*.

**Quantile/Interval Metrics**:
- Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359-378.
- Koenker, R. & Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Winkler, R.L. (1972). A decision-theoretic approach to interval estimation. *JASA*, 67(337), 187-191.

**Financial Metrics**:
- Sharpe, W.F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49-58.
- Goodwin, T.H. (1998). The information ratio. *Financial Analysts Journal*, 54(4), 34-43.

**Asymmetric Loss**:
- Varian, H.R. (1975). A Bayesian approach to real estate assessment. *Studies in Bayesian Econometrics*.
- Zellner, A. (1986). Bayesian estimation and prediction using asymmetric loss functions. *JASA*, 81(394), 446-451.

**Volatility**:
- J.P. Morgan (1996). *RiskMetrics Technical Document*.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
