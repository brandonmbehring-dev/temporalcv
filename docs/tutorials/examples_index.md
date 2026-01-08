# Examples Index

A comprehensive gallery of 21 examples demonstrating temporalcv's capabilities across different domains and use cases.

---

## Quick Navigation

| Category | Examples | Best For |
|----------|----------|----------|
| [Core Concepts](#core-concepts) | 00-05 | First-time users, understanding fundamentals |
| [Production Workflows](#production-workflows) | 06-10 | Building real ML pipelines |
| [Domain-Specific](#domain-specific) | 11-15 | Specific industries and data types |
| [Failure Cases](#failure-cases) | 16-20 | Learning from common mistakes |

---

## Core Concepts

Foundational examples that introduce temporalcv's key features.

### 00: Quickstart
**File**: [`examples/00_quickstart.py`](../../examples/00_quickstart.py)

The minimal example to get started with temporalcv. Demonstrates:
- Basic `WalkForwardCV` usage with gap enforcement
- Running validation gates (`gate_shuffled_target`, `gate_suspicious_improvement`)
- Interpreting HALT/WARN/PASS results

**When to use**: Your first introduction to the library.

---

### 01: Leakage Detection
**File**: [`examples/01_leakage_detection.py`](../../examples/01_leakage_detection.py)

The shuffled target test: the definitive leakage detector. Demonstrates:
- How lookahead bias corrupts model evaluation
- The shuffled target test catches what manual inspection misses
- Before/after comparison showing the fix

**Key insight**: If your model beats a permuted target, your features encode target position.

---

### 02: Walk-Forward CV
**File**: [`examples/02_walk_forward_cv.py`](../../examples/02_walk_forward_cv.py)

Proper temporal cross-validation with gap enforcement. Demonstrates:
- Sliding vs expanding windows
- Gap parameter for h-step forecasting
- sklearn integration with `cross_val_score`

**When to use**: Any time-series model evaluation.

---

### 03: Statistical Tests
**File**: [`examples/03_statistical_tests.py`](../../examples/03_statistical_tests.py)

Is your model improvement statistically significant? Demonstrates:
- Diebold-Mariano test with HAC variance
- Pesaran-Timmermann test for directional accuracy
- When to use each test

**When to use**: Comparing model forecasts, publishing results.

---

### 04: High Persistence
**File**: [`examples/04_high_persistence.py`](../../examples/04_high_persistence.py)

Metrics for sticky series where standard MAE misleads. Demonstrates:
- MASE (Mean Absolute Scaled Error)
- MC-SS (Move-Conditional Skill Score)
- Direction accuracy metrics

**When to use**: Financial returns, macro data, any series with ACF(1) > 0.8.

---

### 05: Conformal Prediction
**File**: [`examples/05_conformal_prediction.py`](../../examples/05_conformal_prediction.py)

Distribution-free prediction intervals. Demonstrates:
- Split conformal for stationary data
- Adaptive conformal for distribution shift
- Coverage guarantees and their limits in time series

**When to use**: Uncertainty quantification, risk management.

---

## Production Workflows

Complete pipelines for real-world ML systems.

### 06: Financial CV
**File**: [`examples/06_financial_cv.py`](../../examples/06_financial_cv.py)

Cross-validation for overlapping financial labels. Demonstrates:
- `PurgedKFold` for label overlap
- Embargo periods to prevent leakage
- Multi-day forward returns

**When to use**: Any finance ML with labels spanning multiple periods.

---

### 07: Nested CV Tuning
**File**: [`examples/07_nested_cv_tuning.py`](../../examples/07_nested_cv_tuning.py)

Hyperparameter selection without leakage. Demonstrates:
- Nested walk-forward CV structure
- Inner loop for tuning, outer loop for evaluation
- Avoiding optimistic bias from tuning on test data

**When to use**: Model selection, hyperparameter optimization.

---

### 08: Regime Stratified
**File**: [`examples/08_regime_stratified.py`](../../examples/08_regime_stratified.py)

Ensuring models are tested across market conditions. Demonstrates:
- Volatility regime detection
- Stratified validation across regimes
- Per-regime performance analysis

**When to use**: Financial models, any domain with distinct operating regimes.

---

### 09: Multi-Horizon
**File**: [`examples/09_multi_horizon.py`](../../examples/09_multi_horizon.py)

Finding the predictability horizon. Demonstrates:
- `compare_horizons()` for systematic horizon analysis
- Decay of predictive power with horizon
- Optimal horizon selection

**When to use**: Determining how far ahead to forecast.

---

### 10: End-to-End Pipeline
**File**: [`examples/10_end_to_end_pipeline.py`](../../examples/10_end_to_end_pipeline.py)

Complete ML pipeline from data to deployment. Demonstrates:
- Full workflow: data → gates → CV → test → deploy
- Integration of all temporalcv components
- Production-ready patterns

**When to use**: Template for real projects.

---

## Domain-Specific

Examples tailored to specific industries and data types.

### 11: Web Traffic
**File**: [`examples/11_web_traffic.py`](../../examples/11_web_traffic.py)

Forecasting with strong weekly seasonality. Demonstrates:
- MASE evaluation against seasonal naive
- Weekly pattern handling
- Traffic-specific feature engineering

**Domain**: Web analytics, e-commerce, marketing.

---

### 12: IoT Sensor
**File**: [`examples/12_iot_sensor.py`](../../examples/12_iot_sensor.py)

High-frequency data with anomalies. Demonstrates:
- Anomaly-aware feature engineering
- Masking anomalies before rolling calculations
- Minute-level forecasting

**Domain**: Industrial IoT, predictive maintenance.

---

### 13: Macro GDP
**File**: [`examples/13_macro_gdp.py`](../../examples/13_macro_gdp.py)

Low-frequency macroeconomic forecasting. Demonstrates:
- Clark-West test for nested model comparison
- Small sample considerations (n < 100)
- Quarterly data handling

**Domain**: Economics, central banking.

---

### 14: Energy Load
**File**: [`examples/14_energy_load.py`](../../examples/14_energy_load.py)

Multi-step ahead load forecasting. Demonstrates:
- Calendar effects (hour, day, month)
- Multi-step gap enforcement
- Energy-specific feature patterns

**Domain**: Utilities, grid operations.

---

### 15: Crypto Volatility
**File**: [`examples/15_crypto_volatility.py`](../../examples/15_crypto_volatility.py)

Forecasting under extreme regime shifts. Demonstrates:
- Adaptive conformal for regime changes
- Split vs adaptive comparison during crashes
- Per-regime coverage analysis

**Domain**: Cryptocurrency, high-volatility assets.

---

## Failure Cases

Learn from common mistakes. Each example shows what goes wrong and how to fix it.

### 16: Rolling Stats Failure
**File**: [`examples/16_failure_rolling_stats.py`](../../examples/16_failure_rolling_stats.py)

**Problem**: Computing rolling features on the full series leaks future information.

```python
# WRONG - uses future data
df['ma_20'] = df['price'].rolling(20).mean()

# CORRECT - shift to use only past
df['ma_20'] = df['price'].shift(1).rolling(20).mean()
```

**Gate that catches it**: `gate_shuffled_target()` returns HALT.

---

### 17: Threshold Leak Failure
**File**: [`examples/17_failure_threshold_leak.py`](../../examples/17_failure_threshold_leak.py)

**Problem**: Computing regime thresholds on the full dataset leaks future information.

```python
# WRONG - uses future to define threshold
high_vol = df['volatility'] > df['volatility'].quantile(0.8)

# CORRECT - expanding quantile using only past
high_vol = df['volatility'] > df['volatility'].expanding().quantile(0.8).shift(1)
```

**Gate that catches it**: Regime validation gates.

---

### 18: Nested DM Test Failure
**File**: [`examples/18_failure_nested_dm.py`](../../examples/18_failure_nested_dm.py)

**Problem**: The Diebold-Mariano test is biased for nested models under the null hypothesis.

```python
# WRONG - DM test for AR(1) vs AR(1)+X
dm_result = dm_test(errors_ar1, errors_ar1_x)  # Biased!

# CORRECT - Clark-West test for nested models
cw_result = cw_test(errors_ar1, errors_ar1_x, nested=True)
```

**Why**: Under H0, the nesting constraint creates bias in the loss differential variance.

---

### 19: Missing Gap Failure
**File**: [`examples/19_failure_missing_gap.py`](../../examples/19_failure_missing_gap.py)

**Problem**: For h-step forecasting, you need a gap of at least h between train and test.

```python
# WRONG - no gap for 5-step ahead forecast
cv = WalkForwardCV(n_splits=5)  # test[0] is train[-1]+1

# CORRECT - gap enforces separation
cv = WalkForwardCV(n_splits=5, horizon=5)  # test[0] is train[-1]+5
```

**Gate that catches it**: `gate_temporal_boundary()` validates gap.

---

### 20: KFold Trap Failure
**File**: [`examples/20_failure_kfold.py`](../../examples/20_failure_kfold.py)

**Problem**: Using sklearn's KFold on time series destroys temporal order.

```python
# WRONG - random splits on time series
cv = KFold(n_splits=5, shuffle=True)  # 47.8% fake improvement!

# CORRECT - temporal order preserved
cv = WalkForwardCV(n_splits=5)
```

**Why**: Future observations leak into training data when order is randomized.

---

## Running Examples

All examples use synthetic data and require no external datasets:

```bash
# Run a single example
python examples/00_quickstart.py

# Run all examples
for f in examples/*.py; do python "$f"; done
```

## See Also

- [Failure Cases Guide](failure_cases.md) - Deep dive into common mistakes
- [Algorithm Decision Tree](../guide/algorithm_decision_tree.md) - Choosing the right approach
- [Common Pitfalls](../guide/common_pitfalls.md) - Anti-patterns to avoid
