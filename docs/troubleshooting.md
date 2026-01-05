# Troubleshooting & FAQ

Common issues and their solutions for temporalcv.

## Installation Issues

### "ModuleNotFoundError: No module named 'temporalcv'"

**Problem**: Package not installed or wrong Python environment.

**Solution**:
```bash
pip install temporalcv
```

Verify installation:
```python
import temporalcv
print(temporalcv.__version__)  # Should print "1.0.0-rc1"
```

### Python version requirements

temporalcv requires Python 3.9 or higher. Check your version:
```bash
python --version  # Must be >= 3.9
```

### Optional dependencies

Some features require additional packages:

| Feature | Install Command |
|---------|-----------------|
| FRED data | `pip install temporalcv[fred]` |
| Changepoint detection | `pip install temporalcv[changepoint]` |
| Model comparison | `pip install temporalcv[compare]` |
| All extras | `pip install temporalcv[all]` |

---

## Validation Gates

### "Why does gate_suspicious_improvement HALT?"

**Problem**: Improvement >20% over baseline triggers a halt.

**Common causes**:
1. **Leakage in feature engineering** — Features inadvertently include future information
2. **Overfitting to test period** — Model memorized test patterns during development
3. **Weak baseline** — Wrong persistence lag or inappropriate baseline model

**Solution**: Run `gate_shuffled_target()` first to detect leakage:
```python
from temporalcv.gates import gate_shuffled_target

result = gate_shuffled_target(model, X, y, n_shuffles=100)
if result.status == GateStatus.HALT:
    print("Leakage detected! Check feature engineering.")
```

### "gate_shuffled_target returns high p-value even with leakage"

**Problem**: Insufficient shuffles to detect small effects.

**Explanation**:
- Minimum p-value is `1/(n_shuffles+1)`
- For p<0.05, need `n_shuffles >= 19`
- For p<0.01, need `n_shuffles >= 99`

**Solution**: Use more shuffles:
```python
# Default is 100, which allows p-values down to ~0.01
result = gate_shuffled_target(model, X, y, n_shuffles=100)
```

### "gate_temporal_boundary returns HALT"

**Problem**: Gap between train and test is smaller than forecast horizon.

**Solution**: Ensure `gap >= horizon`:
```python
from temporalcv import WalkForwardCV

# For 12-step ahead forecasting
cv = WalkForwardCV(n_splits=5, horizon=12, extra_gap=0, test_size=12)
```

---

## Cross-Validation

### "WalkForwardCV returns fewer splits than expected"

**Problem**: Not enough data for requested configuration.

**Causes**:
1. Gap enforcement reduces available splits
2. Train/test sizes are too large for dataset

**Solution**: Check with `get_n_splits()`:
```python
cv = WalkForwardCV(n_splits=10, horizon=5, extra_gap=0, test_size=20)
n_actual = cv.get_n_splits(X, strict=False)  # Returns actual count
print(f"Requested 10, got {n_actual}")
```

### "get_n_splits raises ValueError"

**Problem**: Strict mode (new in v1.0.0) raises errors when splits can't be computed.

**Solution**: Either increase data size or use non-strict mode:
```python
# Option 1: Non-strict mode returns 0 instead of error
n = cv.get_n_splits(X, strict=False)

# Option 2: Reduce n_splits or test_size
cv = WalkForwardCV(n_splits=3, test_size=10)
```

### "CrossFitCV yields fewer splits than n_splits"

**Problem**: CrossFitCV yields `n_splits - 1` pairs by design.

**Explanation**: Fold 0 has no training data (all samples are in test), so it's skipped.

```python
cv = CrossFitCV(n_splits=5)
splits = list(cv.split(X))
assert len(splits) == 4  # Not 5!
```

---

## Statistical Tests

### "DM test p-value is 1.0"

**Problem**: Models have identical performance (no loss differential).

**Causes**:
- Predictions are identical
- All loss differentials are zero

**Solution**: Verify predictions differ:
```python
import numpy as np

loss_diff = errors1**2 - errors2**2
print(f"Mean difference: {np.mean(loss_diff):.6f}")
print(f"All zero: {np.allclose(loss_diff, 0)}")
```

### "HAC variance is zero"

**Problem**: All loss differentials are identical.

**Causes**:
- Models make identical predictions
- Actuals have no variation

**Solution**: Check data variability and model predictions.

### "DM test gives unexpected sign"

**Problem**: Positive statistic means model 1 is *worse*.

**Explanation**: DM tests `E[loss1 - loss2] = 0`. Positive values mean model 1 has higher loss.

```python
result = dm_test(errors1, errors2)
if result.statistic > 0:
    print("Model 1 is worse than Model 2")
else:
    print("Model 1 is better than Model 2")
```

---

## Metrics

### "compute_mase returns inf"

**Problem**: `naive_mae` is zero (series is constant).

**Solution**: Check if series has variation:
```python
naive_errors = np.diff(actuals)  # lag-1 differences
if np.allclose(naive_errors, 0):
    print("Series is constant, MASE undefined")
else:
    naive_mae = np.mean(np.abs(naive_errors))
    mase = compute_mase(predictions, actuals, naive_mae)
```

### "Move-conditional MAE returns NaN"

**Problem**: No moves above threshold in data.

**Solution**: Lower threshold or use more data:
```python
from temporalcv.persistence import compute_move_threshold

# Use lower percentile for threshold
threshold = compute_move_threshold(actuals, percentile=50)  # vs default 70
```

### "compute_mc_ss gives negative values"

**Problem**: MC-SS can be negative when model is worse than persistence.

**Explanation**: MC-SS = 1 - (model_mae / persistence_mae). If model_mae > persistence_mae, MC-SS < 0.

**Interpretation**: Values <0 mean the model is worse than simple persistence on significant moves.

---

## FRED API

### "fredapi authentication failed"

**Problem**: Missing or invalid FRED API key.

**Solution**:
1. Get free API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable:
```bash
export FRED_API_KEY=your_key_here
```

Or pass directly:
```python
from temporalcv.benchmarks import load_fred_rates

rates = load_fred_rates(api_key='your_key_here')
```

### "Examples work without FRED API"

All examples have synthetic data fallback. Real FRED data is optional:

```python
# This will use synthetic data if FRED_API_KEY is not set
rates = load_fred_rates()  # Falls back gracefully
```

---

## High-Persistence Time Series

### "Model MAE is higher than persistence"

**Problem**: This is common for high-persistence series (ACF(1) > 0.9).

**Explanation**: For near-random-walk series:
- Persistence MAE ≈ σ_ε (innovation standard deviation)
- This is a very strong baseline
- MASE > 1 is expected for most models

**Solution**: Use appropriate metrics:
```python
from temporalcv.persistence import compute_move_conditional_metrics

# MC-SS focuses on significant moves
result = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)
mc_ss = result.skill_score  # MC-SS (Move-Conditional Skill Score)
```

### "Validation gates pass but model underperforms"

**Problem**: Gates check for leakage/methodology issues, not predictive skill.

**Explanation**:
- Gates ensure you haven't cheated
- A model can be methodologically sound but still underperform

**Solution**: Use statistical tests to compare models:
```python
from temporalcv.statistical_tests import dm_test

result = dm_test(model_errors, persistence_errors)
print(f"DM statistic: {result.statistic:.3f} (p={result.pvalue:.3f})")
```

---

## Performance

### "gate_shuffled_target is slow"

**Problem**: Permutation test requires fitting model many times.

**Solution**: Use effect_size method for development, permutation for production:
```python
# Fast development check (heuristic)
result = gate_shuffled_target(model, X, y, method="effect_size")

# Rigorous production check (statistical p-value)
result = gate_shuffled_target(model, X, y, method="permutation", n_shuffles=100)
```

### "Conformal prediction is slow"

**Problem**: Walk-forward calibration refits at each step.

**Solution**: Use adaptive conformal with larger adaptation window:
```python
from temporalcv.conformal import AdaptiveConformalPredictor

# Initialize with larger window (fewer parameter updates)
predictor = AdaptiveConformalPredictor(
    alpha=0.10,  # 90% intervals
    gamma=0.01   # Slower adaptation = more stable (vs default 0.05)
)

# Initialize with calibration set
predictor.initialize(cal_preds, cal_actuals)
```

---

## Still stuck?

1. **Check examples**: `examples/` directory has working scripts for each feature
2. **Read API docs**: Each function has detailed docstrings with usage examples
3. **File an issue**: https://github.com/brandonmbehring-dev/temporalcv/issues
