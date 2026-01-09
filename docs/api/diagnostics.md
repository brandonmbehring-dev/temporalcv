# Diagnostics

Tools for understanding what drives validation results and how sensitive they are to parameter choices.

## Overview

The diagnostics module provides two key capabilities:

1. **Influence diagnostics** - Identify which observations disproportionately affect test statistics
2. **Sensitivity analysis** - Assess how results change with different parameter choices

## Influence Diagnostics

When a DM test shows significant difference between models, you should ask: "Is this driven by a few outliers or consistent across the sample?"

```python
from temporalcv.diagnostics import compute_dm_influence

# Get influence scores for each observation
influence = compute_dm_influence(errors1, errors2, h=1)

# Check for high-influence observations
print(f"High-influence observations: {influence.n_high_influence_obs}")
print(f"High-influence blocks: {influence.n_high_influence_blocks}")

# Investigate specific observations
high_obs = np.where(influence.observation_high_mask)[0]
print(f"Indices of high-influence observations: {high_obs}")
```

### Interpreting Influence Results

The `InfluenceDiagnostic` provides two views:

- **`observation_influence`**: HAC-adjusted per-observation scores (for exploration)
- **`block_influence`**: Block jackknife scores (recommended for decisions)

Use block influence for decisions because:
- It respects temporal dependence structure
- It's theoretically justified for serially correlated data
- Observation-level influence can be misleading with autocorrelation

## Gap Sensitivity Analysis

How robust are your results to the gap parameter choice?

```python
from temporalcv.diagnostics import gap_sensitivity_analysis

result = gap_sensitivity_analysis(
    model=my_model,
    X=X,
    y=y,
    gap_range=range(0, 15),
    metric="mae",
    degradation_threshold=0.10,
)

print(f"Break-even gap: {result.break_even_gap}")
print(f"Sensitivity score: {result.sensitivity_score:.3f}")

# Plot sensitivity curve
import matplotlib.pyplot as plt
plt.plot(result.gap_values, result.metrics)
plt.axhline(result.baseline_metric * 1.1, color='r', linestyle='--', label='10% degradation')
plt.xlabel("Gap")
plt.ylabel("MAE")
plt.title("Gap Sensitivity Analysis")
plt.legend()
```

### Interpreting Sensitivity Results

- **`break_even_gap`**: First gap where performance degrades beyond threshold
- **`sensitivity_score`**: Coefficient of variation (higher = more sensitive)

A model with `break_even_gap=1` is concerning—performance drops immediately with any gap, suggesting possible lookahead bias.

## Best Practices

1. **Always run influence diagnostics** when publishing results
   - High-influence observations should be investigated
   - Consider leave-one-out robustness checks

2. **Gap sensitivity is a leakage detector**
   - Sharp degradation at extra_gap=1 suggests temporal leakage
   - Gradual degradation is expected as forecasting horizon increases

3. **Combine with validation gates**
   - Run `gate_signal_verification()` first
   - Use diagnostics for deeper investigation after gates pass

## API Reference

See the function signatures below for complete documentation.
