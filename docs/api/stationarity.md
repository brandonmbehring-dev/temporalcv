# Stationarity Tests

Unit root and stationarity tests with unified interface.

## Overview

Provides wrappers around statsmodels unit root tests with joint interpretation logic.

## Tests

| Test | Null Hypothesis | When to Use |
|------|----------------|-------------|
| **ADF** | Unit root (non-stationary) | General purpose |
| **KPSS** | Stationary | Complement to ADF |
| **PP** | Unit root | Robust to autocorrelation |

## Joint Interpretation

Running ADF and KPSS together gives 4 cases:

| ADF Result | KPSS Result | Conclusion |
|-----------|-------------|------------|
| Rejects | Fails to reject | **Stationary** |
| Fails to reject | Rejects | **Non-stationary (unit root)** |
| Both reject | Both reject | **Difference-stationary** |
| Both fail | Both fail | **Insufficient evidence** |

## Usage

```python
from temporalcv.stationarity import (
    run_adf_test,
    run_kpss_test,
    joint_stationarity_test,
    StationarityConclusion,
)

# Individual tests
adf_result = run_adf_test(series)
kpss_result = run_kpss_test(series)

# Joint test with interpretation
conclusion = joint_stationarity_test(series)
if conclusion == StationarityConclusion.STATIONARY:
    print("Series is stationary")
```

## Complete Example: Testing and Differencing

```python
import numpy as np
from temporalcv.stationarity import (
    run_adf_test,
    joint_stationarity_test,
    StationarityConclusion,
)

# Generate non-stationary random walk
np.random.seed(42)
n = 500
random_walk = np.cumsum(np.random.randn(n))

# Test stationarity
print("Original series:")
adf_result = run_adf_test(random_walk)
print(f"  ADF statistic: {adf_result.statistic:.4f}")
print(f"  p-value: {adf_result.pvalue:.4f}")
print(f"  Conclusion: {'Stationary' if adf_result.pvalue < 0.05 else 'Non-stationary'}")

# Joint test
conclusion = joint_stationarity_test(random_walk)
print(f"  Joint conclusion: {conclusion.name}")

# If non-stationary, try differencing
if conclusion != StationarityConclusion.STATIONARY:
    diff_series = np.diff(random_walk)

    print("\nDifferenced series:")
    adf_diff = run_adf_test(diff_series)
    print(f"  ADF statistic: {adf_diff.statistic:.4f}")
    print(f"  p-value: {adf_diff.pvalue:.4f}")
    print(f"  Conclusion: {'Stationary' if adf_diff.pvalue < 0.05 else 'Non-stationary'}")

    conclusion_diff = joint_stationarity_test(diff_series)
    print(f"  Joint conclusion: {conclusion_diff.name}")
```

## When to Test Stationarity

1. **Before lag selection** - PACF/ACF assume stationarity
2. **Before AR modeling** - AR models require stationarity
3. **Model diagnostics** - Check residuals are stationary
4. **Feature engineering** - Decide if differencing is needed

## See Also

- [Lag Selection](lag_selection.md) - Requires stationary series
- [Diagnostics](diagnostics.md) - Residual stationarity checks

## References

- Dickey & Fuller (1979). JASA 74(366), 427-431.
- Kwiatkowski et al. (1992). J. Econometrics 54, 159-178.
- Phillips & Perron (1988). Biometrika 75(2), 335-346.
