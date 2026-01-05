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

## References

- Dickey & Fuller (1979). JASA 74(366), 427-431.
- Kwiatkowski et al. (1992). J. Econometrics 54, 159-178.
- Phillips & Perron (1988). Biometrika 75(2), 335-346.
