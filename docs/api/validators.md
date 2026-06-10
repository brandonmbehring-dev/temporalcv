# Validators

Validation against theoretical statistical limits, plus hard numeric guards
for impossible arithmetic.

## Overview

The validators subpackage hosts two distinct vocabularies:

1. **Theoretical bounds** (gate-returning): detect "impossibly good" results
   that indicate leakage — the right response is *investigate*, so these
   return `GateResult` (PASS/HALT).
2. **Numeric output guards** (hard raises): police arithmetically impossible
   statistical outputs — if one fires, the upstream computation is wrong, so
   these raise `ValueError` immediately.

This is also distinct from the conformance suite's `check_*` functions
(`AssertionError`, dev-facing seam contracts).

**Knowledge Tier**: [T1] — AR(p) forecast-error bounds (Hamilton 1994, Ch. 4);
elementary properties of standard errors, covariance matrices, confidence
intervals, and coverage rates.

## Theoretical bounds (gate-returning)

### `check_against_ar1_bounds`

```python
from temporalcv import check_against_ar1_bounds, GateStatus

result = check_against_ar1_bounds(model_mse=0.5, phi=0.9, sigma_sq=1.0)
if result.status == GateStatus.HALT:
    print("Model beats the theoretical minimum — investigate for leakage")
```

Also exported: `theoretical_ar1_mse_bound`, `theoretical_ar1_mae_bound`,
`theoretical_ar2_mse_bound`, and the synthetic-series helpers
`generate_ar1_series` / `generate_ar2_series` (which delegate to
[`simulate_ar`](simulators.md)).

## Numeric output guards (hard `ValueError` raises)

Validate-and-return guards for the arithmetic boundary of an inference
pipeline. Each returns the normalized (`np.asarray`-ed) input on success, so
it composes as a pass-through. All reject empty input (an empty statistic at
an inference boundary is an upstream bug, not a vacuous pass).

### `finite_se`

```python
from temporalcv import finite_se

se = finite_se(se)                      # passes: finite and > 0
finite_se(-0.1, name="theta_se")        # raises ValueError
```

### `psd`

```python
from temporalcv import psd

cov = psd(cov)                          # symmetric PSD within tol
psd([[1.0, 2.0], [2.0, 1.0]])           # raises: indefinite
```

### `ci_ordered`

```python
from temporalcv import ci_ordered

lo, hi = ci_ordered(lo, hi)             # elementwise lower <= upper
ci_ordered(-np.inf, 1.96)               # one-sided intervals are fine
ci_ordered(2.0, 1.0)                    # raises: inverted bounds
```

### `coverage_in_unit`

```python
from temporalcv import coverage_in_unit

cov_rate = coverage_in_unit(0.95)       # in [0, 1] inclusive
coverage_in_unit(1.05)                  # raises
```
