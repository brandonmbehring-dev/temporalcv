# HAC

Heteroskedasticity and Autocorrelation Consistent (Newey-West) covariance
estimation: kernel weights, automatic bandwidth, a matrix-accepting long-run
covariance core, and mean / regression-sandwich standard errors with
optional AR(1) prewhitening.

## Overview

Ported and modernized from `dml_ts/dml/hac.py` (v2.0): functions plus a
frozen `HACResult` — no stateful estimator class. Golden-parity with dml_ts
is pinned for every upstream path that was correct; the deliberate
fail-loud deviations are documented in the module docstring.

**Knowledge Tier**: [T1] — Newey & West (1987, 1994); Andrews (1991);
Andrews & Monahan (1992).

## Semantics (read this first)

`HACResult.se` is the standard error **of the mean / coefficient** — final
and directly usable. `HACResult.variance` is `se**2`;
`HACResult.long_run_variance` is `Omega = variance * n` (mean-mode only).
**Never divide `se` or `variance` by n again** — conflating `Omega` with
`Omega/n` understates standard errors by `sqrt(n)` (exactly the bug this
explicit split exists to prevent).

## Functions

### `newey_west_se`

```python
from temporalcv import newey_west_se

# SE of the mean of an influence-score series (psi):
result = newey_west_se(psi, bandwidth="auto", kernel="bartlett")
result.se          # final HAC standard error of mean(psi)
result.bandwidth   # bandwidth actually used

# AR(1) prewhitening (Andrews & Monahan 1992, scalar form):
result = newey_west_se(psi, bandwidth=3, prewhiten=True)
result.ar_coef     # fitted AR(1) coefficient
```

### `newey_west_covariance`

```python
from temporalcv import newey_west_covariance

result = newey_west_covariance(residuals, X, bandwidth="auto")
result.covariance  # (k, k) sandwich covariance
result.se          # sqrt of the leading diagonal element
```

Fails loud on rank-deficient designs (no silent pseudo-inverse) and on a
negative leading variance (possible with the QS kernel in small samples).

### `long_run_covariance`

The matrix-accepting, panel-ready core:

```python
from temporalcv import long_run_covariance

omega = long_run_covariance(scores, bandwidth=6)   # (n, k) -> (k, k)
```

Returns `Omega` (NOT divided by n). Always 2-D, including `(1, 1)` for a
single series. Bartlett/Parzen guarantee PSD up to floating-point
round-off; QS may not be PSD in small samples (negative variance estimates
raise) — validate with [`psd`](validators.md) if PSD-ness is required.

### `optimal_bandwidth`

```python
from temporalcv import optimal_bandwidth

optimal_bandwidth(residuals)                      # floor(T^(1/3)) heuristic
optimal_bandwidth(residuals, method="andrews")    # AR(1) plug-in, Andrews 1991
```

`floor(T^(1/3))` is a common heuristic at the Bartlett-optimal growth rate
(ported from dml_ts); Newey-West (1994)'s own rule of thumb is
`4·(T/100)^(2/9)` — that one is used by `compute_hac_variance`. The Andrews
plug-in uses the kernel-correct constants (`alpha(1)` for Bartlett —
dml_ts applied `alpha(2)` to all kernels, so Andrews+Bartlett bandwidths
deliberately differ from dml_ts).

### Kernels

`bartlett_kernel`, `parzen_kernel`, `quadratic_spectral_kernel` —
`(lag, bandwidth) -> float` weights.

## Relationship to `compute_hac_variance`

`temporalcv.compute_hac_variance` (in `statistical_tests`) is the
DM-test-scoped scalar variant and deliberately does **not** delegate here:
it uses `sum/(n-j)` autocovariances and the `4*(n/100)^(2/9)` bandwidth
rule, and silently changing `dm_test`'s variance estimator would shift test
statistics. Use this module for everything else.
