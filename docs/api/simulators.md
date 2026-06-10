# Simulators

Generic AR/ARMA time-series path generators for testing, benchmarking, and
building downstream test DGPs.

## Overview

These are mechanism-level simulators only: no treatment effects and no causal
structure (causal known-theta DGPs are a consumer concern, e.g. `dml_ts`).

**Knowledge Tier**: [T1] — standard ARMA generation by recursion + burn-in
(Hamilton 1994; Box & Jenkins 1970).

## Conventions

- Coefficients use the **recursion convention**
  (`y_t = ar[0]*y_{t-1} + ... + e_t + ma[0]*e_{t-1} + ...`) — NOT the
  statsmodels lag-polynomial convention.
- Innovations are Gaussian `N(0, sigma^2)`.
- Paths start from zero initial conditions; a burn-in prefix
  (default `max(100, 10 * (p + q))`) is discarded, so returned paths are
  approximately stationary draws.
- Output is **always** a 2-D `(n_paths, n)` matrix — index `[0]` for a single
  series.
- Non-stationary AR coefficients raise `ValueError` (fail-loud); MA
  coefficients are never stationarity-checked (MA is stationary for any
  finite coefficients).

## Functions

### `simulate_arma`

```python
from temporalcv import simulate_arma

paths = simulate_arma(
    ar=[0.5],          # phi coefficients
    ma=[0.3],          # theta coefficients
    n=500,             # observations per path
    n_paths=10,        # independent paths
    sigma=1.0,         # innovation std
    rng=42,            # Generator | int seed | None
)
paths.shape  # (10, 500)
```

### `simulate_ar`

Convenience wrapper with no MA part:

```python
from temporalcv import simulate_ar

path = simulate_ar([0.9], n=1000, rng=42)[0]   # single AR(1) series
panel = simulate_ar([0.6, -0.2], n=300, n_paths=50, rng=7)  # AR(2) panel
```

## Relationship to `validators.generate_ar*_series`

`temporalcv.validators.generate_ar1_series` / `generate_ar2_series` (the
theoretical-bounds test helpers) delegate to `simulate_ar` — a single AR
implementation library-wide, pinned by equality tests
(`tests/test_simulators.py::TestDelegationPins`).
