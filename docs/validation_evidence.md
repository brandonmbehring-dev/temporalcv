# Validation Evidence

This document provides verifiable evidence of temporalcv's correctness for users who need to audit statistical computations. All evidence is reproducible via the test suite.

---

## Executive Summary

| Validation Layer | Tests | Purpose |
|-----------------|-------|---------|
| **Golden Reference** | 5 cases | Verify against R's `forecast` package |
| **Monte Carlo Calibration** | 20+ tests | Type I/II error rates |
| **Property-Based** | 30+ tests | Statistical invariants (Hypothesis) |
| **Anti-Pattern Detection** | 10+ tests | Leakage scenarios caught |
| **Benchmark Suite** | M4 Competition | 4,773 series validation |

**Reproduction**: All tests run via `pytest tests/ -v --cov=temporalcv`

---

## 1. Golden Reference Tests

**Location**: `tests/test_golden_reference.py`, `tests/fixtures/golden_reference.json`

Pre-computed values from R's `forecast::dm.test()`. Any deviation from these frozen values fails CI.

### DM Test Cases

| Case | Scenario | Expected Behavior |
|------|----------|-------------------|
| `case_001` | Equal forecasters (null true) | p-value in [0.4, 0.6] range |
| `case_002` | Model 1 clearly better | Statistic < -2, low p-value |
| `case_003` | Multi-step with HAC | h=3 correction applied |

### Wild Bootstrap Cases

| Case | Scenario | Expected Behavior |
|------|----------|-------------------|
| `case_001` | Positive mean fold stats | CI excludes zero → reject |
| `case_002` | Zero-mean fold stats | CI includes zero → fail to reject |

**Regeneration**: `tests/cross_validation/r_reference/generate_reference.R`

---

## 2. Monte Carlo Calibration

**Location**: `tests/monte_carlo/`

These tests run 400-500 simulations each to verify statistical properties.

### DM Test Type I Error Control

| Test | Simulations | Target | Acceptance Range |
|------|-------------|--------|------------------|
| `test_dm_null_fail_to_reject_rate` | 500 | 90-98% | Must not reject true null |
| `test_dm_type_i_error_control` | 500 | ~5% | 3-7% Type I error |
| `test_dm_type_i_with_autocorrelated_errors` | 500 | 85-99% | HAC correction works |

### DM Test Power Under Alternative

| Test | Effect Size | Expected Power |
|------|-------------|----------------|
| `test_dm_power_moderate_difference` | 0.8/1.2 MAE ratio | > 40% |
| `test_dm_power_small_difference` | 0.9/1.1 MAE ratio | > 10% |
| `test_dm_power_large_sample` | n=300 | > 70% |

### Conformal Prediction Coverage

| Test | Target Coverage | Acceptance Range |
|------|-----------------|------------------|
| `test_coverage_95_homoscedastic` | 95% | 93-99% |
| `test_coverage_90_homoscedastic` | 90% | 85-99% |
| `test_small_calibration_set` | 95% (n=30) | 88-99% |
| `test_large_calibration_set` | 95% (n=150) | 92-99% |

### Wild Bootstrap Coverage

| Test | Folds | Acceptance Range |
|------|-------|------------------|
| `test_type_i_error_5_folds` | 5 | 1-15% |
| `test_type_i_error_10_folds` | 10 | 2-12% |
| `test_type_i_error_20_folds` | 20 | 2-10% |

---

## 3. Property-Based Tests (Hypothesis)

**Location**: `tests/property/`

Uses [Hypothesis](https://hypothesis.readthedocs.io/) for exhaustive property testing.

### Gate Composition Invariants

```python
# These properties hold for ALL valid gate inputs:
- HALT dominates WARN dominates PASS
- Composed report status = max(individual statuses)
- Report contains all input gates (no loss)
- Gate names preserved in report
```

### Suspicious Improvement Gate

```python
# These properties are verified:
- Zero or negative improvement → never HALT
- >90% improvement → always HALT
- Threshold boundary respected exactly
```

### CV Split Invariants

```python
# Verified across n_samples ∈ [50, 200], n_splits ∈ [3, 10]:
- No train/test overlap
- Temporal order preserved (max(train) < min(test))
- Gap parameter respected
```

---

## 4. Anti-Pattern Detection Tests

**Location**: `tests/anti_patterns/`

Tests that intentionally introduce leakage to verify gates catch it.

| Bug Category | Test Method | Expected Result |
|--------------|-------------|-----------------|
| **Lag leakage** | Compute lag on full series | Gate HALT |
| **Threshold leakage** | Percentiles on future data | Gate HALT |
| **Missing gap** | train_end == test_start | Gate HALT |
| **Feature selection on target** | Correlation with y[full] | Gate HALT |
| **Regime lookahead** | Use future regime labels | Gate HALT |

These tests prove the gates work by verifying they catch known bugs.

---

## 5. Synthetic AR(1) Bounds

**Location**: `tests/validation/test_synthetic_ar1.py`

For AR(1) with known parameters (φ, σ), the theoretical minimum MAE is:

```
MAE_optimal = σ × √(2/π)
```

| Test | Predictor | Expected Behavior |
|------|-----------|-------------------|
| `test_mean_predictor_passes` | Unconditional mean | MAE >> optimal → PASS |
| `test_optimal_predictor_passes` | AR(1) with true φ | MAE ≈ optimal → PASS |
| `test_different_phi_values` | φ ∈ {0.1, 0.5, 0.9, 0.99} | All pass bounds |

**Implication**: If your model beats the theoretical bound, it's overfitting or has leakage.

---

## 6. Benchmark Results

**Location**: `docs/benchmarks.md`, `benchmarks/results/`

### M4 Competition Validation

| Metric | Value |
|--------|-------|
| **Series** | 4,773 |
| **Frequencies** | Yearly, Quarterly, Monthly, Weekly, Daily, Hourly |
| **Models Compared** | 9 (Naive, AutoARIMA, AutoETS, AutoTheta, etc.) |
| **Runtime** | 14.3 minutes (128-core AMD EPYC) |

### Key Findings

| Finding | Evidence |
|---------|----------|
| AutoETS most robust | Wins 3/6 frequencies |
| AutoARIMA best mean MAE | -12.9% vs Naive |
| Daily is hardest | Naive wins (complex models overfit) |
| Hourly benefits from ARIMA | -31% vs Naive |

---

## 7. Academic Citations

All statistical tests cite foundational papers:

| Test | Primary Reference | Correction |
|------|-------------------|------------|
| Diebold-Mariano | Diebold & Mariano (1995) | Harvey (1997) small-sample |
| Pesaran-Timmermann | Pesaran & Timmermann (1992) | — |
| Giacomini-White | Giacomini & White (2006) | Conditional ability |
| Clark-West | Clark & West (2007) | Nested models |
| HAC Variance | Newey-West (1987) | Autocorrelation-robust |
| Block Bootstrap | Künsch (1989), Politis & Romano (1994) | Preserve dependence |

---

## 8. Known Caveats

These are documented limitations, not bugs:

| Caveat | Location | Mitigation |
|--------|----------|------------|
| 3-class PT test variance is heuristic | `statistical_tests.py` | Runtime warning emitted |
| Conformal coverage is approximate for time series | `conformal.py` | Documented in docstring |
| `gate_signal_verification` HALT is expected for valid models | `gates.py` | Interpretation guide in docstring |

---

## 9. Reproduction Instructions

### Run Full Test Suite

```bash
# All tests (fast)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=temporalcv --cov-report=html

# Monte Carlo tests only (slow, ~10 min)
pytest tests/monte_carlo/ -v --run-slow
```

### Regenerate R Reference Values

```bash
cd tests/cross_validation/r_reference/
Rscript generate_reference.R
```

### Run Benchmarks

```bash
python -m temporalcv.benchmarks.run --dataset m4_subset --output results/
```

---

## 10. Audit Checklist

For users auditing this library:

- [ ] Run `pytest tests/test_golden_reference.py -v` — Verify R agreement
- [ ] Run `pytest tests/monte_carlo/ -v --run-slow` — Verify calibration
- [ ] Check `tests/fixtures/golden_reference.json` — Review frozen values
- [ ] Read `docs/benchmarks.md` — Verify benchmark claims
- [ ] Search for `[T3]` tags — Review heuristic components

---

**Last Updated**: 2026-01-09
**Coverage**: 83% (318 tests passing)
