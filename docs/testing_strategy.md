# Testing Strategy

**Version**: 1.0.0 | **Last Updated**: 2025-01-05

This document defines the testing approach for temporalcv, a library focused on temporal validation with leakage protection.

---

## Core Testing Principles

### 1. Real Tests Only

**NO stubs, TODOs, or placeholder tests**. Every test must:
- Actually execute the code being tested
- Assert meaningful correctness properties
- Use real data (not just "doesn't crash")

```python
# ❌ BAD - Stub test
def test_walk_forward_cv():
    pass  # TODO: implement

# ✅ GOOD - Real test
def test_walk_forward_cv():
    X = np.random.randn(100, 5)
    cv = WalkForwardCV(n_splits=3, test_size=10)
    splits = list(cv.split(X))
    assert len(splits) == 3
    # Verify temporal ordering
    for train_idx, test_idx in splits:
        assert train_idx[-1] < test_idx[0]
```

### 2. Coverage Targets

| Code Type | Target Coverage | Rationale |
|-----------|----------------|-----------|
| **Modules** | 80%+ | Core functionality must be well-tested |
| **Scripts** | 60%+ | Scripts have more I/O, lower bar |
| **Core systems** (gates, CV) | 90%+ | Critical path for leakage prevention |

### 3. Test Categories

temporalcv uses a **6-layer validation architecture** adapted from `~/Claude/lever_of_archimedes/patterns/testing.md`:

```
Layer 1: Unit tests (pure functions, isolated logic)
Layer 2: Integration tests (CV + metrics, gates + validators)
Layer 3: Anti-pattern tests (detect leakage bugs)
Layer 4: Property tests (statistical properties hold)
Layer 5: Benchmarks (performance + accuracy on M4/M5)
Layer 6: End-to-end validation (full workflow)
```

---

## Testing Layers

### Layer 1: Unit Tests

**Purpose**: Test individual functions in isolation.

**Location**: `tests/test_*.py` (one file per module)

**Coverage**:
- All public functions in `temporalcv.*`
- Edge cases (empty arrays, single sample, etc.)
- Parameter validation (raises on invalid input)

**Example**:
```python
def test_compute_mae():
    predictions = np.array([1.0, 2.0, 3.0])
    actuals = np.array([1.5, 2.5, 2.5])
    mae = compute_mae(predictions, actuals)
    np.testing.assert_almost_equal(mae, 0.5)
```

### Layer 2: Integration Tests

**Purpose**: Test interactions between components.

**Location**: `tests/integration/test_*.py`

**Coverage**:
- CV splitters + model fitting
- Gates + metrics
- Statistical tests + HAC variance

**Example**:
```python
def test_walk_forward_with_gates(self):
    """Verify gates pass on valid walk-forward CV."""
    cv = WalkForwardCV(n_splits=3, horizon=2, extra_gap=0, test_size=10)
    for train_idx, test_idx in cv.split(X):
        gate = gate_temporal_boundary(
            train_end_idx=train_idx[-1],
            test_start_idx=test_idx[0],
            horizon=2,
            extra_gap=0
        )
        assert gate.status == GateStatus.PASS
```

### Layer 3: Anti-Pattern Tests

**Purpose**: Verify that leakage bugs are **caught** by gates.

**Location**: `tests/anti_patterns/test_*.py`

**Coverage**:
- 10 bug categories from `data_leakage_prevention.md`
- Each anti-pattern must HALT validation

**Example**:
```python
def test_lag_leakage_detected():
    """Gate should HALT when lag computed on full series."""
    # Intentional leakage: compute lag-1 on FULL series (train + test)
    X_full = np.random.randn(100, 1)
    X_leaky = np.column_stack([X_full, np.roll(X_full[:, 0], 1)])

    # Shuffled target gate should catch this
    result = gate_signal_verification(DummyModel(), X_leaky, y, n_shuffles=100)
    assert result.status == GateStatus.HALT
```

**Bug categories tested** (from `docs/knowledge/leakage_audit_trail.md`):
1. Lag leakage (full series computation)
2. Temporal boundary violations (no gap)
3. Threshold leakage (full series percentiles)
4. Feature selection on target
5. Regime computation lookahead

### Layer 4: Property Tests

**Purpose**: Verify statistical properties hold over many random inputs.

**Location**: `tests/property/test_*.py` (uses `hypothesis` or manual property checks)

**Coverage**:
- Gates always respect temporal boundaries
- CV splits are non-overlapping
- Statistical test p-values are uniform under H0

**Example**:
```python
@pytest.mark.parametrize("n_samples", [50, 100, 200])
@pytest.mark.parametrize("n_splits", [3, 5, 10])
def test_walk_forward_no_overlap(n_samples, n_splits):
    """Verify no train/test overlap across random configurations."""
    X = np.random.randn(n_samples, 5)
    cv = WalkForwardCV(n_splits=n_splits, test_size=10)

    for train_idx, test_idx in cv.split(X):
        # Property: train and test indices must be disjoint
        assert len(set(train_idx) & set(test_idx)) == 0
```

### Layer 5: Benchmarks

**Purpose**: Validate accuracy and performance on real datasets.

**Location**: `benchmarks/` (separate from unit tests)

**Coverage**:
- M4 Competition (6 frequencies, 100k series)
- M5 Competition (30k hierarchical series)
- Monash Time Series Repository

**Metrics**:
- Forecast accuracy (MAE, RMSE)
- Runtime (seconds per series)
- Memory usage

**See**: `docs/benchmarks/methodology.md` for full details.

### Layer 6: End-to-End Validation

**Purpose**: Test complete user workflows from data loading to deployment.

**Location**: `tests/integration/test_e2e_*.py`

**Coverage**:
- Load data → Run gates → CV → Metrics → Statistical tests → Deploy
- Notebook execution (`jupyter nbconvert --execute`)
- Example scripts run without errors

**Example**:
```python
def test_full_validation_workflow():
    """End-to-end: gates → CV → metrics → DM test."""
    # 1. Run gates
    gates = [
        gate_signal_verification(model, X, y, n_shuffles=100),
        gate_suspicious_improvement(model_mae=0.10, baseline_mae=0.15),
    ]
    report = run_gates(gates)
    assert report.status == GateStatus.PASS

    # 2. Walk-forward CV
    cv = WalkForwardCV(n_splits=5, horizon=1, extra_gap=0, test_size=20)
    results = walk_forward_evaluate(model, X, y, cv)

    # 3. Compute metrics
    mae = compute_mae(results.predictions, results.actuals)

    # 4. Statistical test vs baseline
    baseline_errors = compute_naive_error(y)
    model_errors = results.predictions - results.actuals
    dm = dm_test(model_errors, baseline_errors)

    assert dm.pvalue < 0.05  # Model significantly better
```

---

## Test Organization

```
tests/
├── test_cv.py                      # Layer 1: CV unit tests
├── test_gates.py                   # Layer 1: Gate unit tests
├── test_statistical_tests.py       # Layer 1: Statistical test unit tests
├── test_metrics.py                 # Layer 1: Metrics unit tests
├── test_conformal.py               # Layer 1: Conformal prediction unit tests
├── integration/
│   ├── test_full_workflow.py       # Layer 2 & 6: Integration + E2E
│   └── test_cv_gates.py            # Layer 2: CV + gates integration
├── anti_patterns/
│   ├── test_lag_leakage.py         # Layer 3: Lag computation leakage
│   ├── test_threshold_leakage.py   # Layer 3: Threshold leakage
│   └── test_feature_selection.py   # Layer 3: Feature selection leakage
├── property/
│   └── test_cv_properties.py       # Layer 4: Statistical properties
└── validation/
    └── test_shuffled_target.py     # Layer 4: Shuffled target gate properties
```

---

## Running Tests

### Quick Validation (CI)

```bash
# Run all tests with coverage
pytest tests/ -v --cov=temporalcv --cov-report=term-missing

# Fail if coverage < 80%
pytest tests/ --cov=temporalcv --cov-fail-under=80
```

### Selective Testing

```bash
# Unit tests only (fast)
pytest tests/test_*.py -v

# Integration tests
pytest tests/integration/ -v

# Anti-pattern tests (leakage detection)
pytest tests/anti_patterns/ -v

# Property tests (may be slow)
pytest tests/property/ -v -m property

# End-to-end validation
pytest tests/integration/test_e2e_*.py -v
```

### Benchmarks (separate from CI)

```bash
# Quick benchmark (~4 minutes)
python scripts/run_benchmark.py --quick

# Full benchmark (~15 minutes)
python scripts/run_benchmark.py --full
```

---

## Continuous Integration (CI)

### GitHub Actions Workflow

**Triggers**:
- Every PR
- Push to `main`
- Nightly (for full benchmarks)

**Matrix**:
- **OS**: Ubuntu, macOS, Windows
- **Python**: 3.9, 3.10, 3.11, 3.12
- **Dependencies**: Minimum versions, latest versions

**Jobs**:
1. **Unit tests**: Layer 1-2 (fast, <5 min)
2. **Anti-pattern tests**: Layer 3 (critical path)
3. **Integration tests**: Layer 6 (full workflow)
4. **Notebook validation**: Execute all notebooks
5. **Documentation build**: Sphinx with `-W` (warnings as errors)
6. **Type checking**: `mypy src/temporalcv`
7. **Linting**: `ruff check src/ tests/`

---

## Test Data Strategy

### Synthetic Data (Default)

Most tests use synthetic data for reproducibility:

```python
np.random.seed(42)
X = np.random.randn(100, 5)
y = np.random.randn(100)
```

**Pros**:
- Fast generation
- Deterministic (reproducible)
- No external dependencies

**Cons**:
- May miss real-world edge cases

### Real Data (Benchmarks Only)

Benchmarks use real datasets:
- M4 Competition (via `datasetsforecast`)
- M5 Competition (manual download)
- Monash Repository (via `temporalcv.benchmarks`)

**Pros**:
- Validates on production-like data
- Catches real-world issues

**Cons**:
- Slower (not for CI unit tests)
- May have licensing restrictions

---

## Assertions Style

### Preferred Patterns

```python
# ✅ Use numpy.testing for float comparisons
np.testing.assert_almost_equal(result, expected, decimal=6)
np.testing.assert_allclose(result, expected, rtol=1e-5)

# ✅ Use pytest.raises for error checking
with pytest.raises(ValueError, match="n_splits must be >= 2"):
    WalkForwardCV(n_splits=1)

# ✅ Use descriptive assertion messages
assert len(splits) == 3, f"Expected 3 splits, got {len(splits)}"
```

### Anti-Patterns

```python
# ❌ Direct float comparison (floating point errors)
assert result == 0.333333  # May fail due to precision

# ❌ Vague assertions
assert result  # What property are we testing?

# ❌ Multiple unrelated assertions in one test
def test_everything():
    assert cv.n_splits == 5
    assert gate.status == "PASS"
    assert mae < 0.1
    # Too many concerns → split into separate tests
```

---

## Common Testing Patterns

### Testing Gates

```python
def test_gate_pass():
    """Gate passes on valid configuration."""
    gate = gate_temporal_boundary(
        train_end_idx=79,
        test_start_idx=85,
        horizon=5,
        extra_gap=0
    )
    assert gate.status == GateStatus.PASS

def test_gate_halt():
    """Gate halts on invalid configuration."""
    gate = gate_temporal_boundary(
        train_end_idx=79,
        test_start_idx=82,  # Gap of 2, but horizon=5
        horizon=5,
        extra_gap=0
    )
    assert gate.status == GateStatus.HALT
    assert "actual_gap (2) < required_gap (5)" in gate.message
```

### Testing Statistical Tests

```python
def test_dm_test_identical_models():
    """DM test p-value = 1.0 for identical predictions."""
    errors1 = np.array([0.1, -0.2, 0.3, -0.1])
    errors2 = errors1.copy()  # Identical

    result = dm_test(errors1, errors2, horizon=1)

    assert result.statistic == 0.0
    assert result.pvalue == 1.0

def test_dm_test_significantly_different():
    """DM test detects significant difference."""
    errors1 = np.zeros(100)  # Perfect model
    errors2 = np.random.randn(100)  # Random errors

    result = dm_test(errors1, errors2, horizon=1)

    assert result.pvalue < 0.01  # Highly significant
```

---

## Test-Driven Development (TDD)

For new features, follow the **Red-Green-Refactor** cycle:

1. **Red**: Write failing test
2. **Green**: Implement minimal code to pass
3. **Refactor**: Clean up while keeping tests passing

**Example**: Adding a new gate

```python
# Step 1: RED - Write test first
def test_gate_residual_autocorrelation_pass():
    """Gate passes when residuals are white noise."""
    residuals = np.random.randn(100)  # White noise
    gate = gate_residual_diagnostics(residuals)
    assert gate.status == GateStatus.PASS

# Step 2: GREEN - Implement
def gate_residual_diagnostics(residuals):
    from scipy.stats import acorr_ljungbox
    lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10)
    status = GateStatus.PASS if lb_pvalue[-1] > 0.05 else GateStatus.HALT
    return GateResult(status=status, message=f"Ljung-Box p={lb_pvalue[-1]:.3f}")

# Step 3: REFACTOR - Add edge cases, improve implementation
```

---

## Randomness and Reproducibility

### Seeds in Tests

**ALL tests must be deterministic**. Use fixed random seeds:

```python
def test_with_random_data():
    np.random.seed(42)  # Fixed seed for reproducibility
    X = np.random.randn(100, 5)
    # ... test logic
```

### Seeds in Production Code

Functions that use randomness accept `random_state` parameter:

```python
def gate_signal_verification(model, X, y, n_shuffles=100, random_state=None):
    """
    Parameters
    ----------
    random_state : int or None
        Random seed for reproducibility. If None, use system randomness.
    """
    rng = np.random.default_rng(random_state)
    # ... use rng for all random operations
```

---

## Future Enhancements

### Planned Additions (v1.1+)

1. **Property-based testing** with `hypothesis` for exhaustive edge case coverage
2. **Mutation testing** to verify test suite quality
3. **Performance regression tests** to catch slowdowns
4. **Notebook execution CI** for tutorial validation
5. **Visual regression tests** for plots in examples

---

## References

1. **6-Layer Validation Architecture**: `~/Claude/lever_of_archimedes/patterns/testing.md`
2. **Data Leakage Prevention**: `~/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md`
3. **pytest Documentation**: https://docs.pytest.org
4. **numpy.testing Guide**: https://numpy.org/doc/stable/reference/routines.testing.html
5. **hypothesis Property Testing**: https://hypothesis.readthedocs.io
