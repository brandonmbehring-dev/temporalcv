# temporalcv

**Temporal cross-validation with leakage protection for time-series ML.**

[![CI](https://github.com/brandonmbehring-dev/temporalcv/actions/workflows/ci.yml/badge.svg)](https://github.com/brandonmbehring-dev/temporalcv/actions)
[![PyPI](https://img.shields.io/pypi/v/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Python](https://img.shields.io/pypi/pyversions/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](docs/testing_strategy.md)
[![Tests](https://img.shields.io/badge/tests-318%20passing-brightgreen)](tests/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmbehring-dev/temporalcv/blob/main/notebooks/demo.ipynb)

**▶️ [See it in action](notebooks/01_why_temporal_cv.ipynb)** — Watch the validation gates catch leakage and guide you to a fix (GitHub renders with outputs).

---

## Why temporalcv?

Time-series ML has a leakage problem. Standard cross-validation doesn't respect temporal order, and even "proper" walk-forward implementations often miss subtle bugs:

- **Lag features computed on full series** (leaks future information)
- **No gap between train and test** (target leaks into features)
- **Thresholds computed on full series** (future information in classification)

temporalcv provides **validation gates** that catch these bugs before they corrupt your results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VALIDATION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Data + Model                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │                    VALIDATION GATES                          │     │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │     │
│   │  │  Shuffled    │  │  Temporal    │  │  Suspicious  │        │     │
│   │  │  Target Test │  │  Boundary    │  │  Improvement │        │     │
│   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │     │
│   │         │                 │                 │                │     │
│   │         └─────────────────┼─────────────────┘                │     │
│   │                           ▼                                  │     │
│   │              ┌───────────────────────┐                       │     │
│   │              │   HALT / WARN / PASS  │                       │     │
│   │              └───────────────────────┘                       │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                           │                                             │
│          HALT ◄───────────┼───────────► PASS                            │
│            │              │               │                             │
│            ▼              ▼               ▼                             │
│      ┌─────────┐    ┌─────────┐    ┌─────────────────────────────┐     │
│      │ STOP &  │    │  WARN   │    │      CONTINUE TO:           │     │
│      │INVESTIGATE│   │  USER   │    │  - Walk-Forward CV          │     │
│      └─────────┘    └─────────┘    │  - Statistical Tests (DM/PT)│     │
│                                    │  - Conformal Prediction      │     │
│                                    │  - Deployment                │     │
│                                    └─────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Gate Priority

| Status | Meaning | Action |
|--------|---------|--------|
| **HALT** | Critical failure detected | Stop immediately, investigate |
| **WARN** | Suspicious signal | Proceed with caution, verify externally |
| **PASS** | Validation passed | Continue to next stage |

---

## What Makes This Unique

1. **Shuffled Target Test** — The definitive leakage detector
   - If your model beats a permuted baseline, features encode target position
   - Catches: rolling stats on full series, lookahead bias, centered windows

2. **HALT/WARN/PASS Framework** — Actionable validation status
   - Not just metrics, but decisions
   - Prioritized: HALT > WARN > PASS

3. **Temporal-Aware Conformal Prediction**
   - Adaptive conformal for distribution shift (Gibbs & Candès 2021)
   - Approximate coverage for time series (exact guarantees require exchangeability)

4. **High-Persistence Metrics** — For sticky series (ACF(1) > 0.9)
   - MASE, MC-SS ratio, directional accuracy
   - Standard metrics mislead on near-unit-root data

5. **sklearn Integration** — Drop-in replacement
   - `WalkForwardCV` works with `cross_val_score`, `GridSearchCV`
   - Proper gap enforcement for h-step forecasting

---

## Julia Implementation

The Julia version of this library is available in a separate repository: **[temporalcv.jl](https://github.com/brandondebehring/temporalcv.jl)**.

It provides native Julia implementations of the same core validation gates and statistical tests.

---

## Comparison vs sklearn TimeSeriesSplit

| Feature | temporalcv | sklearn | Winner |
|---------|------------|---------|--------|
| Gap Enforcement | ✅ Native | ✅ v1.0+ | Both |
| Window Types | Expanding + Sliding | Expanding only | **temporalcv** |
| Leakage Detection | 3 validation gates | None | **temporalcv** |
| Statistical Tests | DM, PT, HAC | None | **temporalcv** |
| Conformal Prediction | Split + Adaptive | External (MAPIE) | **temporalcv** |
| Financial CV | Purging + Embargo | None | **temporalcv** |
| Split Speed | ~0.035 ms | ~0.012 ms | sklearn |

**Key Insight**: sklearn's `TimeSeriesSplit` handles basic temporal splits well. temporalcv adds the validation layer that catches bugs *before* they corrupt your results.

---

## Installation

```bash
pip install temporalcv
```

For development:
```bash
pip install temporalcv[dev]
```

### Optional Dependencies

temporalcv has modular dependencies for specific features:

| Feature | Install Command | When Needed |
|---------|----------------|-------------|
| **Benchmarks** | `pip install temporalcv[benchmarks]` | Running M4/M5 benchmarks |
| **Changepoint** | `pip install temporalcv[changepoint]` | PELT algorithm (requires `ruptures`) |
| **Model Comparison** | `pip install temporalcv[compare]` | Benchmark runner with DM tests |
| **Development** | `pip install temporalcv[dev]` | Testing, linting, type checking |
| **All Features** | `pip install temporalcv[all]` | Everything above |

**Core dependencies** (always installed):
- `numpy >= 1.23.0`
- `scipy >= 1.9.0`
- `scikit-learn >= 1.1.0`
- `pandas >= 1.5.0`

### Platform Compatibility

| Platform | Status | Tested Versions |
|----------|--------|-----------------|
| **Linux** | ✅ Fully supported | Ubuntu 20.04+, Debian 11+ |
| **macOS** | ✅ Fully supported | macOS 11+ (Intel & Apple Silicon) |
| **Windows** | ✅ Fully supported | Windows 10+, Windows Server 2019+ |

**Python versions**: 3.9, 3.10, 3.11, 3.12

**CI Matrix**: All combinations tested on every PR via GitHub Actions.

---

## Quick Example

```python
from temporalcv import run_gates, WalkForwardCV
from temporalcv.gates import gate_signal_verification, gate_suspicious_improvement

# Validate your model doesn't have leakage
# Step 1: Compute gate results
# Note: n_shuffles>=100 required for statistical power in permutation mode (default)
gate_results = [
    gate_signal_verification(my_model, X, y, n_shuffles=100),
    gate_suspicious_improvement(model_mae, persistence_mae, threshold=0.20),
]

# Step 2: Aggregate into report
report = run_gates(gate_results)

if report.status == "HALT":
    raise ValueError(f"Leakage detected: {report.summary()}")

# Walk-forward CV with proper gap enforcement
cv = WalkForwardCV(
    window_type="sliding",
    window_size=104,
    horizon=2,  # Minimum required separation for 2-step forecasting
    extra_gap=0,  # Optional: add safety margin (default: 0)
    test_size=1
)

for train_idx, test_idx in cv.split(X, y):
    # Guaranteed: train_idx[-1] + gap < test_idx[0]
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

---

## Features

### Validation Gates
- **Shuffled target test** - Definitive leakage detection
- **Synthetic AR(1) bounds** - Theoretical validation
- **Suspicious improvement detection** - >20% = investigate
- **Temporal boundary audit** - No future in features

### Statistical Tests
- **Diebold-Mariano test** - With HAC variance estimation
- **Pesaran-Timmermann test** - Direction accuracy (3-class)

### Walk-Forward CV
- Sliding and expanding windows
- Gap parameter enforcement
- sklearn-compatible splitter API

### High-Persistence Metrics
- **MC-SS** - Move-Conditional Skill Score
- **Move-only MAE** - Error when target moved
- **Direction Brier** - Probabilistic direction accuracy

---

## Examples

21 real-world case studies organized by use case. See [Examples Index](docs/tutorials/examples_index.md) for full descriptions.

### Core Concepts

| # | Example | Key Concept |
|---|---------|-------------|
| 00 | [Quickstart](examples/00_quickstart.py) | Basic WalkForwardCV + validation gates |
| 01 | [Leakage Detection](examples/01_leakage_detection.py) | Shuffled target test catches lookahead bias |
| 02 | [Walk-Forward CV](examples/02_walk_forward_cv.py) | Gap enforcement for h-step forecasting |
| 03 | [Statistical Tests](examples/03_statistical_tests.py) | DM test: is improvement significant? |
| 04 | [High Persistence](examples/04_high_persistence.py) | MASE metrics for sticky series |
| 05 | [Conformal Prediction](examples/05_conformal_prediction.py) | Adaptive intervals under distribution shift |

### Production Workflows

| # | Example | Key Concept |
|---|---------|-------------|
| 06 | [Financial CV](examples/06_financial_cv.py) | PurgedKFold, embargo, label overlap |
| 07 | [Nested CV Tuning](examples/07_nested_cv_tuning.py) | Hyperparameter selection without leakage |
| 08 | [Regime Stratified](examples/08_regime_stratified.py) | Volatility regimes, stratified gates |
| 09 | [Multi-Horizon](examples/09_multi_horizon.py) | `compare_horizons()`, predictability horizon |
| 10 | [End-to-End Pipeline](examples/10_end_to_end_pipeline.py) | Full data→gates→CV→deploy |

### Domain-Specific

| # | Example | Domain | Challenge |
|---|---------|--------|-----------|
| 11 | [Web Traffic](examples/11_web_traffic.py) | Web/Tech | Weekly seasonality, MASE |
| 12 | [IoT Sensor](examples/12_iot_sensor.py) | IoT | Anomaly-aware features |
| 13 | [Macro GDP](examples/13_macro_gdp.py) | Macro | Low-frequency, CW test |
| 14 | [Energy Load](examples/14_energy_load.py) | Energy | Calendar effects, multi-step |
| 15 | [Crypto Volatility](examples/15_crypto_volatility.py) | Crypto | Adaptive conformal |

### ⚠️ Failure Cases (Learn from Mistakes)

| # | Example | What Goes Wrong |
|---|---------|-----------------|
| 16 | [Rolling Stats](examples/16_failure_rolling_stats.py) | Leakage from `.rolling()` without `.shift()` |
| 17 | [Threshold Leak](examples/17_failure_threshold_leak.py) | Regime boundary computed on full data |
| 18 | [Nested DM Test](examples/18_failure_nested_dm.py) | DM bias for nested models (use CW test) |
| 19 | [Missing Gap](examples/19_failure_missing_gap.py) | No gap for h-step forecasting |
| 20 | [KFold Trap](examples/20_failure_kfold.py) | 47.8% fake improvement from random CV |

See [Failure Cases Guide](docs/tutorials/failure_cases.md) for detailed lessons.

**Interactive Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmbehring-dev/temporalcv/blob/main/notebooks/demo.ipynb)

---

## Benchmark Comparison

### Feature Matrix

| Feature | temporalcv | sklearn | sktime | Darts |
|---------|------------|---------|--------|-------|
| **Gap enforcement** | ✅ Built-in | ❌ Manual | ❌ Manual | ❌ Manual |
| **Leakage detection** | ✅ Gates | ❌ None | ❌ None | ❌ None |
| **Horizon validation** | ✅ Warnings | ❌ None | ❌ None | ❌ None |
| **Statistical tests (DM)** | ✅ HAC variance | ❌ None | ✅ Basic | ❌ None |
| **Conformal prediction** | ✅ Adaptive | ❌ None | ❌ None | ✅ Split |
| **sklearn compatible** | ✅ Full | ✅ Native | ✅ Full | ❌ Partial |

### Why Not Just sklearn's TimeSeriesSplit?

```python
from sklearn.model_selection import TimeSeriesSplit

# sklearn: No gap, no horizon validation
cv = TimeSeriesSplit(n_splits=5)  # Target leakage possible for h>1

# temporalcv: Gap enforcement + validation
from temporalcv import WalkForwardCV
cv = WalkForwardCV(n_splits=5, horizon=2, extra_gap=0)  # total_separation = horizon + extra_gap
```

### Benchmark Runner

Compare models across datasets:

```python
from temporalcv.benchmarks import create_synthetic_dataset
from temporalcv.compare import run_benchmark_suite, NaiveAdapter

datasets = [create_synthetic_dataset(seed=i) for i in range(3)]
report = run_benchmark_suite(datasets, [NaiveAdapter()], include_dm_test=True)
print(report.to_markdown())
```

---

## Documentation

### Getting Started
- [**Quickstart Guide**](docs/quickstart.md) - Get started in 5 minutes

### Tutorials
- [Leakage Detection](docs/tutorials/leakage_detection.md) - Catch data leakage with validation gates
- [Walk-Forward CV](docs/tutorials/walk_forward_cv.md) - Proper temporal cross-validation
- [High-Persistence Metrics](docs/tutorials/high_persistence.md) - Metrics for sticky series
- [Uncertainty Quantification](docs/tutorials/uncertainty.md) - Prediction intervals with coverage guarantees

### API Reference
- [Validation Gates](docs/api/gates.md) - HALT/PASS/WARN framework
- [Walk-Forward CV](docs/api/cv.md) - sklearn-compatible temporal CV
- [Statistical Tests](docs/api/statistical_tests.md) - DM test, PT test, HAC variance
- [High-Persistence Metrics](docs/api/persistence.md) - MC-SS, move-conditional MAE
- [Regime Classification](docs/api/regimes.md) - Volatility and direction regimes
- [Conformal Prediction](docs/api/conformal.md) - Distribution-free intervals
- [Bagging](docs/api/bagging.md) - Time-series-aware bagging
- [Event Metrics](docs/api/metrics.md) - Brier score, PR-AUC

### Internal
- [Planning Documentation](docs/plans/INDEX.md)
- [Ecosystem Gap Analysis](docs/plans/reference/ecosystem_gaps.md)

### Validation & Quality Assurance

temporalcv's statistical computations are validated against established libraries and academic references:

| Validation Type | Reference Source | What It Validates |
|-----------------|------------------|-------------------|
| **DM Test golden values** | R `forecast::dm.test()` | Statistic and p-value computation |
| **Monte Carlo Type I error** | 500 simulations | 5% nominal error rate (±2%) |
| **Conformal coverage** | Synthetic AR(1) | 95% nominal coverage achieved |
| **Harvey small-sample** | Harvey (1997) | Student-t p-value correction |

**Test Coverage**:
- **83% line coverage** across 318 tests
- **Core modules**: 89-94% coverage (gates, CV, statistical tests)
- **6-layer validation architecture**: Unit → Integration → Anti-pattern → Property → Monte Carlo → Benchmark

**Benchmark Results**:
- Validated on **M4 Competition** (4,773 series across 6 frequencies)
- See [Full Results](docs/benchmarks.md) | [Methodology](docs/benchmarks/methodology.md)

For complete validation evidence, see [Testing Strategy](docs/testing_strategy.md) and [Validation Evidence](docs/validation_evidence.md).

### Help & Support
- [**Troubleshooting Guide**](docs/troubleshooting.md) - Common issues and solutions
- [**Testing Strategy**](docs/testing_strategy.md) - How temporalcv is tested
- [**Benchmark Methodology**](docs/benchmarks/methodology.md) - How benchmark results are generated
- [**GitHub Issues**](https://github.com/brandonmbehring-dev/temporalcv/issues) - Report bugs or request features

---

## Citation

If you use temporalcv in your research, please cite:

```bibtex
@software{temporalcv2025,
  author       = {Behring, Brandon},
  title        = {temporalcv: Temporal cross-validation with leakage protection},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/brandonmbehring-dev/temporalcv},
  version      = {1.0.0}
}
```

See [CITATION.cff](CITATION.cff) for additional citation formats.

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
