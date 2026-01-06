# temporalcv

**Temporal cross-validation with leakage protection for time-series ML.**

[![CI](https://github.com/brandonmbehring-dev/temporalcv/actions/workflows/ci.yml/badge.svg)](https://github.com/brandonmbehring-dev/temporalcv/actions)
[![PyPI](https://img.shields.io/pypi/v/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Python](https://img.shields.io/pypi/pyversions/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmbehring-dev/temporalcv/blob/main/notebooks/demo.ipynb)

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
from temporalcv.gates import gate_shuffled_target, gate_suspicious_improvement

# Validate your model doesn't have leakage
# Step 1: Compute gate results
# Note: n_shuffles>=100 required for statistical power in permutation mode (default)
gate_results = [
    gate_shuffled_target(my_model, X, y, n_shuffles=100),
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

Real-world case studies demonstrating key features:

| Example | Description |
|---------|-------------|
| [01_leakage_detection.py](examples/01_leakage_detection.py) | Shuffled target test catches lookahead bias |
| [02_walk_forward_cv.py](examples/02_walk_forward_cv.py) | Gap enforcement for h-step forecasting |
| [03_statistical_tests.py](examples/03_statistical_tests.py) | DM test: is improvement significant? |
| [04_high_persistence.py](examples/04_high_persistence.py) | MASE metrics for sticky series |
| [05_conformal_prediction.py](examples/05_conformal_prediction.py) | Adaptive intervals under distribution shift |

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
