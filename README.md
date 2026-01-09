# temporalcv

**Temporal cross-validation with leakage protection for time-series ML.**

[![CI](https://github.com/brandonmbehring-dev/temporalcv/actions/workflows/ci.yml/badge.svg)](https://github.com/brandonmbehring-dev/temporalcv/actions)
[![PyPI](https://img.shields.io/pypi/v/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Docs](https://readthedocs.org/projects/temporalcv/badge/?version=latest)](https://temporalcv.readthedocs.io)
[![Python](https://img.shields.io/pypi/pyversions/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-green)](docs/testing_strategy.md)
[![Tests](https://img.shields.io/badge/tests-318%20passing-brightgreen)](tests/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brandonmbehring-dev/temporalcv/blob/main/notebooks/demo.ipynb)

**[Full Documentation](https://temporalcv.readthedocs.io)** | **[See it in action](notebooks/01_why_temporal_cv.ipynb)**

---

## The Time Series Trap

You're an ML practitioner. You build a model, run cross-validation, get great metrics... then it fails in production.

**Sound familiar?** Time series breaks standard ML validation in ways that aren't obvious until you've been burned. This library helps you avoid the traps.

---

## What Goes Wrong

### The KFold Trap

Standard cross-validation randomly shuffles data. For time series, this means your model trains on *future* data to predict the *past*.

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Your time series data
X, y = load_your_time_series()

# This looks fine...
cv = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(RandomForestRegressor(), X, y, cv=cv)
print(f"R² = {scores.mean():.2f}")  # R² = 0.73 - Looks great!

# But in production...
# Model performs terribly. What happened?
```

**Result**: Up to 47% fake improvement that vanishes in production.

The problem? KFold shuffled your data. Your model saw 2024 data while "predicting" 2023. That's not forecasting—that's cheating.

sklearn's `TimeSeriesSplit` helps, but doesn't catch everything...

### Common Leakage Patterns

| Pattern | What Happens | Why It's Bad |
|---------|--------------|--------------|
| Rolling stats on full series | `.rolling().mean()` sees future | Features encode tomorrow's info |
| No gap for h-step forecast | Train ends at t, predict t+1 | Target leaks into lagged features |
| Threshold on full data | Regime boundary uses future | Classification cheats |

**These bugs don't throw errors.** Your model trains, evaluates, and looks great—until deployment.

---

## How temporalcv Protects You

temporalcv adds a **validation layer** that catches these bugs before they corrupt your results.

### Validation Gates: HALT / WARN / PASS

Before you trust any result, run it through validation gates:

| Gate | What It Catches | Status |
|------|-----------------|--------|
| **Shuffled Target Test** | Features encode target position | HALT if model beats permuted baseline |
| **Suspicious Improvement** | Too-good-to-be-true results | WARN if >20% better than persistence |
| **Temporal Boundary Audit** | Future information in features | HALT if boundary violated |

```python
from temporalcv import run_gates
from temporalcv.gates import gate_signal_verification, gate_suspicious_improvement

# Step 1: Run validation gates
gate_results = [
    gate_signal_verification(model, X, y, n_shuffles=100),
    gate_suspicious_improvement(model_mae, persistence_mae, threshold=0.20),
]

# Step 2: Check the verdict
report = run_gates(gate_results)
print(report.status)  # HALT, WARN, or PASS
```

| Status | Meaning | Action |
|--------|---------|--------|
| **HALT** | Critical failure | Stop immediately. You have leakage. |
| **WARN** | Suspicious signal | Proceed with caution. Verify externally. |
| **PASS** | Validation passed | Continue to walk-forward CV. |

### The Validation Pipeline

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

---

## Key Concepts

### Walk-Forward CV with Gap Enforcement

```
Training window                    Test
├─────────────────────────┤  GAP  ├───┤
         104 weeks          2 wks  1 wk
```

The `gap` parameter ensures your model can't cheat:

```python
from temporalcv import WalkForwardCV

cv = WalkForwardCV(
    window_type="sliding",   # or "expanding"
    window_size=104,         # 2 years of weekly data
    horizon=2,               # Predicting 2 steps ahead
    extra_gap=0,             # Additional safety margin
    test_size=1              # 1 observation per fold
)

for train_idx, test_idx in cv.split(X, y):
    # Guaranteed: train_idx[-1] + horizon < test_idx[0]
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

### Why Gaps Matter for h-Step Forecasting

If you're predicting 2 steps ahead, you need `horizon >= 2`.

**Without gap**: Your lag-1 feature at test time includes the target you're predicting.
**With gap**: Clean separation ensures no information leaks through lagged features.

### High-Persistence Metrics

When your series is "sticky" (ACF(1) > 0.9), standard metrics lie:

- **MAE looks great** because predicting "same as yesterday" works
- **But your model adds no value** over a simple persistence baseline

temporalcv provides metrics that measure *actual* predictive skill:

| Metric | What It Measures | When To Use |
|--------|------------------|-------------|
| **MASE** | Error relative to naive forecast | Always—scale-free comparison |
| **MC-SS** | Skill only when target moved | High-persistence series |
| **Direction Brier** | Probabilistic direction accuracy | Directional forecasts |

---

## Installation

```bash
pip install temporalcv
```

### Optional Dependencies

| Feature | Install Command | When Needed |
|---------|----------------|-------------|
| Benchmarks | `pip install temporalcv[benchmarks]` | Running M4/M5 benchmarks |
| Changepoint | `pip install temporalcv[changepoint]` | PELT algorithm |
| Comparison | `pip install temporalcv[compare]` | Benchmark runner |
| Development | `pip install temporalcv[dev]` | Testing, linting |
| All Features | `pip install temporalcv[all]` | Everything |

**Core dependencies**: numpy >= 1.23, scipy >= 1.9, scikit-learn >= 1.1, pandas >= 1.5

**Platforms**: Linux, macOS, Windows | **Python**: 3.9, 3.10, 3.11, 3.12

---

## Quick Start: Your First Validated Model

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from temporalcv import run_gates, WalkForwardCV
from temporalcv.gates import gate_signal_verification, gate_suspicious_improvement

# 1. Load your data
X, y = load_your_time_series()  # Shape: (n_samples, n_features), (n_samples,)

# 2. Run validation gates FIRST
#    This catches leakage before you waste time on CV
model = RandomForestRegressor(n_estimators=50, random_state=42)
persistence_mae = np.mean(np.abs(np.diff(y)))  # Naive baseline

gate_results = [
    gate_signal_verification(model, X, y, n_shuffles=100),
    gate_suspicious_improvement(
        model_score=0.85,           # Your model's score
        baseline_score=0.70,        # Persistence baseline
        threshold=0.20              # >20% improvement is suspicious
    ),
]

report = run_gates(gate_results)

if report.status == "HALT":
    raise ValueError(f"Leakage detected! {report.summary()}")
elif report.status == "WARN":
    print(f"Warning: {report.summary()}")
    # Proceed with caution...

# 3. Only if gates pass: Run walk-forward CV
cv = WalkForwardCV(
    window_type="sliding",
    window_size=104,
    horizon=2,
    test_size=1
)

predictions = []
actuals = []

for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    predictions.extend(pred)
    actuals.extend(y[test_idx])

# 4. Evaluate with appropriate metrics
from temporalcv.metrics import mase, mc_skill_score

print(f"MASE: {mase(actuals, predictions, y):.3f}")
print(f"MC-SS: {mc_skill_score(actuals, predictions):.3f}")
```

---

## Examples

21 real-world case studies organized by use case.

**[Full Examples Gallery](https://temporalcv.readthedocs.io/en/latest/auto_examples/)**

| Category | Examples | Key Learning |
|----------|----------|--------------|
| Core Concepts | 00-05 | Gates, CV, statistical tests, metrics, conformal |
| Production Workflows | 06-10 | Financial CV, nested tuning, multi-horizon, pipelines |
| Domain-Specific | 11-15 | Web traffic, IoT sensors, macro GDP, energy, crypto |

### Learn from Mistakes

These examples show **what goes wrong** so you can avoid the same traps:

| # | Example | What Goes Wrong |
|---|---------|-----------------|
| 16 | [Rolling Stats](examples/16_failure_rolling_stats.py) | `.rolling()` without `.shift()` leaks future |
| 17 | [Threshold Leak](examples/17_failure_threshold_leak.py) | Regime boundary computed on full data |
| 18 | [Nested DM Test](examples/18_failure_nested_dm.py) | DM test bias for nested models |
| 19 | [Missing Gap](examples/19_failure_missing_gap.py) | No gap for h-step forecasting |
| 20 | [KFold Trap](examples/20_failure_kfold.py) | 47.8% fake improvement from random CV |

---

## Documentation

**[temporalcv.readthedocs.io](https://temporalcv.readthedocs.io)**

### Learning Path

| If You Want To... | Start Here |
|-------------------|------------|
| Get started fast | [Quickstart Guide](https://temporalcv.readthedocs.io/en/latest/quickstart.html) |
| Understand leakage | [Leakage Detection Tutorial](https://temporalcv.readthedocs.io/en/latest/tutorials/leakage_detection.html) |
| See real examples | [Examples Gallery](https://temporalcv.readthedocs.io/en/latest/auto_examples/) |
| Look up API | [API Reference](https://temporalcv.readthedocs.io/en/latest/api/) |

### API Reference

- [Validation Gates](https://temporalcv.readthedocs.io/en/latest/api/gates.html) — HALT/PASS/WARN framework
- [Walk-Forward CV](https://temporalcv.readthedocs.io/en/latest/api/cv.html) — sklearn-compatible temporal CV
- [Statistical Tests](https://temporalcv.readthedocs.io/en/latest/api/statistical_tests.html) — DM test, PT test, HAC variance
- [Metrics](https://temporalcv.readthedocs.io/en/latest/api/metrics.html) — MASE, MC-SS, high-persistence metrics
- [Conformal Prediction](https://temporalcv.readthedocs.io/en/latest/api/conformal.html) — Distribution-free intervals

---

## Validation & Quality

temporalcv's statistical computations are validated against established references:

| Validation | Reference | What It Checks |
|------------|-----------|----------------|
| DM test golden values | R `forecast::dm.test()` | Statistic and p-value |
| Type I error | 500 Monte Carlo sims | 5% nominal rate (±2%) |
| Conformal coverage | Synthetic AR(1) | 95% nominal achieved |
| Benchmark | M4 Competition | 4,773 series, 6 frequencies |

**Test Coverage**: 83% across 318 tests

For details: [Testing Strategy](https://temporalcv.readthedocs.io/en/latest/testing_strategy.html) | [Validation Evidence](https://temporalcv.readthedocs.io/en/latest/validation_evidence.html)

---

## Julia Implementation

The Julia version is available at **[temporalcv.jl](https://github.com/brandondebehring/temporalcv.jl)** with native implementations of validation gates and statistical tests.

---

## Citation

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

See [CITATION.cff](CITATION.cff) for additional formats.

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
