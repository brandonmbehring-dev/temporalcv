# temporalcv Documentation

**Temporal cross-validation with leakage protection for time-series ML.**

---

## Why temporalcv?

Time-series ML has a leakage problem. Standard cross-validation doesn't respect temporal order, and even "proper" walk-forward implementations often miss subtle bugs:

| Bug | What Happens | How temporalcv Helps |
|-----|--------------|---------------------|
| Lag features from full series | Future leaks into training | Validation gates detect it |
| No gap between train/test | Target leaks into features | Gap parameter enforced |
| Thresholds from full series | Future in classification | Training-only computation |
| "Too good" results unchallenged | Bugs ship to production | >20% improvement triggers HALT |

**temporalcv provides validation gates that catch these bugs before they corrupt your results.**

---

## Quick Example

```python
from temporalcv import WalkForwardCV, run_gates
from temporalcv.gates import gate_shuffled_target, gate_suspicious_improvement

# 1. Validate your model doesn't have leakage
report = run_gates([
    gate_shuffled_target(model, X, y, n_shuffles=5),
    gate_suspicious_improvement(model_mae, baseline_mae, threshold=0.20),
])

if report.status == "HALT":
    raise ValueError(f"Leakage detected: {report.summary()}")

# 2. Walk-forward CV with proper gap enforcement
cv = WalkForwardCV(
    window_type="sliding",
    window_size=104,
    horizon=2,      # Minimum separation for 2-step forecasts
    extra_gap=0,    # Optional safety margin (default: 0)
    test_size=1
)

for train_idx, test_idx in cv.split(X, y):
    # Guaranteed: no overlap, proper temporal ordering
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
```

---

## Documentation

### Getting Started
- **[Quickstart Guide](quickstart.md)** — Install and run your first validation

### Tutorials
- **[Leakage Detection](tutorials/leakage_detection.md)** — Catch data leakage with validation gates
- **[Walk-Forward CV](tutorials/walk_forward_cv.md)** — Proper temporal cross-validation
- **[High-Persistence Series](tutorials/high_persistence.md)** — Metrics for sticky time series
- **[Uncertainty Quantification](tutorials/uncertainty.md)** — Conformal prediction and bagging

### API Reference
- **[Gates](api/gates.md)** — Validation gates and report framework
- **[Cross-Validation](api/cv.md)** — WalkForwardCV and split utilities
- **[Statistical Tests](api/statistical_tests.md)** — DM test, PT test, HAC variance
- **[Persistence Metrics](api/persistence.md)** — Move-conditional metrics, MC-SS
- **[Regime Classification](api/regimes.md)** — Volatility and direction regimes
- **[Conformal Prediction](api/conformal.md)** — Distribution-free intervals
- **[Bagging](api/bagging.md)** — Time-series-aware ensemble methods
- **[Event Metrics](api/metrics.md)** — Brier score, PR-AUC
- **[Benchmarks](api/benchmarks.md)** — Dataset loaders and synthetic data
- **[Model Comparison](api/compare.md)** — Cross-package model comparison

---

## What Makes temporalcv Unique

### Truly Novel (no existing package provides these)
- **Shuffled target test** — Definitive leakage detection
- **HALT/PASS/WARN/SKIP gates** — Composable validation framework
- **>20% improvement trigger** — Automated suspicion protocol
- **MC-SS + move thresholds** — Event-aware metrics for high-persistence series

### Integration Differentiators (components exist, orchestration doesn't)
- **DM test + gates** — Statistical testing integrated with validation
- **Conformal + regime** — Regime-aware calibration
- **Gap enforcement + audit** — Cross-package validation

---

## Installation

```bash
pip install temporalcv
```

With optional dependencies:
```bash
pip install temporalcv[dev]      # Development tools
pip install temporalcv[pandas]   # Pandas support
pip install temporalcv[compare]  # statsforecast for comparison
pip install temporalcv[all]      # Everything
```

---

## License

MIT License — see [LICENSE](../LICENSE) for details.
