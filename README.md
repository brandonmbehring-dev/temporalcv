# temporalcv

**Temporal cross-validation with leakage protection for time-series ML.**

[![CI](https://github.com/yourusername/temporalcv/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/temporalcv/actions)
[![PyPI](https://img.shields.io/pypi/v/temporalcv.svg)](https://pypi.org/project/temporalcv/)
[![Python](https://img.shields.io/pypi/pyversions/temporalcv.svg)](https://pypi.org/project/temporalcv/)

---

## Why temporalcv?

Time-series ML has a leakage problem. Standard cross-validation doesn't respect temporal order, and even "proper" walk-forward implementations often miss subtle bugs:

- **Lag features computed on full series** (leaks future information)
- **No gap between train and test** (target leaks into features)
- **Thresholds computed on full series** (future information in classification)

temporalcv provides **validation gates** that catch these bugs before they corrupt your results.

---

## Installation

```bash
pip install temporalcv
```

For development:
```bash
pip install temporalcv[dev]
```

---

## Quick Example

```python
from temporalcv import run_gates, WalkForwardCV
from temporalcv.gates import gate_shuffled_target, gate_suspicious_improvement

# Validate your model doesn't have leakage
report = run_gates(
    model=my_model,
    X=X, y=y,
    gates=[
        gate_shuffled_target(n_shuffles=5),
        gate_suspicious_improvement(threshold=0.20),
    ]
)

if report.status == "HALT":
    raise ValueError(f"Leakage detected: {report.failures}")

# Walk-forward CV with proper gap enforcement
cv = WalkForwardCV(
    window_type="sliding",
    window_size=104,
    gap=2,  # Enforces gap >= horizon
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

## Documentation

- [API Reference](docs/api/)
- [Planning Documentation](docs/plans/INDEX.md)
- [Ecosystem Gap Analysis](docs/plans/reference/ecosystem_gaps.md)

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
