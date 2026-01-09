# API Design

**Purpose**: Ideal API examples for temporalcv.

---

## Core Validation Workflow

```python
from temporalcv import ValidationReport, run_gates
from temporalcv.gates import (
    gate_signal_verification,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
    gate_temporal_boundary,
)
from temporalcv.tests import dm_test, pt_test
from temporalcv.cv import WalkForwardCV

# === Core validation workflow ===
# Pre-compute gates, then aggregate
gates = [
    gate_signal_verification(model=my_model, X=X, y=y, n_shuffles=5, random_state=42),
    gate_synthetic_ar1(model=my_model, phi=0.95, random_state=42),
    gate_suspicious_improvement(model_metric=model_mae, baseline_metric=baseline_mae),
]

report = run_gates(gates)

if report.status == "HALT":
    raise ValueError(f"Validation failed: {report.failures}")
```

## Statistical Testing

```python
# === Diebold-Mariano test ===
dm_result = dm_test(
    errors_model=errors_1,
    errors_baseline=errors_2,
    h=2,  # Horizon for HAC bandwidth
    loss="squared"
)
print(f"DM statistic: {dm_result.statistic}, p-value: {dm_result.pvalue}")

# === Pesaran-Timmermann test ===
pt_result = pt_test(
    actual=actual_changes,
    predicted=predicted_changes,
    move_threshold=0.01  # 1bp minimum move
)
print(f"Direction accuracy: {pt_result.accuracy:.2%}")
```

## Walk-Forward CV

```python
# === Walk-forward with gap ===
cv = WalkForwardCV(
    window_type="sliding",
    window_size=104,  # 2 years
    gap=2,            # h=2 horizon gap
    test_size=1
)

for train_idx, test_idx in cv.split(X, y):
    # Guaranteed: train_idx[-1] + gap < test_idx[0]
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
```

## High-Persistence Regime Detection

```python
from temporalcv.regimes import classify_volatility_regime

regimes = classify_volatility_regime(
    rates=rates,
    train_end=split_date,  # Thresholds from training only
    window=4,
    basis="changes"  # NOT levels (prevents BUG-005)
)
```

## Import Structure

```python
# Top-level imports
from temporalcv import run_gates, WalkForwardCV

# Gates submodule
from temporalcv.gates import (
    gate_signal_verification,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
    gate_temporal_boundary,
)

# Statistical tests
from temporalcv.tests import dm_test, pt_test

# Metrics
from temporalcv.metrics import mc_ss, move_only_mae, direction_brier

# CV splitters
from temporalcv.cv import WalkForwardCV, SlidingWindowSplitter, ExpandingWindowSplitter
```

## CLI Examples

```bash
# Audit a model script
temporalcv audit ./my_model.py

# Run walk-forward CV
temporalcv cv --gap 2 --window sliding --window-size 104

# Check for leakage
temporalcv check-leakage --model ridge --data train.csv
```

## Dependency Strategy

| Dependency | Required? | When Used |
|------------|-----------|-----------|
| numpy | Yes | Core arrays |
| scipy | Yes | Statistics |
| pandas | Optional | DataFrame support |
| scikit-learn | Optional | API compatibility |
| statsmodels | No | Avoid circular deps |
