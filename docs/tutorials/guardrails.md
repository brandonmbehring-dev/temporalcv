# Guardrails: The HALT/WARN/PASS Framework

A decision-tree guide for using temporalcv's validation gates to catch leakage before it corrupts your results.

---

## Philosophy

> "Run gates first, train models second."

Validation gates are **pre-flight checks** for your ML pipeline. They catch problems *before* you waste time on model development that will fail in production.

---

## The Gate Decision Tree

```{mermaid}
graph TD
    A[Start: New ML Pipeline] --> B{Running gates?}
    B -->|No| C[Run gate_signal_verification first]
    B -->|Yes| D{What did gates return?}

    D -->|HALT| E[STOP: Critical issue detected]
    D -->|WARN| F[Proceed with caution]
    D -->|PASS| G[Continue to model training]

    E --> H{Which gate HALTed?}
    H -->|shuffled_target| I[Features encode target position]
    H -->|temporal_boundary| J[Gap violation for h-step]
    H -->|suspicious_improvement| K[Unrealistic performance]

    I --> L[Fix: Check .shift on rolling features]
    J --> M[Fix: Set horizon parameter]
    K --> N[Fix: Investigate data pipeline]

    L --> C
    M --> C
    N --> C

    F --> O[Verify externally before deploying]
    G --> P[Proceed to WalkForwardCV]
```

---

## Gate Reference

### `gate_signal_verification` — The Definitive Leakage Detector

**What it tests**: Whether features encode information about target position (not just value).

**How it works**:
1. Shuffle the target labels randomly
2. Train your model on shuffled data
3. If model still performs well → features leak target position

```python
from temporalcv.gates import gate_signal_verification

result = gate_signal_verification(
    model=my_model,
    X=X,
    y=y,
    n_shuffles=100,  # Statistical power requires ≥100
    test_size=0.2
)

print(f"Status: {result.status}")
print(f"p-value: {result.pvalue:.4f}")
```

**Interpretation**:

| Status | p-value | Meaning |
|--------|---------|---------|
| HALT | < 0.05 | Model beats shuffled baseline → leakage detected |
| WARN | 0.05-0.10 | Borderline, investigate further |
| PASS | > 0.10 | No evidence of position encoding |

**Common causes of HALT**:
- Rolling stats without `.shift()`
- Centered moving averages
- Lookahead in feature engineering
- Target leakage in pipeline

---

### `gate_temporal_boundary` — Gap Enforcement

**What it tests**: Whether train/test splits have sufficient separation for h-step forecasting.

**The rule**: For h-step ahead forecasting, you need `gap >= h`.

```python
from temporalcv.gates import gate_temporal_boundary

result = gate_temporal_boundary(
    cv=WalkForwardCV(n_splits=5),
    horizon=5,  # 5-step ahead forecast
    X=X
)

if result.status == "HALT":
    print(f"Gap too small: {result.actual_gap} < {result.required_gap}")
```

**Interpretation**:

| Status | Meaning | Action |
|--------|---------|--------|
| HALT | Gap < horizon | Set `horizon` parameter in CV |
| PASS | Gap >= horizon | Continue |

---

### `gate_suspicious_improvement` — Reality Check

**What it tests**: Whether model improvement over baseline is unrealistically large.

**The heuristic**: >20% improvement on first attempt is suspicious.

```python
from temporalcv.gates import gate_suspicious_improvement

result = gate_suspicious_improvement(
    model_metric=model_mae,
    baseline_metric=persistence_mae,
    threshold=0.20  # 20% improvement threshold
)
```

**Interpretation**:

| Improvement | Status | Meaning |
|-------------|--------|---------|
| < 5% | PASS | Realistic, typical for good models |
| 5-20% | PASS | Good improvement, likely valid |
| 20-50% | WARN | Investigate — possibly valid, often leakage |
| > 50% | HALT | Almost certainly leakage or bug |

---

## Running Gates in Sequence

The recommended order:

```python
from temporalcv.gates import (
    gate_signal_verification,
    gate_temporal_boundary,
    gate_suspicious_improvement
)
from temporalcv import run_gates

# 1. Collect gate results
gate_results = [
    gate_signal_verification(model, X, y, n_shuffles=100),
    gate_temporal_boundary(cv, horizon=h),
    gate_suspicious_improvement(model_mae, baseline_mae),
]

# 2. Aggregate into report
report = run_gates(gate_results)

# 3. Act on status
if report.status == "HALT":
    print(f"BLOCKED: {report.summary()}")
    raise ValueError("Fix leakage before proceeding")
elif report.status == "WARN":
    print(f"CAUTION: {report.summary()}")
    # Proceed but verify externally
else:
    print("PASS: Proceeding to model training")
```

---

## When to Run Gates

| Phase | Gates to Run | Why |
|-------|--------------|-----|
| **Initial exploration** | `shuffled_target` | Catch pipeline bugs early |
| **Before CV** | `temporal_boundary` | Verify gap is correct |
| **After CV** | `suspicious_improvement` | Reality check results |
| **Before deployment** | All gates | Final validation |

---

## Gate False Positives

Gates can occasionally produce false signals. Here's how to diagnose:

### `shuffled_target` False HALT

**Symptoms**: Gate HALTs but you've verified features are correct.

**Possible causes**:
- Very strong signal (rare but possible)
- Deterministic features that naturally correlate with position

**Resolution**:
1. Increase `n_shuffles` to 500+
2. Check p-value distribution across multiple runs
3. If consistently < 0.01, it's likely real leakage

### `suspicious_improvement` False WARN

**Symptoms**: Gate WARNs but improvement is legitimate.

**Possible causes**:
- Weak baseline (not using best available)
- Domain where large improvements are common

**Resolution**:
1. Verify baseline is reasonable (persistence for time series)
2. Check literature for typical improvements in your domain
3. Document justification if proceeding

---

## See Also

- [Algorithm Decision Tree](../guide/algorithm_decision_tree.md) — Complete workflow
- [Common Pitfalls](../guide/common_pitfalls.md) — What gates catch
- [Failure Cases](failure_cases.md) — Examples of caught leakage
- [Example 16-20](examples_index.md#failure-cases) — Failure case studies
