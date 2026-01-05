# gate_shuffled_target Model Card

**Version**: 1.0.0
**Module**: `temporalcv.gates`
**Type**: Validation gate (HALT/PASS/WARN/SKIP)
**License**: MIT
**Knowledge Tier**: [T1] Permutation test principle; [T2] Shuffled target as leakage test

---

## Component Details

`gate_shuffled_target` is the **definitive leakage detection test** for time-series ML pipelines. It shuffles the target vector to destroy temporal relationships, then tests whether the model can still beat the shuffled baseline.

**Key Insight**: If a model beats a shuffled target, it's using information about target *position* rather than legitimate predictive signal—this indicates data leakage.

---

## Intended Use

### Primary Use Cases

- Definitive data leakage detection in time-series ML pipelines
- First-stage external validation before internal metrics
- Pre-deployment sanity check for forecasting models
- Publication-quality validation (use `method="permutation"`, `strict=True`)

### Out-of-Scope Uses

- **Detecting overfitting**: Use held-out test set instead
- **Feature importance**: Use SHAP or permutation importance
- **Model comparison**: Use DM test instead
- **Non-temporal leakage**: e.g., target encoding in cross-sectional data

### Target Users

- ML engineers deploying time-series forecasting pipelines
- Researchers requiring rigorous validation
- **Prerequisites**: Understanding of permutation tests and temporal validation

---

## Parameters

| Parameter | Type | Default | Description | Tier |
|-----------|------|---------|-------------|------|
| `model` | FitPredictModel | required | sklearn-compatible model | - |
| `X` | ArrayLike | required | Feature matrix (n_samples, n_features) | - |
| `y` | ArrayLike | required | Target vector (n_samples,) | - |
| `method` | str | "permutation" | "permutation" or "effect_size" | [T1/T3] |
| `n_shuffles` | int | auto | Number of shuffles (see below) | [T1] |
| `alpha` | float | 0.05 | Significance level (permutation method) | [T3] |
| `threshold` | float | 0.05 | Max improvement ratio (effect_size method) | [T3] |
| `permutation` | str | "block" | "iid" or "block" | [T1] |
| `block_size` | int/"auto" | "auto" | Block size for block permutation | [T1] |
| `n_cv_splits` | int | 3 | Walk-forward CV splits | [T2] |
| `strict` | bool | False | If True, n_shuffles >= 199 | [T1] |
| `random_state` | int | None | Random seed for reproducibility | - |

### Default n_shuffles by Method

| Method | Default | Justification |
|--------|---------|---------------|
| `"effect_size"` | 5 | Fast heuristic check |
| `"permutation"` | 100 | Rigorous statistical test |
| `"permutation"` + `strict=True` | 199 | p-value resolution of 0.005 |

### Method Selection Guide

| Method | Answers | Speed | Use When |
|--------|---------|-------|----------|
| `"permutation"` | "What's the probability of seeing this by chance?" | Slower | Publication, production |
| `"effect_size"` | "How much better is model than shuffled?" | Fast | Development, quick checks |

---

## Assumptions

| Assumption | Required For | Violation Consequence | Validation Method |
|------------|--------------|----------------------|-------------------|
| Temporal relationship in data | Shuffling destroys signal | False HALT | Domain knowledge |
| No NaN values | Valid computation | `ValueError` raised | Validated at entry |
| Model can be cloned | Fresh fit per shuffle | State leakage | Uses `sklearn.base.clone()` |
| `X.shape[0] == len(y)` | Paired data | `ValueError` raised | Validated at entry |

---

## Performance Characteristics

### Time Complexity

- **O(n_shuffles × n_cv_splits × model_fit_time)**
- Dominant factor is model training
- Example: 100 shuffles × 3 CV splits = 300 model fits

### Space Complexity

- O(n) for data copies during permutation
- Models are cloned, not modified in place

### Sample Size Requirements

| Context | Minimum | Recommended | Justification |
|---------|---------|-------------|---------------|
| Permutation test (p < 0.05) | n_shuffles >= 19 | n_shuffles >= 100 | Min p-value = 1/(n+1) |
| Effect size mode | n_shuffles >= 3 | n_shuffles >= 5 | Variance reduction |
| Data samples | 30+ | 100+ | CV fold requirements |

---

## Output Interpretation

### GateResult Status

| Status | Meaning | Action |
|--------|---------|--------|
| **HALT** | Model beats shuffled target significantly | **STOP** - Investigate leakage |
| **PASS** | Model does NOT beat shuffled target | Proceed to next gate |
| **WARN** | Marginal result | Review with caution |
| **SKIP** | Insufficient data | Provide more samples |

### Details Dictionary

```python
result.details = {
    "model_mae": 0.123,           # Model MAE on real target
    "shuffled_mae_mean": 0.145,   # Mean MAE on shuffled targets
    "shuffled_mae_std": 0.012,    # Std of shuffled MAEs
    "improvement_ratio": 0.15,    # How much model beats shuffled
    "pvalue": 0.03,               # (permutation method only)
    "n_shuffles": 100,            # Actual shuffles performed
}
```

---

## Limitations and Caveats

### Known Limitations

1. **Computationally expensive**: n_shuffles × n_cv_splits model fits required
2. **Assumes leakage causes feature-target alignment**: May miss other leakage types
3. **Block permutation heuristics**: Block size n^(1/3) is rule of thumb
4. **IID permutation limitation**: May false-positive on legitimately persistent series

### When NOT to Use

- Data is already known to have temporal structure destroyed
- Model training is extremely expensive (consider effect_size mode)
- You want to detect non-temporal leakage patterns

### Common Misconfigurations

| Mistake | Problem | Fix |
|---------|---------|-----|
| Using `permutation="iid"` on persistent series | False positives | Use `permutation="block"` (default) |
| `n_shuffles=10` for permutation test | Min p-value = 0.09, can't detect at α=0.05 | Use `n_shuffles >= 100` (or `method="effect_size"` for fast checks) |
| Ignoring HALT result | Deploying leaky model | **Always investigate HALT** |
| Using effect_size for publication | Not statistically rigorous | Use `method="permutation", strict=True` |

---

## Examples

### Quick Check During Development

```python
from temporalcv.gates import gate_shuffled_target

result = gate_shuffled_target(
    model, X, y,
    method="effect_size",  # Fast heuristic
)
print(f"Status: {result.status}")
print(f"Improvement: {result.metric_value:.1%}")
```

### Rigorous Testing for Publication

```python
result = gate_shuffled_target(
    model, X, y,
    method="permutation",  # Statistical rigor
    strict=True,           # n_shuffles >= 199
    random_state=42,       # Reproducibility
)
print(f"p-value: {result.details['pvalue']:.4f}")
if result.status == "HALT":
    print("WARNING: Model beats shuffled target - investigate leakage!")
```

### In Validation Pipeline

```python
from temporalcv.gates import gate_shuffled_target, gate_synthetic_ar1

# Stage 1: External validation (run first)
result1 = gate_shuffled_target(model, X, y)
if result1.status == "HALT":
    raise ValueError("Leakage detected - do not proceed")

# Stage 2: Synthetic validation
result2 = gate_synthetic_ar1(model, ar_coef=0.9)

# Stage 3: Internal validation (only if stages 1-2 pass)
# ... proceed with normal evaluation
```

---

## References

### [T1] Academic Sources

- Kunsch, H.R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. *Annals of Statistics*, 17(3), 1217-1241.
- Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. *JASA*, 89(428), 1303-1313.
- Phipson, B. & Smyth, G.K. (2010). Permutation P-values Should Never Be Zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

### [T2] Empirical Sources

- External-first validation ordering validated in myga-forecasting-v2
- Block permutation default validated across multiple time-series domains

### [T3] Heuristics

- `alpha=0.05`: Conventional significance level
- `threshold=0.05`: Conservative improvement threshold
- `block_size=n^(1/3)`: Kunsch (1989) rule of thumb

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-Q1 | Two-mode API (permutation + effect_size) |
| 0.4.0 | 2024-12 | Added `strict` mode for publication |
| 0.3.0 | 2024-11 | Block permutation default |
| 0.2.0 | 2024-10 | Initial permutation test implementation |

---

## See Also

- `gate_synthetic_ar1`: Test against theoretical AR(1) bounds
- `gate_suspicious_improvement`: Check for implausible improvement ratios
- `WalkForwardCV`: The CV strategy used internally
- `gate_temporal_boundary`: Verify gap enforcement
