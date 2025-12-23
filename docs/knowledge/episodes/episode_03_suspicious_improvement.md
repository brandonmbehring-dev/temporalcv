# Episode 03: Suspicious Improvement Threshold

**Category**: Bug Categories 1-5 (Multiple)
**Origin**: Empirical observation across 10+ projects
**Impact**: Prevents deployment of leaky models

---

## The Pattern

Across multiple forecasting projects, a consistent pattern emerged:

| Improvement over Persistence | Outcome |
|------------------------------|---------|
| < 10% | Usually legitimate |
| 10-20% | Often legitimate, verify carefully |
| > 20% | Almost always a bug |

The 20% threshold became the "too good to be true" heuristic.

---

## Why 20%?

### Theoretical Justification

For high-persistence series (ACF(1) > 0.9):
- Persistence (predict no change) is a strong baseline
- The theoretical optimal forecast is `φ × y_{t-1}` where φ ≈ 0.95
- Expected improvement: `(1 - φ) × σ_y / σ_ε` ≈ 5-15%

Beating persistence by >20% means either:
1. Persistence is not the right baseline (rare for financial data)
2. There's information leakage (common)

### Empirical Evidence

From myga-forecasting postmortems:

| Project | Claimed Improvement | Bug? | Root Cause |
|---------|---------------------|------|------------|
| v1-beta | 35% | Yes | MIDAS architecture mismatch |
| v1-final | 18% | No | Legitimate (verified externally) |
| v2-beta | 45% | Yes | Regime threshold from full series |
| v2-final | 12% | No | After bug fix |
| v3-beta | 28% | Yes | Rolling features on full series |
| v3-final | 9% | No | After bug fix |

Every case of >20% improvement was a bug.

---

## The Gate

```python
from temporalcv import gate_suspicious_improvement

result = gate_suspicious_improvement(
    model_error=0.042,
    baseline_error=0.055,
    threshold=0.20,      # HALT if > 20%
    warn_threshold=0.10  # WARN if 10-20%
)

# improvement = (0.055 - 0.042) / 0.055 = 23.6%
print(result.status)   # HALT
print(result.message)  # "Model 23.6% better than baseline (max: 20%)"
```

---

## Calibrating the Threshold

The 20% threshold is [T3] (assumption needing justification). You may need to adjust for your domain:

### When to Lower (Stricter)
- Very high persistence (ACF(1) > 0.98)
- Financial/economic data with efficient markets
- Data where persistence is theoretically optimal

### When to Raise (More Lenient)
- Lower persistence series (ACF(1) < 0.8)
- Data with strong seasonal patterns
- Situations where external information genuinely helps

**Always document your justification** in SPECIFICATION.md Amendment History.

---

## False Positive Prevention

The threshold can trigger false positives in legitimate cases:

1. **Seasonal data**: Strong seasonality can legitimately beat persistence
   - Solution: Use seasonal persistence as baseline

2. **Regime changes**: Model captures regime shifts
   - Solution: Verify with `gate_shuffled_target()` first

3. **External features**: Genuinely informative external data
   - Solution: If shuffled target passes, improvement may be real

---

## The Decision Flowchart

```
Model shows 25% improvement over persistence
    |
    v
[gate_suspicious_improvement] → HALT
    |
    v
Run gate_shuffled_target()
    |
    ├── HALT → Feature encodes target position → BUG
    |
    └── PASS → Continue investigation
              |
              v
         Run gate_synthetic_ar1()
              |
              ├── HALT → Beating theoretical bounds → BUG
              |
              └── PASS → May be legitimate
                        |
                        v
                   Verify on held-out data
                        |
                        ├── Replicates → Legitimate improvement!
                        |
                        └── Doesn't replicate → Hidden bug
```

---

## Test Case

```python
def test_suspicious_improvement_detection():
    """Gate should HALT on >20% improvement."""
    # Model appears to beat baseline by 25%
    model_mae = 0.045
    baseline_mae = 0.060

    result = gate_suspicious_improvement(
        model_error=model_mae,
        baseline_error=baseline_mae,
        threshold=0.20
    )

    assert result.status == GateStatus.HALT
    assert "25" in result.message  # 25% improvement
```

---

## Related

- [Leakage Audit Trail](../leakage_audit_trail.md) - Full bug category list
- SPECIFICATION.md Section 1.1 - Threshold values
- [Episode 01: Lag Leakage](episode_01_lag_leakage.md) - Common cause of high improvement
- [Episode 02: Boundary Violations](episode_02_boundary_violations.md) - Another common cause

---

## Key Insight

> "If your time series model beats persistence by more than 20%, you probably have a bug. If it beats persistence by more than 30%, you definitely have a bug."

This heuristic has saved weeks of debugging time by catching leakage early.
