# Strategic Plan: Time-Series Package Ecosystem Impact Analysis

**Created**: 2025-12-04
**Status**: ARCHIVED (Content extracted to component files)
**Original location**: `~/.claude/plans/warm-chasing-coral.md`

---

> **Note**: This document preserves the original comprehensive research plan.
> Active planning is now in component files. See `../INDEX.md` for navigation.

---

## Package Naming Exploration

**Package Vision**: A rigorous, general-use time-series ML prediction package with:
- Proper CV rigor (walk-forward, gap enforcement)
- Bootstrapping options (block bootstrap, stationary bootstrap)
- Leakage protection (shuffled target, temporal boundary audits)
- Statistical testing (DM, PT tests with HAC variance)
- High-persistence handling (MC-SS, move thresholds)

### PyPI Availability Research (Verified 2025-12-04)

#### Available Names

| Name | Chars | Theme | Notes |
|------|-------|-------|-------|
| **temporalcv** | 10 | Temporal + CV | **SELECTED** |
| forecastguard | 12 | Guard/protection | Professional |
| tsguard | 7 | Guard + time-series | Short |
| tsvalidate | 10 | Time-series + validate | Clear |
| walkforward | 11 | CV method | Method-specific |
| foldcast | 8 | Fold + forecast | Creative |
| tsrigor | 7 | Time-series + rigor | Memorable |

#### Taken Names

| Name | Issue |
|------|-------|
| tscv | TAKEN (scikit-learn extension) |
| timeseries-cv | TAKEN |
| mlforecast | TAKEN (Nixtla) |
| pytorch-forecasting | TAKEN |
| pyts | TAKEN |
| tslearn | TAKEN |
| tsfresh | TAKEN |

---

## Executive Summary

**Package Name**: `temporalcv` (Python) → `TemporalCV.jl` (Julia later)

**Positioning**: Statistically rigorous ML time-series for highly persistent series with exogenous data support.

**Unique Value**:
1. Integrated validation framework — Leakage detection + statistical testing + CV
2. High-persistence handling — MC-SS, move thresholds
3. Comprehensive Python ML connection — Bridges gap between rigor and sklearn
4. Exogenous data support — Future-known vs future-unknown distinction

---

## 1. Ecosystem Gap Analysis

### 1.1 Critical Gaps (No Solution Anywhere)

| Gap | Python | R | Julia | temporalcv |
|-----|--------|---|-------|------------|
| Leakage detection framework | ❌ | ❌ | ❌ | `gates.py` |
| Shuffled target test | ❌ | ❌ | ❌ | `gate_shuffled_target()` |
| Suspicious improvement detection | ❌ | ❌ | ❌ | `gate_suspicious_improvement()` |
| 3-stage validation gates | ❌ | ❌ | ❌ | External → Internal → Statistical |

### 1.2 Partial Coverage

| Gap | Existing | temporalcv Addition |
|-----|----------|---------------------|
| Walk-forward with gap | sktime, Darts | Cross-package validation |
| DM test | dieboldmariano PyPI | Integrated with gates |
| Conformal | MAPIE | Regime-aware calibration |

---

## 2. Components Ranked by Impact

### Tier 1: High Impact

| Component | Impact | Target |
|-----------|--------|--------|
| Validation Gate Framework | 🔴 Critical | Python + Julia |
| Anti-Pattern Test Suite | 🔴 Critical | All |
| Walk-Forward CV | 🟠 High | Julia primarily |
| Statistical Tests | 🟠 High | Julia (no DM test!) |

### Tier 2: Strong Differentiators

| Component | Impact | Notes |
|-----------|--------|-------|
| Regime Classification | 🟡 Medium | Changes-based volatility |
| Time Series Bagging | 🟡 Medium | Block bootstrap |
| Conformal Prediction | 🟡 Medium | Complements ConformalPrediction.jl |

---

## 4. Implementation Roadmap

### Phase 0: Repository Setup (Day 1) ✅ COMPLETE

- Directory structure created
- pyproject.toml with hatch
- CI workflow
- Community docs

### Phase 1: Core Foundation (Weeks 1-4)

| Module | Files |
|--------|-------|
| `temporalcv.gates` | `gates.py` |
| `temporalcv.tests` | `statistical_tests.py` |
| `temporalcv.metrics` | `metrics.py` |

### Phase 2: Walk-Forward (Weeks 5-8)

| Module | Files |
|--------|-------|
| `temporalcv.cv` | `cv.py` |
| `temporalcv.splitters` | `splitters.py` |

### Phase 3: High-Persistence (Weeks 9-12)

| Module | Files |
|--------|-------|
| `temporalcv.regimes` | `regimes.py` |
| `temporalcv.persistence` | `persistence.py` |

### Phase 4: Uncertainty (Weeks 13-16)

| Module | Files |
|--------|-------|
| `temporalcv.conformal` | `conformal.py` |
| `temporalcv.bagging` | `bagging/` |

### Phase 5: Benchmarks (Weeks 17-20)

| Module | Files |
|--------|-------|
| `temporalcv.benchmarks` | loaders |
| `temporalcv.metrics.event` | MC-SS, move-only |

### Phase 6: Julia Port (Weeks 21-28)

Native Julia implementation.

---

## 5. Package API Design

```python
from temporalcv import ValidationReport, run_gates
from temporalcv.gates import (
    gate_shuffled_target,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
)
from temporalcv.tests import dm_test, pt_test
from temporalcv.cv import WalkForwardCV

# Core validation
report = run_gates(
    model=my_model,
    X=X, y=y,
    gates=[
        gate_shuffled_target(n_shuffles=5),
        gate_synthetic_ar1(phi=0.95),
        gate_suspicious_improvement(threshold=0.20),
    ]
)

if report.status == "HALT":
    raise ValueError(f"Validation failed: {report.failures}")

# Statistical testing
dm_result = dm_test(errors_1, errors_2, h=2, loss="squared")

# Walk-forward with gap
cv = WalkForwardCV(window_type="sliding", window_size=104, gap=2, test_size=1)
```

---

## 6. Differentiation

### vs sktime

| Feature | sktime | temporalcv |
|---------|--------|------------|
| Walk-forward CV | ✅ Good | ✅ + strict temporal safety |
| Leakage detection | ❌ | ✅ Novel |
| DM test | ❌ | ✅ With HAC |

### vs Darts / Nixtla / GluonTS

| Feature | Others | temporalcv |
|---------|--------|------------|
| Focus | Models | Validation |
| Leakage prevention | ❌ | ✅ |
| Statistical testing | ❌ | ✅ |

---

## 7. Benchmark Strategy

### Datasets

| Source | Key Feature |
|--------|-------------|
| M5 (Walmart) | Rich exogenous |
| FRED panels | High persistence |
| GluonTS | Probabilistic standard |

### Novel Metrics

| Metric | Purpose |
|--------|---------|
| MC-SS | Skill on moves only |
| Move-only MAE | Error when target moved |
| Direction Brier | Probabilistic direction |

---

## 8. Reference Files (myga-forecasting-v3)

| File | Module | Pattern |
|------|--------|---------|
| `validation/gates.py` | gates | HALT/PASS/WARN design |
| `evaluation/statistical_tests.py` | tests | HAC, Harvey correction |
| `pipeline/walk_forward.py` | cv | Gap enforcement |
| `features/regimes.py` | regimes | Training-only thresholds |

---

## 9. Julia Porting

### Feasibility: 8/10

| Dimension | Score |
|-----------|-------|
| Dependency Coverage | 9/10 |
| Performance Gain | 8/10 |
| Code Reusability | 6/10 |

### Key Packages

| Python | Julia |
|--------|-------|
| numpy | Arrays |
| scipy.stats | Distributions.jl |
| pandas | DataFrames.jl |
| scikit-learn | MLJ.jl |

### Timeline: 8-12 weeks

---

## 11. Academic Literature

### Essential Papers

| Paper | Key Insight |
|-------|-------------|
| Tashman (2000) | Walk-forward validation |
| Hewamalage et al. (2023) | 6 Evaluation Pitfalls |
| Diebold & Mariano (1995) | DM Test |
| Harvey et al. (1997) | Small-sample correction |

### Essential Books

| Title | Author |
|-------|--------|
| FPP3 | Hyndman & Athanasopoulos |
| Time Series Analysis | Hamilton (1994) |

---

## Key Decisions Made

1. **Package name**: `temporalcv` (technical precision, CV-focused)
2. **Documentation**: Component-based with INDEX hub
3. **Target size**: INDEX ~80 lines, components ~150-300 lines
4. **Tiered approach**: Active plans separate from reference

---

## Bugs to Avoid (from myga-forecasting-v3)

| Bug | What | Fix |
|-----|------|-----|
| #1 | Lags from full series | Compute within splits |
| #2 | No horizon gap | gap >= horizon |
| #3 | Thresholds from full series | Training-only |
| #5 | 1-week volatility | Use training window |
