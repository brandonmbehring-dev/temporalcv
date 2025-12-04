# Implementation Roadmap

**Purpose**: Phase timeline for temporalcv development.

---

## Phase Summary

| Phase | Weeks | Focus | Deliverable |
|-------|-------|-------|-------------|
| 0 | Done | Repository setup | pyproject.toml, CI, CLAUDE.md |
| 1 | 1-4 | Core gates + tests | `pip install temporalcv` with validation |
| 2 | 5-8 | Walk-forward CV | Temporal splitters |
| 3 | 9-12 | High-persistence | Regime handling, move metrics |
| 4 | 13-16 | Uncertainty | Conformal, bagging |
| 5 | 17-20 | Benchmarks | M5, FRED, event metrics |
| 6 | 21-28 | Julia port | TemporalCV.jl |

---

## Phase 1: Core Foundation (Weeks 1-4)

**Goal**: Standalone leakage detection + statistical testing

| Module | Files | Reference |
|--------|-------|-----------|
| `temporalcv.gates` | `gates.py` | myga: `validation/gates.py` |
| `temporalcv.tests` | `statistical_tests.py` | myga: `evaluation/statistical_tests.py` |
| `temporalcv.metrics` | `metrics.py` | myga: `evaluation/metrics.py` |

**Dependencies**: numpy, scipy only

---

## Phase 2: Walk-Forward Infrastructure (Weeks 5-8)

**Goal**: Temporal CV with leakage prevention

| Module | Files | Reference |
|--------|-------|-----------|
| `temporalcv.cv` | `cv.py` | myga: `pipeline/walk_forward.py` |
| `temporalcv.splitters` | `splitters.py` | sklearn-compatible |

**Key feature**: Gap parameter enforcement for h-step horizons

**Dependencies**: + pandas (optional), scikit-learn (optional)

---

## Phase 3: High-Persistence Handling (Weeks 9-12)

**Goal**: Specialized tools for "imbalanced" time series

| Module | Files | Reference |
|--------|-------|-----------|
| `temporalcv.regimes` | `regimes.py` | myga: `features/regimes.py` |
| `temporalcv.persistence` | `persistence.py` | New |

**Novel features**:
- Changes-based volatility (not levels)
- Move threshold for fair persistence baseline
- Training-only threshold computation

---

## Phase 4: Uncertainty + Ensemble (Weeks 13-16)

**Goal**: Full framework completion

| Module | Files | Reference |
|--------|-------|-----------|
| `temporalcv.conformal` | `conformal.py` | myga: `evaluation/uncertainty.py` |
| `temporalcv.bagging` | `bagging/` | myga: `models/bagging/*` |

---

## Phase 5: Benchmark Suite (Weeks 17-20)

**Goal**: Validation against public benchmarks

| Module | Source |
|--------|--------|
| `temporalcv.benchmarks` | New (loaders for M3/M4/M5/FRED) |
| `temporalcv.metrics.event` | New (MC-SS, move-only MAE, Brier) |
| `temporalcv.compare` | New (cross-package comparison) |

**Novel metrics**:
- MC-SS (Move-Conditional Skill Score)
- Move-only MAE
- Direction Brier

---

## Phase 6: Julia Port (Weeks 21-28)

**Goal**: Native Julia implementation

See `julia_strategy.md` for detailed analysis.

---

## Deferred Items

**Phase 2+**:
- Property-based tests (hypothesis)
- Seed-sweep tests
- Windows CI job

**Phase 3+**:
- Performance budgets
- Memory/time limits
- Numba optimization

**Phase 5+**:
- MLflow/Airflow adapters
- Drift monitoring hooks

**v1.0+**:
- Deprecation policy
- API contract tests
- Plugin interface

---

## Bugs to Avoid (from myga-forecasting-v3)

| Bug | What | Fix |
|-----|------|-----|
| #1 | Lags from full series | Compute within splits |
| #2 | No horizon gap | gap >= horizon enforced |
| #3 | Thresholds from full series | Training-only |
| #5 | 1-week volatility window | Use training window |
