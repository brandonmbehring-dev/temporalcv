# Julia Porting Strategy

**Purpose**: Analysis for TemporalCV.jl native implementation.

---

## Feasibility Summary

| Dimension | Score | Notes |
|-----------|-------|-------|
| Overall Feasibility | 8/10 | Achievable in 8-12 weeks |
| Dependency Coverage | 9/10 | 95%+ has Julia equivalent |
| Performance Gain | 8/10 | 2-5x loops, 10-30x bootstrap |
| Code Reusability | 6/10 | 40% direct port, 60% redesign |

---

## Dependency Mapping

| Python | Julia | Status |
|--------|-------|--------|
| numpy | Arrays + LinearAlgebra | ✅ Trivial |
| scipy.stats | Distributions.jl | ✅ Excellent |
| scipy.optimize | Optimization.jl | ✅ Good |
| pandas | DataFrames.jl + TimeSeries.jl | 🟡 Biggest challenge |
| scikit-learn | MLJ.jl | 🟡 API redesign |
| statsmodels | StateSpaceModels.jl | ✅ Good |
| numba | Julia native JIT | ✅ Superior |

---

## Three Categories of Effort

### Easy (70% effort savings)

- Statistical tests (HAC, DM, PT) — 1:1 translation
- Bootstrap algorithms — faster in Julia
- ARIMA models — StateSpaceModels.jl
- Conformal prediction — ConformalPrediction.jl
- Metrics — trivial
- Test suite — Test.jl simpler than pytest

### Requires Redesign (2-3 weeks)

- DataFrame operations → TimeSeries.jl paradigm
- Model API → MLJ interface (multiple dispatch)
- Walk-forward CV → manual with temporal safety
- Feature engineering → MLJ `@pipeline`

### No Julia Equivalent

- GridSearchCV → MLJ `tuned_model()` (better design)
- pandas `.iloc` → DataFrames slicing (verbose)
- pytest fixtures → Test.jl `@testset`

---

## Key Technical Challenges

### DataFrame Indexing

```python
# Python - elegant
X_train = X.iloc[train_start_idx:train_end_idx]
```

```julia
# Julia - more verbose
X_train = X[train_start_idx:train_end_idx, :]
```

### scikit-learn → MLJ

```python
# Python: OOP
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

```julia
# Julia: Multiple dispatch
mach = machine(model, X_train, y_train)
fit!(mach)
preds = predict(mach, X_test)
```

---

## Julia Advantages

1. **MarSwitching.jl** — 6.7x faster than Python statsmodels
2. **No Numba needed** — Native JIT superior for bootstrap
3. **SciML ecosystem** — Autodiff, scientific computing
4. **Parallel computing** — `Threads.@threads` cleaner

---

## Packages to Use Directly

| Function | Julia Package |
|----------|---------------|
| ARIMA | StateSpaceModels.jl |
| Ridge/RF | MLJ.jl |
| DM test | ForecastEval.jl |
| Time series data | TSFrames.jl |
| Conformal | ConformalPrediction.jl |
| Regime switching | MarSwitching.jl |

---

## Must Build from Scratch

| Component | Effort |
|-----------|--------|
| Walk-forward validator | Medium |
| Shuffled target test | Low |
| Synthetic AR(1) | Low |
| Temporal split | Low |

---

## Timeline

| Phase | Weeks | Scope |
|-------|-------|-------|
| Foundation | 1-2 | Package structure, tests, metrics |
| Core | 3-4 | Models (MLJ), walk-forward CV |
| Features | 5-6 | Lag construction, engineering |
| Validation | 7-8 | Test suite, docs, registry |

**Total**: 8-12 weeks for full port
