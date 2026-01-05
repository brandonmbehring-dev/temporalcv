# Temporal Validation Ecosystem — Execution Plan

**Created**: 2025-12-26
**Status**: APPROVED — Ready for implementation
**Execution Order**: MCP Setup → Phase 2 Julia work

---

## Phase 0: MCP Setup (NEXT)

### Task 0.1: Create MCP Configuration for temporalcv

**File**: `/home/brandon_behring/Claude/temporalcv/.mcp.json`

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/brandon_behring/Claude/temporalcv"
      ],
      "description": "Direct filesystem access to temporalcv project"
    },
    "research-kb": {
      "command": "python",
      "args": ["-m", "research_kb_mcp.server"],
      "cwd": "/home/brandon_behring/Claude/research-kb",
      "description": "Search statistical/ML literature - 120 papers + 76 textbooks with FTS+vector+graph"
    }
  }
}
```

**Effort**: 5 minutes

### Task 0.2: Verify MCP Connectivity

1. Restart Claude Code in temporalcv directory
2. Test research-kb search: Query "Diebold-Mariano test"
3. Verify filesystem access works

**Effort**: 5 minutes

### Research-KB Knowledge Available for temporalcv

| Domain | Examples | Use Case |
|--------|----------|----------|
| Statistical Testing | DM test, HAC variance, p-value computation | Validate gate implementations |
| Time-Series | AR(1), stationarity, autocorrelation | Conformal/bootstrap theory |
| Validation | Cross-validation, temporal splits | CV splitter design |
| Leakage Detection | 10 bug categories, feature selection | Gate threshold justification |

**Search Tools**:
- `research_kb_search(query, context_type="auditing")` - Hybrid FTS+vector
- `research_kb_graph_neighborhood(concept)` - Knowledge graph traversal
- `research_kb_list_concepts(type="METHOD")` - Browse statistical methods

---

## Phase 1: Python Audit Fixes ✅ COMPLETE

All 6 tasks completed:
- [x] Task 1.1: Two-mode permutation test
- [x] Task 1.2: Strict get_n_splits
- [x] Task 1.3: Release metadata
- [x] Task 1.4: Logging in runner.py
- [x] Task 1.5: Per-series aggregation
- [x] Task 1.6: Coverage diagnostics

---

## Decisions Finalized (/iterate session)

| Decision | Choice |
|----------|--------|
| Permutation test | Two modes: `method="effect_size"` (n=5) + `method="permutation"` (n=100, default) |
| Julia conformal | Standalone core first, MLJ wrapper in Sprint 7 |
| Stationarity tests | Implement ADF/KPSS/PP from scratch, unified `StationarityResult` API |
| P3 features | All in order listed (full parity) |
| Documentation | Default Documenter.jl + GitHub Pages |
| Execution order | Python first, then Julia |

---

## Phase 1: Python Audit Fixes (Days 1-7)

### Task 1.1: Shuffled Target → Two-Mode Permutation Test
**Files**: `src/temporalcv/gates.py`, `tests/test_gates.py`

```python
def gate_shuffled_target(
    ...,
    method: Literal["effect_size", "permutation"] = "permutation",
    n_shuffles: int | None = None,  # 5 for effect_size, 100 for permutation
    alpha: float = 0.05,  # only used in permutation mode
) -> GateResult:
```

**Changes**:
1. Add `method` parameter with two modes
2. `effect_size` mode: Current behavior (fast, returns improvement ratio)
3. `permutation` mode: True p-value via `(1 + sum(shuffled >= observed)) / (1 + n_shuffles)`
4. Update default from `threshold=0.05` to `alpha=0.05` in permutation mode
5. Update SPECIFICATION.md Section 1.2
6. Update `examples/01_leakage_detection.py`

**Effort**: 5h

### Task 1.2: Fix Silent Failure in get_n_splits
**File**: `src/temporalcv/cv.py:697`

```python
def get_n_splits(self, X=None, y=None, groups=None, *, strict: bool = True) -> int:
    try:
        ...
    except ValueError as e:
        if strict:
            raise ValueError(f"Cannot compute n_splits: {e}") from e
        return 0
```

**Effort**: 1h

### Task 1.3: Update Release Metadata
**Files**: `CHANGELOG.md`, `pyproject.toml`

1. Add `[1.0.0-rc1]` section to CHANGELOG
2. Update classifier from "Alpha" to "Beta"
3. Document breaking changes (shuffled target default)

**Effort**: 1h

### Task 1.4: Replace print with logging
**File**: `src/temporalcv/compare/runner.py`

```python
import logging
logger = logging.getLogger(__name__)

# Replace: print(f"Warning: ...")
# With: logger.warning(f"...")
```

**Effort**: 2h

### Task 1.5: Per-Series Aggregation
**File**: `src/temporalcv/compare/runner.py`

Add `aggregation_mode` parameter:
- `"flatten"` (current default)
- `"per_series_mean"`
- `"per_series_median"`

**Effort**: 4h

### Task 1.6: Conformal Coverage Diagnostics
**File**: `src/temporalcv/conformal.py`

Add `compute_coverage_diagnostics()`:
- Empirical coverage by time window
- Coverage by regime (if regimes provided)
- Warning if coverage < target - 0.05

**Effort**: 3h

---

## Phase 2: Julia Conformal & Bootstrap (Days 8-21)

### Task 2.1: Conformal Submodule Structure
**Files**:
- `src/conformal/Conformal.jl`
- `src/conformal/split.jl`
- `src/conformal/adaptive.jl`

```julia
# src/conformal/Conformal.jl
module Conformal

using ..TemporalValidation: DEFAULT_ALPHA, DEFAULT_CALIBRATION_FRACTION

include("split.jl")
include("adaptive.jl")

export SplitConformalPredictor, AdaptiveConformalPredictor
export calibrate!, predict_interval, coverage

end
```

**Effort**: 1h

### Task 2.2: SplitConformalPredictor
**File**: `src/conformal/split.jl`

```julia
struct SplitConformalPredictor{M}
    model::M
    quantile::Float64
    scores::Vector{Float64}
    alpha::Float64
end

function calibrate!(cp::SplitConformalPredictor, X_cal, y_cal)
    # Compute nonconformity scores
    # Calculate quantile: ceil((n+1)(1-α))/n
end

function predict_interval(cp::SplitConformalPredictor, X_test)
    # Return (lower, upper) bounds
end
```

**Effort**: 4h

### Task 2.3: AdaptiveConformalPredictor
**File**: `src/conformal/adaptive.jl`

Gibbs & Candès (2021) online update rule:
```julia
struct AdaptiveConformalPredictor{M}
    model::M
    quantile::Float64
    gamma::Float64  # learning rate, default 0.1
    alpha::Float64
end

function update!(acp::AdaptiveConformalPredictor, y_true, interval)
    covered = interval[1] <= y_true <= interval[2]
    if covered
        acp.quantile -= acp.gamma * acp.alpha
    else
        acp.quantile += acp.gamma * (1 - acp.alpha)
    end
end
```

**Effort**: 4h

### Task 2.4: Bagging Submodule Structure
**Files**:
- `src/bagging/Bagging.jl`
- `src/bagging/block_bootstrap.jl`
- `src/bagging/stationary_bootstrap.jl`
- `src/bagging/strategies.jl`

**Effort**: 1h

### Task 2.5: MovingBlockBootstrap
**File**: `src/bagging/block_bootstrap.jl`

Kunsch (1989) block bootstrap:
```julia
struct MovingBlockBootstrap
    block_size::Int  # default: ceil(n^(1/3))
    n_bootstrap::Int
    random_state::Union{Int, Nothing}
end

function bootstrap_sample(mbb::MovingBlockBootstrap, data::AbstractVector)
    # Sample blocks with replacement
    # Concatenate to original length
end
```

**Effort**: 4h

### Task 2.6: StationaryBootstrap
**File**: `src/bagging/stationary_bootstrap.jl`

Politis & Romano (1994):
```julia
struct StationaryBootstrap
    mean_block_size::Float64  # expected block length
    n_bootstrap::Int
    random_state::Union{Int, Nothing}
end

function bootstrap_sample(sb::StationaryBootstrap, data::AbstractVector)
    # Geometric distribution for block lengths
end
```

**Effort**: 4h

### Task 2.7: TimeSeriesBagger
**File**: `src/bagging/Bagging.jl`

```julia
struct TimeSeriesBagger{M, S<:BootstrapStrategy}
    base_model::M
    strategy::S
    n_estimators::Int
    models::Vector{M}
end

function fit!(bagger::TimeSeriesBagger, X, y)
    for i in 1:bagger.n_estimators
        X_boot, y_boot = bootstrap_sample(bagger.strategy, X, y)
        fit!(bagger.models[i], X_boot, y_boot)
    end
end
```

**Effort**: 4h

---

## Phase 3: Julia Financial CV & Stationarity (Days 22-35)

### Task 3.1: PurgedKFold
**File**: `src/cv/financial.jl`

Lopez de Prado (2018) purged K-fold:
```julia
struct PurgedKFold <: MLJBase.ResamplingStrategy
    n_folds::Int
    purge_gap::Int
    embargo_pct::Float64
end
```

**Effort**: 4h

### Task 3.2: CombinatorialPurgedCV
**File**: `src/cv/financial.jl`

CPCV for combinatorial path testing.

**Effort**: 4h

### Task 3.3: PurgedWalkForward
**File**: `src/cv/financial.jl`

Walk-forward with purge gap.

**Effort**: 3h

### Task 3.4: Stationarity Tests (from scratch)
**File**: `src/stationarity.jl`

```julia
struct StationarityResult
    test_name::Symbol  # :ADF, :KPSS, :PP
    statistic::Float64
    pvalue::Float64
    is_stationary::Bool
    critical_values::Dict{String, Float64}
    n_lags::Int
end

function adf_test(y::AbstractVector; maxlag=nothing, regression=:c)
function kpss_test(y::AbstractVector; regression=:c, nlags=nothing)
function pp_test(y::AbstractVector; regression=:c)
function check_stationarity(y; tests=[:ADF, :KPSS])
```

**Effort**: 10h (implementing from scratch)

### Task 3.5: Lag Selection
**File**: `src/lag_selection.jl`

```julia
function select_lag_pacf(y; alpha=0.05, max_lag=nothing)
function select_lag_aic(y; max_lag=nothing)
function select_lag_bic(y; max_lag=nothing)
```

**Effort**: 3h

### Task 3.6: Changepoint Detection
**File**: `src/changepoint.jl`

```julia
function detect_variance_changepoints(y; min_segment=10)
function detect_changepoints_pelt(y; penalty=:bic)  # optional Ruptures.jl
```

**Effort**: 4h

### Task 3.7: Guardrails
**File**: `src/guardrails.jl`

```julia
function check_suspicious_improvement(baseline_error, model_error; threshold=0.20)
function check_minimum_sample_size(n; min_n=30, test=:dm)
function check_residual_autocorrelation(residuals; max_lag=10)
function run_all_guardrails(...)
```

**Effort**: 4h

---

## Phase 4: Julia P3 Metrics (Days 36-49)

### Task 4.1-4.5: Metric Modules
| Task | File | Functions | Effort |
|------|------|-----------|--------|
| 4.1 | `metrics/quantile.jl` | pinball_loss, crps, interval_score, winkler_score | 3h |
| 4.2 | `metrics/financial.jl` | sharpe_ratio, max_drawdown, profit_factor, calmar_ratio | 3h |
| 4.3 | `metrics/asymmetric.jl` | linex_loss, directional_loss, huber_loss | 3h |
| 4.4 | `metrics/event.jl` | brier_score, pr_auc, direction_brier | 3h |
| 4.5 | `metrics/volatility_weighted.jl` | vol_normalized_mae, vol_weighted_mae | 3h |

---

## Phase 5: Julia Compare & Benchmarks (Days 50-56)

### Task 5.1: ForecastAdapter Protocol
**File**: `src/compare/base.jl`

```julia
abstract type ForecastAdapter end

function fit!(adapter::ForecastAdapter, X, y) end
function predict(adapter::ForecastAdapter, X) end
```

**Effort**: 3h

### Task 5.2: Comparison Runner
**File**: `src/compare/runner.jl`

**Effort**: 4h

### Task 5.3: Benchmark Loaders
**Files**: `src/benchmarks/*.jl`

**Effort**: 6h

---

## Phase 6: Documentation & Release (Days 57-70)

### Task 6.1: Documenter.jl Setup
- `docs/make.jl`
- `docs/src/index.md`
- GitHub Pages deployment

**Effort**: 4h

### Task 6.2: API Documentation
- All exported functions documented

**Effort**: 6h

### Task 6.3: Tutorials
- Gates tutorial
- CV tutorial
- Conformal tutorial

**Effort**: 8h

### Task 6.4: MLJ ConformalModel Wrapper
**File**: `src/conformal/mlj_wrapper.jl`

```julia
struct ConformalModel{M} <: MLJModelInterface.Probabilistic
    model::M
    alpha::Float64
end
```

**Effort**: 2h

### Task 6.5: Examples & README
**Effort**: 6h

### Task 6.6: Registry Submission
- @JuliaRegistrator
- Discourse post
- Blog post

**Effort**: 7h

---

## Critical File Paths

### Python (Phase 1)
- `src/temporalcv/gates.py` — shuffled target fix
- `src/temporalcv/cv.py` — strict failure mode
- `src/temporalcv/compare/runner.py` — logging + aggregation
- `src/temporalcv/conformal.py` — coverage diagnostics
- `SPECIFICATION.md` — update Section 1.2
- `CHANGELOG.md` — add 1.0.0-rc1 section

### Julia (Phases 2-6)
- `src/conformal/Conformal.jl` — new submodule
- `src/conformal/split.jl` — SplitConformalPredictor
- `src/conformal/adaptive.jl` — AdaptiveConformalPredictor
- `src/bagging/Bagging.jl` — new submodule
- `src/cv/financial.jl` — PurgedKFold, CPCV
- `src/stationarity.jl` — ADF, KPSS, PP from scratch
- `src/guardrails.jl` — unified validation

---

## Test Targets

| Phase | New Tests | Running Total |
|-------|-----------|---------------|
| Python Phase 1 | +50 | 1,391 |
| Julia Phase 2 | +200 | 1,931 |
| Julia Phase 3 | +250 | 2,181 |
| Julia Phase 4 | +150 | 2,331 |
| Julia Phase 5 | +100 | 2,431 |
| **Final** | | **~3,800 combined** |
