# temporalcv critique (codex)

## Scope and method
- Reviewed source, tests, and docs under `/home/brandon_behring/Claude/temporalcv`.
- Static review only; no tests, notebooks, or benchmarks were executed.
- Focused on leakage risks, evaluation methodology, benchmarking rigor, documentation accuracy, and test depth.

## What is already strong
- Core modules have detailed docstrings and explicit assumptions, with consistent T1/T2/T3 tagging in code and docs.
- Test suite is deep: property-based CV invariants, Monte Carlo coverage, adversarial tests, and integration flows are present.
- Many functions validate shapes, NaNs, and sample sizes, reducing silent failure risk.

## High-impact issues

### 1) Quickstart and workflow examples teach leakage and a mismatched baseline
**Evidence**:
- `docs/quickstart.md:29-35` builds lag features from the full series (leaky).
- `docs/quickstart.md:74-96` runs walk-forward CV on those leaky features.
- `docs/quickstart.md:111-118` uses a zero baseline for *levels* data (persistence for levels should be last value, not zero).
- `docs/quickstart.md:173-181` applies `compute_move_threshold` to levels instead of changes.
- `docs/quickstart.md:215-238` repeats leaky feature construction in the “Complete Workflow.”

**Why it matters**: This contradicts the package’s core promise (leakage prevention) and can lead users to publish invalid results. It also misuses persistence baselines, which are central to the package’s evaluation logic [R1][R2][R3].

**Options**:
- **Option A: Rewrite quickstart using fold-local feature engineering** (compute lags inside each CV split, with training context for test lags).
  - Pros: Aligns with leakage guidance; safe defaults for users.
  - Cons: Longer example; slightly more complex.
- **Option B: Provide two explicit tracks (levels vs changes)** with separate baselines and metrics.
  - Pros: Removes ambiguity; supports both common workflows.
  - Cons: More doc maintenance, larger surface area.

### 2) Manual API docs are stale and reference non-existent APIs
**Evidence**:
- `docs/api/benchmarks.md:31-55` references `load_fred_series`, `load_m5_sample`, `load_monash_dataset` (not present).
- `docs/api/compare.md:10-66` references `ComparisonRunner` and model names not supported by `StatsforecastAdapter`.

**Why it matters**: A publishable package cannot ship with broken or misleading docs. These pages will cause runtime errors and undermine trust.

**Options**:
- **Option A: Delete or redirect `docs/api/*.md` to Sphinx autodoc**.
  - Pros: Eliminates drift; single source of truth.
  - Cons: Loses custom narrative unless rebuilt elsewhere.
- **Option B: Update the markdown docs to match current APIs** and add doc tests in CI.
  - Pros: Keeps narrative; improves user onboarding.
  - Cons: Requires ongoing maintenance and CI setup.

### 3) Gap/horizon semantics conflict across docs vs spec vs code
**Evidence**:
- `docs/tutorials/walk_forward_cv.md:100-106` and `docs/api/cv.md:172-188` say `gap >= h - 1`.
- `SPECIFICATION.md:61-72` and `CLAUDE.md` require `gap >= horizon`.
- `docs/tutorials/leakage_detection.md:161-167` uses a different gap formula than `gate_temporal_boundary`.

**Why it matters**: Inconsistent guidance around gap is a direct leakage risk; it can also trigger false HALTs. Time-series evaluation depends on correct gap/horizon semantics [R1][R2].

**Options**:
- **Option A: Standardize all docs to the spec (`gap >= horizon`) and explain why**.
  - Pros: Clear, conservative guidance; aligns with code validation.
  - Cons: Might be stricter than some users expect.
- **Option B: Rename `gate_temporal_boundary(gap=...)` to `extra_gap` and document it explicitly**.
  - Pros: Removes ambiguity between “required gap” vs “additional gap.”
  - Cons: API change (but can be backward-compatible via alias).

### 4) Shuffled-target gate is described as a p-value test but implemented as a ratio
**Evidence**:
- `SPECIFICATION.md:31-42` says threshold is a “p-value” for a shuffled test.
- `src/temporalcv/gates.py:405-438` compares an improvement *ratio* against a threshold; no permutation p-value is computed.

**Why it matters**: This is a statistical interpretation error. Users will interpret “p < 0.05” as a calibrated test when it is not, which undermines the claim that the gate is “definitive” [R3].

**Options**:
- **Option A: Compute an empirical permutation p-value** (fraction of shuffles where MAE <= real MAE).
  - Pros: Statistically correct interpretation; preserves 0.05 semantics.
  - Cons: Slightly higher variance for small `n_shuffles`.
- **Option B: Rename parameters and docs to “improvement_ratio_threshold”** and remove p-value language from the spec.
  - Pros: Clearer, minimal code change.
  - Cons: Loses inferential framing; may need new heuristics.

### 5) Benchmarking is not yet “competition-grade” or end-to-end
**Evidence**:
- `src/temporalcv/compare/runner.py:102-140` uses a single train/test split (no walk-forward CV).
- `src/temporalcv/benchmarks/monash.py:127-151` truncates series to min length but still sets `official_split=True`.
- `src/temporalcv/benchmarks/m5.py:123-147` sets `official_split=True` even when `aggregate=True` (not competition-comparable).

**Why it matters**: Benchmark claims are only meaningful if they match published protocols [R4][R5][R6][R7]. Single-split evaluation and truncation break comparability.

**Options**:
- **Option A: Add a benchmark protocol runner** that uses walk-forward evaluation with fixed horizons and documented splits.
  - Pros: Enables fair, reproducible comparisons; aligns with research norms.
  - Cons: More compute; more code.
- **Option B: Mark `official_split=False` when truncating/aggregating** and document comparability limits.
  - Pros: Honest metadata; minimal code change.
  - Cons: Still leaves the evaluation protocol to users.

## Medium-impact issues

### 6) `gate_synthetic_ar1` reuses the same model across CV folds
**Evidence**: `src/temporalcv/gates.py:555-559` fits the same model instance in each split.

**Why it matters**: Stateful or warm-start models can leak state across folds, biasing the gate.

**Options**:
- **Option A: Clone model per fold (like `gate_signal_verification`)**.
  - Pros: Consistent behavior; safer for stateful models.
  - Cons: Slight overhead for models without `clone`.
- **Option B: Accept a `model_factory` callable** and instantiate per fold.
  - Pros: Explicit; works without sklearn.
  - Cons: API expansion.

### 7) `BootstrapUncertainty` uses fixed seeding and omits time-series assumptions
**Evidence**: `src/temporalcv/conformal.py:511-569` defaults `random_state=42` and uses residual bootstrap without stating i.i.d. assumptions.

**Why it matters**: Default determinism conflicts with the project seed policy, and residual bootstrap can misrepresent uncertainty for autocorrelated data [R9].

**Options**:
- **Option A: Default `random_state=None` and switch to `np.random.default_rng`**.
  - Pros: Aligns with project seed policy; modern RNG.
  - Cons: Non-deterministic defaults.
- **Option B: Add a block bootstrap option or warning about dependence**.
  - Pros: Statistically safer for time series; consistent with project’s bootstrap expertise.
  - Cons: Additional implementation and docs.

### 8) API docs for gates are missing important parameters
**Evidence**: `docs/api/gates.md:70-120` omits `n_cv_splits`, `permutation`, and `block_size` for `gate_signal_verification`, and omits `n_cv_splits` for `gate_synthetic_ar1`.

**Why it matters**: Users cannot discover or correctly tune key parameters, undermining reproducibility.

**Options**:
- **Option A: Update the markdown to match signatures.**
  - Pros: Simple fix.
  - Cons: Ongoing maintenance.
- **Option B: Generate docs from autodoc only.**
  - Pros: Single source of truth.
  - Cons: Less narrative control.

## Unstated assumptions and overlooked risks
- **Temporal ordering and regular sampling are assumed but not checked**. `WalkForwardCV` and gates operate on index positions; irregular time steps can invalidate horizon logic.
- **Targets are often assumed to be changes/returns**, but only some APIs enforce this (`compute_move_threshold`). Quickstart and some metrics do not guard against “level” misuse.
- **PT test assumes independence** and does not HAC-adjust for serial correlation, which can be optimistic for h > 1 [R2].
- **Benchmark comparability depends on official splits and horizons**, but truncation/aggregation can silently break that [R4][R5][R6][R7].

## Testing and benchmarking gaps
- **Notebook and example scripts are not tested in CI**, so they may drift or break quietly. This is at odds with “publishable package” expectations [R8].
- **Integration coverage is loose in some critical areas** (e.g., adaptive conformal coverage passes at 60% for a 90% target; `tests/test_integration.py:187-189`). This is too weak to catch regressions.
- **Optional dependency loaders are barely tested** (mostly error-path tests for FRED/M5). There are no tests for M3/M4/Monash or GluonTS loaders when dependencies are present.
- **No true end-to-end benchmark test** that runs: dataset loader -> adapter -> evaluation -> gates -> statistical tests on a real (or pinned) dataset.

## Recommendations (prioritized)
1) **Fix doc drift and leakage in quickstart/examples** (`docs/quickstart.md`, `docs/api/*.md`).
2) **Normalize gap/horizon semantics across docs and gates** and clarify “gap” vs “extra gap.”
3) **Calibrate the shuffled-target gate** (either compute p-values or remove p-value language).
4) **Upgrade benchmarking to a walk-forward protocol** and mark comparability explicitly in metadata.
5) **Add CI coverage for notebooks/examples** (smoke tests via `nbval` or `papermill`).
6) **Tighten integration test assertions** to catch coverage and leakage regressions.

## References
- [R1] Tashman, L.J. (2000). *Out-of-sample tests of forecasting accuracy: an analysis and review*. DOI: 10.1016/S0169-2070(00)00065-0
- [R2] Bergmeir, C., & Benitez, J.M. (2012). *On the use of cross-validation for time series predictor evaluation*. DOI: 10.1016/j.ins.2011.12.028
- [R3] Hewamalage, H., Bergmeir, C., & Bandara, K. (2022). *Forecast evaluation for data scientists: common pitfalls and best practices*. DOI: 10.1007/s10618-022-00894-5
- [R4] Makridakis, S., & Hibon, M. (2000). *The M3-Competition: results, conclusions and implications*. DOI: 10.1016/S0169-2070(00)00057-1
- [R5] Makridakis, S., et al. (2018). *The M4 Competition: Results, findings, conclusion and way forward*. DOI: 10.1016/j.ijforecast.2018.06.001
- [R6] Makridakis, S., et al. (2022). *M5 accuracy competition: Results, findings, and conclusions*. DOI: 10.1016/j.ijforecast.2021.11.013
- [R7] Monash Time Series Forecasting Repository. https://forecastingdata.org/
- [R8] Breck, E., et al. (2017). *The ML Test Score: A rubric for ML production systems*. https://arxiv.org/abs/1706.02168
- [R9] Politis, D.N., & Romano, J.P. (1994). *The Stationary Bootstrap*. DOI: 10.1080/01621459.1994.10476870
