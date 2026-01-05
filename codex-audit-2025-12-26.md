# temporalcv Comprehensive Audit (2025-12-26)

## Scope and method
- Reviewed core modules, docs, tests, examples, benchmarks, and the demo notebook.
- Ran a docstring completeness scan on public functions/classes/methods (no missing docstrings found).
- Tests and benchmarks were not executed during this audit.

## Executive summary
- Code quality is strong and consistent; publishable with a few targeted fixes.
- Test suite is deep and non-superficial, but lacks true end-to-end data benchmarks and benchmark regression thresholds.
- Documentation is thorough, but there are inconsistencies between SPECIFICATION and implementation for the shuffled target gate.
- Benchmarking and comparative evaluation are the biggest maturity gaps.

## Strengths
- Clear architecture and consistent API patterns.
- Robust multi-layer testing: property-based, Monte Carlo, adversarial, golden reference, and determinism.
- Detailed docstrings with knowledge tiers and citations.
- Guardrails and validation gates are thoughtfully organized.

## Key findings (prioritized)

### 1) Shuffled target gate semantics and spec mismatch (High)
Evidence:
- `SPECIFICATION.md:33` defines the threshold as a p-value, but the implementation uses an improvement ratio.
- `src/temporalcv/gates.py:440` evaluates improvement_ratio against threshold.
- `README.md:73` markets this as a "definitive leakage detector".
- `examples/01_leakage_detection.py:228` sets threshold=0.90 because clean models beat shuffled targets on persistent series.

Risk:
- Users may interpret this as a p-value test and get frequent false HALTs on legitimate signal.
- Documentation, defaults, and examples pull in different directions.

Options:
- Option A: Convert to a true permutation test and compare p-value to alpha.
  Pros: statistically principled and aligned with permutation test guidance [11].
  Cons: requires larger n_shuffles and more compute.
- Option B: Keep effect-size logic but rename and reframe the gate (e.g., "shuffled_effect_size"),
  update defaults (e.g., 0.5 to 0.9), and add calibration guidance.
  Pros: minimal code change and clearer semantics.
  Cons: still heuristic and domain-dependent.
- Option C: Return both effect size and permutation p-value, and let users choose the decision rule.
  Pros: flexible and transparent.
  Cons: more complexity and compute.

### 2) Benchmarking is not end-to-end or reproducible (High)
Evidence:
- `scripts/benchmark_comparison.py` is a microbenchmark and feature check, not an accuracy benchmark.
- `tests/benchmarks/*` measure speed but do not enforce regression budgets or record baselines.
- Benchmark loaders require external data (FRED API, Kaggle M5) and/or sampling/truncation
  (`src/temporalcv/benchmarks/monash.py:116`, `src/temporalcv/benchmarks/monash.py:127`).

Risk:
- Performance claims are not reproducible or comparable to published benchmarks (M4/M5) [8][9].
- Users cannot validate "better than sklearn" claims with real datasets.

Options:
- Option A: Add a benchmark runner that uses official splits and official metrics
  (M4: sMAPE + MASE [7][8], M5: WRMSSE [9]) and publish a results table.
  Pros: credible, comparable, publishable.
  Cons: data size, licensing, and compute cost.
- Option B: Bundle a small, reproducible dataset with fixed splits for CI, plus a script to
  download full datasets for full benchmarks.
  Pros: reproducible and CI-friendly.
  Cons: limited external validity.
- Option C: Add a caching/downloading layer with dataset versioning and hashes, and save
  results with environment metadata.
  Pros: reproducible across machines.
  Cons: more maintenance and dependencies.

### 3) Model comparison flattens multi-series data and uses a single split (Medium)
Evidence:
- `src/temporalcv/compare/runner.py:126` flattens multi-series arrays before computing metrics.
- `run_comparison()` uses a single train/test split.

Risk:
- Metrics are weighted by series length and may not match benchmark rules (M4 uses per-series
  sMAPE and MASE [7][8]).
- Single split increases variance and can mis-rank models.

Options:
- Option A: Compute metrics per series and aggregate according to benchmark rules.
  Pros: correct and comparable.
  Cons: more implementation complexity.
- Option B: Add aggregation modes (per-series mean, length-weighted, benchmark-specific).
  Pros: flexible.
  Cons: more API surface.
- Option C: Keep flattening but emit warnings when n_series > 1 or metadata.official_split is True.
  Pros: low effort.
  Cons: still not comparable.

### 4) Silent failure in WalkForwardCV.get_n_splits (Medium)
Evidence:
- `src/temporalcv/cv.py:697` returns 0 on ValueError.

Risk:
- Misconfiguration can be silently hidden and propagate downstream.

Options:
- Option A: Raise ValueError with context.
  Pros: explicit and safe.
  Cons: could break callers relying on 0.
- Option B: Return 0 but issue a warning.
  Pros: backward compatible.
  Cons: easy to miss.
- Option C: Add a strict flag defaulting to True in v1.0.
  Pros: user control.
  Cons: more API complexity.

### 5) Release metadata is inconsistent (Medium)
Evidence:
- `pyproject.toml:3` sets version to 1.0.0-rc1 but classifier is Alpha.
- `CHANGELOG.md:12` only documents up to 0.4.0.

Risk:
- Packaging ambiguity undermines "publishable package" positioning.

Options:
- Option A: Add a 1.0.0-rc1 section to the changelog and update classifiers to Beta/RC.
  Pros: clear release signal.
  Cons: requires curation.
- Option B: Downgrade version to 0.4.x until RC release is complete.
  Pros: consistent with changelog.
  Cons: less marketing impact.

### 6) Benchmark dataset truncation and sampling reduce comparability (Medium)
Evidence:
- `src/temporalcv/benchmarks/monash.py:127` truncates series to minimum length.
- `src/temporalcv/benchmarks/monash.py:165` defaults to sample_size=100 for M4.

Risk:
- Results do not match official benchmark protocols and can bias comparisons [8].

Options:
- Option A: Preserve per-series length and evaluate per-series metrics.
  Pros: faithful to benchmark design.
  Cons: more data handling complexity.
- Option B: Keep truncation but require explicit user opt-in and mark results as non-comparable.
  Pros: low effort and safer messaging.
  Cons: still non-standard.

### 7) Logging and error reporting use print (Low/Medium)
Evidence:
- `src/temporalcv/compare/runner.py:121` and `src/temporalcv/compare/runner.py:305` print warnings.

Risk:
- Hard to integrate into applications and silence in production.

Options:
- Option A: Use warnings.warn.
  Pros: standard.
  Cons: warnings can be filtered silently.
- Option B: Use a module logger.
  Pros: user-controllable and structured.
  Cons: requires logging configuration.

### 8) Conformal and bootstrap assumptions under dependence (Medium)
Evidence:
- Conformal methods are documented as approximate for time series, but no block or
  prequential conformal is implemented [10].

Risk:
- Users may assume coverage guarantees that do not hold under dependence.

Options:
- Option A: Add block or sliding-window conformal for dependent data.
  Pros: improved validity.
  Cons: more code and compute.
- Option B: Add guardrail checks and stronger warnings (residual autocorrelation,
  coverage diagnostics).
  Pros: safer usage.
  Cons: no methodological improvement.

### 9) Testing gaps and permissive assertions (Low)
Evidence:
- Some edge-case tests accept either exception or success (`tests/test_edge_cases.py`).
- No end-to-end tests that use benchmark loaders in a full pipeline.
- Benchmarks do not enforce regression budgets.

Risk:
- Error-handling regressions and dataset integration issues can slip through.

Options:
- Option A: Add known-answer tests for error conditions and message quality.
  Pros: stronger guarantees.
  Cons: more maintenance.
- Option B: Add a small E2E test using a bundled dataset to run gates, CV, and metrics.
  Pros: true end-to-end coverage.
  Cons: CI time.

### 10) RNG usage consistency (Low)
Evidence:
- `src/temporalcv/conformal.py:646` uses RandomState with fixed seed per call for bootstrap intervals.

Risk:
- Repeated calls yield identical intervals; mixed RNG styles complicate reproducibility guidance.

Options:
- Option A: Standardize on np.random.Generator and allow RNG injection.
  Pros: modern and consistent.
  Cons: minor breaking change.
- Option B: Keep current API but advance internal RNG state per call.
  Pros: minimal API change.
  Cons: slightly less deterministic.

## Unstated or weakly enforced assumptions
- Temporal spacing is treated as uniform; gap and horizon are index-based, not time-based [1][2].
- Feature engineering is assumed to be performed inside each split; global transforms are not detected.
- Many metrics and tests assume finite inputs; NaN is checked, but inf is not consistently validated.
- DM/PT tests assume stationarity and weak dependence of loss differentials [3][4][5][6].
- Conformal methods assume exchangeability or slow drift; this is acknowledged but not enforced [10].
- Regime and move thresholds must be computed on training data only; not enforced by API.
- Financial CV assumes horizon-based label overlap; event-based label spans are not modeled [12].

## Testing and quality assessment
- Strength: multi-layer test strategy (property-based, Monte Carlo, adversarial, golden reference,
  reproducibility) is excellent and aligns with evaluation best practices [13].
- Gaps: microbenchmarks do not enforce regression budgets; limited use of real data; some assertions
  are permissive rather than deterministic.

## Documentation and docstrings
- Docstrings are comprehensive for the public API and consistently styled.
- Tutorials cover leakage, CV, persistence, and uncertainty, but not benchmarks, compare,
  financial CV, or bagging (`docs/tutorials/index.md`).
- Messaging inconsistencies around the shuffled target gate should be resolved.

## Code quality and publishability
- The core codebase is clean, typed, and well tested.
- The main blockers to "publishable package" maturity are benchmarking rigor, spec consistency,
  and release metadata alignment.
- With those addressed, code quality is appropriate for publication.

## Recommended next steps (shortlist)
1. Resolve shuffled target gate semantics (Option A or C above).
2. Ship a reproducible benchmark runner with at least one official dataset and one bundled dataset.
3. Fix release metadata consistency (version, changelog, classifier).
4. Add a small end-to-end test and a benchmark regression policy.
5. Add tutorials for compare, benchmarks, and financial CV.

## References
[1] Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy. International Journal of Forecasting.
[2] Bergmeir, C. and Benitez, J.M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences.
[3] Diebold, F.X. and Mariano, R.S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics.
[4] Harvey, D., Leybourne, S., and Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting.
[5] Newey, W.K. and West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica.
[6] Pesaran, M.H. and Timmermann, A. (1992). A simple nonparametric test of predictive performance. JBES.
[7] Hyndman, R.J. and Koehler, A.B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting.
[8] Makridakis, S. et al. (2018). The M4 Competition. International Journal of Forecasting.
[9] Makridakis, S. et al. (2020). The M5 Competition. International Journal of Forecasting.
[10] Gibbs, I. and Candes, E. (2021). Adaptive conformal inference under distribution shift. NeurIPS.
[11] Phipson, B. and Smyth, G.K. (2010). Permutation P-values Should Never Be Zero. Statistical Applications in Genetics and Molecular Biology.
[12] Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
[13] Hewamalage, H., Bergmeir, C., and Bandara, K. (2023). Forecast evaluation for data scientists. International Journal of Forecasting.
