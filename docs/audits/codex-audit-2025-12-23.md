# temporalcv audit critique (codex)

## Scope and method
- Reviewed core modules in `src/temporalcv/` (gates, cv, statistical_tests, conformal, persistence, metrics, benchmarks, compare, diagnostics, inference).
- Reviewed `SPECIFICATION.md`, `CLAUDE.md`, `README.md`, tutorials, and plans under `docs/`.
- Reviewed examples under `examples/` and notebook `notebooks/demo.ipynb` for API correctness and methodology.
- Reviewed test suite structure and CI workflows (unit, property-based, Monte Carlo, integration).
- Static analysis only; no tests, notebooks, or benchmarks executed.

## High-impact findings (correctness and methodology)

1) Compare DM test is effectively broken by wrong argument names.
- Evidence: `src/temporalcv/compare/runner.py:226-233` calls `dm_test(errors1=..., errors2=..., horizon=...)`, but the function expects `errors_1`, `errors_2`, and `h`.
- Impact: DM results in comparisons are always an error payload, yet tests do not assert this. This silently disables statistical inference in benchmarks.
- Fix priority: High (small change, big impact).

2) Shuffled target gate is labeled as a definitive leakage detector but functions as a signal detection heuristic.
- Evidence: `src/temporalcv/gates.py:242-323` and `README.md:71-84` describe it as definitive; implementation computes an improvement ratio vs a fully shuffled target, not a permutation p-value.
- Why it matters: For time series with legitimate predictive signal (especially high persistence), a model should beat a fully shuffled target. That is expected, not necessarily leakage. This risks false HALT outcomes or forces users to inflate thresholds (see `examples/01_leakage_detection.py` and notebook use of 0.95 thresholds).
- Methodology risk: Random shuffling breaks autocorrelation; for time series, nulls should preserve dependence (block or circular permutations) [1][2][4][5][13].

3) Gap and horizon semantics are inconsistent and not enforced.
- Evidence: `SPECIFICATION.md:208-215` requires gap >= horizon, while `docs/tutorials/walk_forward_cv.md:100-116` recommends gap >= h-1 and `docs/tutorials/leakage_detection.md:161-167` shows a different formula. `docs/knowledge/assumptions.md:78-82` claims enforcement that does not exist in `src/temporalcv/cv.py`.
- Impact: Users can follow docs and still introduce target leakage for multi-step forecasting, undermining the core promise of temporal safety [1][2].

4) Benchmark loaders do not preserve official train/test splits and truncate series.
- Evidence: `src/temporalcv/benchmarks/monash.py:120-133` and `src/temporalcv/benchmarks/gluonts.py:73-86` truncate to the minimum length and invent a split, ignoring documented splits.
- Impact: Results are not comparable to published benchmarks, and truncation can distort distributional properties [14][15][16][17].

5) Spec and assumptions drift on conformal minimum sample sizes and CV enforcement.
- Evidence: `SPECIFICATION.md:208-215` says conformal min n=50; `docs/knowledge/assumptions.md:55-59` says n>=30; code enforces only n>=10 in `src/temporalcv/conformal.py:259-273`.
- Impact: The project's "single source of truth" is violated and users get inconsistent guidance on reliability.

6) Baseline definitions are inconsistent across modules and docs.
- Evidence: `src/temporalcv/persistence.py:323-333` assumes persistence predicts 0 change; `src/temporalcv/compare/base.py:367-405` uses last value (levels). Quickstart uses level series but compares to zero-change baseline (`docs/quickstart.md:101-125`).
- Impact: Comparisons and tests can be invalid if users mix level and change targets; this is a high-risk source of methodological error.

7) Tutorials and quickstart include feature engineering patterns that contradict leakage guidance.
- Evidence: `docs/tutorials/walk_forward_cv.md:182-187` builds lag features with `np.roll` on the full series and asserts this is "proper"; `docs/quickstart.md:17-33` constructs lag features before any split.
- Impact: This normalizes full-series feature construction, a known leakage vector when rolling stats, scaling, or target-based transforms are added later [1][2][3].

## Medium-impact issues (consistency, API drift, and silent failures)

- Docs and plans reference APIs that do not exist or have wrong signatures.
  - Examples: `src/temporalcv/__init__.py:20-25` and `docs/plans/reference/api_design.md:9-33` show `run_gates(model=..., gates=[gate_shuffled_target(...)])`, but `run_gates` only accepts `List[GateResult]`.
  - `docs/plans/reference/api_design.md:17, 101-108` references `temporalcv.tests` and splitters not implemented; CLI examples are aspirational.
  - `docs/tutorials/leakage_detection.md:64-69` uses `result.details['real_mae']` and `['shuffled_mae']`, but code emits `mae_real` and `mae_shuffled_avg`.

- "NEVER FAIL SILENTLY" is violated in several places.
  - `src/temporalcv/statistical_tests.py:347-358` returns `pvalue=1.0` and `statistic=nan` when variance is degenerate, with no warning.
  - `src/temporalcv/persistence.py:354-369` and following can return NaNs for empty or no-move cases without warning.
  - `gate_shuffled_target` returns 0.0 MAE if no splits are possible (`src/temporalcv/gates.py:278-287`), which can create false PASS.

- PT test effective sample size is unchecked.
  - `src/temporalcv/statistical_tests.py:535-555` filters zeros for 2-class mode but does not validate `n_effective`, potentially far below the stated minimum. P-values can be meaningless [8].

- WalkForwardCV does not enforce horizon constraints.
  - `src/temporalcv/cv.py` does not accept `horizon`; enforcement relies on users manually setting gap, but docs imply enforcement [1][2].

- gate_shuffled_target reuses the same model instance across folds and shuffles.
  - `src/temporalcv/gates.py:278-295` calls `model.fit` repeatedly without cloning; stateful or warm-start models can leak across shuffles, biasing the gate.

- PR-AUC implementation uses trapezoidal integration, not average precision, and this difference is not emphasized.
  - `src/temporalcv/metrics/event.py:520-543` computes area via trapezoidal rule; `sklearn` users may assume average precision, which can differ materially [18].

- README overstates conformal guarantees for time series.
  - `README.md:81-84` states "coverage guarantee without distributional assumptions," but exchangeability is violated in time series. Code warns, but README does not [10][11][12].

- Seed policy drift.
  - `CLAUDE.md` declares `random_state=None` defaults, but bagging and bootstrap defaults are fixed at 42 (`src/temporalcv/bagging/base.py:160-203`, `src/temporalcv/bagging/__init__.py`). This is inconsistent with the stated policy.

## Benchmarking and end-to-end validation gaps

- Benchmark loaders are not aligned with official splits or dataset metadata (Monash, GluonTS). This undermines comparability to published results [14][15][16][17].
- `run_benchmark_suite` uses a single train/test split per dataset; it does not enforce gap/horizon, run leakage gates, or perform end-to-end temporal validation.
- No published benchmark results or reproducible scripts are present; "Benchmark Strategy" includes non-existent functions (`docs/plans/reference/benchmark_strategy.md:72-89`).
- Optional adapters (e.g., statsforecast) are not tested in CI, so benchmark paths are unverified.

## Tests, notebooks, and examples (quality and gaps)

Strengths:
- Property-based and Monte Carlo tests exist for CV invariants and statistical tests.
- Dedicated CI workflows for regular and nightly (Monte Carlo/slow) tests.

Gaps:
- No test asserts that compare DM tests actually run (current bug is untested).
- No tests for dataset loaders that require optional dependencies or external data (Monash, GluonTS, FRED), so benchmark integrity is unverified.
- No tests for doc examples or the notebook (no doctest/nbval), allowing drift.
- No tests for irregular timestamps, missing values, or change-target alignment edge cases.
- Conformal tests are i.i.d. only; no time-series coverage tests for autocorrelation or distribution shift.

## Unstated or weakly enforced assumptions

- Temporal ordering and equal spacing are assumed across modules; no validation exists for timestamps or irregular intervals.
- No NaN handling in most core functions; NaNs can silently propagate.
- Stationarity and weak dependence are assumed for HAC variance and block bootstrap methods [9][4][5][13].
- Gaussian innovations are assumed for AR(1) theoretical bounds; heavy tails or heteroskedasticity can invalidate bounds.
- Exchangeability is assumed for split conformal; time series violate this [10][11].
- Models are assumed to be reset on `fit()`; gates reuse the same instance across splits/shuffles.

## Recommendations with options (pros and cons)

1) Fix compare DM test path immediately.
- Option A: Correct parameter names and add a unit test in `tests/test_compare.py`.
  - Pros: Restores statistical inference; small change.
  - Cons: None.

2) Redesign or reframe the shuffled target gate.
- Option A: Use block or circular permutation and compute an empirical p-value (time-series-safe null) [4][5].
  - Pros: Reduces false HALT for legitimate signal; more defensible.
  - Cons: More compute; requires block-size tuning.
- Option B: Split into two gates: `gate_signal_presence` (shuffled target) and `gate_leakage_specific` (feature leakage checks).
  - Pros: Clear semantics; avoids overclaiming.
  - Cons: Adds surface area.

3) Standardize gap/horizon semantics and enforce them.
- Option A: Add `horizon` parameter to `WalkForwardCV` and auto-set/validate gap.
  - Pros: Prevents common leakage; aligns with spec and assumptions.
  - Cons: API change; needs migration notes.
- Option B: Keep API but add helper functions (e.g., `align_target_for_horizon`) and warnings when gap < horizon.
  - Pros: Backwards compatible; reduces user error.
  - Cons: Enforcement still optional.

4) Make benchmarks comparable to official splits.
- Option A: Load official train/test splits from Monash and GluonTS; do not truncate to min length.
  - Pros: Benchmark fidelity; reproducibility.
  - Cons: Requires per-series handling and more complex dataset container.
- Option B: Keep current loaders but label them as "approximate" and add explicit warnings in loader docstrings.
  - Pros: Low effort.
  - Cons: Benchmarks remain non-comparable.

5) Align spec/assumptions with code for minimum sample sizes and error handling.
- Option A: Update code to enforce spec minima (e.g., conformal n>=50) and add warnings for smaller n.
  - Pros: Consistent reliability policy.
  - Cons: Behavior change; possible test updates.
- Option B: Update spec and assumptions to match code and add "T3" tags for low-n operation.
  - Pros: Minimal code change.
  - Cons: Weaker guardrails.

6) Unify baseline definitions for levels vs changes.
- Option A: Add explicit `target_mode` or separate APIs for level vs change targets and persistences.
  - Pros: Prevents silent misinterpretation.
  - Cons: API expansion.
- Option B: Provide `persistence_level` and `persistence_change` helpers and update quickstart/examples accordingly.
  - Pros: Quick fix; clearer defaults.
  - Cons: Duplication.

7) Strengthen "never fail silently."
- Option A: Emit warnings or SKIP results when degenerate cases occur (e.g., var_d<=0, no splits, n_effective too small).
  - Pros: Users see reliability issues immediately.
  - Cons: Slightly noisier logs.
- Option B: Add `strict=True` to raise instead of warn.
  - Pros: Enforces correctness for production usage.
  - Cons: More configuration overhead.

8) Improve example and notebook quality checks.
- Option A: Add CI smoke tests for `examples/` and `notebooks/demo.ipynb` (nbval or run-once with small data).
  - Pros: Prevents API drift; increases trust.
  - Cons: Longer CI time.
- Option B: Convert key examples into doctests in docs.
  - Pros: Tightens doc-code alignment.
  - Cons: Requires doc build in CI.

9) Fix documentation patterns that normalize full-series feature engineering.
- Option A: Rewrite examples to compute features inside each CV split and add explicit "do not precompute rolling stats" warnings [1][2][3].
  - Pros: Reduces leakage risk for new users; aligns with project principles.
  - Cons: Examples become longer.
- Option B: Provide a helper function (e.g., `make_lags_within_split`) and standardize usage across tutorials.
  - Pros: Cleaner examples; makes safe patterns easy.
  - Cons: Adds API surface.

10) Clone models inside gate_shuffled_target.
- Option A: Clone per fold and per shuffle using sklearn.clone when available.
  - Pros: Prevents state leakage across fits; more correct for warm-start models.
  - Cons: Slight overhead; non-sklearn models need fallback.

## References (external)
[1] Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. International Journal of Forecasting, 16(4), 437-450.
[2] Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences, 191, 192-213.
[3] Hewamalage, H., Bergmeir, C., & Bandara, K. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. International Journal of Forecasting, 39(3), 1238-1268.
[4] Kunsch, H.R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. Annals of Statistics, 17(3), 1217-1241.
[5] Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313.
[6] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253-263.
[7] Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291.
[8] Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. Journal of Business & Economic Statistics, 10(4), 461-465.
[9] Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.
[10] Romano, Y., Patterson, E., & Candes, E.J. (2019). Conformalized quantile regression. NeurIPS.
[11] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
[12] Gibbs, I. & Candes, E.J. (2021). Adaptive conformal inference under distribution shift. NeurIPS.
[13] Lahiri, S.N. (2003). Resampling Methods for Dependent Data. Springer.
[14] Makridakis, S. & Hibon, M. (2000). The M3-Competition: results, conclusions and implications. International Journal of Forecasting, 16(4), 451-476.
[15] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). The M4 Competition: Results, findings, conclusion. International Journal of Forecasting, 34(4), 802-808.
[16] Monash Time Series Forecasting Archive. https://forecastingdata.org/
[17] Alexandrov, A., Benidis, K., Bohlke-Schneider, M., et al. (2020). GluonTS: Probabilistic and Neural Time Series Modeling in Python. Journal of Machine Learning Research, 21(116), 1-6.
[18] Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. ICML.

## Internal references
- `SPECIFICATION.md`, `CLAUDE.md`, `README.md`
- `docs/knowledge/assumptions.md`, `docs/tutorials/*.md`, `docs/plans/reference/*.md`
- `src/temporalcv/**` (modules cited above)
