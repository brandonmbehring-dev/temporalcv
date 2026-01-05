# temporalcv methodology + pedagogy review (codex)

## Findings (ordered by severity)

### High
1) Shuffled target gate examples and docs are now wrong after the default switched to permutation mode, and many examples effectively disable the gate.
- Evidence: `examples/01_leakage_detection.py:231-273`, `examples/00_quickstart.py:55`, `notebooks/demo.ipynb:157-171`, `docs/quickstart.md:52-58`, `README.md:133-134`, `docs/index.rst:21-29`, `docs/api/gates.md:74-94`.
- Why this matters: In permutation mode, `metric_value` is a p-value and `threshold` is ignored; with n_shuffles in {5, 10, 19} the minimum p-value is >= 0.05, so the gate can never HALT at alpha=0.05 even for blatant leakage.
- Fix: Update all examples/docs to either (a) use `method="effect_size"` and print `details["improvement_ratio"]`, or (b) keep `method="permutation"` and print `details["pvalue"]` with `alpha`, and use n_shuffles >= 20 (>= 100 for publication-quality resolution). Update the API doc to include `method`, `permutation`, `alpha`, `strict`, and p-value semantics.
- References: Phipson and Smyth 2010 (permutation p-values), https://doi.org/10.2202/1544-6115.1585

2) Gap/horizon semantics are inconsistent between core code, spec, and docs, so guidance and gate checks can disagree.
- Evidence: `src/temporalcv/gates.py:850-877` (required_gap = horizon + gap), `src/temporalcv/cv.py:495-531` (gap >= horizon when horizon is set), `SPECIFICATION.md:87-99` (required_gap = horizon), `docs/api/cv.md:172-188` (gap >= h - 1), `docs/tutorials/walk_forward_cv.md:255-262`, `docs/tutorials/leakage_detection.md:163-166`, `notebooks/02_gap_enforcement.ipynb:6`, `notebooks/08_validation_workflow.ipynb:300-307`, `notebooks/demo.ipynb:474-476`.
- Why this matters: Users can follow the docs and still fail gate_temporal_boundary, or pass gates while still leaking. This is a core correctness rule for multi-step forecasting.
- Fix: Choose one definition (gap as total gap vs extra gap beyond horizon). If extra gap is intended, rename parameter to `extra_gap` in gate_temporal_boundary and docs. If total gap is intended, change the gate logic to `required_gap = gap`. Align SPECIFICATION, docs, notebooks, and examples. Add `horizon` to WalkForwardCV docs and show a canonical formula.
- References: Bergmeir and Benitez 2012 (time series CV), https://doi.org/10.1016/j.ins.2012.12.028; Tashman 2000 (rolling origin), https://doi.org/10.1016/S0169-2070(00)00065-0

3) Several examples and notebooks compute errors or baselines in-sample or using full-series data, which invalidates DM/MASE/suspicious improvement claims.
- Evidence: `examples/03_statistical_tests.py:91-92` (predicts on full X after training on first half), `examples/01_leakage_detection.py:308-311` (model_mae computed in-sample before gate), `notebooks/03_persistence_baseline.ipynb:657-659` (naive MAE from full series), `examples/04_high_persistence.py:91-94` (MASE denominator from test set), `docs/quickstart.md:109-118` (persistence baseline assumes change targets while data are levels).
- Why this matters: In-sample errors make models look better than they are, and the gates/tests are designed for out-of-sample errors. This undermines the core "methodological correctness" promise.
- Fix: Use walk-forward or holdout predictions everywhere, compute naive error from training only, and align baseline errors to the same forecast indices as model predictions. For DM, feed out-of-sample errors only.
- References: Diebold and Mariano 1995 (predictive accuracy), https://doi.org/10.1080/07350015.1995.10524599; Hyndman and Koehler 2006 (MASE), https://doi.org/10.1016/j.ijforecast.2006.03.001

### Medium
1) High-persistence docs use level data for change-based metrics (MC-SS, direction accuracy, persistence MAE), which produces meaningless results or silent misuse.
- Evidence: `docs/tutorials/high_persistence.md:68-139`, `docs/quickstart.md:171-181`.
- Fix: Convert to changes (np.diff) before calling compute_move_threshold, compute_move_conditional_metrics, compute_direction_accuracy, and compute_persistence_mae. Add a one-line reminder that these functions expect changes/returns.
- References: Hyndman and Koehler 2006 (MASE), https://doi.org/10.1016/j.ijforecast.2006.03.001

2) Shuffled-target notebook uses detail keys that no longer exist.
- Evidence: `notebooks/05_shuffled_target_gate.ipynb:790-791` (model_mae/mean_shuffled_mae).
- Fix: Use `mae_real` and `mae_shuffled_avg`, or `details["improvement_ratio"]` / `details["pvalue"]` depending on method.

3) "Safe" feature engineering validation fits on the full series, leaking training statistics.
- Evidence: `notebooks/06_feature_engineering_pitfalls.ipynb:614-615`.
- Fix: Fit the engineer on each fold's training data, then transform train/test with training stats, even in the validation gate example.

4) Regime threshold alignment likely off by lag offset, which mislabels regimes and breaks per-regime metrics.
- Evidence: `notebooks/07_threshold_leakage.ipynb:602-607`.
- Fix: Carry lag offsets through (e.g., return aligned indices from create_lag_features or add `n_lags` to series indices before computing regimes).

5) Gap calculations are off by one in notebooks, reinforcing the gap/horizon confusion.
- Evidence: `notebooks/08_validation_workflow.ipynb:300-307`, `notebooks/demo.ipynb:474-476`.
- Fix: Show actual gap as `test_start - train_end - 1` and make comments match.

6) Documentation and API drift will mislead users into calling non-existent APIs or using old signatures.
- Evidence: `docs/api/benchmarks.md:31-55` (load_fred_series/load_m5_sample/load_monash_dataset), `docs/api/compare.md:10-66` (ComparisonRunner + statsforecast model names), `docs/api/gates.md:74-103` (missing method/permutation/alpha/strict), `docs/api/cv.md:172-188` (gap rule + missing horizon), `docs/troubleshooting.md:262-315` (compute_mc_ss/WalkForwardConformal), `docs/index.rst:21-29` (run_gates signature), `docs/knowledge/assumptions.md:34-37` and `docs/api/statistical_tests.md:146-158` (PT n >= 20 vs code n >= 30).
- Fix: Sweep docs to match the current API, and add an API-drift checklist to release notes. Consider generating the "API Guides" from docstrings to avoid repeated drift.

7) Benchmark metadata marks "official_split=True" even when the data are truncated or aggregated.
- Evidence: `src/temporalcv/benchmarks/monash.py:120-151`, `src/temporalcv/benchmarks/monash.py:220-248`, `src/temporalcv/benchmarks/m5.py:118-146`.
- Fix: If truncated or aggregated, set official_split=False (or add a new flag like `derived_split=True`).

8) gate_synthetic_ar1 reuses the same model instance across folds, which can leak state for warm-start or adaptive models.
- Evidence: `src/temporalcv/gates.py:667-672`.
- Fix: Clone or re-instantiate the model per fold, as done in gate_shuffled_target.

9) Version/consistency drift across spec and citations.
- Evidence: `SPECIFICATION.md:3` (version 0.1.0), `src/temporalcv/__init__.py:36` (1.0.0-rc1), `CITATION.cff:13` (1.0.0).
- Fix: Decide a single authoritative version and update all three.

10) Module coverage gaps in the docs reduce discoverability.
- Evidence: Missing docs for guardrails, stationarity, lag_selection, changepoint, cv_financial, inference; `docs/index.md:64-74` (no diagnostics, guardrails, etc), `docs/api` lacks these modules.
- Fix: Add API guide pages or point to api_reference pages for these modules.

### Low
- Conformal docs suggest "expect coverage within +/-5%" for time series without evidence.
  - Evidence: `docs/api/conformal.md:5`.
  - Fix: Soften language or cite a source; emphasize that coverage can deviate materially under dependence.
- Placeholder or missing reference links reduce pedagogy quality.
  - Evidence: `notebooks/01_why_temporal_cv.ipynb:822`.
  - Fix: Add DOI/arXiv links and internal doc links throughout notebooks and docs.
- A few pedagogy warnings are over-broad (lags computed on full series are not inherently leaky if strictly backward-looking).
  - Evidence: `docs/tutorials/walk_forward_cv.md:291-295`.
  - Fix: Clarify that the real leakage is centered windows, global normalization, or target-derived features.

## Suggested reference links to add
- Phipson and Smyth 2010 (permutation p-values): https://doi.org/10.2202/1544-6115.1585
- Bergmeir and Benitez 2012 (time series CV): https://doi.org/10.1016/j.ins.2012.12.028
- Tashman 2000 (rolling origin): https://doi.org/10.1016/S0169-2070(00)00065-0
- Diebold and Mariano 1995 (DM test): https://doi.org/10.1080/07350015.1995.10524599
- Harvey, Leybourne, Newbold 1997 (DM small sample): https://doi.org/10.1016/S0169-2070(96)00763-4
- Newey and West 1987 (HAC): https://doi.org/10.2307/1913210
- Pesaran and Timmermann 1992 (PT test): https://doi.org/10.1080/07350015.1992.10509954
- Hyndman and Koehler 2006 (MASE): https://doi.org/10.1016/j.ijforecast.2006.03.001
- Kunsch 1989 (block bootstrap): https://projecteuclid.org/journals/annals-of-statistics/volume-17/issue-3/The-Jackknife-and-the-Bootstrap-for-General-Stationary/10.1214/aos/1176347242.full
- Politis and Romano 1994 (stationary bootstrap): https://doi.org/10.1080/01621459.1994.10476870
- Romano et al 2019 (CQR): https://arxiv.org/abs/1905.03222
- Gibbs and Candes 2021 (adaptive conformal): https://arxiv.org/abs/2107.07511
- Internal links: `docs/knowledge/mathematical_foundations.md`, `docs/knowledge/assumptions.md`, `docs/knowledge/leakage_audit_trail.md`, `SPECIFICATION.md`

## Questions / assumptions
- Should `gap` represent total separation or extra gap beyond horizon? If extra, is renaming to `extra_gap` acceptable?
- Do you want `temporalcv[all]` to include the `changepoint` extra and `docs`, or is "all" intentionally minimal?
- Should `official_split` mean "exact competition protocol" or "protocol applied after truncation/aggregation"?

## Strengths (brief)
- The knowledge-tiering and audit trail docs are rare and valuable.
- The gate framework is cohesive and test coverage is extensive.
- Notebook sequencing and the PROBLEM -> FAILURE -> SOLUTION pedagogy is strong once the API drift is fixed.
