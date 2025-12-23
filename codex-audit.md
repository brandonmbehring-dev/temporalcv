# temporalcv audit critique (codex)

## Scope and method
- Reviewed repository docs, source, and tests under /home/brandon_behring/Claude/temporalcv.
- Reviewed plan file /home/brandon_behring/.claude/plans/hazy-finding-moore.md.
- Did not run tests or benchmarks; findings are static review only.

## High-impact issues (correctness and statistical validity)

1) SPECIFICATION.md is authoritative but diverges from code and docs.
- Examples: gate_shuffled_target default n_shuffles is 100 in SPECIFICATION.md vs 5 in code and docs; DM/PT minimum sample sizes differ between spec, assumptions, and code; bagging auto block length is sqrt(n) in spec vs n^(1/3) in code.
- Consequence: "single source of truth" is violated and results are not reproducible across readers.

2) gate_shuffled_target is not a permutation test and uses in-sample evaluation.
- Fits and evaluates on the same data; improvement ratio is treated like a p-value threshold even though no permutation p-value is computed.
- High-capacity models can memorize shuffled targets, leading to false PASS even with leakage, while in-sample bias inflates observed improvement.
- Use out-of-sample evaluation or time-series CV for this gate [1][2].

3) gate_synthetic_ar1 can false HALT and lacks basic input validation.
- The gate evaluates on the training data it just fit, so flexible models can beat the theoretical bound by overfitting, not leakage.
- No validation for phi in (-1, 1) or n_samples > n_lags; phi >= 1 can yield infinite or NaN initial variance.
- Theoretical bound assumes Gaussian innovations; not enforced or documented.

4) dm_test uses normal p-values even when Harvey correction is enabled.
- Harvey et al. recommend a small-sample correction and t distribution (df = n - 1) for inference; the current implementation uses the normal distribution, which can misstate significance for small n or h > 1 [4].

5) SplitConformalPredictor quantile selection can under-cover.
- Finite-sample guarantees require using the order-statistic (a "higher" quantile) rather than default linear interpolation; np.quantile default can under-cover by construction [7][8].

6) Gap/horizon semantics are inconsistent.
- Spec says gap >= horizon; WalkForwardCV has no horizon context and does not enforce; gate_temporal_boundary treats gap as an additional buffer, which can double-count if the user passes gap=horizon.
- This can create false HALT or a false sense of safety.

## Medium-impact issues (consistency, reproducibility, UX)

- Seed policy mismatch: CLAUDE.md says random_state defaults to None, but bagging and BootstrapUncertainty default to 42, and there is no consistent behavior across modules.
- run_gates examples are incorrect in docstrings (gates.py and __init__.py). They pass model/X/y into run_gates and pass uncalled gate functions, but run_gates only accepts GateResult objects.
- compute_move_conditional_metrics computes threshold from full actuals when threshold=None without warning; this contradicts the "training only" leakage guidance.
- pt_test filters zeros for 2-class but only checks n >= 20 before filtering; n_effective can be far smaller with no warning.
- Many functions return NaN or "pvalue=1.0" without warnings (dm_test for var_d <= 0, compute_move_conditional_metrics for zero moves), which weakens the "NEVER FAIL SILENTLY" principle.
- PR-AUC uses trapezoidal integration rather than average precision; results will differ from sklearn.metrics.average_precision_score. This is fine, but needs explicit documentation.

## Unstated or weakly enforced assumptions

- Time order is correct and evenly spaced. WalkForwardCV uses index positions, so irregular timestamps can violate intended horizons.
- No missing values or NaNs. Most functions do not validate and can silently propagate NaNs.
- Stationarity and weak dependence for HAC variance and block bootstrap methods [5][9][10].
- Model errors are well-behaved (finite variance, mild tails). Heavy-tailed errors can break DM and PT asymptotics [3][6].
- Targets are changes, not levels. Several metrics (persistence baseline, direction accuracy) assume change targets without enforcing.
- Conformal coverage is approximate for time series; exchangeability is violated even when users may read "coverage guarantee" in README [7][8].

## Plan audit: /home/brandon_behring/.claude/plans/hazy-finding-moore.md

- Redundant items: Harvey adjustment is already implemented (harvey_correction in dm_test), and SciPy is already a required dependency.
- Residual diagnostics gate: plan cites scipy.stats.acorr_ljungbox, but this function lives in statsmodels, not SciPy. This implies a new dependency or a custom Ljung-Box implementation [13].
- Theoretical AR(1) bounds on real data: estimating phi via ACF(1) is not enough to infer innovation sigma; the theoretical MAE uses the innovation variance, not the series variance. The gate needs a clear estimation step and assumptions about innovation distribution.
- Wild cluster bootstrap: CV folds are not independent clusters because training windows overlap. A wild cluster bootstrap may be invalid without a justification for dependence structure [11].
- Influence diagnostics for DM: influence on d_t ignores HAC autocorrelation; the influence function must account for serial correlation or it will mis-rank observations.
- Cross-fit validation: naive K-fold cross-fitting violates temporal order. Time-series cross-fitting must use blocked or forward splits and must state dependence assumptions [12].
- Regime-stratified reporting: should mask low-n regimes (already supported in regimes.py) and compute thresholds on training only to avoid leakage.

## Recommendations with options (pros and cons)

1) Resolve spec drift.
- Option 1: Update code to match SPECIFICATION.md defaults. Pros: restores "single source of truth". Cons: behavior changes for existing users.
- Option 2: Amend SPECIFICATION.md to match code and document rationale. Pros: minimal code churn. Cons: weakens the authoritative spec posture.
- Option 3: Version the spec and add deprecation warnings for mismatched defaults. Pros: explicit transition path. Cons: more governance overhead.

2) Redesign gate_shuffled_target.
- Option 1: Use a single temporal holdout (last split) for real and shuffled targets. Pros: simple and fast. Cons: higher variance, split choice matters.
- Option 2: Use WalkForwardCV inside the gate and compute a permutation p-value over folds. Pros: closer to out-of-sample leakage detection [1][2]. Cons: slower and more complex.
- Option 3: Accept user-provided predictions and scores. Pros: flexibility for complex pipelines. Cons: more burden on users.

3) Fix gate_synthetic_ar1 evaluation and validation.
- Option 1: Evaluate on an out-of-sample segment of the synthetic series. Pros: reduces overfit false HALT. Cons: slightly noisier.
- Option 2: Compare to the known optimal AR(1) predictor on a holdout and gate on the gap. Pros: clearer statistical meaning. Cons: additional implementation work.

4) Improve dm_test small-sample inference.
- Option 1: Use t distribution (df = n - 1) when harvey_correction=True. Pros: aligns with Harvey et al. [4]. Cons: changes p-values.
- Option 2: Keep normal but add a parameter for distribution choice and warn when n < 100. Pros: backwards compatible. Cons: leaves biased defaults.

5) Make conformal quantile selection conservative.
- Option 1: Use np.quantile(method="higher") or the explicit order statistic. Pros: restores finite-sample coverage [7][8]. Cons: slight behavior change for existing tests.
- Option 2: Keep current behavior but document approximate coverage for non-exchangeable data. Pros: no code change. Cons: users may misinterpret coverage guarantees.

6) Reduce leakage risk in move-conditional metrics.
- Option 1: Require threshold explicitly for compute_move_conditional_metrics (raise if None). Pros: prevents accidental leakage. Cons: breaks backward compatibility.
- Option 2: Keep default but warn loudly when threshold is computed from the same data. Pros: preserves API. Cons: warnings can be ignored.

7) Align seed policy with documentation.
- Option 1: Change defaults to random_state=None across stochastic components. Pros: consistent with CLAUDE.md. Cons: non-deterministic results by default.
- Option 2: Update CLAUDE.md to acknowledge deterministic defaults. Pros: keeps reproducibility. Cons: departs from documented policy.

8) Tighten plan items before implementation.
- Option 1: Add a short "assumptions and dependencies" subsection to each planned feature (e.g., statsmodels for Ljung-Box, blocked splits for cross-fit). Pros: reduces surprises. Cons: extra planning effort.
- Option 2: Defer inference-heavy features until user demand is clearer. Pros: focuses on correctness and spec drift first. Cons: slower feature growth.

## Round 2: Cross-repo alignment and public-facing consistency

### High-impact cross-repo issues

1) Leakage in tutorials and inconsistent gap semantics.
- `docs/tutorials/walk_forward_cv.md` uses `np.roll` on the full series to build lags and asserts that this is "proper since we're using CV correctly." This is the exact leakage pattern documented in myga postmortems: full-series feature computation and rolling stats contaminate training splits [18][22]. The same tutorial says "gap at least h-1," while other docs and the internal bug fix require gap = horizon, creating confusion for internal users [19][22].

2) Documentation/API mismatches undermine public usability.
- `docs/knowledge/episodes/episode_02_boundary_violations.md` calls `gate_temporal_boundary(cv, X, y, horizon=2)` (signature does not exist) and claims "temporalcv enforces gap >= horizon" even though the code has no horizon awareness [24]. `docs/tutorials/leakage_detection.md` uses incorrect detail keys (`real_mae`, `shuffled_mae`) and a gap formula that conflicts with the implementation [23]. `docs/plans/reference/api_design.md` references `temporalcv.tests`, CLI commands, and splitters that do not exist [25]. `docs/plans/reference/benchmark_strategy.md` cites gates (`gate_horizon_gap`, `gate_exogenous_alignment`) and `run_validation_suite()` that are not implemented [26]. This violates the "test-driven documentation" and "single source of truth" governance principles used across your stack [20][21].

3) Internal validation requirements are not covered by temporalcv gates.
- myga and annuity research rules mandate: gap defaults to horizon, warn if gap < horizon, and enforce `last_target_idx = train_end - horizon` (prevents target leakage) [17][19]. temporalcv has no horizon-aware splitter or gate that inspects all splits. 
- Change-target sanity checks (e.g., mean near zero) are required to prevent level/changes confusion but no such gate exists [17].
- Lag-0 prohibition, per-split feature computation, and rolling-stat leakage detection are explicitly documented as critical bug categories but have no direct gate support [18][20].
- myga flags MC-SS > 0.35 as suspicious, but temporalcv has no MC-SS anomaly gate [29].
- Your public portfolio explicitly calls out a "label leakage gate" (target-aware scaling / rolling-stat leakage) as a next OSS step; it does not exist yet [27].

4) Horizon semantics are not enforced, despite prior critical bugs.
- annuity_forecasting documents a major failure from interpreting `h` as calendar days instead of time periods; temporalcv does not accept timestamps or frequency to prevent this, and its own docs are inconsistent on gap/horizon semantics [16][22].

5) Public narrative drift.
- The portfolio appendix references a `temporalcv/src/temporalcv/validation.py` and a `TemporalValidator` "multi-agent" architecture that does not exist in the codebase, which can mislead external users and reviewers [28].

### Medium-impact cross-repo issues

- No internal adoption yet: a repo-wide search shows no code imports of temporalcv outside its own docs. If it is the intended validation core, you likely need adapters or replacement of local validation modules in annuity_forecasting and myga-forecasting-v4 (observational finding).
- Regime/threshold leakage remains easy to trigger: temporalcv exposes training-only thresholds as guidance but does not enforce them, while internal standards treat this as a critical bug [17][18].
- "Single source of truth" drift: annuity and myga enforce immutable specs (PROJECT_NORTH_STAR/CONSTITUTION), while temporalcv has SPECIFICATION plus multiple aspirational reference docs with incompatible APIs [15][17][25].

### Additional recommendations (internal + public alignment)

1) Build a concrete `TemporalValidator` (or similar) that matches the portfolio narrative and internal needs.
- Include gate orchestration over full CV splits, horizon-aware checks, and explicit, testable HALT/WARN/PASS logic.

2) Add missing high-risk gates from internal postmortems.
- Target sanity (level vs change), lag-0 feature detection, rolling-stat/scale leakage, horizon-gradient anomaly, and MC-SS anomaly thresholds [17][18][29].

3) Make `WalkForwardCV` horizon-aware (or add a wrapper).
- `gap=None` defaults to horizon, warn if gap < horizon, and expose helpers for `last_target_idx = train_end - horizon` [19].

4) Enforce "test-driven documentation."
- Add doctests or pytest-based doc examples so API drift cannot ship silently [21].

5) Provide internal adapters.
- Add minimal integration layers or recipes for annuity_forecasting and myga-forecasting-v4 to make temporalcv the shared validation core (short-term wrappers, long-term module extraction).

## References
[1] Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. International Journal of Forecasting, 16(4), 437-450.
[2] Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for time series predictor evaluation. Information Sciences, 191, 192-213.
[3] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. Journal of Business and Economic Statistics, 13(3), 253-263.
[4] Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291.
[5] Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.
[6] Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of predictive performance. Journal of Business and Economic Statistics, 10(4), 461-465.
[7] Romano, Y., Patterson, E., & Candes, E.J. (2019). Conformalized quantile regression. NeurIPS.
[8] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.
[9] Kunsch, H.R. (1989). The Jackknife and the Bootstrap for General Stationary Observations. Annals of Statistics, 17(3), 1217-1241.
[10] Politis, D.N. & Romano, J.P. (1994). The Stationary Bootstrap. Journal of the American Statistical Association, 89(428), 1303-1313.
[11] Cameron, A.C., Gelbach, J.B., & Miller, D.L. (2008). Bootstrap-based improvements for inference with clustered errors. Review of Economics and Statistics, 90(3), 414-427.
[12] Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
[13] Ljung, G.M. & Box, G.E.P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297-303.
[14] Jarque, C.M. & Bera, A.K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters, 6(3), 255-259.
[15] /home/brandon_behring/Claude/annuity_forecasting/docs/PROJECT_NORTH_STAR.md
[16] /home/brandon_behring/Claude/annuity_forecasting/docs/HORIZON_SEMANTICS.md
[17] /home/brandon_behring/Claude/myga-forecasting-v4/CONSTITUTION.md
[18] /home/brandon_behring/Claude/myga-forecasting-v4/docs/episodes/BUG-001_lag_leakage.md
[19] /home/brandon_behring/Claude/myga-forecasting-v4/docs/episodes/BUG-002_target_gap.md
[20] /home/brandon_behring/Claude/lever_of_archimedes/patterns/data_leakage_prevention.md
[21] /home/brandon_behring/Claude/ml-governance-toolkit/README.md
[22] /home/brandon_behring/Claude/temporalcv/docs/tutorials/walk_forward_cv.md
[23] /home/brandon_behring/Claude/temporalcv/docs/tutorials/leakage_detection.md
[24] /home/brandon_behring/Claude/temporalcv/docs/knowledge/episodes/episode_02_boundary_violations.md
[25] /home/brandon_behring/Claude/temporalcv/docs/plans/reference/api_design.md
[26] /home/brandon_behring/Claude/temporalcv/docs/plans/reference/benchmark_strategy.md
[27] /home/brandon_behring/Claude/agentic_ai_portfolio/README.md
[28] /home/brandon_behring/Claude/agentic_ai_portfolio/docs/appendix_detailed_patterns.md
[29] /home/brandon_behring/Claude/myga-forecasting-v4/docs/knowledge/domain/metrics.md
