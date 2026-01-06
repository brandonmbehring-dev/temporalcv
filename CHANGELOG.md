# Changelog

All notable changes to temporalcv will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes.

---

## [1.1.0] - 2026-01-01

Statistical extensions and enhanced features (implemented, documented retroactively).

### Added

#### Statistical Tests
- `variance_method="self_normalized"` for `dm_test()` — bandwidth-free DM test (Shao 2010)
- `gw_test()` — Giacomini-White conditional predictive ability test (GW 2006)
- `cw_test()` — Clark-West nested model comparison test (CW 2007)

#### Multi-Horizon Analysis
- `compare_horizons()` — two-model comparison across forecast horizons
- `compare_models_horizons()` — multi-model comparison with best-per-horizon
- `MultiHorizonResult` dataclass with `degradation_pattern`, `significant_horizons`

#### Nested Cross-Validation
- `NestedWalkForwardCV` — unbiased hyperparameter selection (Bergmeir 2012)
- Voting-based parameter selection with `params_stability` metric
- Both grid search and randomized search modes

#### Block Bootstrap CI
- `bootstrap_ci=True` parameter for `gate_shuffled_target()` and `gate_synthetic_ar1()`
- Returns `ci_lower`, `ci_upper`, `bootstrap_std` in gate details
- Moving block bootstrap preserving temporal dependence (Kunsch 1989)

#### Conformal Diagnostics
- `CoverageDiagnostics` dataclass for coverage analysis
- `compute_coverage_diagnostics()` — by time window and regime

#### Documentation
- Model cards for `WalkForwardCV` and `gate_shuffled_target` (docs/model_cards/)

### Technical Notes

**Test Count**: 1,698 tests (up from 1,453)

**Note**: These features were implemented during v1.0.0-rc1 development but not documented in CHANGELOG until this retroactive update.

---

## [1.2.0] - 2026-01-01

Advanced statistical tests and conformal prediction methods.

### Added

#### Multiple Comparison Tests
- `reality_check_test()` — White's Reality Check for data snooping (White 2000)
- `spa_test()` — Hansen's Superior Predictive Ability test with 3 p-values (Hansen 2005)
- `RealityCheckResult`, `SPATestResult` dataclasses

#### Forecast Encompassing
- `forecast_encompassing_test()` — pairwise encompassing test (Harvey et al. 1998)
- `forecast_encompassing_bidirectional()` — returns recommendation: use_a/use_b/combine/equivalent
- `EncompassingTestResult`, `BidirectionalEncompassingResult` dataclasses

#### Bellman Conformal Inference
- `BellmanConformalPredictor` — DP-based optimal prediction intervals (Yang, Candès & Lei 2024)
- Multi-step forecast handling with bounded interval widths
- `initialize()`, `predict_interval()`, `predict_intervals_batch()` methods

#### Research-kb Integration
- MCP-based integration hooks documented in AI_CONTEXT.md
- Example usage for querying methodological background

### Technical Notes

**Test Count**: 1,741 tests (up from 1,698)

**Note**: v1.2.0 features implemented 2025-12-31, documented retroactively.

---

## [1.0.0-rc1] - 2025-12-26

Release candidate 1 for v1.0.0 - statistical rigor improvements and API stabilization.

### Breaking Changes

#### `gate_shuffled_target()` - New Default Behavior
- **Default method changed from heuristic to true permutation test**
- Old behavior: `method="effect_size"` (fast, n_shuffles=5, improvement ratio)
- New behavior: `method="permutation"` (rigorous, n_shuffles=100, p-value)
- **Migration**: Add `method="effect_size"` to preserve old behavior

#### `get_n_splits()` - Strict by Default
- Now raises `ValueError` on insufficient data instead of silently returning 0
- **Migration**: Use `get_n_splits(X, strict=False)` for old behavior

### Added

#### Two-Mode Shuffled Target Gate
- `method="permutation"`: True permutation test with p-value (Phipson & Smyth 2010)
  - Default `n_shuffles=100` for statistical power
  - `alpha=0.05` significance level
  - `strict=True` uses n_shuffles=199 for p-value resolution of 0.005
- `method="effect_size"`: Fast heuristic (original behavior)
  - Default `n_shuffles=5` for quick iteration
  - `threshold=0.05` improvement ratio threshold
- Statistical power warnings when n_shuffles too low for target alpha

#### Strict Mode for CV Methods
- `get_n_splits(strict=True)`: Raises on insufficient data (default)
- `get_n_splits(strict=False)`: Returns 0 on failure (backward compatible)
- Clear error messages with suggested fixes

### Changed

- Development Status: Alpha → Beta
- SPECIFICATION.md Section 1.2: Documented both shuffled target methods

### Technical Notes

**Test Count**: 1,453+ tests

**Breaking Change Impact**: Low - most users use default thresholds

---

## [0.4.0] - 2025-12-24

Hardening release with robustness testing and cross-platform CI.

### Added

#### Performance Benchmarks
- pytest-benchmark infrastructure for critical path testing
- `tests/benchmarks/test_cv_benchmarks.py` - CV splitting performance
- `tests/benchmarks/test_gate_benchmarks.py` - gate evaluation timing
- `tests/benchmarks/test_metric_benchmarks.py` - metric computation benchmarks

#### Cross-Platform CI
- Windows CI job in GitHub Actions matrix
- `os: [ubuntu-latest, windows-latest]` matrix support
- Python 3.10, 3.11, 3.12 on both platforms

#### Seed-Sweep Reproducibility Tests
- `tests/reproducibility/test_seed_determinism.py` - 20-seed sweeps
- Determinism tests for gates, bootstrap, conformal prediction
- Bootstrap strategy reproducibility (MovingBlock, Stationary, Residual)
- TimeSeriesBagger prediction determinism

#### Expanded Property-Based Tests
- `tests/property/test_stationarity_invariants.py` - ADF, KPSS, PP invariants
- `tests/property/test_financial_cv_invariants.py` - PurgedKFold, CPCV, PurgedWalkForward
- `tests/property/test_metric_invariants.py` - MAE, MSE, RMSE, MAPE, pinball, Huber, LinEx

### Technical Notes

**Test Count**: 1,428 tests (up from 1,243)

**Coverage**: Maintained 85%+

---

## [0.3.0] - 2025-12-24

Capability gaps release with comprehensive time-series analysis toolkit.

### Added

#### Stationarity Testing
- `adf_test()` - Augmented Dickey-Fuller test
- `kpss_test()` - KPSS stationarity test
- `pp_test()` - Phillips-Perron test
- `check_stationarity()` - unified stationarity assessment
- `difference_until_stationary()` - automatic differencing
- `StationarityConclusion` enum for conclusions

#### Lag Selection
- `select_lag_pacf()` - PACF-based lag selection
- `select_lag_aic()` - AIC minimization
- `select_lag_bic()` - BIC minimization

#### Changepoint Detection
- `detect_variance_changepoints()` - variance-based detection
- `detect_changepoints_pelt()` - PELT algorithm (ruptures optional)

#### Residual Bootstrap
- `ResidualBootstrap` class with STL decomposition support
- `seasonal_period` parameter for seasonal adjustment
- Integration with `TimeSeriesBagger`

#### Financial Cross-Validation
- `PurgedKFold` - K-fold with purge gap and embargo
- `CombinatorialPurgedCV` (CPCV) - combinatorial path testing
- `PurgedWalkForward` - walk-forward with purge gap
- `compute_label_overlap()` - label overlap matrix
- `estimate_purge_gap()` - gap estimation from horizon

#### Guardrails Module
- `check_suspicious_improvement()` - >20% improvement trigger
- `check_minimum_sample_size()` - statistical power validation
- `check_stratified_sample_size()` - per-regime minimum n
- `check_forecast_horizon_consistency()` - horizon alignment
- `check_residual_autocorrelation()` - Ljung-Box check
- `run_all_guardrails()` - unified validation suite

#### Validators Module
- `validate_predictions()` - prediction array validation
- `validate_actuals()` - actuals array validation
- `validate_prediction_pair()` - paired validation
- `validate_positive()` - positivity constraint

### Technical Notes

**Test Count**: 1,243 tests (up from 1,042)

**Coverage**: 85%+

---

## [0.2.0] - 2025-12-24

Major metrics expansion release with comprehensive evaluation toolkit.

### Added

#### Core Metrics
- `compute_mae()` - mean absolute error
- `compute_mse()` - mean squared error
- `compute_rmse()` - root mean squared error
- `compute_mape()` - mean absolute percentage error
- `compute_smape()` - symmetric MAPE (bounded 0-200%)
- `compute_bias()` - mean error (systematic bias detection)
- `compute_naive_error()` - naive forecast MAE for MASE denominator
- `compute_mase()` - mean absolute scaled error (Hyndman & Koehler 2006)
- `compute_mrae()` - mean relative absolute error
- `compute_theils_u()` - Theil's U statistic (Theil 1966)
- `compute_forecast_correlation()` - Pearson/Spearman correlation
- `compute_r_squared()` - coefficient of determination

#### Event & Direction Metrics
- `compute_calibrated_direction_brier()` - Brier score with calibration adjustment
- `convert_predictions_to_direction_probs()` - point predictions to direction probabilities

#### Quantile & Interval Metrics
- `compute_pinball_loss()` - quantile regression loss (Koenker & Bassett 1978)
- `compute_crps()` - Continuous Ranked Probability Score with scipy auto-detect (Gneiting & Raftery 2007)
- `compute_interval_score()` - proper scoring rule for intervals (Gneiting & Raftery 2007)
- `compute_quantile_coverage()` - empirical coverage of prediction intervals
- `compute_winkler_score()` - alias for interval score (Winkler 1972)

#### Financial & Trading Metrics
- `compute_sharpe_ratio()` - annualized risk-adjusted return (Sharpe 1966)
- `compute_max_drawdown()` - peak-to-trough decline
- `compute_cumulative_return()` - geometric/arithmetic total return
- `compute_information_ratio()` - active return vs tracking error (Goodwin 1998)
- `compute_hit_rate()` - directional accuracy (sign matching)
- `compute_profit_factor()` - gross profit / gross loss
- `compute_calmar_ratio()` - return / max drawdown

#### Asymmetric Loss Functions
- `compute_linex_loss()` - linear-exponential asymmetric loss (Varian 1975, Zellner 1986)
- `compute_asymmetric_mape()` - weighted over/under MAPE
- `compute_directional_loss()` - custom UP miss vs DOWN miss penalties
- `compute_squared_log_error()` - mean squared log error (MSLE)
- `compute_huber_loss()` - robust quadratic/linear loss

#### Volatility-Weighted Metrics
- `VolatilityEstimator` Protocol - extensibility interface for custom estimators
- `RollingVolatility` class - rolling standard deviation estimator
- `EWMAVolatility` class - exponentially weighted estimator (RiskMetrics 1996)
- `compute_local_volatility()` - unified estimation with rolling_std/ewm/garch methods
- `compute_volatility_normalized_mae()` - scale-invariant MAE across regimes
- `compute_volatility_weighted_mae()` - inverse/importance weighting schemes
- `VolatilityStratifiedResult` dataclass - tercile breakdown with interpretation
- `compute_volatility_stratified_metrics()` - MAE/RMSE by volatility regime

#### Multi-Model Comparison
- `MultiModelComparisonResult` dataclass - pairwise comparison results
- `compare_multiple_models()` - Bonferroni-corrected pairwise DM tests

#### Regime-Stratified Analysis
- `StratifiedMetricsResult` dataclass - per-regime breakdown
- `compute_stratified_metrics()` - MAE/RMSE by custom regimes

#### Cross-Validation
- `SplitResult` dataclass - single split results with predictions/actuals
- `WalkForwardResults` dataclass - aggregated CV results with lazy metrics
- `walk_forward_evaluate()` - unified evaluation function with date support

#### Guardrails & Validation
- `GuardrailResult` dataclass - unified check results
- `check_suspicious_improvement()` - >20% improvement trigger
- `check_minimum_sample_size()` - statistical power validation
- `check_stratified_sample_size()` - per-regime minimum n
- `check_forecast_horizon_consistency()` - horizon alignment check
- `check_residual_autocorrelation()` - Ljung-Box based check
- `run_all_guardrails()` - unified validation suite

#### Theoretical Bounds
- `theoretical_ar1_mse_bound()` - minimum MSE for AR(1) process
- `theoretical_ar1_mae_bound()` - minimum MAE for AR(1) process
- `theoretical_ar2_mse_bound()` - minimum MSE for AR(2) process
- `check_against_ar1_bounds()` - leakage detection via bounds
- `generate_ar1_series()` - synthetic AR(1) for testing
- `generate_ar2_series()` - synthetic AR(2) for testing

#### Diagnostics & Inference
- `gate_residual_diagnostics()` - autocorrelation, normality, bias checks
- `gate_theoretical_bounds()` - AR(1) theoretical minimum validation
- `compute_dm_influence()` - observation and block influence diagnostics
- `gap_sensitivity_analysis()` - gap parameter sensitivity analysis
- `wild_cluster_bootstrap()` - few-fold inference with auto weight selection

#### Testing Infrastructure
- 6-tier test architecture: known-answer, Monte Carlo, golden reference, adversarial
- Nightly CI workflow for Monte Carlo validation
- Golden reference JSON with R-validated DM test cases
- Monte Carlo calibration tests for DM, wild bootstrap, conformal coverage
- 1042 tests total (1041 passing, 1 skipped)

#### Documentation
- Sphinx infrastructure with napoleon, autodoc, intersphinx
- Comprehensive metrics API reference (`docs/api/metrics.md`)
- Knowledge Tier glossary (T1/T2/T3)
- ReadTheDocs deployment configuration

### Changed
- Enhanced Suspicious Results Protocol with concrete thresholds
- `SplitInfo` dataclass now frozen (immutable)
- Expanded `__init__.py` exports (70+ public symbols)

### Fixed
- DM test uses t-distribution when `harvey_correction=True` (was normal)
- Gates use WalkForwardCV internally for out-of-sample evaluation (was in-sample)
- Conformal quantile uses `method="higher"` for conservative coverage
- PT test minimum samples increased to 30 (was 20)
- phi validation (-1 < phi < 1) added to `gate_synthetic_ar1`
- SPECIFICATION.md synced with code parameters

## [0.1.0-alpha] - 2025-12-23

### Added

#### Phase 5: Benchmarks and Event Metrics
- Benchmark dataset loaders: FRED, M5, Monash (M3/M4), GluonTS
- Model comparison framework with adapter pattern
- Event metrics: Brier score, PR-AUC, direction probability
- `create_synthetic_ar1()` for controlled testing
- 115 new tests (366 total)

#### Phase 4: Uncertainty and Ensemble Methods
- Split conformal prediction with coverage guarantee
- Adaptive conformal inference for distribution shift (Gibbs & Candès 2021)
- Bootstrap uncertainty estimation
- Time-series bagging: block bootstrap, stationary bootstrap, feature bagging
- `walk_forward_conformal()` for online prediction intervals
- `evaluate_interval_quality()` metrics
- 89 new tests (251 total)

#### Phase 3: High-Persistence Handling
- Move-Conditional Skill Score (MC-SS) for sticky series
- `compute_move_threshold()` with training-only enforcement
- `classify_moves()` for UP/DOWN/FLAT classification
- `compute_move_only_mae()` excluding flat periods
- `compute_direction_accuracy()` with 2-class and 3-class modes
- Regime classification: volatility and direction regimes
- `mask_low_n_regimes()` for small-sample protection
- 58 new tests (162 total)

#### Phase 2: Walk-Forward CV Infrastructure
- `WalkForwardCV` with sliding and expanding windows
- Gap parameter enforcement (`gap >= horizon`)
- `SplitInfo` dataclass for split metadata
- sklearn-compatible splitter API
- 40 new tests (104 total)

#### Phase 1: Validation Gates and Statistical Tests
- `gate_shuffled_target()` - definitive leakage detection
- `gate_synthetic_ar1()` - theoretical bounds validation
- `gate_suspicious_improvement()` - >20% improvement trigger
- `gate_temporal_boundary()` - gap enforcement validation
- `run_gates()` - aggregation with HALT > WARN > PASS priority
- `dm_test()` - Diebold-Mariano with HAC variance
- `pt_test()` - Pesaran-Timmermann direction test
- `compute_hac_variance()` - Bartlett kernel estimator
- `GateStatus`, `GateResult`, `ValidationReport` types
- 64 tests with 97% coverage

#### Phase 0: Repository Setup
- Package structure with `src/` layout
- PEP 561 type markers (`py.typed`)
- pyproject.toml with hatchling build
- Pre-commit hooks (ruff, mypy)
- GitHub Actions CI (Python 3.9-3.12)
- AI_CONTEXT.md with project principles
- Knowledge tier system documentation

### Technical Notes

**Dependencies**:
- Core: numpy>=1.21, scipy>=1.7, scikit-learn>=1.0
- Optional: pandas, fredapi, gluonts, statsforecast

**Coverage**: 86% overall, 95%+ on core modules

**Test Count**: 628 tests (627 passing, 1 skipped)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 1.2.0 | 2026-01-01 | RC/SPA tests, Encompassing, Bellman Conformal, research-kb hooks |
| 1.1.0 | 2026-01-01 | Statistical extensions (GW, CW, self-normalized DM), NestedCV, bootstrap CI |
| 1.0.0-rc1 | 2025-12-26 | Release candidate (two-mode shuffled target, strict get_n_splits) |
| 0.4.0 | 2025-12-24 | Hardening (benchmarks, Windows CI, reproducibility, property tests) |
| 0.3.0 | 2025-12-24 | Capability gaps (stationarity, financial CV, guardrails) |
| 0.2.0 | 2025-12-24 | Major metrics expansion (40+ new functions) |
| 0.1.0-alpha | 2025-12-23 | Complete feature set (gates, CV, conformal, bagging) |

---

## Migration Notes

### From Unreleased to 0.1.0

No breaking changes. New features are additive.

### Future Breaking Changes (v1.0)

The following may change before v1.0:
- Default threshold values (currently [T3])
- CLI exit codes
- Minimum n requirements for tests

See SPECIFICATION.md for current frozen values.
