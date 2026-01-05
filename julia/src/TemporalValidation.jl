"""
TemporalValidation.jl - Temporal validation for time-series ML in Julia.

Provides rigorous validation tools for time-series forecasting, including:
- Conformal prediction intervals with coverage guarantees
- Time-series-aware bootstrap methods (block, stationary)
- Bagging with temporal dependence preservation

This is the Julia companion to the Python `temporalcv` package.

# Quick Example
```julia
using TemporalValidation

# Conformal prediction
cp = SplitConformalPredictor(alpha=0.1)
calibrate!(cp, predictions_cal, actuals_cal)
intervals = predict_interval(cp, predictions_test)

# Time-series bagging
bagger = TimeSeriesBagger(
    base_model=MyModel(),
    strategy=StationaryBootstrap(expected_block_length=10.0),
    n_estimators=50
)
fit!(bagger, X_train, y_train)
mean_pred, std_pred = predict_with_uncertainty(bagger, X_test)
```
"""
module TemporalValidation

using Statistics
using Random
using Distributions
using LinearAlgebra

# =============================================================================
# Constants
# =============================================================================

"""Default miscoverage rate for conformal prediction."""
const DEFAULT_ALPHA = 0.1

"""Default calibration fraction for split conformal."""
const DEFAULT_CALIBRATION_FRACTION = 0.2

"""Default learning rate for adaptive conformal."""
const DEFAULT_GAMMA = 0.1

"""Default number of bootstrap estimators."""
const DEFAULT_N_ESTIMATORS = 50

# =============================================================================
# Submodules
# =============================================================================

include("conformal/Conformal.jl")
include("bagging/Bagging.jl")
include("cv/CV.jl")
include("statistical_tests/StatisticalTests.jl")
include("gates/Gates.jl")
include("stationarity/Stationarity.jl")
include("lag_selection.jl")
include("changepoint.jl")
include("metrics/Metrics.jl")
include("validators/Validators.jl")
include("diagnostics/Diagnostics.jl")
include("cv_financial/CVFinancial.jl")
include("guardrails/Guardrails.jl")
include("inference/Inference.jl")
include("regimes/Regimes.jl")
include("compare/Compare.jl")
include("benchmarks/Benchmarks.jl")

using .Conformal
using .Bagging
using .CV
using .StatisticalTests
using .Gates
using .Stationarity
using .Metrics
using .Validators
using .Diagnostics
using .CVFinancial
using .Guardrails
using .Inference
using .Regimes
using .Compare
using .Benchmarks

# =============================================================================
# Re-exports from Conformal
# =============================================================================

export PredictionInterval
export SplitConformalPredictor, AdaptiveConformalPredictor
export calibrate!, initialize!, predict_interval, update!
export width, mean_width, coverage

# =============================================================================
# Re-exports from Bagging
# =============================================================================

export BootstrapStrategy
export MovingBlockBootstrap, StationaryBootstrap
export TimeSeriesBagger
export bootstrap_sample, fit!, predict, predict_with_uncertainty

# =============================================================================
# Re-exports from CV
# =============================================================================

export SplitInfo, SplitResult, WalkForwardResults
export WalkForwardCV, CrossFitCV
export split, get_n_splits, get_split_info, get_fold_indices
export train_size, test_size, gap, errors, absolute_errors
export mae, rmse, bias, mse, n_splits, predictions, actuals, total_samples
export to_split_info

# =============================================================================
# Re-exports from StatisticalTests
# =============================================================================

export DMTestResult, PTTestResult
export significant_at_05, significant_at_01, skill
export compute_hac_variance, bartlett_kernel
export dm_test, pt_test

# =============================================================================
# Re-exports from Gates
# =============================================================================

export GateStatus, HALT, WARN, PASS, SKIP
export GateResult, ValidationReport
export status, failures, warnings  # summary conflicts, use qualified
export gate_suspicious_improvement, gate_temporal_boundary, gate_residual_diagnostics
export compute_acf, ljung_box_test
export run_gates

# =============================================================================
# Re-exports from Stationarity
# =============================================================================

export StationarityConclusion, STATIONARY, NON_STATIONARY
export DIFFERENCE_STATIONARY, INSUFFICIENT_EVIDENCE
export StationarityTestResult, JointStationarityResult
export adf_test, kpss_test
export check_stationarity, difference_until_stationary, integration_order

# =============================================================================
# Re-exports from Lag Selection
# =============================================================================

export LagSelectionResult
export select_lag_pacf, select_lag_aic, select_lag_bic
export auto_select_lag, suggest_cv_gap
export compute_pacf

# =============================================================================
# Re-exports from Changepoint Detection
# =============================================================================

export Changepoint, ChangepointResult
export detect_changepoints, detect_changepoints_variance
export classify_regimes, get_segment_boundaries
export create_regime_indicators

# =============================================================================
# Re-exports from Metrics
# =============================================================================

# Types
export MoveDirection, UP, DOWN, FLAT
export MoveConditionalResult
export BrierScoreResult
export VolatilityStratifiedResult
export IntervalScoreResult

# Type accessors
export n_total, n_moves, is_reliable, move_fraction
export decomposition_valid

# Core metrics
export compute_mae, compute_mse, compute_rmse
export compute_mape, compute_smape, compute_bias

# Scale-invariant metrics
export compute_naive_error, compute_mase, compute_mrae, compute_theils_u

# Correlation metrics
export compute_forecast_correlation, compute_r_squared

# Persistence metrics
export compute_move_threshold
export classify_moves
export compute_move_conditional_metrics
export compute_direction_accuracy
export compute_move_only_mae
export compute_persistence_mae

# Quantile metrics
export compute_pinball_loss
export compute_crps
export compute_interval_score
export compute_winkler_score
export compute_quantile_coverage

# Event metrics
export compute_direction_brier
export compute_calibrated_direction_brier
export skill_score

# Asymmetric loss functions
export compute_linex_loss
export compute_asymmetric_mape
export compute_directional_loss
export compute_squared_log_error
export compute_huber_loss

# Financial metrics
export compute_sharpe_ratio
export compute_max_drawdown
export compute_cumulative_return
export compute_information_ratio
export compute_hit_rate
export compute_profit_factor
export compute_calmar_ratio

# Volatility metrics
export VolatilityEstimator
export RollingVolatility
export EWMAVolatility
export estimate

export compute_local_volatility
export compute_volatility_normalized_mae
export compute_volatility_weighted_mae
export compute_volatility_stratified_metrics

# =============================================================================
# Re-exports from Validators
# =============================================================================

# Types
export AR1Bounds, AR2Bounds, BoundsCheckResult

# AR(1) functions
export compute_ar1_mse_bound, compute_ar1_mae_bound, compute_ar1_rmse_bound
export generate_ar1_series, estimate_ar1_params
export compute_ar1_bounds

# AR(2) functions
export compute_ar2_mse_bound, generate_ar2_series
export compute_ar2_bounds

# Checking
export check_against_ar1_bounds, check_against_ar2_bounds

# =============================================================================
# Re-exports from Diagnostics
# =============================================================================

# Types
export InfluenceDiagnostic, SensitivityResult, StabilityReport

# Influence functions
export compute_dm_influence, compute_block_influence
export identify_influential_points

# Sensitivity functions
export compute_parameter_sensitivity
export compute_stability_report
export bootstrap_metric_variance

# =============================================================================
# Re-exports from CVFinancial
# =============================================================================

# Types
export PurgedSplit
export PurgedKFold, CombinatorialPurgedCV, PurgedWalkForward

# Functions
export compute_label_overlap, estimate_purge_gap
export apply_purge_and_embargo
export get_train_test_indices, total_purged_samples, total_embargoed_samples

# =============================================================================
# Re-exports from Guardrails
# =============================================================================

# Types
export GuardrailResult, GuardrailSummary
export pass_result, fail_result, warn_result, skip_result

# Checks
export check_suspicious_improvement
export check_minimum_sample_size
export check_stratified_sample_size
export check_forecast_horizon_consistency
export check_residual_autocorrelation
export run_all_guardrails

# =============================================================================
# Re-exports from Inference
# =============================================================================

# Types
export WildBootstrapResult

# Functions
export rademacher_weights, webb_weights
export wild_cluster_bootstrap, wild_cluster_bootstrap_difference

# =============================================================================
# Re-exports from Regimes
# =============================================================================

# Types
export VolatilityRegime, VOL_LOW, VOL_MED, VOL_HIGH
export DirectionRegime, DIR_UP, DIR_DOWN, DIR_FLAT
export StratifiedMetricsResult

# Functions
export regime_string
export classify_volatility_regime, classify_direction_regime
export get_combined_regimes, get_regime_counts, mask_low_n_regimes
export compute_stratified_metrics

# =============================================================================
# Re-exports from Compare
# =============================================================================

# Types
export ModelResult, ComparisonResult, ComparisonReport
export ForecastAdapter, NaiveAdapter, SeasonalNaiveAdapter

# Functions (types.jl)
export get_metric, get_ranking, to_dict, to_markdown

# Functions (adapters.jl)
export model_name, package_name, fit_predict, get_params
export compute_comparison_metrics

# Functions (runner.jl)
export run_comparison, run_benchmark_suite, compare_to_baseline

# =============================================================================
# Re-exports from Benchmarks
# =============================================================================

# Types
export DatasetNotFoundError, DatasetMetadata, TimeSeriesDataset

# Type utilities
export n_obs, has_exogenous, get_train_test_split, validate_dataset

# Generators
export create_ar1_series
export create_synthetic_dataset
export create_electricity_like_dataset

# Bundled datasets
export create_bundled_test_datasets

# Benchmark integration
export to_benchmark_tuple, to_benchmark_tuples

# =============================================================================
# Module version
# =============================================================================

const VERSION = v"0.1.0"

end # module TemporalValidation
