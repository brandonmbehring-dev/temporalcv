using TemporalValidation
using Test
using Random
using Statistics

# Import from Conformal submodule
import TemporalValidation.Conformal: PredictionInterval, SplitConformalPredictor, AdaptiveConformalPredictor
import TemporalValidation.Conformal: calibrate!, initialize!, update!
import TemporalValidation.Conformal: width, mean_width, coverage
import TemporalValidation.Conformal: predict_interval as conformal_predict_interval

# Import from Bagging submodule
import TemporalValidation.Bagging: MovingBlockBootstrap, StationaryBootstrap, TimeSeriesBagger
import TemporalValidation.Bagging: bootstrap_sample, fit!, predict, predict_with_uncertainty
import TemporalValidation.Bagging: predict_interval as bagger_predict_interval

# Import from CV submodule
import TemporalValidation.CV: SplitInfo, SplitResult, WalkForwardResults
import TemporalValidation.CV: WalkForwardCV, CrossFitCV
import TemporalValidation.CV: split, get_n_splits, get_split_info, get_fold_indices
import TemporalValidation.CV: train_size, test_size, gap, errors, absolute_errors
import TemporalValidation.CV: mae, rmse, bias, mse, n_splits, predictions, actuals
import TemporalValidation.CV: total_samples, to_split_info

# Import from StatisticalTests submodule
import TemporalValidation.StatisticalTests: DMTestResult, PTTestResult
import TemporalValidation.StatisticalTests: significant_at_05, significant_at_01, skill
import TemporalValidation.StatisticalTests: compute_hac_variance, bartlett_kernel
import TemporalValidation.StatisticalTests: dm_test, pt_test

# Import from Gates submodule
import TemporalValidation.Gates: GateStatus, HALT, WARN, PASS, SKIP
import TemporalValidation.Gates: GateResult, ValidationReport
import TemporalValidation.Gates: status, failures, warnings
import TemporalValidation.Gates: gate_suspicious_improvement, gate_temporal_boundary
import TemporalValidation.Gates: gate_residual_diagnostics
import TemporalValidation.Gates: compute_acf, ljung_box_test
import TemporalValidation.Gates: run_gates
import TemporalValidation.Gates

# Import from Stationarity submodule
import TemporalValidation.Stationarity: StationarityConclusion
import TemporalValidation.Stationarity: STATIONARY, NON_STATIONARY
import TemporalValidation.Stationarity: DIFFERENCE_STATIONARY, INSUFFICIENT_EVIDENCE
import TemporalValidation.Stationarity: StationarityTestResult, JointStationarityResult
import TemporalValidation.Stationarity: adf_test, kpss_test
import TemporalValidation.Stationarity: check_stationarity, difference_until_stationary
import TemporalValidation.Stationarity: integration_order

# Import from Lag Selection (top-level)
import TemporalValidation: LagSelectionResult
import TemporalValidation: select_lag_pacf, select_lag_aic, select_lag_bic
import TemporalValidation: auto_select_lag, suggest_cv_gap
import TemporalValidation: compute_pacf

# Import from Changepoint Detection (top-level)
import TemporalValidation: Changepoint, ChangepointResult
import TemporalValidation: detect_changepoints, detect_changepoints_variance
import TemporalValidation: classify_regimes, get_segment_boundaries
import TemporalValidation: create_regime_indicators

# Import from Metrics submodule
import TemporalValidation.Metrics: MoveDirection, UP, DOWN, FLAT
import TemporalValidation.Metrics: MoveConditionalResult, BrierScoreResult
import TemporalValidation.Metrics: VolatilityStratifiedResult, IntervalScoreResult
import TemporalValidation.Metrics: n_total, n_moves, is_reliable, move_fraction
import TemporalValidation.Metrics: decomposition_valid
import TemporalValidation.Metrics: compute_mae, compute_mse, compute_rmse
import TemporalValidation.Metrics: compute_mape, compute_smape, compute_bias
import TemporalValidation.Metrics: compute_naive_error, compute_mase, compute_mrae, compute_theils_u
import TemporalValidation.Metrics: compute_forecast_correlation, compute_r_squared
import TemporalValidation.Metrics: compute_move_threshold, classify_moves
import TemporalValidation.Metrics: compute_move_conditional_metrics
import TemporalValidation.Metrics: compute_direction_accuracy
import TemporalValidation.Metrics: compute_move_only_mae, compute_persistence_mae
import TemporalValidation.Metrics: compute_pinball_loss, compute_crps
import TemporalValidation.Metrics: compute_interval_score, compute_winkler_score
import TemporalValidation.Metrics: compute_quantile_coverage
import TemporalValidation.Metrics: compute_direction_brier, compute_calibrated_direction_brier
import TemporalValidation.Metrics: skill_score

# Import from Asymmetric metrics
import TemporalValidation.Metrics: compute_linex_loss, compute_asymmetric_mape
import TemporalValidation.Metrics: compute_directional_loss, compute_squared_log_error
import TemporalValidation.Metrics: compute_huber_loss

# Import from Financial metrics
import TemporalValidation.Metrics: compute_sharpe_ratio, compute_max_drawdown
import TemporalValidation.Metrics: compute_cumulative_return, compute_information_ratio
import TemporalValidation.Metrics: compute_hit_rate, compute_profit_factor
import TemporalValidation.Metrics: compute_calmar_ratio

# Import from Volatility metrics
import TemporalValidation.Metrics: VolatilityEstimator, RollingVolatility, EWMAVolatility
import TemporalValidation.Metrics: estimate, compute_local_volatility
import TemporalValidation.Metrics: compute_volatility_normalized_mae
import TemporalValidation.Metrics: compute_volatility_weighted_mae
import TemporalValidation.Metrics: compute_volatility_stratified_metrics

# Import from Validators submodule
import TemporalValidation.Validators: AR1Bounds, AR2Bounds, BoundsCheckResult
import TemporalValidation.Validators: compute_ar1_mse_bound, compute_ar1_mae_bound, compute_ar1_rmse_bound
import TemporalValidation.Validators: generate_ar1_series, estimate_ar1_params
import TemporalValidation.Validators: compute_ar2_mse_bound, generate_ar2_series
import TemporalValidation.Validators: check_against_ar1_bounds, check_against_ar2_bounds
import TemporalValidation.Validators: compute_ar1_bounds, compute_ar2_bounds

# Import from Diagnostics submodule
import TemporalValidation.Diagnostics: InfluenceDiagnostic, SensitivityResult, StabilityReport
import TemporalValidation.Diagnostics: compute_dm_influence, compute_block_influence
import TemporalValidation.Diagnostics: identify_influential_points
import TemporalValidation.Diagnostics: compute_parameter_sensitivity
import TemporalValidation.Diagnostics: compute_stability_report
import TemporalValidation.Diagnostics: bootstrap_metric_variance

# Import from CVFinancial submodule
import TemporalValidation.CVFinancial: PurgedSplit
import TemporalValidation.CVFinancial: PurgedKFold, CombinatorialPurgedCV, PurgedWalkForward
import TemporalValidation.CVFinancial: compute_label_overlap, estimate_purge_gap
import TemporalValidation.CVFinancial: apply_purge_and_embargo
import TemporalValidation.CVFinancial: get_train_test_indices, total_purged_samples, total_embargoed_samples
# Note: split and get_n_splits conflict with CV module, use qualified access in tests
import TemporalValidation.CVFinancial
const CVFin = TemporalValidation.CVFinancial

# Import from Guardrails submodule
import TemporalValidation.Guardrails: GuardrailResult, GuardrailSummary
import TemporalValidation.Guardrails: pass_result, fail_result, warn_result, skip_result
import TemporalValidation.Guardrails: check_suspicious_improvement
import TemporalValidation.Guardrails: check_minimum_sample_size
import TemporalValidation.Guardrails: check_stratified_sample_size
import TemporalValidation.Guardrails: check_forecast_horizon_consistency
import TemporalValidation.Guardrails: check_residual_autocorrelation
import TemporalValidation.Guardrails: run_all_guardrails

# Import from Inference submodule
import TemporalValidation.Inference: WildBootstrapResult
import TemporalValidation.Inference: rademacher_weights, webb_weights
import TemporalValidation.Inference: wild_cluster_bootstrap, wild_cluster_bootstrap_difference

# Import from Regimes submodule
import TemporalValidation.Regimes: VolatilityRegime, VOL_LOW, VOL_MED, VOL_HIGH
import TemporalValidation.Regimes: DirectionRegime, DIR_UP, DIR_DOWN, DIR_FLAT
import TemporalValidation.Regimes: StratifiedMetricsResult
import TemporalValidation.Regimes: regime_string
import TemporalValidation.Regimes: classify_volatility_regime, classify_direction_regime
import TemporalValidation.Regimes: get_combined_regimes, get_regime_counts, mask_low_n_regimes
import TemporalValidation.Regimes: compute_stratified_metrics
import TemporalValidation.Regimes
const RegimeSummary = TemporalValidation.Regimes.summary  # Avoid collision with Statistics.summary

# Import from Compare submodule
import TemporalValidation.Compare: ModelResult, ComparisonResult, ComparisonReport
import TemporalValidation.Compare: ForecastAdapter, NaiveAdapter, SeasonalNaiveAdapter
import TemporalValidation.Compare: get_metric, get_ranking, to_dict, to_markdown
import TemporalValidation.Compare: model_name, package_name, fit_predict, get_params
import TemporalValidation.Compare: compute_comparison_metrics
import TemporalValidation.Compare: run_comparison, run_benchmark_suite, compare_to_baseline

# Import from Benchmarks submodule
import TemporalValidation.Benchmarks: DatasetNotFoundError, DatasetMetadata, TimeSeriesDataset
import TemporalValidation.Benchmarks: n_obs, has_exogenous, get_train_test_split, validate_dataset
import TemporalValidation.Benchmarks: create_ar1_series, create_synthetic_dataset
import TemporalValidation.Benchmarks: create_electricity_like_dataset
import TemporalValidation.Benchmarks: create_bundled_test_datasets
import TemporalValidation.Benchmarks: to_benchmark_tuple, to_benchmark_tuples

# Create unified predict_interval that dispatches on type
predict_interval(cp::SplitConformalPredictor, preds) = conformal_predict_interval(cp, preds)
predict_interval(acp::AdaptiveConformalPredictor, pred) = conformal_predict_interval(acp, pred)
predict_interval(bagger::TimeSeriesBagger, X; alpha=0.1) = bagger_predict_interval(bagger, X; alpha=alpha)

@testset "TemporalValidation.jl" begin

    @testset "Conformal" begin
        include("conformal/test_types.jl")
        include("conformal/test_split.jl")
        include("conformal/test_adaptive.jl")
    end

    @testset "Bagging" begin
        include("bagging/test_block_bootstrap.jl")
        include("bagging/test_stationary_bootstrap.jl")
        include("bagging/test_bagger.jl")
    end

    @testset "CV" begin
        include("cv/test_types.jl")
        include("cv/test_walk_forward.jl")
        include("cv/test_cross_fit.jl")
    end

    @testset "StatisticalTests" begin
        include("statistical_tests/test_hac.jl")
        include("statistical_tests/test_dm.jl")
        include("statistical_tests/test_pt.jl")
    end

    @testset "Gates" begin
        include("gates/test_types.jl")
        include("gates/test_gates.jl")
    end

    @testset "Stationarity" begin
        include("stationarity/test_types.jl")
        include("stationarity/test_stationarity.jl")
    end

    @testset "LagSelection" begin
        include("test_lag_selection.jl")
    end

    @testset "Changepoint" begin
        include("test_changepoint.jl")
    end

    @testset "Metrics" begin
        include("metrics/test_types.jl")
        include("metrics/test_core.jl")
        include("metrics/test_persistence.jl")
        include("metrics/test_quantile.jl")
        include("metrics/test_event.jl")
        include("metrics/test_asymmetric.jl")
        include("metrics/test_financial.jl")
        include("metrics/test_volatility.jl")
    end

    @testset "Validators" begin
        include("validators/test_theoretical.jl")
    end

    @testset "Diagnostics" begin
        include("diagnostics/test_influence.jl")
        include("diagnostics/test_sensitivity.jl")
    end

    @testset "CVFinancial" begin
        include("cv_financial/test_purged.jl")
    end

    @testset "Guardrails" begin
        include("guardrails/test_guardrails.jl")
    end

    @testset "Inference" begin
        include("inference/test_wild_bootstrap.jl")
    end

    @testset "Regimes" begin
        include("regimes/test_regimes.jl")
    end

    @testset "Compare" begin
        include("compare/test_compare.jl")
    end

    @testset "Benchmarks" begin
        include("benchmarks/test_types.jl")
        include("benchmarks/test_synthetic.jl")
    end

end
