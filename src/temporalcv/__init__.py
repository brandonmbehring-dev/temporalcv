"""
temporalcv: Temporal cross-validation with leakage protection for time-series ML.

This package provides rigorous validation tools for time-series forecasting,
including:

- Validation gates for detecting data leakage
- Walk-forward cross-validation with gap enforcement
- Statistical tests (Diebold-Mariano, Pesaran-Timmermann)
- High-persistence series handling (MC-SS, move thresholds)
- Regime classification (volatility, direction)
- Conformal prediction intervals with coverage guarantees
- Time-series-aware bagging with bootstrap strategies

Example
-------
>>> from temporalcv import run_gates, WalkForwardCV
>>> from temporalcv.gates import gate_signal_verification
>>>
>>> # Signal verification: does model have predictive power?
>>> result = gate_signal_verification(model=my_model, X=X, y=y, random_state=42)
>>> if result.status.name == "HALT":
...     # Model has signal - could be legitimate or leakage
...     print("Model has signal - investigate source")
>>>
>>> # Aggregate multiple gates
>>> report = run_gates([result])
>>> if report.status == "HALT":
...     print(f"Investigation needed: {report.failures}")

>>> # Move-conditional metrics for high-persistence series
>>> from temporalcv import compute_move_threshold, compute_move_conditional_metrics
>>> threshold = compute_move_threshold(train_actuals)  # From training only!
>>> mc = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)
>>> print(f"MC-SS: {mc.skill_score:.3f}")

>>> # Conformal prediction intervals
>>> from temporalcv import SplitConformalPredictor
>>> conformal = SplitConformalPredictor(alpha=0.05)
>>> conformal.calibrate(cal_preds, cal_actuals)
>>> intervals = conformal.predict_interval(test_preds)
>>> print(f"Coverage: {intervals.coverage(test_actuals):.1%}")
"""

from __future__ import annotations

__version__ = "1.0.0"

# Gates module exports
# Bagging exports
from temporalcv.bagging import (
    BootstrapStrategy,
    FeatureBagging,
    MovingBlockBootstrap,
    ResidualBootstrap,
    StationaryBootstrap,
    TimeSeriesBagger,
    create_block_bagger,
    create_feature_bagger,
    create_residual_bagger,
    create_stationary_bagger,
)

# Changepoint detection exports
from temporalcv.changepoint import (
    Changepoint,
    ChangepointResult,
    classify_regimes_from_changepoints,
    create_regime_indicators,
    detect_changepoints,
    detect_changepoints_pelt,
    detect_changepoints_variance,
    get_segment_boundaries,
)

# Conformal prediction exports
from temporalcv.conformal import (
    AdaptiveConformalPredictor,
    BellmanConformalPredictor,
    BootstrapUncertainty,
    CoverageDiagnostics,
    PredictionInterval,
    SplitConformalPredictor,
    compute_coverage_diagnostics,
    evaluate_interval_quality,
    walk_forward_conformal,
)

# Cross-validation exports
from temporalcv.cv import (
    CrossFitCV,
    NestedCVResult,
    NestedWalkForwardCV,
    SplitInfo,
    SplitResult,
    WalkForwardCV,
    WalkForwardResults,
    walk_forward_evaluate,
)

# Financial CV exports
from temporalcv.cv_financial import (
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedSplit,
    PurgedWalkForward,
    compute_label_overlap,
    estimate_purge_gap,
)

# Diagnostics exports
from temporalcv.diagnostics import (
    GapSensitivityResult,
    InfluenceDiagnostic,
    compute_dm_influence,
    gap_sensitivity_analysis,
)
from temporalcv.gates import (
    GateResult,
    GateStatus,
    StratifiedValidationReport,
    ValidationReport,
    gate_residual_diagnostics,
    gate_signal_verification,
    gate_suspicious_improvement,
    gate_synthetic_ar1,
    gate_temporal_boundary,
    gate_theoretical_bounds,
    run_gates,
    run_gates_stratified,
)

# Guardrails exports (unified validation)
from temporalcv.guardrails import (
    GuardrailResult,
    check_forecast_horizon_consistency,
    check_minimum_sample_size,
    check_residual_autocorrelation,
    check_stratified_sample_size,
    check_suspicious_improvement,
    run_all_guardrails,
)

# Inference exports
from temporalcv.inference import (
    WildBootstrapResult,
    wild_cluster_bootstrap,
)

# Lag selection exports
from temporalcv.lag_selection import (
    LagSelectionResult,
    auto_select_lag,
    select_lag_aic,
    select_lag_bic,
    select_lag_pacf,
    suggest_cv_gap,
)

# Core metrics exports
# Quantile/interval metrics exports
# Financial/trading metrics exports
# Asymmetric loss exports
# Volatility-weighted metrics exports
from temporalcv.metrics import (
    EWMAVolatility,
    RollingVolatility,
    VolatilityEstimator,
    VolatilityStratifiedResult,
    compute_asymmetric_mape,
    compute_bias,
    compute_calmar_ratio,
    compute_crps,
    compute_cumulative_return,
    compute_directional_loss,
    compute_forecast_correlation,
    compute_hit_rate,
    compute_huber_loss,
    compute_information_ratio,
    compute_interval_score,
    compute_linex_loss,
    compute_local_volatility,
    compute_mae,
    compute_mape,
    compute_mase,
    compute_max_drawdown,
    compute_mrae,
    compute_mse,
    compute_naive_error,
    compute_pinball_loss,
    compute_profit_factor,
    compute_quantile_coverage,
    compute_r_squared,
    compute_rmse,
    compute_sharpe_ratio,
    compute_smape,
    compute_squared_log_error,
    compute_theils_u,
    compute_volatility_normalized_mae,
    compute_volatility_stratified_metrics,
    compute_volatility_weighted_mae,
    compute_winkler_score,
)

# High-persistence metrics exports
from temporalcv.persistence import (
    MoveConditionalResult,
    MoveDirection,
    classify_moves,
    compute_direction_accuracy,
    compute_move_conditional_metrics,
    compute_move_only_mae,
    compute_move_threshold,
    compute_persistence_mae,
)

# Regime classification exports
from temporalcv.regimes import (
    StratifiedMetricsResult,
    classify_direction_regime,
    classify_volatility_regime,
    compute_stratified_metrics,
    get_combined_regimes,
    get_regime_counts,
    mask_low_n_regimes,
)

# Stationarity tests exports
from temporalcv.stationarity import (
    JointStationarityResult,
    StationarityConclusion,
    StationarityTestResult,
    adf_test,
    check_stationarity,
    difference_until_stationary,
    kpss_test,
    pp_test,
)

# Statistical tests exports
from temporalcv.statistical_tests import (
    BidirectionalEncompassingResult,
    CWTestResult,
    DMTestResult,
    EncompassingTestResult,
    GWTestResult,
    MultiHorizonResult,
    MultiModelComparisonResult,
    MultiModelHorizonResult,
    PTTestResult,
    RealityCheckResult,
    SPATestResult,
    compare_horizons,
    compare_models_horizons,
    compare_multiple_models,
    compute_hac_variance,
    cw_test,
    dm_test,
    forecast_encompassing_bidirectional,
    forecast_encompassing_test,
    gw_test,
    pt_test,
    reality_check_test,
    spa_test,
)

# Validators exports (theoretical bounds)
from temporalcv.validators import (
    check_against_ar1_bounds,
    generate_ar1_series,
    generate_ar2_series,
    theoretical_ar1_mae_bound,
    theoretical_ar1_mse_bound,
    theoretical_ar2_mse_bound,
)

__all__ = [
    "__version__",
    # Gates
    "GateStatus",
    "GateResult",
    "ValidationReport",
    "StratifiedValidationReport",
    "gate_signal_verification",
    "gate_synthetic_ar1",
    "gate_suspicious_improvement",
    "gate_temporal_boundary",
    "gate_residual_diagnostics",
    "gate_theoretical_bounds",
    "run_gates",
    "run_gates_stratified",
    # Statistical tests
    "DMTestResult",
    "PTTestResult",
    "GWTestResult",
    "CWTestResult",
    "EncompassingTestResult",
    "BidirectionalEncompassingResult",
    "RealityCheckResult",
    "SPATestResult",
    "MultiHorizonResult",
    "MultiModelHorizonResult",
    "MultiModelComparisonResult",
    "dm_test",
    "pt_test",
    "gw_test",
    "cw_test",
    "compare_multiple_models",
    "compare_horizons",
    "compare_models_horizons",
    "forecast_encompassing_test",
    "forecast_encompassing_bidirectional",
    "reality_check_test",
    "spa_test",
    "compute_hac_variance",
    # Cross-validation
    "SplitInfo",
    "SplitResult",
    "WalkForwardResults",
    "NestedCVResult",
    "WalkForwardCV",
    "CrossFitCV",
    "NestedWalkForwardCV",
    "walk_forward_evaluate",
    # Regime classification
    "classify_volatility_regime",
    "classify_direction_regime",
    "get_combined_regimes",
    "get_regime_counts",
    "mask_low_n_regimes",
    "StratifiedMetricsResult",
    "compute_stratified_metrics",
    # High-persistence metrics
    "MoveDirection",
    "MoveConditionalResult",
    "compute_move_threshold",
    "classify_moves",
    "compute_move_conditional_metrics",
    "compute_direction_accuracy",
    "compute_move_only_mae",
    "compute_persistence_mae",
    # Core metrics
    "compute_mae",
    "compute_mse",
    "compute_rmse",
    "compute_mape",
    "compute_smape",
    "compute_bias",
    "compute_naive_error",
    "compute_mase",
    "compute_mrae",
    "compute_theils_u",
    "compute_forecast_correlation",
    "compute_r_squared",
    # Quantile/interval metrics
    "compute_pinball_loss",
    "compute_crps",
    "compute_interval_score",
    "compute_quantile_coverage",
    "compute_winkler_score",
    # Financial/trading metrics
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "compute_cumulative_return",
    "compute_information_ratio",
    "compute_hit_rate",
    "compute_profit_factor",
    "compute_calmar_ratio",
    # Asymmetric loss functions
    "compute_linex_loss",
    "compute_asymmetric_mape",
    "compute_directional_loss",
    "compute_squared_log_error",
    "compute_huber_loss",
    # Volatility-weighted metrics
    "VolatilityEstimator",
    "RollingVolatility",
    "EWMAVolatility",
    "compute_local_volatility",
    "compute_volatility_normalized_mae",
    "compute_volatility_weighted_mae",
    "VolatilityStratifiedResult",
    "compute_volatility_stratified_metrics",
    # Conformal prediction
    "PredictionInterval",
    "SplitConformalPredictor",
    "AdaptiveConformalPredictor",
    "BellmanConformalPredictor",
    "BootstrapUncertainty",
    "evaluate_interval_quality",
    "walk_forward_conformal",
    "CoverageDiagnostics",
    "compute_coverage_diagnostics",
    # Bagging
    "BootstrapStrategy",
    "TimeSeriesBagger",
    "MovingBlockBootstrap",
    "StationaryBootstrap",
    "FeatureBagging",
    "ResidualBootstrap",
    "create_block_bagger",
    "create_stationary_bagger",
    "create_feature_bagger",
    "create_residual_bagger",
    # Diagnostics
    "InfluenceDiagnostic",
    "compute_dm_influence",
    "GapSensitivityResult",
    "gap_sensitivity_analysis",
    # Inference
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
    # Validators (theoretical bounds)
    "theoretical_ar1_mse_bound",
    "theoretical_ar1_mae_bound",
    "theoretical_ar2_mse_bound",
    "check_against_ar1_bounds",
    "generate_ar1_series",
    "generate_ar2_series",
    # Guardrails (unified validation)
    "GuardrailResult",
    "check_suspicious_improvement",
    "check_minimum_sample_size",
    "check_stratified_sample_size",
    "check_forecast_horizon_consistency",
    "check_residual_autocorrelation",
    "run_all_guardrails",
    # Stationarity tests
    "StationarityTestResult",
    "StationarityConclusion",
    "JointStationarityResult",
    "adf_test",
    "kpss_test",
    "pp_test",
    "check_stationarity",
    "difference_until_stationary",
    # Lag selection
    "LagSelectionResult",
    "select_lag_pacf",
    "select_lag_aic",
    "select_lag_bic",
    "auto_select_lag",
    "suggest_cv_gap",
    # Changepoint detection
    "Changepoint",
    "ChangepointResult",
    "detect_changepoints",
    "detect_changepoints_variance",
    "detect_changepoints_pelt",
    "classify_regimes_from_changepoints",
    "create_regime_indicators",
    "get_segment_boundaries",
    # Financial CV
    "PurgedSplit",
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "PurgedWalkForward",
    "compute_label_overlap",
    "estimate_purge_gap",
]
