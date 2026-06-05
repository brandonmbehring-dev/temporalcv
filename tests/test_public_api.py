"""Public-API stability test (issue #16).

The stable v2.0 surface is the top-level ``temporalcv`` namespace exported via ``__all__``
(ADR 0002). This test snapshots that surface so that **any** change to it — an addition, a
removal, or a rename — fails loudly and must be made deliberately by updating
``EXPECTED_PUBLIC_API`` below. That makes a public-contract change a reviewable diff, not an
accident.

It also guards two structural invariants: every exported name must be importable from the
top-level package (no dangling exports), and no private (``_``-prefixed) name may leak into the
public surface (the sole allowed dunder is ``__version__``).

If this test fails because you intentionally changed the public API, update
``EXPECTED_PUBLIC_API`` to match (the assertion message prints the added/removed names).
"""

from __future__ import annotations

import temporalcv

# The frozen v2.0 public surface (sorted). Update DELIBERATELY when adding/removing an export.
EXPECTED_PUBLIC_API: frozenset[str] = frozenset(
    {
        "AdaptiveConformalPredictor",
        "BellmanConformalPredictor",
        "BidirectionalEncompassingResult",
        "BlockedTimeSeriesCV",
        "BootstrapStrategy",
        "BootstrapUncertainty",
        "CWTestResult",
        "Changepoint",
        "ChangepointResult",
        "CombinatorialPurgedCV",
        "CoverageDiagnostics",
        "CrossFitCV",
        "CrossFitter",
        "DMTestResult",
        "EWMAVolatility",
        "EncompassingTestResult",
        "FeatureBagging",
        "GWTestResult",
        "GapSensitivityResult",
        "GateResult",
        "GateStatus",
        "GuardrailResult",
        "InfluenceDiagnostic",
        "JointStationarityResult",
        "LagSelectionResult",
        "MoveConditionalResult",
        "MoveDirection",
        "MovingBlockBootstrap",
        "MultiHorizonResult",
        "MultiModelComparisonResult",
        "MultiModelHorizonResult",
        "NestedCVResult",
        "NestedWalkForwardCV",
        "PTTestResult",
        "PredictionInterval",
        "PurgedKFold",
        "PurgedSplit",
        "PurgedWalkForward",
        "RealityCheckResult",
        "ResidualBootstrap",
        "RollingVolatility",
        "SPATestResult",
        "SplitConformalPredictor",
        "SplitInfo",
        "SplitResult",
        "Splitter",
        "StationarityConclusion",
        "StationarityTestResult",
        "StationaryBootstrap",
        "StratifiedMetricsResult",
        "StratifiedValidationReport",
        "SupportsBootstrap",
        "SupportsFitPredict",
        "SupportsForecast",
        "TemporalTags",
        "TimeSeriesBagger",
        "TimeSeriesCrossValidator",
        "ValidationReport",
        "VolatilityEstimator",
        "VolatilityStratifiedResult",
        "WalkForwardCV",
        "WalkForwardResults",
        "WildBootstrapResult",
        "__version__",
        "adf_test",
        "auto_select_lag",
        "check_against_ar1_bounds",
        "check_bootstrap_strategy",
        "check_forecast_adapter",
        "check_forecast_horizon_consistency",
        "check_minimum_sample_size",
        "check_residual_autocorrelation",
        "check_stationarity",
        "check_stratified_sample_size",
        "check_suspicious_improvement",
        "check_temporal_estimator",
        "check_temporal_splitter",
        "classify_direction_regime",
        "classify_moves",
        "classify_regimes_from_changepoints",
        "classify_volatility_regime",
        "compare_horizons",
        "compare_models_horizons",
        "compare_multiple_models",
        "compute_asymmetric_mape",
        "compute_bias",
        "compute_calmar_ratio",
        "compute_coverage_diagnostics",
        "compute_crps",
        "compute_cumulative_return",
        "compute_direction_accuracy",
        "compute_directional_loss",
        "compute_dm_influence",
        "compute_forecast_correlation",
        "compute_hac_variance",
        "compute_hit_rate",
        "compute_huber_loss",
        "compute_information_ratio",
        "compute_interval_score",
        "compute_label_overlap",
        "compute_linex_loss",
        "compute_local_volatility",
        "compute_mae",
        "compute_mape",
        "compute_mase",
        "compute_max_drawdown",
        "compute_move_conditional_metrics",
        "compute_move_only_mae",
        "compute_move_threshold",
        "compute_mrae",
        "compute_mse",
        "compute_naive_error",
        "compute_persistence_mae",
        "compute_pinball_loss",
        "compute_profit_factor",
        "compute_quantile_coverage",
        "compute_r_squared",
        "compute_rmse",
        "compute_sharpe_ratio",
        "compute_smape",
        "compute_squared_log_error",
        "compute_stratified_metrics",
        "compute_theils_u",
        "compute_volatility_normalized_mae",
        "compute_volatility_stratified_metrics",
        "compute_volatility_weighted_mae",
        "compute_winkler_score",
        "create_block_bagger",
        "create_feature_bagger",
        "create_regime_indicators",
        "create_residual_bagger",
        "create_stationary_bagger",
        "cross_fit_residualize",
        "cw_test",
        "detect_changepoints",
        "detect_changepoints_pelt",
        "detect_changepoints_variance",
        "difference_until_stationary",
        "dm_test",
        "estimate_purge_gap",
        "evaluate_interval_quality",
        "forecast_encompassing_bidirectional",
        "forecast_encompassing_test",
        "gap_sensitivity_analysis",
        "gate_residual_diagnostics",
        "gate_signal_verification",
        "gate_suspicious_improvement",
        "gate_synthetic_ar1",
        "gate_temporal_boundary",
        "gate_theoretical_bounds",
        "generate_ar1_series",
        "generate_ar2_series",
        "get_combined_regimes",
        "get_regime_counts",
        "get_segment_boundaries",
        "gw_test",
        "kpss_test",
        "mask_low_n_regimes",
        "pp_test",
        "pt_test",
        "reality_check_test",
        "run_all_guardrails",
        "run_gates",
        "run_gates_stratified",
        "select_lag_aic",
        "select_lag_bic",
        "select_lag_pacf",
        "spa_test",
        "suggest_cv_gap",
        "theoretical_ar1_mae_bound",
        "theoretical_ar1_mse_bound",
        "theoretical_ar2_mse_bound",
        "walk_forward_conformal",
        "walk_forward_evaluate",
        "wild_cluster_bootstrap",
    }
)


def test_public_api_matches_snapshot() -> None:
    """The exported public surface must match the frozen snapshot exactly (fail-loud on drift)."""
    actual = set(temporalcv.__all__)
    added = actual - EXPECTED_PUBLIC_API
    removed = EXPECTED_PUBLIC_API - actual
    assert actual == EXPECTED_PUBLIC_API, (
        "temporalcv public API (__all__) drifted from the v2.0 snapshot.\n"
        f"  ADDED (new exports not in snapshot):   {sorted(added)}\n"
        f"  REMOVED (snapshot names now missing):  {sorted(removed)}\n"
        "If this change is intentional, update EXPECTED_PUBLIC_API in tests/test_public_api.py "
        "(and confirm it is a deliberate public-contract change per ADR 0002)."
    )


def test_public_api_has_no_duplicates() -> None:
    """``__all__`` must not list any name twice."""
    all_list = list(temporalcv.__all__)
    seen: set[str] = set()
    dupes = sorted({n for n in all_list if n in seen or seen.add(n)})  # type: ignore[func-returns-value]
    assert not dupes, f"temporalcv.__all__ contains duplicate names: {dupes}"


def test_every_exported_name_is_importable() -> None:
    """No dangling exports: every name in ``__all__`` must resolve on the package."""
    missing = sorted(n for n in temporalcv.__all__ if not hasattr(temporalcv, n))
    assert not missing, f"temporalcv.__all__ lists names not importable from the package: {missing}"


def test_no_private_name_leaked() -> None:
    """No ``_``-prefixed name may be public except the ``__version__`` dunder."""
    leaked = sorted(n for n in temporalcv.__all__ if n.startswith("_") and n != "__version__")
    assert not leaked, f"private names leaked into temporalcv.__all__: {leaked}"
