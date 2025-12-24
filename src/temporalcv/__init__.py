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
>>> from temporalcv.gates import gate_shuffled_target
>>>
>>> # Pre-compute gates, then aggregate
>>> gates = [gate_shuffled_target(model=my_model, X=X, y=y, n_shuffles=5, random_state=42)]
>>> report = run_gates(gates)
>>> if report.status == "HALT":
...     raise ValueError(f"Leakage detected: {report.failures}")

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

__version__ = "0.1.0"

# Gates module exports
from temporalcv.gates import (
    GateStatus,
    GateResult,
    ValidationReport,
    StratifiedValidationReport,
    gate_shuffled_target,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
    gate_temporal_boundary,
    gate_residual_diagnostics,
    gate_theoretical_bounds,
    run_gates,
    run_gates_stratified,
)

# Statistical tests exports
from temporalcv.statistical_tests import (
    DMTestResult,
    PTTestResult,
    dm_test,
    pt_test,
    compute_hac_variance,
)

# Cross-validation exports
from temporalcv.cv import (
    SplitInfo,
    WalkForwardCV,
    CrossFitCV,
)

# Regime classification exports
from temporalcv.regimes import (
    classify_volatility_regime,
    classify_direction_regime,
    get_combined_regimes,
    get_regime_counts,
    mask_low_n_regimes,
)

# High-persistence metrics exports
from temporalcv.persistence import (
    MoveDirection,
    MoveConditionalResult,
    compute_move_threshold,
    classify_moves,
    compute_move_conditional_metrics,
    compute_direction_accuracy,
    compute_move_only_mae,
    compute_persistence_mae,
)

# Conformal prediction exports
from temporalcv.conformal import (
    PredictionInterval,
    SplitConformalPredictor,
    AdaptiveConformalPredictor,
    BootstrapUncertainty,
    evaluate_interval_quality,
    walk_forward_conformal,
)

# Bagging exports
from temporalcv.bagging import (
    BootstrapStrategy,
    TimeSeriesBagger,
    MovingBlockBootstrap,
    StationaryBootstrap,
    FeatureBagging,
    create_block_bagger,
    create_stationary_bagger,
    create_feature_bagger,
)

# Diagnostics exports
from temporalcv.diagnostics import (
    InfluenceDiagnostic,
    compute_dm_influence,
    GapSensitivityResult,
    gap_sensitivity_analysis,
)

# Inference exports
from temporalcv.inference import (
    WildBootstrapResult,
    wild_cluster_bootstrap,
)

__all__ = [
    "__version__",
    # Gates
    "GateStatus",
    "GateResult",
    "ValidationReport",
    "StratifiedValidationReport",
    "gate_shuffled_target",
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
    "dm_test",
    "pt_test",
    "compute_hac_variance",
    # Cross-validation
    "SplitInfo",
    "WalkForwardCV",
    "CrossFitCV",
    # Regime classification
    "classify_volatility_regime",
    "classify_direction_regime",
    "get_combined_regimes",
    "get_regime_counts",
    "mask_low_n_regimes",
    # High-persistence metrics
    "MoveDirection",
    "MoveConditionalResult",
    "compute_move_threshold",
    "classify_moves",
    "compute_move_conditional_metrics",
    "compute_direction_accuracy",
    "compute_move_only_mae",
    "compute_persistence_mae",
    # Conformal prediction
    "PredictionInterval",
    "SplitConformalPredictor",
    "AdaptiveConformalPredictor",
    "BootstrapUncertainty",
    "evaluate_interval_quality",
    "walk_forward_conformal",
    # Bagging
    "BootstrapStrategy",
    "TimeSeriesBagger",
    "MovingBlockBootstrap",
    "StationaryBootstrap",
    "FeatureBagging",
    "create_block_bagger",
    "create_stationary_bagger",
    "create_feature_bagger",
    # Diagnostics
    "InfluenceDiagnostic",
    "compute_dm_influence",
    "GapSensitivityResult",
    "gap_sensitivity_analysis",
    # Inference
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
]
