# =============================================================================
# Metrics Module
# =============================================================================

"""
    Metrics

Forecast evaluation metrics for time series validation.

Provides comprehensive metrics for evaluating time series forecasts:
- Core metrics: MAE, MSE, RMSE, MAPE, SMAPE, bias
- Scale-invariant: MASE, Theil's U, MRAE, R²
- Persistence/high-persistence: MC-SS, move-conditional MAE
- Direction accuracy: 2-class and 3-class

# Knowledge Tiers
- [T1] Core metrics are standard statistical formulations
- [T1] MASE: Hyndman & Koehler (2006)
- [T2] MC-SS: Move-Conditional Skill Score (myga-forecasting-v2)
- [T2] 70th percentile threshold for "significant" moves

# Example
```julia
using TemporalValidation.Metrics

# Basic metrics
mae = compute_mae(predictions, actuals)
rmse = compute_rmse(predictions, actuals)

# Scale-invariant
naive_mae = compute_naive_error(train_values)
mase = compute_mase(predictions, actuals, naive_mae)

# Move-conditional (high-persistence series)
threshold = compute_move_threshold(train_actuals)
mc = compute_move_conditional_metrics(predictions, actuals; threshold=threshold)
println("MC-SS: ", mc.skill_score)
```
"""
module Metrics

using Statistics
using LinearAlgebra

# =============================================================================
# Include source files
# =============================================================================

include("types.jl")
include("core.jl")
include("persistence.jl")
include("quantile.jl")
include("event.jl")
include("asymmetric.jl")
include("financial.jl")
include("volatility.jl")

# =============================================================================
# Exports: Types
# =============================================================================

export MoveDirection, UP, DOWN, FLAT
export MoveConditionalResult
export BrierScoreResult
export VolatilityStratifiedResult
export IntervalScoreResult

# Type accessors
export n_total, n_moves, is_reliable, move_fraction
export decomposition_valid

# =============================================================================
# Exports: Core Metrics
# =============================================================================

# Point forecast metrics
export compute_mae, compute_mse, compute_rmse
export compute_mape, compute_smape, compute_bias

# Scale-invariant metrics
export compute_naive_error, compute_mase, compute_mrae, compute_theils_u

# Correlation metrics
export compute_forecast_correlation, compute_r_squared

# =============================================================================
# Exports: Persistence Metrics
# =============================================================================

export compute_move_threshold
export classify_moves
export compute_move_conditional_metrics
export compute_direction_accuracy
export compute_move_only_mae
export compute_persistence_mae

# =============================================================================
# Exports: Quantile Metrics
# =============================================================================

export compute_pinball_loss
export compute_crps
export compute_interval_score
export compute_winkler_score
export compute_quantile_coverage

# =============================================================================
# Exports: Event Metrics
# =============================================================================

export compute_direction_brier
export compute_calibrated_direction_brier
export skill_score

# =============================================================================
# Exports: Asymmetric Metrics
# =============================================================================

export compute_linex_loss
export compute_asymmetric_mape
export compute_directional_loss
export compute_squared_log_error
export compute_huber_loss

# =============================================================================
# Exports: Financial Metrics
# =============================================================================

export compute_sharpe_ratio
export compute_max_drawdown
export compute_cumulative_return
export compute_information_ratio
export compute_hit_rate
export compute_profit_factor
export compute_calmar_ratio

# =============================================================================
# Exports: Volatility Metrics
# =============================================================================

export VolatilityEstimator
export RollingVolatility
export EWMAVolatility
export estimate

export compute_local_volatility
export compute_volatility_normalized_mae
export compute_volatility_weighted_mae
export compute_volatility_stratified_metrics

# =============================================================================
# Module version
# =============================================================================

const VERSION = v"0.1.0"

end # module Metrics
