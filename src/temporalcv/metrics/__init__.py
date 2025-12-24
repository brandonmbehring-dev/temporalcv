"""
Metrics Subpackage for Time Series Forecast Evaluation.

Comprehensive metrics covering:
- **Core metrics**: MAE, MSE, RMSE, MAPE, SMAPE, bias
- **Scale-invariant**: MASE, MRAE, Theil's U
- **Correlation**: Pearson r, Spearman ρ, R²
- **Direction/Event metrics**: Brier score, PR-AUC

Example
-------
>>> from temporalcv.metrics import (
...     compute_rmse,
...     compute_mape,
...     compute_mase,
...     compute_naive_error,
...     compute_direction_brier,
... )
>>>
>>> # Basic metrics
>>> rmse = compute_rmse(predictions, actuals)
>>> mape = compute_mape(predictions, actuals)
>>>
>>> # Scale-invariant (compare across series)
>>> naive_mae = compute_naive_error(train_values)
>>> mase = compute_mase(predictions, actuals, naive_mae)
>>>
>>> # Direction prediction
>>> result = compute_direction_brier(pred_probs, actual_directions)
>>> print(f"Brier: {result.brier_score:.4f}")

References
----------
- Hyndman & Koehler (2006). Another look at measures of forecast accuracy.
- Brier (1950). Verification of forecasts expressed in terms of probability.
- Theil (1966). Applied Economic Forecasting.
"""

from temporalcv.metrics.core import (
    compute_bias,
    compute_forecast_correlation,
    compute_mae,
    compute_mape,
    compute_mase,
    compute_mrae,
    compute_mse,
    compute_naive_error,
    compute_r_squared,
    compute_rmse,
    compute_smape,
    compute_theils_u,
)

from temporalcv.metrics.event import (
    BrierScoreResult,
    PRAUCResult,
    compute_calibrated_direction_brier,
    compute_direction_brier,
    compute_pr_auc,
    convert_predictions_to_direction_probs,
)

from temporalcv.metrics.quantile import (
    compute_crps,
    compute_interval_score,
    compute_pinball_loss,
    compute_quantile_coverage,
    compute_winkler_score,
)

from temporalcv.metrics.financial import (
    compute_calmar_ratio,
    compute_cumulative_return,
    compute_hit_rate,
    compute_information_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
)

from temporalcv.metrics.asymmetric import (
    compute_asymmetric_mape,
    compute_directional_loss,
    compute_huber_loss,
    compute_linex_loss,
    compute_squared_log_error,
)

from temporalcv.metrics.volatility_weighted import (
    EWMAVolatility,
    RollingVolatility,
    VolatilityEstimator,
    VolatilityStratifiedResult,
    compute_local_volatility,
    compute_volatility_normalized_mae,
    compute_volatility_stratified_metrics,
    compute_volatility_weighted_mae,
)

__all__ = [
    # Core metrics
    "compute_mae",
    "compute_mse",
    "compute_rmse",
    "compute_mape",
    "compute_smape",
    "compute_bias",
    # Scale-invariant metrics
    "compute_naive_error",
    "compute_mase",
    "compute_mrae",
    "compute_theils_u",
    # Correlation metrics
    "compute_forecast_correlation",
    "compute_r_squared",
    # Direction/Event metrics
    "BrierScoreResult",
    "PRAUCResult",
    "compute_direction_brier",
    "compute_pr_auc",
    "compute_calibrated_direction_brier",
    "convert_predictions_to_direction_probs",
    # Quantile/Interval metrics
    "compute_pinball_loss",
    "compute_crps",
    "compute_interval_score",
    "compute_quantile_coverage",
    "compute_winkler_score",
    # Financial/Trading metrics
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
]
