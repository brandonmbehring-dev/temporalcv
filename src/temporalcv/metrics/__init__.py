"""
Metrics Subpackage for Time Series Forecast Evaluation.

Comprehensive metrics covering:
- **Core metrics**: MAE, MSE, RMSE, MAPE, SMAPE, bias
- **Scale-invariant**: MASE, MRAE, Theil's U
- **Correlation**: Pearson r, Spearman ρ, R²
- **Direction/Event metrics**: Brier score, PR-AUC

Example
-------
>>> import numpy as np
>>> from temporalcv.metrics import (
...     compute_rmse,
...     compute_mape,
...     compute_mase,
...     compute_naive_error,
...     compute_direction_brier,
... )
>>>
>>> actuals = np.array([100.0, 101.0, 102.0, 103.0])
>>> predictions = np.array([100.5, 100.5, 102.5, 102.5])
>>> train_values = np.array([97.0, 98.0, 99.0, 100.0])
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
>>> pred_probs = np.array([0.8, 0.2, 0.6, 0.9])
>>> actual_directions = np.array([1, 0, 1, 1])
>>> result = compute_direction_brier(pred_probs, actual_directions)
>>> print(f"Brier: {result.brier_score:.4f}")
Brier: 0.0625

References
----------
- Hyndman & Koehler (2006). Another look at measures of forecast accuracy.
- Brier (1950). Verification of forecasts expressed in terms of probability.
- Theil (1966). Applied Economic Forecasting.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from temporalcv.metrics.asymmetric import (
    compute_asymmetric_mape,
    compute_directional_loss,
    compute_huber_loss,
    compute_linex_loss,
    compute_squared_log_error,
)
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
from temporalcv.metrics.financial import (
    compute_calmar_ratio,
    compute_cumulative_return,
    compute_hit_rate,
    compute_information_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
)
from temporalcv.metrics.quantile import (
    compute_crps,
    compute_interval_score,
    compute_pinball_loss,
    compute_quantile_coverage,
    compute_winkler_score,
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

# ---------------------------------------------------------------------------
# Convenience aliases for README Quick Start
# ---------------------------------------------------------------------------

#: Alias for :func:`compute_mase` — scale-free error relative to naive forecast.
mase = compute_mase


def mc_skill_score(
    actuals: ArrayLike,
    predictions: ArrayLike,
    *,
    threshold: float | None = None,
    threshold_percentile: float = 70.0,
) -> float:
    """Return the Move-Conditional Skill Score as a scalar float.

    Convenience wrapper around :func:`~temporalcv.persistence.compute_move_conditional_metrics`
    for quick one-liner evaluation.

    Parameters
    ----------
    actuals : array-like
        Actual changes / returns.
    predictions : array-like
        Predicted changes / returns.
    threshold : float, optional
        Move threshold.  If *None*, auto-computed from *actuals*
        at *threshold_percentile*.
    threshold_percentile : float, default=70.0
        Percentile for auto-threshold (used only when *threshold* is None).

    Returns
    -------
    float
        MC-SS ∈ (-∞, 1].  Higher is better; 0 = no skill over persistence.

    Examples
    --------
    >>> import numpy as np
    >>> from temporalcv.metrics import mc_skill_score
    >>> actual = np.array([1.0, -1.0, 2.0, -2.0, 1.5, -1.5, 1.0, -1.0])
    >>> predicted = np.array([0.9, -0.8, 1.8, -1.9, 1.4, -1.3, 0.9, -0.9])
    >>> print(f"MC-SS: {mc_skill_score(actual, predicted):.3f}")
    MC-SS: 0.925
    """
    import numpy as np

    from temporalcv.persistence import compute_move_conditional_metrics

    result = compute_move_conditional_metrics(
        predictions=np.asarray(predictions),
        actuals=np.asarray(actuals),
        threshold=threshold,
        threshold_percentile=threshold_percentile,
    )
    return result.skill_score


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
    # Convenience aliases
    "mase",
    "mc_skill_score",
]
