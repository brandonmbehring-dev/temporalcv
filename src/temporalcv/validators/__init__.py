"""
Validators Subpackage for Theoretical Bounds Checking.

Provides validation against theoretical statistical limits:
- **AR(p) Bounds**: Minimum achievable MSE/MAE for AR processes
- **Series Generation**: Synthetic AR series for testing

These validators catch "impossibly good" results that indicate leakage.

Knowledge Tiers
---------------
[T1] AR(p) MSE bounds: Standard time series theory (Hamilton 1994, Ch. 4)
[T1] For AR(1) with parameter φ and innovation variance σ²:
     - 1-step ahead MSE = σ² (irreducible forecast error)
     - h-step ahead MSE = σ² · (1 - φ^(2h)) / (1 - φ²)
[T1] For AR(1), optimal 1-step MAE = σ · √(2/π) ≈ 0.798σ

Example
-------
>>> from temporalcv.validators import (
...     theoretical_ar1_mse_bound,
...     check_against_ar1_bounds,
...     generate_ar1_series,
... )
>>>
>>> # Check if model beats theoretical limits (indicates leakage)
>>> phi, sigma = 0.9, 1.0
>>> result = check_against_ar1_bounds(model_mse=0.5, phi=phi, sigma_sq=sigma**2)
>>> if result.status == GateStatus.HALT:
...     print("Model beats theoretical minimum - investigate for leakage")

References
----------
[T1] Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
[T1] Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting
     and Control. Holden-Day.
"""

from temporalcv.validators.theoretical import (
    check_against_ar1_bounds,
    generate_ar1_series,
    generate_ar2_series,
    theoretical_ar1_mae_bound,
    theoretical_ar1_mse_bound,
    theoretical_ar2_mse_bound,
)

__all__ = [
    "theoretical_ar1_mse_bound",
    "theoretical_ar1_mae_bound",
    "theoretical_ar2_mse_bound",
    "check_against_ar1_bounds",
    "generate_ar1_series",
    "generate_ar2_series",
]
