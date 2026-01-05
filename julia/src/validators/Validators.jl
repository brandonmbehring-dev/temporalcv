"""
    Validators

Theoretical Bounds Validation for Time Series Forecasts.

Provides theoretical minimum error bounds for AR(p) processes.
A model that beats these bounds is likely suffering from data leakage.

# Knowledge Tiers
- [T1] AR(1) 1-step MSE = sigma^2 (innovation variance is irreducible)
- [T1] AR(1) h-step MSE = sigma^2 * (1 - phi^(2h)) / (1 - phi^2)
- [T1] AR(1) 1-step MAE = sigma * sqrt(2/pi) (half-normal mean)
- [T1] AR(2) theory: Hamilton (1994), stationary if roots outside unit circle
- [T3] Tolerance factor 1.5 allows for finite-sample variation (empirical choice)

# Theory
For an AR(1) process: y_t = phi * y_{t-1} + eps_t where eps_t ~ N(0, sigma^2)

The optimal h-step ahead forecast E[y_{t+h} | y_t, y_{t-1}, ...] = phi^h * y_t

The h-step ahead forecast error is:
  e_{t+h} = y_{t+h} - E[y_{t+h} | y_t] = sum(phi^i * eps_{t+h-i} for i=0..h-1)

Since epsilon terms are independent:
  Var(e_{t+h}) = sigma^2 * sum(phi^(2i) for i=0..h-1) = sigma^2 * (1 - phi^(2h)) / (1 - phi^2)

For h=1: MSE = sigma^2 (just the innovation variance)
For h->inf: MSE -> sigma^2 / (1 - phi^2) = Var(y) (unconditional variance)

# References
- [T1] Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
  Chapter 4: Linear Stationary Time Series Models.
- [T1] Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis.

# Example
```julia
using TemporalValidation.Validators

# Compute theoretical bounds
mse_bound = compute_ar1_mse_bound(0.9, 1.0; horizon=1)
mae_bound = compute_ar1_mae_bound(0.9, 1.0; horizon=1)

# Check model against bounds
result = check_against_ar1_bounds(model_mse=0.5, phi=0.9, sigma_sq=1.0)
if result.status == :HALT
    println("Possible leakage detected!")
end
```
"""
module Validators

using Statistics
using Random

# =============================================================================
# Types
# =============================================================================

include("types.jl")

# =============================================================================
# Theoretical Bounds
# =============================================================================

include("theoretical.jl")

# =============================================================================
# Exports
# =============================================================================

# Types
export AR1Bounds, AR2Bounds, BoundsCheckResult

# AR(1) functions
export compute_ar1_mse_bound, compute_ar1_mae_bound, compute_ar1_rmse_bound
export generate_ar1_series, estimate_ar1_params

# AR(2) functions
export compute_ar2_mse_bound, generate_ar2_series

# Checking
export check_against_ar1_bounds, check_against_ar2_bounds

end # module Validators
