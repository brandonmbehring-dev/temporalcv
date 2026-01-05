"""
    CV

Walk-Forward Cross-Validation Module.

Provides temporal cross-validation with gap enforcement
for h-step forecasting scenarios.

# Knowledge Tiers
- [T1] Walk-forward validation is the standard for time-series (Tashman 2000)
- [T1] Gap >= horizon prevents information leakage for h-step forecasts
- [T1] Expanding window vs sliding window are both valid approaches (Tashman 2000)
- [T2] Gap enforcement: train_end + gap < test_start prevents lookahead

# Exports
- `SplitInfo`: Metadata for a single CV split
- `SplitResult`: Result from a single walk-forward split
- `WalkForwardResults`: Aggregated results
- `WalkForwardCV`: Walk-forward CV with gap enforcement
- `CrossFitCV`: Temporal cross-fitting for debiased predictions

# Example
```julia
using TemporalValidation.CV

cv = WalkForwardCV(n_splits=5, gap=2, window_type=:sliding, window_size=104)
for (train_idx, test_idx) in split(cv, 500)
    @assert last(train_idx) + cv.gap < first(test_idx)  # Gap enforced
end
```

# References
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy.
  International Journal of Forecasting, 16(4), 437-450.
- Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for
  time series predictor evaluation. Information Sciences, 191, 192-213.
"""
module CV

using Statistics

# =============================================================================
# Types
# =============================================================================

include("types.jl")

# =============================================================================
# CV Strategies
# =============================================================================

include("walk_forward.jl")
include("cross_fit.jl")

# =============================================================================
# Exports
# =============================================================================

# Types
export SplitInfo, SplitResult, WalkForwardResults

# Type accessors
export train_size, test_size, gap, errors, absolute_errors
export mae, rmse, bias, mse
export n_splits, predictions, actuals, total_samples
export to_split_info

# CV strategies
export WalkForwardCV, CrossFitCV

# Functions
export split, get_n_splits, get_split_info, get_fold_indices

end # module CV
