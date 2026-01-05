"""
    StatisticalTests

Statistical tests for forecast evaluation.

Implements:
- **Diebold-Mariano test** (DM 1995): Compare predictive accuracy of two models
- **Pesaran-Timmermann test** (PT 1992): Test directional accuracy
- **HAC variance** (Newey-West 1987): Correct for serial correlation in h>1 forecasts

# Knowledge Tiers
- [T1] DM test core methodology (Diebold & Mariano 1995)
- [T1] Harvey small-sample adjustment (Harvey et al. 1997)
- [T1] HAC variance with Bartlett kernel (Newey & West 1987)
- [T1] PT test 2-class formulas (Pesaran & Timmermann 1992)
- [T1] Automatic bandwidth selection (Andrews 1991)
- [T2] Minimum sample size n >= 30 for DM, n >= 20 for PT
- [T3] PT 3-class mode is ad-hoc extension

# Example
```julia
using TemporalValidation.StatisticalTests

# Compare model to baseline
result = dm_test(model_errors, baseline_errors, h=2)
println("DM statistic: \$(round(result.statistic, digits=3)), p-value: \$(round(result.pvalue, digits=4))")

# Test directional accuracy
pt_result = pt_test(actual_changes, predicted_changes, move_threshold=0.01)
println("Direction accuracy: \$(round(pt_result.accuracy * 100, digits=1))%")
```

# References
- Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy.
  Journal of Business & Economic Statistics, 13(3), 253-263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality
  of prediction mean squared errors. International Journal of Forecasting.
- Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test
  of predictive performance. Journal of Business & Economic Statistics.
- Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
"""
module StatisticalTests

using Statistics
using Distributions

# =============================================================================
# Types
# =============================================================================

include("types.jl")

# =============================================================================
# Components
# =============================================================================

include("hac.jl")
include("dm_test.jl")
include("pt_test.jl")

# =============================================================================
# Exports
# =============================================================================

# Result types
export DMTestResult, PTTestResult

# Type accessors
export significant_at_05, significant_at_01, skill

# Functions
export compute_hac_variance, bartlett_kernel
export dm_test, pt_test

end # module StatisticalTests
