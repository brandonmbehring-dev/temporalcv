"""
    Diagnostics

Diagnostic tools for temporal validation analysis.

Provides influence diagnostics for statistical tests and sensitivity analysis
for cross-validation parameters.

# Knowledge Tiers
- [T1] Influence functions: Cook (1977), well-established for regression
- [T2] Block jackknife for time series: Künsch (1989)
- [T2] HAC-adjusted influence: empirical best practice
- [T2] Gap sensitivity analysis: empirical best practice

# References
- [T1] Cook, R.D. (1977). Detection of Influential Observation in Linear Regression.
  Technometrics, 19(1), 15-18.
- [T2] Künsch, H.R. (1989). The Jackknife and the Bootstrap for General Stationary
  Observations. Annals of Statistics, 17(3), 1217-1241.

# Example
```julia
using TemporalValidation.Diagnostics

# Compute influence on DM test
errors1 = model_predictions .- actuals
errors2 = baseline_predictions .- actuals
diag = compute_dm_influence(errors1, errors2; horizon=4)

if diag.n_high_influence_blocks > 0
    println("Warning: \$(diag.n_high_influence_blocks) high-influence blocks detected")
end
```
"""
module Diagnostics

using Statistics
using Random

# Import HAC variance from sibling module
import ..StatisticalTests: compute_hac_variance, bartlett_kernel

# =============================================================================
# Types
# =============================================================================

include("types.jl")

# =============================================================================
# Influence Analysis
# =============================================================================

include("influence.jl")

# =============================================================================
# Sensitivity Analysis
# =============================================================================

include("sensitivity.jl")

# =============================================================================
# Exports
# =============================================================================

# Types
export InfluenceDiagnostic, SensitivityResult, StabilityReport

# Influence functions
export compute_dm_influence, compute_block_influence
export identify_influential_points

# Sensitivity functions
export compute_parameter_sensitivity
export compute_stability_report
export bootstrap_metric_variance

end # module Diagnostics
