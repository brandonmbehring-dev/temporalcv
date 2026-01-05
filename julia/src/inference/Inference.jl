# =============================================================================
# Inference Module - Bootstrap Methods for CV Folds
# =============================================================================
#
# Provides wild cluster bootstrap for inference on cross-validation folds.
# Particularly important when number of folds is small (n < 20).
#
# Based on:
# - Webb (2023): Reworking wild bootstrap for clustered errors
# - Cameron et al. (2008): Bootstrap-based improvements for clustered errors

module Inference

using Statistics
using Random
using Distributions

# =============================================================================
# Types
# =============================================================================

"""
Result of wild cluster bootstrap inference.

# Fields
- `estimate::Float64`: Point estimate (original statistic)
- `se::Float64`: Bootstrap standard error
- `ci_lower::Float64`: Lower confidence interval bound
- `ci_upper::Float64`: Upper confidence interval bound
- `p_value::Float64`: Two-sided p-value
- `n_bootstrap::Int`: Number of bootstrap replications
- `n_clusters::Int`: Number of clusters (folds)
- `weight_type::Symbol`: Weight distribution used (:rademacher or :webb)
- `bootstrap_distribution::Vector{Float64}`: Full bootstrap distribution
"""
struct WildBootstrapResult
    estimate::Float64
    se::Float64
    ci_lower::Float64
    ci_upper::Float64
    p_value::Float64
    n_bootstrap::Int
    n_clusters::Int
    weight_type::Symbol
    bootstrap_distribution::Vector{Float64}
end


# =============================================================================
# Core Functions
# =============================================================================

include("wild_bootstrap.jl")


# =============================================================================
# Exports
# =============================================================================

export WildBootstrapResult
export rademacher_weights, webb_weights
export wild_cluster_bootstrap

end # module Inference
