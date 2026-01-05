# =============================================================================
# Diagnostics Types
# =============================================================================

"""
Result of influence analysis on DM test statistic.

Provides two views of influence:
1. observation_influence: Per-observation scores (granular, for exploration)
2. block_influence: Per-block scores (robust, for decisions)

# Fields
- `observation_influence::Vector{Float64}`: Per-observation influence scores
  using HAC-adjusted formula: psi_i = (d_i - d_bar) / sqrt(HAC_var * n)
- `observation_high_mask::BitVector`: True where |observation_influence| > threshold * std
- `block_influence::Vector{Float64}`: Per-block influence scores using block jackknife
- `block_high_mask::BitVector`: True where |block_influence| > threshold * std
- `block_indices::Vector{Tuple{Int, Int}}`: (start, end) indices for each block
- `n_high_influence_obs::Int`: Count of high-influence observations
- `n_high_influence_blocks::Int`: Count of high-influence blocks
- `influence_threshold::Float64`: Threshold multiplier used (e.g., 2.0 = 2σ)

# Notes
**Recommendation**: Use block_influence for decisions as it properly
accounts for serial correlation in time series. Use observation_influence
for exploratory analysis to identify specific problematic points.
"""
struct InfluenceDiagnostic
    observation_influence::Vector{Float64}
    observation_high_mask::BitVector
    block_influence::Vector{Float64}
    block_high_mask::BitVector
    block_indices::Vector{Tuple{Int, Int}}
    n_high_influence_obs::Int
    n_high_influence_blocks::Int
    influence_threshold::Float64
end


"""
Result of single-parameter sensitivity analysis.

# Fields
- `parameter::Symbol`: Name of the parameter analyzed
- `base_value::Float64`: Original parameter value
- `perturbed_values::Vector{Float64}`: Parameter values tested
- `base_metric::Float64`: Metric at base value
- `perturbed_metrics::Vector{Float64}`: Metric at each perturbed value
- `sensitivity::Float64`: Estimated d(metric)/d(parameter)
"""
struct SensitivityResult
    parameter::Symbol
    base_value::Float64
    perturbed_values::Vector{Float64}
    base_metric::Float64
    perturbed_metrics::Vector{Float64}
    sensitivity::Float64
end


"""
Multi-parameter stability report.

# Fields
- `sensitivities::Dict{Symbol, SensitivityResult}`: Per-parameter sensitivity
- `most_sensitive::Symbol`: Parameter with highest sensitivity
- `stability_score::Float64`: 0-1 score, higher = more stable
- `total_variation::Float64`: Sum of absolute sensitivities
"""
struct StabilityReport
    sensitivities::Dict{Symbol, SensitivityResult}
    most_sensitive::Symbol
    stability_score::Float64
    total_variation::Float64
end
