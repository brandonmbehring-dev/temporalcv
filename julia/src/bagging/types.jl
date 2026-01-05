# =============================================================================
# Bagging Types
# =============================================================================

"""
    BootstrapStrategy

Abstract type for bootstrap resampling strategies.

All concrete strategies must implement:
- `bootstrap_sample(strategy, X, y, n_samples, rng)` -> Vector{Tuple{Matrix, Vector}}

The returned samples are `(X_boot, y_boot)` tuples where both preserve
temporal dependence structure appropriate to the strategy.
"""
abstract type BootstrapStrategy end

"""
    bootstrap_sample(strategy, X, y, n_samples, rng)

Generate bootstrap samples from data.

# Arguments
- `strategy::BootstrapStrategy`: The bootstrap strategy to use
- `X::AbstractMatrix`: Feature matrix (n_obs × n_features)
- `y::AbstractVector`: Target vector (n_obs,)
- `n_samples::Int`: Number of bootstrap samples to generate
- `rng::AbstractRNG`: Random number generator

# Returns
`Vector{Tuple{Matrix{Float64}, Vector{Float64}}}`: List of (X_boot, y_boot) tuples
"""
function bootstrap_sample end
