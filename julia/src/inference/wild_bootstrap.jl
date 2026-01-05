# =============================================================================
# Wild Cluster Bootstrap Implementation
# =============================================================================

"""
    rademacher_weights(n::Int; rng=Random.GLOBAL_RNG) -> Vector{Float64}

Generate Rademacher weights: {-1, +1} with probability 0.5 each.

Used for wild bootstrap when number of clusters >= 13.

# Arguments
- `n::Int`: Number of weights to generate
- `rng::AbstractRNG`: Random number generator

# Returns
Vector of n weights in {-1.0, +1.0}.

# References
[T1] Cameron et al. (2008), Section 3.1
"""
function rademacher_weights(n::Int; rng::AbstractRNG = Random.GLOBAL_RNG)::Vector{Float64}
    n > 0 || error("n must be positive, got $n")
    return [rand(rng, Bool) ? 1.0 : -1.0 for _ in 1:n]
end


"""
    webb_weights(n::Int; rng=Random.GLOBAL_RNG) -> Vector{Float64}

Generate Webb 6-point weights for small cluster counts.

Uses {-√(3/2), -√(2/2), -√(1/2), +√(1/2), +√(2/2), +√(3/2)}
which equals approximately {-1.225, -0.707, -0.408, +0.408, +0.707, +1.225}.

The original Webb paper uses {-1.5, -1, -0.5, +0.5, +1, +1.5} scaled.
We use the normalized version for proper variance properties.

Used for wild bootstrap when number of clusters < 13.

# Arguments
- `n::Int`: Number of weights to generate
- `rng::AbstractRNG`: Random number generator

# Returns
Vector of n weights from Webb 6-point distribution.

# References
[T1] Webb (2023): Reworking wild bootstrap for few clusters
"""
function webb_weights(n::Int; rng::AbstractRNG = Random.GLOBAL_RNG)::Vector{Float64}
    n > 0 || error("n must be positive, got $n")

    # 6-point Webb distribution
    # These weights have mean 0 and variance 1
    webb_points = [-sqrt(3/2), -sqrt(2/2), -sqrt(1/2),
                   sqrt(1/2), sqrt(2/2), sqrt(3/2)]

    return [webb_points[rand(rng, 1:6)] for _ in 1:n]
end


"""
    wild_cluster_bootstrap(
        fold_metrics::Vector{<:Real};
        n_bootstrap::Int=999,
        confidence_level::Float64=0.95,
        rng::AbstractRNG=Random.GLOBAL_RNG,
        weight_type::Union{Symbol, Nothing}=nothing
    ) -> WildBootstrapResult

Perform wild cluster bootstrap for CV fold metrics.

Automatically selects Webb 6-point weights for < 13 folds,
Rademacher weights for >= 13 folds (unless overridden).

# Arguments
- `fold_metrics::Vector{<:Real}`: Metric value for each CV fold
- `n_bootstrap::Int=999`: Number of bootstrap replications
- `confidence_level::Float64=0.95`: Confidence level for interval
- `rng::AbstractRNG`: Random number generator
- `weight_type::Symbol=nothing`: Force :rademacher or :webb (auto if nothing)

# Returns
`WildBootstrapResult` with estimate, SE, CI, and p-value.

# Algorithm
```
1. Compute original statistic: mean(fold_metrics)
2. Center the metrics: d_centered = fold_metrics .- mean
3. For each bootstrap iteration:
   a. Generate cluster-level weights w_k
   b. Compute bootstrap statistic: mean(w_k .* d_centered)
4. SE = std(bootstrap_distribution)
5. CI from percentiles
6. P-value = proportion of |bootstrap| >= |original|
```

# Example
```julia
# Fold MAE values from 5-fold CV
fold_maes = [0.82, 0.91, 0.78, 0.85, 0.88]

result = wild_cluster_bootstrap(fold_maes)
println("MAE: \$(result.estimate) ± \$(result.se)")
println("95% CI: [\$(result.ci_lower), \$(result.ci_upper)]")
```

# References
[T1] Webb (2023): Use 6-point for < 13 clusters
[T1] Cameron et al. (2008): Wild cluster bootstrap theory
"""
function wild_cluster_bootstrap(
    fold_metrics::Vector{<:Real};
    n_bootstrap::Int = 999,
    confidence_level::Float64 = 0.95,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    weight_type::Union{Symbol, Nothing} = nothing
)::WildBootstrapResult
    n_folds = length(fold_metrics)
    n_folds >= 2 || error("Need at least 2 folds, got $n_folds")
    n_bootstrap > 0 || error("n_bootstrap must be positive, got $n_bootstrap")
    0 < confidence_level < 1 || error("confidence_level must be in (0, 1)")

    fold_metrics_float = Float64.(fold_metrics)

    # Determine weight type
    if isnothing(weight_type)
        weight_type = n_folds < 13 ? :webb : :rademacher
    end

    weight_type in [:rademacher, :webb] || error("weight_type must be :rademacher or :webb")

    weight_fn = weight_type == :webb ? webb_weights : rademacher_weights

    # Original estimate
    original_estimate = mean(fold_metrics_float)

    # Center for bootstrap
    centered = fold_metrics_float .- original_estimate

    # Bootstrap loop
    bootstrap_stats = zeros(n_bootstrap)

    for b in 1:n_bootstrap
        # Generate cluster-level weights
        weights = weight_fn(n_folds; rng=rng)

        # Weighted mean of centered values
        bootstrap_stats[b] = mean(weights .* centered)
    end

    # Standard error
    se = std(bootstrap_stats)

    # Percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = quantile(bootstrap_stats, alpha / 2) + original_estimate
    ci_upper = quantile(bootstrap_stats, 1 - alpha / 2) + original_estimate

    # Two-sided p-value for testing H0: mean = 0
    # Proportion of bootstrap stats with |t*| >= |t|
    p_value = mean(abs.(bootstrap_stats) .>= abs(original_estimate))

    return WildBootstrapResult(
        original_estimate,
        se,
        ci_lower,
        ci_upper,
        p_value,
        n_bootstrap,
        n_folds,
        weight_type,
        bootstrap_stats
    )
end


"""
    wild_cluster_bootstrap_difference(
        fold_metrics_a::Vector{<:Real},
        fold_metrics_b::Vector{<:Real};
        kwargs...
    ) -> WildBootstrapResult

Bootstrap inference for difference between two models' fold metrics.

# Arguments
- `fold_metrics_a::Vector{<:Real}`: Metrics for model A
- `fold_metrics_b::Vector{<:Real}`: Metrics for model B
- `kwargs...`: Passed to `wild_cluster_bootstrap`

# Returns
`WildBootstrapResult` for the difference (A - B).

# Example
```julia
model_a_mae = [0.82, 0.91, 0.78, 0.85, 0.88]
model_b_mae = [0.90, 0.95, 0.85, 0.92, 0.93]

result = wild_cluster_bootstrap_difference(model_a_mae, model_b_mae)
if result.p_value < 0.05
    println("Model A significantly better")
end
```
"""
function wild_cluster_bootstrap_difference(
    fold_metrics_a::Vector{<:Real},
    fold_metrics_b::Vector{<:Real};
    n_bootstrap::Int = 999,
    confidence_level::Float64 = 0.95,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    weight_type::Union{Symbol, Nothing} = nothing
)::WildBootstrapResult
    length(fold_metrics_a) == length(fold_metrics_b) ||
        error("Fold metric vectors must have same length")

    # Compute paired differences
    differences = Float64.(fold_metrics_a) .- Float64.(fold_metrics_b)

    return wild_cluster_bootstrap(
        differences;
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        rng=rng,
        weight_type=weight_type
    )
end


# Export the difference function
export wild_cluster_bootstrap_difference
