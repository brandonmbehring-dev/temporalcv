# =============================================================================
# Influence Diagnostics
# =============================================================================

"""
Compute influence of each observation on DM test statistic.

Provides TWO complementary views:

1. **Observation-level** (HAC-adjusted influence function):
   - Granular per-observation scores
   - Formula: psi_i = (d_i - d_bar) / sqrt(HAC_var * n)
   - Best for: Exploratory analysis, identifying specific outliers

2. **Block-level** (Block jackknife):
   - Leave-block-out influence scores
   - Block size = max(horizon, 1) to account for forecast horizon
   - Best for: Decision-making, robust to autocorrelation

# Arguments
- `errors1::AbstractVector{<:Real}`: Forecast errors from model 1 (actual - prediction)
- `errors2::AbstractVector{<:Real}`: Forecast errors from model 2 (baseline)
- `horizon::Int=1`: Forecast horizon. Used for HAC bandwidth and block size.
- `loss::Symbol=:squared`: Loss function (:squared for MSE, :absolute for MAE)
- `influence_threshold::Float64=2.0`: Multiplier for std to flag high-influence points

# Returns
`InfluenceDiagnostic` with both observation-level and block-level diagnostics.

# Knowledge Tier
[T2] Combines standard influence theory with HAC variance for time series robustness.

# Example
```julia
errors1 = model_predictions .- actuals
errors2 = baseline_predictions .- actuals
diag = compute_dm_influence(errors1, errors2; horizon=4)

println("High-influence blocks: \$(diag.n_high_influence_blocks)")
for (i, (start, stop)) in enumerate(diag.block_indices)
    if diag.block_high_mask[i]
        println("Block \$i (indices \$start:\$stop): influence=\$(diag.block_influence[i])")
    end
end
```

# References
- Cook (1977). Detection of Influential Observation in Linear Regression.
- Künsch (1989). The Jackknife and the Bootstrap for General Stationary Observations.
"""
function compute_dm_influence(
    errors1::AbstractVector{<:Real},
    errors2::AbstractVector{<:Real};
    horizon::Int = 1,
    loss::Symbol = :squared,
    influence_threshold::Float64 = 2.0
)::InfluenceDiagnostic
    errors1 = Float64.(errors1)
    errors2 = Float64.(errors2)

    n = length(errors1)

    if length(errors2) != n
        error("Error arrays must have same length: $n vs $(length(errors2))")
    end

    if n < 10
        error("Need at least 10 observations for influence analysis, got $n")
    end

    if loss ∉ (:squared, :absolute)
        error("loss must be :squared or :absolute, got $loss")
    end

    # Compute loss differentials
    d = if loss == :squared
        errors1 .^ 2 .- errors2 .^ 2
    else  # :absolute
        abs.(errors1) .- abs.(errors2)
    end

    d_mean = mean(d)

    # ==========================================================================
    # Observation-level influence (HAC-adjusted)
    # ==========================================================================
    # HAC variance with bandwidth = horizon - 1 (appropriate for h-step forecasts)
    bandwidth = horizon > 1 ? horizon - 1 : nothing
    hac_var = compute_hac_variance(d; bandwidth=bandwidth)

    # Influence function: psi_i = (d_i - d_bar) / sqrt(HAC_var * n)
    obs_influence = if hac_var > 0
        (d .- d_mean) ./ sqrt(hac_var * n)
    else
        # Degenerate case: all d_i identical
        zeros(n)
    end

    # Flag high-influence observations
    obs_std = std(obs_influence)
    obs_high_mask = if obs_std > 0
        BitVector(abs.(obs_influence) .> influence_threshold * obs_std)
    else
        falses(n)
    end

    # ==========================================================================
    # Block-level influence (block jackknife)
    # ==========================================================================
    block_size = max(horizon, 1)
    n_blocks = n ÷ block_size

    if n_blocks < 2
        # Not enough for block jackknife, use single-observation blocks
        block_size = 1
        n_blocks = n
    end

    block_indices = Tuple{Int, Int}[]
    block_influence_list = Float64[]

    # Compute full DM statistic (mean of d)
    full_dm = d_mean

    for b in 1:n_blocks
        start_idx = (b - 1) * block_size + 1
        end_idx = min(b * block_size, n)
        push!(block_indices, (start_idx, end_idx))

        # Leave-block-out: compute DM without this block
        d_without = vcat(d[1:start_idx-1], d[end_idx+1:end])

        influence_b = if length(d_without) > 0
            dm_without = mean(d_without)
            # Influence = (full - leave_out) * sqrt(n)
            (full_dm - dm_without) * sqrt(n)
        else
            0.0
        end

        push!(block_influence_list, influence_b)
    end

    block_influence = block_influence_list

    # Flag high-influence blocks
    block_std = std(block_influence)
    block_high_mask = if block_std > 0
        BitVector(abs.(block_influence) .> influence_threshold * block_std)
    else
        falses(length(block_influence))
    end

    return InfluenceDiagnostic(
        obs_influence,
        obs_high_mask,
        block_influence,
        block_high_mask,
        block_indices,
        sum(obs_high_mask),
        sum(block_high_mask),
        influence_threshold
    )
end


"""
Compute block-based influence for arbitrary loss differentials.

Simpler interface when you already have loss differentials computed.

# Arguments
- `loss_differentials::AbstractVector{<:Real}`: d_i = L(e1_i) - L(e2_i)
- `block_size::Int=1`: Size of each block
- `influence_threshold::Float64=2.0`: Threshold multiplier

# Returns
Tuple of (block_influence, block_high_mask, block_indices)
"""
function compute_block_influence(
    loss_differentials::AbstractVector{<:Real};
    block_size::Int = 1,
    influence_threshold::Float64 = 2.0
)
    d = Float64.(loss_differentials)
    n = length(d)

    if block_size < 1
        error("block_size must be >= 1, got $block_size")
    end

    n_blocks = n ÷ block_size
    if n_blocks < 1
        n_blocks = 1
        block_size = n
    end

    full_dm = mean(d)
    block_indices = Tuple{Int, Int}[]
    block_influence = Float64[]

    for b in 1:n_blocks
        start_idx = (b - 1) * block_size + 1
        end_idx = min(b * block_size, n)
        push!(block_indices, (start_idx, end_idx))

        d_without = vcat(d[1:start_idx-1], d[end_idx+1:end])
        dm_without = length(d_without) > 0 ? mean(d_without) : 0.0
        push!(block_influence, (full_dm - dm_without) * sqrt(n))
    end

    block_std = std(block_influence)
    block_high_mask = if block_std > 0
        BitVector(abs.(block_influence) .> influence_threshold * block_std)
    else
        falses(length(block_influence))
    end

    return (block_influence, block_high_mask, block_indices)
end


"""
Identify influential points based on a threshold.

# Arguments
- `influence_values::AbstractVector{<:Real}`: Influence scores
- `threshold::Float64=2.0`: Multiplier for standard deviation

# Returns
Vector of indices where |influence| > threshold * std(influence)

# Example
```julia
influential_idx = identify_influential_points(diag.observation_influence)
```
"""
function identify_influential_points(
    influence_values::AbstractVector{<:Real};
    threshold::Float64 = 2.0
)::Vector{Int}
    influence_values = Float64.(influence_values)
    influence_std = std(influence_values)

    if influence_std == 0
        return Int[]
    end

    indices = findall(abs.(influence_values) .> threshold * influence_std)
    return indices
end
