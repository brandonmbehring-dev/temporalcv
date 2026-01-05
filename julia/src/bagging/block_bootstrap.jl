# =============================================================================
# Moving Block Bootstrap
# =============================================================================

"""
    MovingBlockBootstrap <: BootstrapStrategy

Moving Block Bootstrap (Kunsch 1989).

Resamples time series by drawing overlapping blocks of fixed length
with replacement, preserving local temporal dependence.

# Fields
- `block_length::Union{Int, Nothing}`: Block length. If `nothing`, auto-compute as `ceil(n^(1/3))`.

# Example
```julia
mbb = MovingBlockBootstrap(block_length=10)
samples = bootstrap_sample(mbb, X, y, 20, rng)
```

# Warning
Moving block bootstrap assumes weak dependence (mixing conditions).
For highly persistent series (ACF(1) > 0.95), consider larger block lengths.

# Reference
Kunsch (1989) "The Jackknife and the Bootstrap for General Stationary Observations"
"""
struct MovingBlockBootstrap <: BootstrapStrategy
    block_length::Union{Int, Nothing}

    function MovingBlockBootstrap(; block_length::Union{Int, Nothing}=nothing)
        if !isnothing(block_length)
            @assert block_length >= 1 "block_length must be >= 1, got $block_length"
        end
        new(block_length)
    end
end

function bootstrap_sample(mbb::MovingBlockBootstrap,
                         X::AbstractMatrix,
                         y::AbstractVector,
                         n_samples::Int,
                         rng::AbstractRNG)
    n = size(X, 1)
    @assert n == length(y) "X and y must have same number of rows"
    @assert n > 0 "need at least one observation"

    # Auto block length: O(n^1/3), clamped
    block_len = if isnothing(mbb.block_length)
        max(1, ceil(Int, n^(1/3)))
    else
        min(mbb.block_length, n)  # Clamp to n
    end

    # Number of blocks needed (ceiling division)
    n_blocks = cld(n, block_len)

    # Available block start positions
    n_starts = n - block_len + 1
    @assert n_starts >= 1 "block_length too large for data size"

    samples = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}()
    sizehint!(samples, n_samples)

    for _ in 1:n_samples
        # Sample block start indices uniformly
        block_starts = rand(rng, 1:n_starts, n_blocks)

        # Build indices by concatenating blocks
        indices = Int[]
        sizehint!(indices, n_blocks * block_len)

        for start in block_starts
            append!(indices, start:(start + block_len - 1))
        end

        # Truncate to original length
        indices = indices[1:n]

        X_boot = Matrix{Float64}(X[indices, :])
        y_boot = Vector{Float64}(y[indices])
        push!(samples, (X_boot, y_boot))
    end

    return samples
end
