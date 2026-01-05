# =============================================================================
# Stationary Bootstrap
# =============================================================================

"""
    StationaryBootstrap <: BootstrapStrategy

Stationary Bootstrap (Politis & Romano 1994).

Uses geometric distribution for block lengths, producing stationary
resampled series. More robust to block length choice than MovingBlockBootstrap.

# Fields
- `expected_block_length::Union{Float64, Nothing}`: Expected block length (1/p where p is geometric parameter).
  If `nothing`, auto-compute as `n^(1/3)`.

# Algorithm
At each step:
1. With probability p = 1/expected_block_length, jump to random position
2. Otherwise, continue to next observation (with circular wrap)

# Example
```julia
sb = StationaryBootstrap(expected_block_length=10.0)
samples = bootstrap_sample(sb, X, y, 20, rng)
```

# Warning
Stationary bootstrap assumes weak dependence (stationarity and mixing).
For highly persistent series (ACF(1) > 0.95), consider larger expected_block_length.

# Reference
Politis & Romano (1994) "The Stationary Bootstrap" JASA 89(428), 1303-1313
"""
struct StationaryBootstrap <: BootstrapStrategy
    expected_block_length::Union{Float64, Nothing}

    function StationaryBootstrap(; expected_block_length::Union{Float64, Nothing}=nothing)
        if !isnothing(expected_block_length)
            @assert expected_block_length >= 1.0 "expected_block_length must be >= 1.0, got $expected_block_length"
        end
        new(expected_block_length)
    end
end

function bootstrap_sample(sb::StationaryBootstrap,
                         X::AbstractMatrix,
                         y::AbstractVector,
                         n_samples::Int,
                         rng::AbstractRNG)
    n = size(X, 1)
    @assert n == length(y) "X and y must have same number of rows"
    @assert n > 0 "need at least one observation"

    # Auto expected length: O(n^1/3), clamped
    exp_len = if isnothing(sb.expected_block_length)
        max(1.0, n^(1/3))
    else
        max(1.0, sb.expected_block_length)
    end

    # Geometric parameter: probability of jumping to new block
    p = 1.0 / exp_len

    samples = Vector{Tuple{Matrix{Float64}, Vector{Float64}}}()
    sizehint!(samples, n_samples)

    for _ in 1:n_samples
        # Per-sample random generation (avoids index exhaustion)
        uniforms = rand(rng, n)
        jump_targets = rand(rng, 1:n, n)
        start_idx = rand(rng, 1:n)

        # Build indices using Markov chain
        indices = Vector{Int}(undef, n)
        i = start_idx

        for j in 1:n
            indices[j] = i
            if uniforms[j] < p
                # Jump to random position
                i = jump_targets[j]
            else
                # Continue to next (circular wrap)
                i = mod1(i + 1, n)
            end
        end

        X_boot = Matrix{Float64}(X[indices, :])
        y_boot = Vector{Float64}(y[indices])
        push!(samples, (X_boot, y_boot))
    end

    return samples
end
