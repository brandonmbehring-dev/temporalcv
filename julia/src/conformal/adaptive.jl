# =============================================================================
# Adaptive Conformal Predictor
# =============================================================================

"""
    AdaptiveConformalPredictor

Adaptive conformal predictor with online quantile updates.

Adapts to distribution shift in time series data using the
update rule from Gibbs & Candès (2021):
- If covered: quantile -= γ × α
- If not covered: quantile += γ × (1 - α)

This ensures long-run average coverage converges to 1-α.

# Fields
- `alpha::Float64`: Target miscoverage rate
- `gamma::Float64`: Learning rate for quantile updates
- `quantile::Float64`: Current quantile for interval construction
- `initialized::Bool`: Whether initialize! has been called

# Example
```julia
acp = AdaptiveConformalPredictor(alpha=0.1, gamma=0.1)
initialize!(acp, initial_predictions, initial_actuals)

for (pred, actual) in zip(predictions, actuals)
    interval = predict_interval(acp, pred)
    update!(acp, actual, interval)
end
```

# Reference
Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
"""
mutable struct AdaptiveConformalPredictor
    alpha::Float64
    gamma::Float64
    quantile::Float64
    initialized::Bool

    function AdaptiveConformalPredictor(;
            alpha::Float64=0.1,
            gamma::Float64=0.1)
        @assert 0 < alpha < 1 "alpha must be in (0, 1), got $alpha"
        @assert gamma > 0 "gamma must be positive, got $gamma"
        new(alpha, gamma, 0.0, false)
    end
end

"""
    initialize!(acp::AdaptiveConformalPredictor, predictions, actuals) -> AdaptiveConformalPredictor

Initialize adaptive predictor from calibration data.

Sets initial quantile using the same finite-sample formula as
SplitConformalPredictor.

# Arguments
- `predictions`: Point predictions on calibration set
- `actuals`: True values on calibration set
"""
function initialize!(acp::AdaptiveConformalPredictor,
                    predictions::AbstractVector,
                    actuals::AbstractVector)
    n = length(predictions)
    @assert n == length(actuals) "predictions and actuals must have same length"
    @assert n > 0 "need at least one calibration sample"

    # Compute initial quantile from calibration data
    scores = abs.(predictions .- actuals)
    quantile_idx = ceil(Int, (n + 1) * (1 - acp.alpha))
    quantile_idx = clamp(quantile_idx, 1, n)

    acp.quantile = sort(scores)[quantile_idx]
    acp.initialized = true

    return acp
end

"""
    predict_interval(acp::AdaptiveConformalPredictor, prediction::Real) -> Tuple{Float64, Float64}

Generate prediction interval for a single new prediction.

# Returns
Tuple `(lower, upper)` for the prediction interval.
"""
function predict_interval(acp::AdaptiveConformalPredictor, prediction::Real)
    @assert acp.initialized "Must call initialize! before predict_interval"

    lower = prediction - acp.quantile
    upper = prediction + acp.quantile

    return (lower, upper)
end

"""
    update!(acp::AdaptiveConformalPredictor, actual::Real, interval::Tuple) -> AdaptiveConformalPredictor

Update quantile based on whether actual value was covered.

Implements Gibbs & Candès (2021) online update:
- If covered: quantile -= γ × α (shrink intervals)
- If not covered: quantile += γ × (1 - α) (widen intervals)

# Arguments
- `actual`: True observed value
- `interval`: Tuple `(lower, upper)` from predict_interval
"""
function update!(acp::AdaptiveConformalPredictor,
                actual::Real,
                interval::Tuple{<:Real, <:Real})
    lower, upper = interval
    covered = lower <= actual <= upper

    if covered
        acp.quantile -= acp.gamma * acp.alpha
    else
        acp.quantile += acp.gamma * (1 - acp.alpha)
    end

    # Ensure quantile stays non-negative
    acp.quantile = max(acp.quantile, 0.0)

    return acp
end
