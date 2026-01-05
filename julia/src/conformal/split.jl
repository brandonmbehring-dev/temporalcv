# =============================================================================
# Split Conformal Predictor
# =============================================================================

"""
    SplitConformalPredictor

Split conformal predictor for i.i.d. data.

Uses nonconformity scores (absolute residuals) and finite-sample
quantile formula for coverage guarantee: `ceil((n+1)(1-α))/n`.

# Fields
- `alpha::Float64`: Miscoverage rate (default 0.1 for 90% coverage)
- `quantile::Float64`: Calibrated quantile for interval construction
- `scores::Vector{Float64}`: Stored nonconformity scores
- `calibrated::Bool`: Whether calibrate! has been called

# Example
```julia
cp = SplitConformalPredictor(alpha=0.1)
calibrate!(cp, predictions_cal, actuals_cal)
intervals = predict_interval(cp, predictions_test)
@assert coverage(intervals, actuals_test) >= 0.9 - 0.05  # Allow margin
```

# Reference
Vovk, Gammerman, Shafer (2005) "Algorithmic Learning in a Random World"
"""
mutable struct SplitConformalPredictor
    alpha::Float64
    quantile::Float64
    scores::Vector{Float64}
    calibrated::Bool

    function SplitConformalPredictor(; alpha::Float64=0.1)
        @assert 0 < alpha < 1 "alpha must be in (0, 1), got $alpha"
        new(alpha, 0.0, Float64[], false)
    end
end

"""
    calibrate!(cp::SplitConformalPredictor, predictions, actuals) -> SplitConformalPredictor

Calibrate conformal predictor on holdout data.

Computes nonconformity scores (absolute residuals) and determines
the quantile for interval construction using the finite-sample formula.

# Arguments
- `predictions`: Point predictions on calibration set
- `actuals`: True values on calibration set

# Returns
The calibrated predictor (mutated in place).
"""
function calibrate!(cp::SplitConformalPredictor,
                   predictions::AbstractVector,
                   actuals::AbstractVector)
    n = length(predictions)
    @assert n == length(actuals) "predictions and actuals must have same length"
    @assert n > 0 "need at least one calibration sample"

    # Nonconformity scores: absolute residuals
    cp.scores = abs.(predictions .- actuals)

    # Finite-sample quantile: ceil((n+1)(1-α))/n
    # This guarantees coverage >= 1-α for exchangeable data
    quantile_idx = ceil(Int, (n + 1) * (1 - cp.alpha))
    quantile_idx = clamp(quantile_idx, 1, n)  # Clamp to valid range

    sorted_scores = sort(cp.scores)
    cp.quantile = sorted_scores[quantile_idx]
    cp.calibrated = true

    return cp
end

"""
    predict_interval(cp::SplitConformalPredictor, predictions) -> PredictionInterval

Generate prediction intervals for new predictions.

# Arguments
- `predictions`: Point predictions to wrap with intervals

# Returns
`PredictionInterval` with `lower = predictions - quantile` and
`upper = predictions + quantile`.

# Throws
- `AssertionError` if predictor not calibrated
"""
function predict_interval(cp::SplitConformalPredictor,
                         predictions::AbstractVector)
    @assert cp.calibrated "Must call calibrate! before predict_interval"

    lower = predictions .- cp.quantile
    upper = predictions .+ cp.quantile

    return PredictionInterval(lower, upper)
end
