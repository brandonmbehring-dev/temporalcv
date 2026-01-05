# =============================================================================
# Conformal Prediction Types
# =============================================================================

"""
    PredictionInterval

Container for prediction intervals with lower and upper bounds.

# Fields
- `lower::Vector{Float64}`: Lower bounds of intervals
- `upper::Vector{Float64}`: Upper bounds of intervals

# Example
```julia
pi = PredictionInterval([1.0, 2.0], [3.0, 4.0])
println(width(pi))      # [2.0, 2.0]
println(mean_width(pi)) # 2.0
```
"""
struct PredictionInterval
    lower::Vector{Float64}
    upper::Vector{Float64}

    function PredictionInterval(lower::AbstractVector, upper::AbstractVector)
        @assert length(lower) == length(upper) "lower and upper must have same length"
        new(collect(Float64, lower), collect(Float64, upper))
    end
end

Base.length(pi::PredictionInterval) = length(pi.lower)

"""
    width(pi::PredictionInterval) -> Vector{Float64}

Compute width of each interval.
"""
function width(pi::PredictionInterval)
    return pi.upper .- pi.lower
end

"""
    mean_width(pi::PredictionInterval) -> Float64

Compute mean interval width.
"""
function mean_width(pi::PredictionInterval)
    return mean(width(pi))
end

"""
    coverage(pi::PredictionInterval, actuals::AbstractVector) -> Float64

Compute empirical coverage: fraction of actuals within intervals.
"""
function coverage(pi::PredictionInterval, actuals::AbstractVector)
    @assert length(pi) == length(actuals) "intervals and actuals must have same length"
    covered = (pi.lower .<= actuals) .& (actuals .<= pi.upper)
    return mean(covered)
end
