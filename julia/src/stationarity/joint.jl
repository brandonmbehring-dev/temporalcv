# =============================================================================
# Joint Stationarity Analysis (ADF + KPSS)
# =============================================================================

"""
    check_stationarity(series; alpha=0.05, regression=:c)

Run ADF + KPSS jointly and interpret results.

Joint interpretation follows the logic:
- ADF rejects (p < α) + KPSS fails to reject (p >= α) → Stationary
- ADF fails + KPSS rejects → Non-stationary
- Both reject → Difference-stationary (or trend-stationary)
- Both fail → Insufficient evidence

# Arguments
- `series::AbstractVector`: Time series data
- `alpha::Float64=0.05`: Significance level for both tests
- `regression::Symbol=:c`: Regression type (`:c` for level, `:ct` for trend)

# Returns
- `JointStationarityResult`: Combined result with interpretation

# Example
```julia
using TemporalValidation.Stationarity
using Random

rng = MersenneTwister(42)

# Stationary series
stationary = randn(rng, 100)
result = check_stationarity(stationary)
result.conclusion  # STATIONARY

# Random walk (non-stationary)
random_walk = cumsum(randn(rng, 100))
result = check_stationarity(random_walk)
result.conclusion  # NON_STATIONARY
```

# Notes
This joint testing approach is recommended because:
- ADF has low power against near-unit-root alternatives
- KPSS has low power against alternatives close to stationarity
- Together they provide more robust inference
"""
function check_stationarity(
    series::AbstractVector{<:Real};
    alpha::Float64 = 0.05,
    regression::Symbol = :c
)::JointStationarityResult
    @assert regression in (:c, :ct) "regression must be :c or :ct for joint test"

    # Run both tests
    adf_result = adf_test(series; regression=regression, alpha=alpha)
    kpss_result = kpss_test(series; regression=regression, alpha=alpha)

    # Joint interpretation
    adf_rejects = adf_result.pvalue < alpha   # Rejects unit root (stationary)
    kpss_rejects = kpss_result.pvalue < alpha # Rejects stationarity (unit root)

    if adf_rejects && !kpss_rejects
        conclusion = STATIONARY
        action = "Series appears stationary. Safe to model without differencing."
    elseif !adf_rejects && kpss_rejects
        conclusion = NON_STATIONARY
        action = "Series has unit root. Consider differencing or cointegration analysis."
    elseif adf_rejects && kpss_rejects
        conclusion = DIFFERENCE_STATIONARY
        action = "Conflicting results: may be difference-stationary or trend-stationary. " *
                 "Try first-differencing and re-testing."
    else
        conclusion = INSUFFICIENT_EVIDENCE
        action = "Neither test conclusive. Series may be borderline stationary. " *
                 "Consider larger sample or alternative tests."
    end

    return JointStationarityResult(
        adf_result,
        kpss_result,
        conclusion,
        action
    )
end

"""
    difference_until_stationary(series; max_diff=2, alpha=0.05)

Difference series until stationary (ADF test passes).

# Arguments
- `series::AbstractVector`: Time series data
- `max_diff::Int=2`: Maximum number of differences
- `alpha::Float64=0.05`: Significance level

# Returns
- `differenced::Vector{Float64}`: The differenced series
- `d::Int`: Number of differences applied

# Throws
- `AssertionError`: If series is not stationary after max_diff differences

# Example
```julia
using TemporalValidation.Stationarity
using Random

rng = MersenneTwister(42)
random_walk = cumsum(randn(rng, 100))
diff_series, d = difference_until_stationary(random_walk)
d  # 1
```
"""
function difference_until_stationary(
    series::AbstractVector{<:Real};
    max_diff::Int = 2,
    alpha::Float64 = 0.05
)::Tuple{Vector{Float64}, Int}
    arr = collect(Float64, series)

    for d in 0:max_diff
        if d > 0
            arr = diff(arr)
        end

        if length(arr) < 20
            error("Series too short after $d differences: n=$(length(arr)). " *
                  "Need at least 20 observations.")
        end

        result = adf_test(arr; alpha=alpha)
        if result.is_stationary
            return arr, d
        end
    end

    error("Series not stationary after $max_diff differences. " *
          "Consider alternative transformations (log, Box-Cox).")
end

"""
    integration_order(series; max_order=2, alpha=0.05)

Determine the integration order d of a series.

Returns the number of differences needed to achieve stationarity.

# Arguments
- `series::AbstractVector`: Time series data
- `max_order::Int=2`: Maximum integration order to test
- `alpha::Float64=0.05`: Significance level

# Returns
- `d::Int`: Integration order (0 = stationary, 1 = I(1), etc.)

# Example
```julia
using TemporalValidation.Stationarity
using Random

rng = MersenneTwister(42)

# I(0) - stationary
stationary = randn(rng, 100)
integration_order(stationary)  # 0

# I(1) - random walk
rw = cumsum(randn(rng, 100))
integration_order(rw)  # 1

# I(2) - integrated random walk
i2 = cumsum(cumsum(randn(rng, 100)))
integration_order(i2)  # 2
```
"""
function integration_order(
    series::AbstractVector{<:Real};
    max_order::Int = 2,
    alpha::Float64 = 0.05
)::Int
    _, d = difference_until_stationary(series; max_diff=max_order, alpha=alpha)
    return d
end
