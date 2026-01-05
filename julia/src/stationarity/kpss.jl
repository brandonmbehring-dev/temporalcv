# =============================================================================
# KPSS Test (Simplified Implementation)
# =============================================================================

"""
    kpss_test(series; regression=:c, nlags=nothing, alpha=0.05)

KPSS test for stationarity.

Tests H0: series is stationary
vs    H1: series has a unit root (non-stationary)

Note: KPSS has the **opposite** null hypothesis from ADF!

# Arguments
- `series::AbstractVector`: Time series data
- `regression::Symbol=:c`: Null hypothesis type
  - `:c` - level stationarity (constant mean)
  - `:ct` - trend stationarity (linear trend)
- `nlags::Union{Int, Nothing}=nothing`: Lags for HAC variance.
  If nothing, uses sqrt(n) heuristic
- `alpha::Float64=0.05`: Significance level

# Returns
- `StationarityTestResult`: Test results

# Example
```julia
using TemporalValidation.Stationarity
using Random

rng = MersenneTwister(42)
stationary = randn(rng, 100)
result = kpss_test(stationary)
result.is_stationary  # true

random_walk = cumsum(randn(rng, 100))
result = kpss_test(random_walk)
result.is_stationary  # false
```

# Notes
- [T1] Kwiatkowski et al. (1992) J. Econometrics 54, 159-178
- For KPSS, is_stationary = true means we FAIL to reject H0 (stationarity)
"""
function kpss_test(
    series::AbstractVector{<:Real};
    regression::Symbol = :c,
    nlags::Union{Int, Nothing} = nothing,
    alpha::Float64 = 0.05
)::StationarityTestResult
    y = collect(Float64, series)
    n = length(y)

    @assert n >= 20 "Series too short for KPSS test: n=$n, need >= 20"
    @assert regression in (:c, :ct) "regression must be :c or :ct"
    @assert 0.0 < alpha < 1.0 "alpha must be between 0 and 1"

    # Determine bandwidth for HAC variance
    lags = isnothing(nlags) ? floor(Int, sqrt(n)) : nlags

    # Detrend the series
    if regression == :c
        # Level stationarity: remove mean
        residuals = y .- mean(y)
    else  # :ct
        # Trend stationarity: remove linear trend
        t = collect(1:n)
        X = hcat(ones(n), t)
        β = X \ y
        residuals = y - X * β
    end

    # Compute partial sums S_t = Σ_{i=1}^t e_i
    S = cumsum(residuals)

    # KPSS statistic: η = (1/n²) Σ S_t² / σ²_LR
    # where σ²_LR is the long-run variance

    # Compute long-run variance using Bartlett kernel
    σ²_lr = _kpss_long_run_variance(residuals, lags)

    # KPSS statistic
    η = sum(S.^2) / (n^2 * σ²_lr)

    # Critical values from Kwiatkowski et al. (1992) Table 1
    critical_values = _kpss_critical_values(regression)

    # P-value approximation
    pvalue = _kpss_pvalue(η, regression)

    # For KPSS, we reject H0 (stationarity) if statistic > critical value
    # is_stationary = true means we fail to reject (p >= alpha)
    is_stationary = pvalue >= alpha

    return StationarityTestResult(
        test_name = :KPSS,
        statistic = η,
        pvalue = pvalue,
        is_stationary = is_stationary,
        lags_used = lags,
        regression = regression,
        critical_values = critical_values
    )
end

"""
Long-run variance using Bartlett kernel for KPSS test.
"""
function _kpss_long_run_variance(residuals::Vector{Float64}, lags::Int)::Float64
    n = length(residuals)

    # Base variance
    γ₀ = sum(residuals.^2) / n

    # Add autocovariance terms with Bartlett weights
    s = γ₀
    for k in 1:lags
        weight = 1.0 - k / (lags + 1)  # Bartlett kernel
        γₖ = sum(residuals[1:(n-k)] .* residuals[(k+1):n]) / n
        s += 2 * weight * γₖ
    end

    return max(s, 1e-10)  # Ensure positive
end

"""
Critical values for KPSS test from Kwiatkowski et al. (1992).
"""
function _kpss_critical_values(regression::Symbol)::Dict{String, Float64}
    if regression == :c
        # Level stationarity
        return Dict(
            "1%" => 0.739,
            "5%" => 0.463,
            "10%" => 0.347
        )
    else  # :ct
        # Trend stationarity
        return Dict(
            "1%" => 0.216,
            "5%" => 0.146,
            "10%" => 0.119
        )
    end
end

"""
Approximate p-value for KPSS test.
"""
function _kpss_pvalue(stat::Float64, regression::Symbol)::Float64
    cv = _kpss_critical_values(regression)

    # For KPSS, larger statistic = more evidence against stationarity
    if stat >= cv["1%"]
        return 0.005  # Very small p-value, reject stationarity
    elseif stat >= cv["5%"]
        # Interpolate between 1% and 5%
        frac = (stat - cv["5%"]) / (cv["1%"] - cv["5%"])
        return 0.05 - frac * 0.04
    elseif stat >= cv["10%"]
        # Interpolate between 5% and 10%
        frac = (stat - cv["10%"]) / (cv["5%"] - cv["10%"])
        return 0.10 - frac * 0.05
    else
        # Below 10% critical value - stationary
        # Simple exponential growth to 1.0
        ratio = stat / cv["10%"]
        return min(1.0, 0.10 + 0.90 * (1 - ratio))
    end
end
