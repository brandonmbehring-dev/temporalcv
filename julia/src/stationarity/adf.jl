# =============================================================================
# Augmented Dickey-Fuller Test (Simplified Implementation)
# =============================================================================

"""
    adf_test(series; max_lags=nothing, regression=:c, alpha=0.05)

Augmented Dickey-Fuller test for unit root.

Tests H0: series has a unit root (non-stationary)
vs    H1: series is stationary

This is a simplified implementation using OLS regression to estimate
the AR coefficient and compare against critical values.

# Arguments
- `series::AbstractVector`: Time series data
- `max_lags::Union{Int, Nothing}=nothing`: Max lags for augmentation.
  If nothing, uses floor(cbrt(n)) heuristic
- `regression::Symbol=:c`: Regression type
  - `:c` - constant only (default)
  - `:ct` - constant and trend
  - `:n` - no constant, no trend
- `alpha::Float64=0.05`: Significance level

# Returns
- `StationarityTestResult`: Test results

# Example
```julia
using TemporalValidation.Stationarity
using Random

rng = MersenneTwister(42)
stationary = randn(rng, 100)
result = adf_test(stationary)
result.is_stationary  # true

random_walk = cumsum(randn(rng, 100))
result = adf_test(random_walk)
result.is_stationary  # false
```

# Notes
- [T1] Dickey & Fuller (1979) JASA 74(366), 427-431
- Critical values are approximate MacKinnon (1994) values
"""
function adf_test(
    series::AbstractVector{<:Real};
    max_lags::Union{Int, Nothing} = nothing,
    regression::Symbol = :c,
    alpha::Float64 = 0.05
)::StationarityTestResult
    y = collect(Float64, series)
    n = length(y)

    @assert n >= 20 "Series too short for ADF test: n=$n, need >= 20"
    @assert regression in (:c, :ct, :n) "regression must be :c, :ct, or :n"
    @assert 0.0 < alpha < 1.0 "alpha must be between 0 and 1"

    # Determine number of lags
    p = isnothing(max_lags) ? max(1, floor(Int, cbrt(n))) : max_lags
    p = min(p, n ÷ 4)  # Don't use more than n/4 lags

    # ADF regression: Δy_t = α + β*t + γ*y_{t-1} + Σδ_i*Δy_{t-i} + ε_t
    # We test H0: γ = 0 (unit root) vs H1: γ < 0 (stationary)

    # Compute differences
    Δy = diff(y)

    # Build design matrix
    effective_n = length(Δy) - p
    @assert effective_n >= 10 "Insufficient data after lagging: effective n = $effective_n"

    # y_{t-1} for t = p+2 to n
    y_lag1 = y[(p+1):(n-1)]

    # Δy_t for t = p+2 to n
    Δy_current = Δy[(p+1):end]

    # Build regressor matrix
    X_cols = Vector{Vector{Float64}}()

    # Constant term
    if regression in (:c, :ct)
        push!(X_cols, ones(effective_n))
    end

    # Trend term
    if regression == :ct
        push!(X_cols, collect(Float64, 1:effective_n))
    end

    # Lagged level y_{t-1}
    push!(X_cols, y_lag1)

    # Lagged differences Δy_{t-i} for i = 1, ..., p
    for i in 1:p
        Δy_lag = Δy[(p+1-i):(end-i)]
        push!(X_cols, Δy_lag)
    end

    # Construct X matrix
    X = hcat(X_cols...)

    # OLS: β = (X'X)^{-1} X'y
    XtX = X' * X
    Xty = X' * Δy_current

    # Use Cholesky for numerical stability
    β = XtX \ Xty

    # Find position of γ (coefficient on y_{t-1})
    γ_idx = regression == :n ? 1 : (regression == :c ? 2 : 3)
    γ = β[γ_idx]

    # Compute residuals and standard error
    residuals = Δy_current - X * β
    σ² = sum(residuals.^2) / (effective_n - length(β))

    # Variance-covariance matrix
    var_cov = σ² * inv(XtX)
    se_γ = sqrt(var_cov[γ_idx, γ_idx])

    # ADF t-statistic
    t_stat = γ / se_γ

    # Critical values (approximate MacKinnon 1994)
    critical_values = _adf_critical_values(regression, effective_n)

    # P-value approximation using interpolation
    pvalue = _adf_pvalue(t_stat, regression, effective_n)

    # Reject H0 (unit root) if t_stat < critical value → stationary
    is_stationary = pvalue < alpha

    return StationarityTestResult(
        test_name = :ADF,
        statistic = t_stat,
        pvalue = pvalue,
        is_stationary = is_stationary,
        lags_used = p,
        regression = regression,
        critical_values = critical_values
    )
end

"""
Approximate MacKinnon (1994) critical values for ADF test.
"""
function _adf_critical_values(regression::Symbol, n::Int)::Dict{String, Float64}
    # Asymptotic critical values (n → ∞)
    if regression == :n
        cv_1 = -2.58
        cv_5 = -1.95
        cv_10 = -1.62
    elseif regression == :c
        cv_1 = -3.43
        cv_5 = -2.86
        cv_10 = -2.57
    else  # :ct
        cv_1 = -3.96
        cv_5 = -3.41
        cv_10 = -3.13
    end

    # Finite sample adjustment (simplified)
    adj = 1.0 + 2.0 / n

    return Dict(
        "1%" => cv_1 * adj,
        "5%" => cv_5 * adj,
        "10%" => cv_10 * adj
    )
end

"""
Approximate p-value for ADF test using interpolation.
"""
function _adf_pvalue(t_stat::Float64, regression::Symbol, n::Int)::Float64
    cv = _adf_critical_values(regression, n)

    # Interpolate p-value
    if t_stat <= cv["1%"]
        return 0.005  # Very significant
    elseif t_stat <= cv["5%"]
        # Linear interpolation between 1% and 5%
        frac = (cv["5%"] - t_stat) / (cv["5%"] - cv["1%"])
        return 0.05 - frac * 0.04
    elseif t_stat <= cv["10%"]
        # Linear interpolation between 5% and 10%
        frac = (cv["10%"] - t_stat) / (cv["10%"] - cv["5%"])
        return 0.10 - frac * 0.05
    else
        # Beyond 10% critical value
        # Simple exponential decay from 0.10
        excess = t_stat - cv["10%"]
        return min(1.0, 0.10 + 0.90 * (1 - exp(-excess)))
    end
end
