# =============================================================================
# Lag Selection for Time Series Models
# =============================================================================

"""
Lag selection module for AR model order selection.

Methods:
- PACF: Partial autocorrelation function significance cutoff
- AIC: Akaike Information Criterion minimization
- BIC: Bayesian Information Criterion minimization

# Knowledge Tiers
- [T1] Box, Jenkins & Reinsel (2015). Time Series Analysis, 5th ed.
- [T1] Akaike (1974). AIC for model identification
- [T1] Schwarz (1978). BIC for dimension estimation

# Example
```julia
using TemporalValidation

# AR(2) process
rng = MersenneTwister(42)
ar2 = zeros(200)
for i in 3:200
    ar2[i] = 0.5 * ar2[i-1] + 0.3 * ar2[i-2] + randn(rng)
end

result = select_lag_bic(ar2)
result.optimal_lag  # 2

# Suggest CV gap
gap = suggest_cv_gap(ar2, horizon=1)
```
"""

using Statistics
using Distributions
using LinearAlgebra

# =============================================================================
# Types
# =============================================================================

"""
    LagSelectionResult

Result of lag selection procedure.

# Fields
- `optimal_lag::Int`: Selected optimal lag order
- `criterion_values::Dict{Int, Float64}`: Criterion value for each lag tested
- `method::Symbol`: Method used (:pacf, :aic, :bic)
- `all_lags_tested::Vector{Int}`: All lag values evaluated
"""
struct LagSelectionResult
    optimal_lag::Int
    criterion_values::Dict{Int, Float64}
    method::Symbol
    all_lags_tested::Vector{Int}
end

function Base.show(io::IO, r::LagSelectionResult)
    print(io, "LagSelectionResult(optimal=$(r.optimal_lag), method=$(r.method), tested=$(length(r.all_lags_tested)) lags)")
end

# =============================================================================
# Utilities
# =============================================================================

"""
    compute_max_lag(n, max_lag)

Compute maximum lag to test using rule of thumb: min(10*log10(n), n/4).
"""
function compute_max_lag(n::Int, max_lag::Union{Int, Nothing})::Int
    if !isnothing(max_lag)
        return min(max_lag, n ÷ 2 - 1)
    end

    # Rule of thumb: min(10*log10(n), n/4)
    rule_of_thumb = min(floor(Int, 10 * log10(n)), n ÷ 4)
    return max(1, min(rule_of_thumb, n ÷ 2 - 1))
end

"""
    compute_pacf(y, max_lag)

Compute partial autocorrelation function using Yule-Walker equations.
Returns PACF values for lags 0 to max_lag.
"""
function compute_pacf(y::AbstractVector{<:Real}, max_lag::Int)::Vector{Float64}
    n = length(y)
    y_centered = y .- mean(y)

    # Compute autocorrelation function first
    acf = zeros(max_lag + 1)
    for k in 0:max_lag
        if k == 0
            acf[k+1] = 1.0
        else
            acf[k+1] = sum(y_centered[1:(n-k)] .* y_centered[(k+1):n]) / sum(y_centered.^2)
        end
    end

    # PACF using Durbin-Levinson algorithm
    pacf = zeros(max_lag + 1)
    pacf[1] = 1.0  # PACF at lag 0 is always 1

    if max_lag >= 1
        pacf[2] = acf[2]  # PACF at lag 1 = ACF at lag 1

        # Durbin-Levinson recursion
        phi = zeros(max_lag)
        phi[1] = acf[2]

        for k in 2:max_lag
            # Compute phi_kk
            num = acf[k+1]
            for j in 1:(k-1)
                num -= phi[j] * acf[k-j+1]
            end

            den = 1.0
            for j in 1:(k-1)
                den -= phi[j] * acf[j+1]
            end

            phi_kk = den ≈ 0.0 ? 0.0 : num / den
            pacf[k+1] = phi_kk

            # Update phi vector
            phi_old = copy(phi[1:(k-1)])
            for j in 1:(k-1)
                phi[j] = phi_old[j] - phi_kk * phi_old[k-j]
            end
            phi[k] = phi_kk
        end
    end

    return pacf
end

"""
    fit_ar_ols(y, p)

Fit AR(p) model using OLS. Returns (coefficients, σ²_mle, aic, bic).
Uses concentrated log-likelihood for information criteria.
"""
function fit_ar_ols(y::Vector{Float64}, p::Int)::NamedTuple{(:β, :σ², :aic, :bic), Tuple{Vector{Float64}, Float64, Float64, Float64}}
    n = length(y)
    @assert p >= 1 "AR order must be at least 1"
    @assert n > p + 10 "Insufficient data for AR($p)"

    # Build design matrix
    n_eff = n - p
    X = ones(n_eff, p + 1)  # Include intercept

    for i in 1:p
        X[:, i+1] = y[(p-i+1):(n-i)]
    end

    y_target = y[(p+1):n]

    # OLS
    β = X \ y_target
    residuals = y_target - X * β
    rss = sum(residuals.^2)
    σ²_mle = rss / n_eff

    # Information criteria using concentrated log-likelihood
    # log L = -n/2 * (log(2π) + log(σ²) + 1)
    # AIC = -2*log L + 2k = n*(log(2π) + log(σ²) + 1) + 2k
    # Dropping constants that don't affect comparison:
    # AIC = n*log(σ²) + 2k
    k = p + 1  # AR coefficients + intercept (not counting σ² as estimated separately)
    aic = n_eff * log(σ²_mle) + 2 * k
    bic = n_eff * log(σ²_mle) + k * log(n_eff)

    return (β=β, σ²=σ²_mle, aic=aic, bic=bic)
end

# =============================================================================
# Selection Functions
# =============================================================================

"""
    select_lag_pacf(series; max_lag=nothing, alpha=0.05)

Select lag using PACF significance cutoff.

Finds the last lag where PACF is significantly different from zero
using Bartlett approximation: ±z_{α/2}/√n.

# Arguments
- `series::AbstractVector`: Time series data
- `max_lag::Union{Int, Nothing}=nothing`: Maximum lag to consider
- `alpha::Float64=0.05`: Significance level for confidence interval

# Returns
- `LagSelectionResult`: Selected lag and PACF values

# Example
```julia
rng = MersenneTwister(42)
ar2 = zeros(200)
for i in 3:200
    ar2[i] = 0.5 * ar2[i-1] + 0.3 * ar2[i-2] + randn(rng)
end
result = select_lag_pacf(ar2)
result.optimal_lag  # 2
```

# Notes
[T1] Box, Jenkins & Reinsel (2015), Chapter 3.
"""
function select_lag_pacf(
    series::AbstractVector{<:Real};
    max_lag::Union{Int, Nothing} = nothing,
    alpha::Float64 = 0.05
)::LagSelectionResult
    y = collect(Float64, series)
    n = length(y)

    @assert n >= 10 "Series too short for PACF: n=$n, need >= 10"

    max_lag_to_test = compute_max_lag(n, max_lag)

    # Compute PACF
    pacf_values = compute_pacf(y, max_lag_to_test)

    # Confidence interval threshold (Bartlett approximation)
    z_critical = quantile(Normal(), 1 - alpha / 2)
    threshold = z_critical / sqrt(n)

    # Build criterion values
    criterion_values = Dict{Int, Float64}()
    all_lags = collect(0:max_lag_to_test)

    for lag in all_lags
        criterion_values[lag] = pacf_values[lag+1]
    end

    # Find optimal lag: last consecutive significant lag starting from 1
    optimal_lag = 0
    for lag in 1:max_lag_to_test
        if abs(criterion_values[lag]) > threshold
            optimal_lag = lag
        else
            # First insignificant lag - stop
            break
        end
    end

    return LagSelectionResult(
        optimal_lag,
        criterion_values,
        :pacf,
        all_lags
    )
end

"""
    select_lag_aic(series; max_lag=nothing)

Select lag minimizing AIC (Akaike Information Criterion).

AIC = 2k - 2ln(L) where k is number of parameters.
Tends to select larger models than BIC.

# Arguments
- `series::AbstractVector`: Time series data
- `max_lag::Union{Int, Nothing}=nothing`: Maximum lag to consider

# Returns
- `LagSelectionResult`: Selected lag and AIC values

# Notes
[T1] Akaike (1974). "A new look at statistical model identification."
"""
function select_lag_aic(
    series::AbstractVector{<:Real};
    max_lag::Union{Int, Nothing} = nothing
)::LagSelectionResult
    y = collect(Float64, series)
    n = length(y)

    @assert n >= 10 "Series too short for lag selection: n=$n, need >= 10"

    max_lag_to_test = compute_max_lag(n, max_lag)

    criterion_values = Dict{Int, Float64}()
    all_lags = collect(1:max_lag_to_test)

    for lag in all_lags
        try
            result = fit_ar_ols(y, lag)
            criterion_values[lag] = result.aic
        catch
            criterion_values[lag] = Inf
        end
    end

    # Find lag with minimum AIC
    if isempty(criterion_values)
        optimal_lag = 1
    else
        optimal_lag = argmin(criterion_values)
    end

    return LagSelectionResult(
        optimal_lag,
        criterion_values,
        :aic,
        all_lags
    )
end

"""
    select_lag_bic(series; max_lag=nothing)

Select lag minimizing BIC (Bayesian Information Criterion).

BIC = k*ln(n) - 2ln(L) where k is number of parameters.
More parsimonious than AIC. Asymptotically consistent.

# Arguments
- `series::AbstractVector`: Time series data
- `max_lag::Union{Int, Nothing}=nothing`: Maximum lag to consider

# Returns
- `LagSelectionResult`: Selected lag and BIC values

# Notes
[T1] Schwarz (1978). "Estimating the Dimension of a Model."
"""
function select_lag_bic(
    series::AbstractVector{<:Real};
    max_lag::Union{Int, Nothing} = nothing
)::LagSelectionResult
    y = collect(Float64, series)
    n = length(y)

    @assert n >= 10 "Series too short for lag selection: n=$n, need >= 10"

    max_lag_to_test = compute_max_lag(n, max_lag)

    criterion_values = Dict{Int, Float64}()
    all_lags = collect(1:max_lag_to_test)

    for lag in all_lags
        try
            result = fit_ar_ols(y, lag)
            criterion_values[lag] = result.bic
        catch
            criterion_values[lag] = Inf
        end
    end

    # Find lag with minimum BIC
    if isempty(criterion_values)
        optimal_lag = 1
    else
        optimal_lag = argmin(criterion_values)
    end

    return LagSelectionResult(
        optimal_lag,
        criterion_values,
        :bic,
        all_lags
    )
end

"""
    auto_select_lag(series; method=:bic, max_lag=nothing, alpha=0.05)

Convenience function returning just the optimal lag.

# Arguments
- `series::AbstractVector`: Time series data
- `method::Symbol=:bic`: Selection method (:aic, :bic, :pacf)
- `max_lag::Union{Int, Nothing}=nothing`: Maximum lag to consider
- `alpha::Float64=0.05`: Significance level for PACF method

# Returns
- `Int`: Optimal lag order

# Example
```julia
rng = MersenneTwister(42)
ar2 = zeros(200)
for i in 3:200
    ar2[i] = 0.5 * ar2[i-1] + 0.3 * ar2[i-2] + randn(rng)
end
auto_select_lag(ar2)  # 2
```
"""
function auto_select_lag(
    series::AbstractVector{<:Real};
    method::Symbol = :bic,
    max_lag::Union{Int, Nothing} = nothing,
    alpha::Float64 = 0.05
)::Int
    if method == :aic
        result = select_lag_aic(series; max_lag=max_lag)
    elseif method == :bic
        result = select_lag_bic(series; max_lag=max_lag)
    elseif method == :pacf
        result = select_lag_pacf(series; max_lag=max_lag, alpha=alpha)
    else
        error("Unknown method: $method. Use :aic, :bic, or :pacf.")
    end

    return result.optimal_lag
end

"""
    suggest_cv_gap(series; horizon=1, method=:bic)

Suggest cross-validation gap based on series autocorrelation.

Gap = max(horizon, optimal_lag)

This ensures temporal separation accounts for both forecast
horizon and series memory.

# Arguments
- `series::AbstractVector`: Time series data
- `horizon::Int=1`: Forecast horizon
- `method::Symbol=:bic`: Method for estimating series memory

# Returns
- `Int`: Suggested gap parameter for walk-forward CV

# Example
```julia
gap = suggest_cv_gap(ar5, horizon=1)
# Use in CV
cv = WalkForwardCV(n_splits=5, test_size=10, gap=gap)
```
"""
function suggest_cv_gap(
    series::AbstractVector{<:Real};
    horizon::Int = 1,
    method::Symbol = :bic
)::Int
    optimal_lag = auto_select_lag(series; method=method)
    return max(horizon, optimal_lag)
end
