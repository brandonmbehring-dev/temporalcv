# =============================================================================
# Volatility-Weighted Metrics
# =============================================================================

"""
Metrics that account for local volatility to provide scale-invariant
evaluation across different market regimes.

# Knowledge Tiers
- [T1] Rolling standard deviation (fundamental statistics)
- [T1] EWMA volatility (RiskMetrics, J.P. Morgan 1996)
- [T2] Volatility-normalized errors (common in quant finance)
- [T3] Volatility tercile stratification (practical heuristic)

# References
- J.P. Morgan (1996). RiskMetrics Technical Document.
"""

# =============================================================================
# Volatility Estimator Types
# =============================================================================

"""
Abstract type for volatility estimation methods.
Allows extensibility for custom volatility estimators.
"""
abstract type VolatilityEstimator end


"""
    RollingVolatility <: VolatilityEstimator

Rolling window standard deviation estimator.

# Fields
- `window::Int`: Rolling window size for standard deviation calculation
- `min_periods::Int`: Minimum observations required for a valid estimate
"""
struct RollingVolatility <: VolatilityEstimator
    window::Int
    min_periods::Int

    function RollingVolatility(window::Int=13, min_periods::Union{Int, Nothing}=nothing)
        if window < 2
            error("window must be >= 2, got $window")
        end
        mp = min_periods === nothing ? div(window, 2) : min_periods
        new(window, mp)
    end
end


"""
    EWMAVolatility <: VolatilityEstimator

Exponentially Weighted Moving Average volatility estimator.

EWMA places more weight on recent observations, making it more
responsive to volatility changes than rolling window methods.

# Fields
- `span::Int`: Decay span for EWMA. Lambda = 2/(span+1)
- `alpha::Float64`: Decay factor derived from span
"""
struct EWMAVolatility <: VolatilityEstimator
    span::Int
    alpha::Float64

    function EWMAVolatility(span::Int=13)
        if span < 1
            error("span must be >= 1, got $span")
        end
        alpha = 2.0 / (span + 1)
        new(span, alpha)
    end
end


"""
    estimate(estimator::RollingVolatility, values) -> Vector{Float64}

Compute rolling standard deviation.
"""
function estimate(estimator::RollingVolatility, values::AbstractVector{<:Real})::Vector{Float64}
    values = Float64.(values)
    n = length(values)

    if n == 0
        return Float64[]
    end

    volatility = fill(NaN, n)

    for i in 1:n
        start_idx = max(1, i - estimator.window + 1)
        window_data = values[start_idx:i]

        if length(window_data) >= estimator.min_periods
            volatility[i] = std(window_data; corrected=true)
        end
    end

    # Forward-fill NaNs at the start
    first_valid = findfirst(!isnan, volatility)
    if first_valid !== nothing && first_valid > 1
        volatility[1:first_valid-1] .= volatility[first_valid]
    end

    return volatility
end


"""
    estimate(estimator::EWMAVolatility, values) -> Vector{Float64}

Compute EWMA volatility (of squared deviations from mean).
"""
function estimate(estimator::EWMAVolatility, values::AbstractVector{<:Real})::Vector{Float64}
    values = Float64.(values)
    n = length(values)

    if n == 0
        return Float64[]
    end

    # Center around mean for variance calculation
    mean_val = mean(values)
    squared_devs = (values .- mean_val).^2

    # EWMA of squared deviations
    ewma_var = zeros(n)
    ewma_var[1] = squared_devs[1]

    for i in 2:n
        ewma_var[i] = estimator.alpha * squared_devs[i] + (1 - estimator.alpha) * ewma_var[i-1]
    end

    volatility = sqrt.(ewma_var)

    return volatility
end


# =============================================================================
# Volatility Computation Functions
# =============================================================================

"""
    compute_local_volatility(values; window=13, method=:rolling_std) -> Vector{Float64}

Compute local volatility estimates.

# Arguments
- `values::AbstractVector{<:Real}`: Input values (typically returns or changes)
- `window::Int=13`: Window size for rolling methods, or span for EWMA
- `method::Symbol=:rolling_std`: Volatility estimation method
  - `:rolling_std`: Rolling window standard deviation
  - `:ewm`: Exponentially weighted moving average

# Returns
Local volatility estimates, same length as input.

# Notes
[T3] Window of 13 (approximately one quarter for weekly data) is a
practical default balancing responsiveness with stability.

# Example
```julia
returns = randn(100) .* 0.02
vol = compute_local_volatility(returns; window=13)
```
"""
function compute_local_volatility(
    values::AbstractVector{<:Real};
    window::Int = 13,
    method::Symbol = :rolling_std
)::Vector{Float64}
    values = Float64.(values)

    if isempty(values)
        return Float64[]
    end

    estimator = if method == :rolling_std
        RollingVolatility(window)
    elseif method == :ewm
        EWMAVolatility(window)
    else
        error("Unknown method '$method'. Use :rolling_std or :ewm")
    end

    return estimate(estimator, values)
end


# =============================================================================
# Volatility-Adjusted Metrics
# =============================================================================

"""
    compute_volatility_normalized_mae(predictions, actuals, volatility;
                                      epsilon=1e-8) -> Float64

Compute volatility-normalized MAE (scale-invariant).

Divides errors by local volatility, making the metric comparable
across different volatility regimes and time series.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual observed values
- `volatility::AbstractVector{<:Real}`: Local volatility estimates
- `epsilon::Float64=1e-8`: Small constant to prevent division by zero

# Returns
Mean volatility-normalized absolute error.

# Formula
```
VN-MAE = mean( |prediction - actual| / volatility )
```

A value of 1.0 means errors are "typical" relative to local volatility.
Lower is better.

# Example
```julia
predictions = [1.0, 2.0, 3.0]
actuals = [1.1, 1.9, 3.2]
volatility = [0.1, 0.15, 0.2]
vnmae = compute_volatility_normalized_mae(predictions, actuals, volatility)
```
"""
function compute_volatility_normalized_mae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    volatility::AbstractVector{<:Real};
    epsilon::Float64 = 1e-8
)::Float64
    if length(predictions) != length(actuals) || length(predictions) != length(volatility)
        error("Array lengths must match. Got predictions=$(length(predictions)), " *
              "actuals=$(length(actuals)), volatility=$(length(volatility))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    abs_errors = abs.(predictions .- actuals)
    normalized_errors = abs_errors ./ (volatility .+ epsilon)

    return mean(normalized_errors)
end


"""
    compute_volatility_weighted_mae(predictions, actuals, volatility;
                                    weighting=:inverse, epsilon=1e-8) -> Float64

Compute volatility-weighted MAE.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual observed values
- `volatility::AbstractVector{<:Real}`: Local volatility estimates
- `weighting::Symbol=:inverse`: How to weight by volatility
  - `:inverse`: Weight low-vol periods more (clearer signal)
  - `:importance`: Weight high-vol periods more (if those matter)
- `epsilon::Float64=1e-8`: Small constant to prevent division by zero

# Returns
Weighted mean absolute error.

# Notes
With inverse weighting, low-volatility periods (where predictions
should be more precise) receive higher weight. This is useful when
the goal is accuracy during stable periods.

With importance weighting, high-volatility periods receive higher
weight. This is useful when performance during turbulent periods
matters most (e.g., risk management).

# Example
```julia
predictions = [1.0, 2.0, 3.0]
actuals = [1.1, 1.9, 3.2]
volatility = [0.1, 0.5, 0.2]
# Weight low-vol periods more
wmae = compute_volatility_weighted_mae(predictions, actuals, volatility;
                                       weighting=:inverse)
```
"""
function compute_volatility_weighted_mae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    volatility::AbstractVector{<:Real};
    weighting::Symbol = :inverse,
    epsilon::Float64 = 1e-8
)::Float64
    if length(predictions) != length(actuals) || length(predictions) != length(volatility)
        error("Array lengths must match. Got predictions=$(length(predictions)), " *
              "actuals=$(length(actuals)), volatility=$(length(volatility))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    if weighting ∉ (:inverse, :importance)
        error("weighting must be :inverse or :importance, got $weighting")
    end

    abs_errors = abs.(predictions .- actuals)

    weights = if weighting == :inverse
        1.0 ./ (volatility .+ epsilon)
    else  # :importance
        Float64.(volatility)
    end

    # Normalize weights
    weights = weights ./ sum(weights)

    weighted_mae = sum(weights .* abs_errors)

    return weighted_mae
end


# =============================================================================
# Volatility-Stratified Metrics
# =============================================================================

"""
    compute_volatility_stratified_metrics(predictions, actuals;
                                          volatility=nothing, window=13,
                                          method=:rolling_std) -> VolatilityStratifiedResult

Compute MAE stratified by volatility terciles.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual observed values
- `volatility::Union{AbstractVector{<:Real}, Nothing}=nothing`:
  Pre-computed volatility estimates. If not provided, will be computed
  from actuals using the specified method.
- `window::Int=13`: Window size for volatility estimation (if not provided)
- `method::Symbol=:rolling_std`: Volatility estimation method

# Returns
`VolatilityStratifiedResult` with stratified metrics.

# Notes
[T3] Tercile stratification uses the 33rd and 67th percentiles of
volatility to create three equally-sized groups.

# Example
```julia
predictions = randn(100)
actuals = randn(100)
result = compute_volatility_stratified_metrics(predictions, actuals)
```
"""
function compute_volatility_stratified_metrics(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    volatility::Union{AbstractVector{<:Real}, Nothing} = nothing,
    window::Int = 13,
    method::Symbol = :rolling_std
)::VolatilityStratifiedResult
    predictions = Float64.(predictions)
    actuals = Float64.(actuals)

    if length(predictions) != length(actuals)
        error("Array lengths must match. Got predictions=$(length(predictions)), " *
              "actuals=$(length(actuals))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    # Compute volatility if not provided
    vol = if volatility === nothing
        compute_local_volatility(actuals; window=window, method=method)
    else
        if length(volatility) != length(predictions)
            error("volatility length must match. Got $(length(volatility)), " *
                  "expected $(length(predictions))")
        end
        Float64.(volatility)
    end

    # Compute tercile thresholds
    p33 = quantile(vol, 0.3333)
    p67 = quantile(vol, 0.6667)

    # Classify into terciles
    low_mask = vol .<= p33
    high_mask = vol .> p67
    med_mask = .!low_mask .& .!high_mask

    # Compute errors
    abs_errors = abs.(predictions .- actuals)

    # Overall metrics
    overall_mae = mean(abs_errors)
    vnmae = mean(abs_errors ./ (vol .+ 1e-8))

    # Stratified MAE (handle empty groups)
    function safe_mean(arr)
        isempty(arr) ? NaN : mean(arr)
    end

    low_vol_mae = safe_mean(abs_errors[low_mask])
    med_vol_mae = safe_mean(abs_errors[med_mask])
    high_vol_mae = safe_mean(abs_errors[high_mask])

    # Match existing VolatilityStratifiedResult: 7 fields
    # mae_low, mae_medium, mae_high, n_low, n_medium, n_high, vol_thresholds
    return VolatilityStratifiedResult(
        low_vol_mae,
        med_vol_mae,
        high_vol_mae,
        sum(low_mask),
        sum(med_mask),
        sum(high_mask),
        (p33, p67)
    )
end
