# =============================================================================
# Core Forecast Evaluation Metrics
# =============================================================================

"""
Core forecast evaluation metrics.

Foundational metrics for time series forecast evaluation:
- Point forecast metrics: MAE, MSE, RMSE, MAPE, SMAPE
- Scale-invariant metrics: MASE, Theil's U, MRAE
- Correlation metrics: R², forecast correlation
- Naive/persistence baselines: compute_naive_error

# Knowledge Tiers
- [T1] All metrics are standard statistical formulations
- [T1] MASE: Hyndman & Koehler (2006), "Another look at measures of forecast accuracy"
- [T1] SMAPE: Armstrong (1985), bounded symmetric alternative to MAPE
- [T1] Theil's U: Theil (1966), relative accuracy to naive forecast

# References
- Hyndman, R.J. & Koehler, A.B. (2006). Another look at measures of
  forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
- Armstrong, J.S. (1985). Long-Range Forecasting: From Crystal Ball to Computer.
- Theil, H. (1966). Applied Economic Forecasting. North-Holland Publishing.
"""

using Statistics
using LinearAlgebra

# =============================================================================
# Input Validation
# =============================================================================

"""
    _validate_inputs(predictions, actuals, name)

Validate and convert inputs to Float64 vectors.

Raises error if:
- Inputs have different lengths
- Inputs contain NaN values
"""
function _validate_inputs(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    name::String
)::Tuple{Vector{Float64}, Vector{Float64}}
    preds = collect(Float64, predictions)
    acts = collect(Float64, actuals)

    if length(preds) != length(acts)
        error("$name: predictions and actuals must have same length " *
              "(got $(length(preds)) and $(length(acts)))")
    end

    if any(isnan, preds)
        error("$name: predictions contain NaN values")
    end

    if any(isnan, acts)
        error("$name: actuals contain NaN values")
    end

    return preds, acts
end

# =============================================================================
# Point Forecast Metrics
# =============================================================================

"""
    compute_mae(predictions, actuals)

Compute Mean Absolute Error.

MAE = mean(|ŷ - y|)

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values

# Returns
- `Float64`: Mean absolute error

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_mae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_mae")
    return mean(abs.(preds .- acts))
end


"""
    compute_mse(predictions, actuals)

Compute Mean Squared Error.

MSE = mean((ŷ - y)²)

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values

# Returns
- `Float64`: Mean squared error

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_mse(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_mse")
    return mean((preds .- acts).^2)
end


"""
    compute_rmse(predictions, actuals)

Compute Root Mean Squared Error.

RMSE = √(mean((ŷ - y)²))

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values

# Returns
- `Float64`: Root mean squared error

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_rmse(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    return sqrt(compute_mse(predictions, actuals))
end


"""
    compute_mape(predictions, actuals; epsilon=1e-8)

Compute Mean Absolute Percentage Error.

MAPE = 100 × mean(|ŷ - y| / |y|)

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values
- `epsilon::Float64=1e-8`: Small value to prevent division by zero

# Returns
- `Float64`: Mean absolute percentage error (as percentage, 0-100+)

# Notes
MAPE has known issues:
- Undefined when actuals = 0
- Asymmetric: penalizes over-prediction more
- Unbounded above 100%

Consider SMAPE for a bounded alternative.

# Knowledge Tier
[T1] Standard percentage error metric.
"""
function compute_mape(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    epsilon::Float64 = 1e-8
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_mape")
    denom = max.(abs.(acts), epsilon)
    return 100.0 * mean(abs.(preds .- acts) ./ denom)
end


"""
    compute_smape(predictions, actuals)

Compute Symmetric Mean Absolute Percentage Error.

SMAPE = 100 × mean(2|ŷ - y| / (|ŷ| + |y|))

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values

# Returns
- `Float64`: Symmetric MAPE (bounded 0-200%)

# Notes
SMAPE is bounded [0, 200%] and symmetric around zero.
When both prediction and actual are zero, that observation is excluded.

# Knowledge Tier
[T1] Armstrong (1985) symmetric alternative to MAPE.
"""
function compute_smape(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_smape")

    denom = abs.(preds) .+ abs.(acts)
    # Exclude cases where both are zero
    mask = denom .> 0

    if !any(mask)
        return 0.0
    end

    numerator = 2.0 .* abs.(preds[mask] .- acts[mask])
    return 100.0 * mean(numerator ./ denom[mask])
end


"""
    compute_bias(predictions, actuals)

Compute mean signed error (bias).

Bias = mean(ŷ - y)

Positive bias indicates over-prediction on average.
Negative bias indicates under-prediction on average.

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values

# Returns
- `Float64`: Mean signed error (positive = over-prediction)

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_bias(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_bias")
    return mean(preds .- acts)
end


# =============================================================================
# Scale-Invariant Metrics
# =============================================================================

"""
    compute_naive_error(values; method=:persistence)

Compute naive forecast MAE for scale normalization.

Used as denominator for MASE and other scale-free metrics.

# Arguments
- `values::AbstractVector`: Training series values
- `method::Symbol=:persistence`: Method for naive forecast
  - `:persistence`: Naive forecast (y[t] = y[t-1])
  - `:mean`: Mean forecast (y[t] = mean(y))

# Returns
- `Float64`: MAE of naive forecast on training data

# Notes
For persistence: MAE = mean(|y[t] - y[t-1]|) for t = 2, ..., n
This represents the "cost of being naive" for MASE normalization.

# Knowledge Tier
[T1] Hyndman & Koehler (2006).
"""
function compute_naive_error(
    values::AbstractVector{<:Real};
    method::Symbol = :persistence
)::Float64
    vals = collect(Float64, values)
    n = length(vals)

    if n < 2
        error("compute_naive_error requires at least 2 values")
    end

    if method == :persistence
        # y[t] - y[t-1] for t >= 2
        return mean(abs.(diff(vals)))
    elseif method == :mean
        mean_val = mean(vals)
        return mean(abs.(vals .- mean_val))
    else
        error("method must be :persistence or :mean, got $method")
    end
end


"""
    compute_mase(predictions, actuals, naive_mae)

Compute Mean Absolute Scaled Error.

MASE = MAE / naive_MAE

Where naive_MAE is typically the in-sample MAE of the naive
(persistence) forecast.

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values
- `naive_mae::Float64`: MAE of naive forecast on training data.
  Compute with `compute_naive_error(train_values)`.

# Returns
- `Float64`: MASE value. <1 means better than naive, >1 means worse.

# Notes
MASE is scale-free and can compare accuracy across different time series.

- MASE < 1: Model beats naive forecast
- MASE = 1: Model equals naive forecast
- MASE > 1: Model worse than naive forecast

# Knowledge Tier
[T1] Hyndman & Koehler (2006).
"""
function compute_mase(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    naive_mae::Float64
)::Float64
    if naive_mae <= 0
        error("naive_mae must be positive, got $naive_mae")
    end

    mae = compute_mae(predictions, actuals)
    return mae / naive_mae
end


"""
    compute_mrae(predictions, actuals, naive_predictions)

Compute Mean Relative Absolute Error.

MRAE = mean(|ŷ - y| / |ŷ_naive - y|)

# Arguments
- `predictions::AbstractVector`: Model predictions
- `actuals::AbstractVector`: Actual values
- `naive_predictions::AbstractVector`: Naive/baseline predictions (same length)

# Returns
- `Float64`: MRAE value. <1 means better than naive.

# Notes
MRAE compares each error to the naive error at that point.
Points where naive_error = 0 are excluded.

# Knowledge Tier
[T1] Standard relative error metric.
"""
function compute_mrae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    naive_predictions::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_mrae")
    naive = collect(Float64, naive_predictions)

    if length(naive) != length(preds)
        error("naive_predictions must have same length as predictions")
    end

    model_errors = abs.(preds .- acts)
    naive_errors = abs.(naive .- acts)

    # Exclude points where naive error is zero
    mask = naive_errors .> 0
    if !any(mask)
        return NaN
    end

    return mean(model_errors[mask] ./ naive_errors[mask])
end


"""
    compute_theils_u(predictions, actuals; naive_predictions=nothing)

Compute Theil's U statistic.

U = RMSE(model) / RMSE(naive)

If naive_predictions not provided, uses persistence (y[t-1]).

# Arguments
- `predictions::AbstractVector`: Model predictions
- `actuals::AbstractVector`: Actual values
- `naive_predictions::Union{AbstractVector, Nothing}=nothing`:
  Naive/baseline predictions. If nothing, uses persistence.

# Returns
- `Float64`: Theil's U. <1 means better than naive, >1 means worse.

# Notes
This is Theil's U2 (1966), comparing to a naive forecast.
U < 1 indicates the model outperforms the naive forecast.

# Knowledge Tier
[T1] Theil (1966).
"""
function compute_theils_u(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    naive_predictions::Union{AbstractVector{<:Real}, Nothing} = nothing
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_theils_u")

    if isnothing(naive_predictions)
        # Use persistence: y[t-1] as prediction for y[t]
        if length(preds) < 2
            error("Need at least 2 observations for persistence baseline")
        end
        # y[0], y[1], ..., y[n-2] predict y[1], ..., y[n-1]
        naive = acts[1:end-1]
        preds = preds[2:end]
        acts = acts[2:end]
    else
        naive = collect(Float64, naive_predictions)
        if length(naive) != length(preds)
            error("naive_predictions must have same length as predictions")
        end
    end

    model_rmse = sqrt(mean((preds .- acts).^2))
    naive_rmse = sqrt(mean((naive .- acts).^2))

    if naive_rmse == 0
        return model_rmse > 0 ? Inf : 1.0
    end

    return model_rmse / naive_rmse
end


# =============================================================================
# Correlation Metrics
# =============================================================================

"""
    compute_forecast_correlation(predictions, actuals; method=:pearson)

Compute correlation between predictions and actuals.

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values
- `method::Symbol=:pearson`: Correlation method (:pearson or :spearman)

# Returns
- `Float64`: Correlation coefficient [-1, 1]

# Notes
Correlation measures association but not accuracy. A model can
have high correlation but large errors (wrong scale/offset).

# Knowledge Tier
[T1] Standard statistical measures.
"""
function compute_forecast_correlation(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    method::Symbol = :pearson
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_forecast_correlation")

    if length(preds) < 2
        return NaN
    end

    if method == :pearson
        # Pearson correlation using Statistics.cor
        return cor(preds, acts)
    elseif method == :spearman
        # Spearman rank correlation
        n = length(preds)
        ranks_pred = sortperm(sortperm(preds))
        ranks_act = sortperm(sortperm(acts))
        return cor(Float64.(ranks_pred), Float64.(ranks_act))
    else
        error("method must be :pearson or :spearman, got $method")
    end
end


"""
    compute_r_squared(predictions, actuals)

Compute R² (coefficient of determination).

R² = 1 - SS_res / SS_tot

Where:
- SS_res = sum((y - ŷ)²)  [residual sum of squares]
- SS_tot = sum((y - mean(y))²)  [total sum of squares]

# Arguments
- `predictions::AbstractVector`: Predicted values
- `actuals::AbstractVector`: Actual values

# Returns
- `Float64`: R² value. Can be negative if model is worse than mean.

# Notes
- R² = 1: Perfect predictions
- R² = 0: Model equals mean forecast
- R² < 0: Model worse than mean forecast

# Knowledge Tier
[T1] Standard statistical measure.
"""
function compute_r_squared(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_inputs(predictions, actuals, "compute_r_squared")

    ss_res = sum((acts .- preds).^2)
    ss_tot = sum((acts .- mean(acts)).^2)

    if ss_tot == 0
        # All actuals are the same
        return ss_res == 0 ? 1.0 : -Inf
    end

    return 1.0 - ss_res / ss_tot
end
