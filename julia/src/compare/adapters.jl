# =============================================================================
# Forecast Adapters - Abstract Interface for Model Comparison
# =============================================================================

"""
    ForecastAdapter

Abstract base type for model adapters.

Adapters wrap different forecasting packages to provide a unified interface
for comparison.

# Required Methods
- `model_name(adapter) -> String`: Return the model name
- `package_name(adapter) -> String`: Return the package name
- `fit_predict(adapter, train_values, test_size, horizon) -> Vector{Float64}`: Fit and predict

# Optional Methods
- `get_params(adapter) -> Dict{String, Any}`: Return model parameters

# Example
```julia
struct MyAdapter <: ForecastAdapter
    param::Float64
end

model_name(::MyAdapter) = "MyModel"
package_name(::MyAdapter) = "MyPackage"

function fit_predict(adapter::MyAdapter, train::AbstractVector{<:Real}, test_size::Int, horizon::Int)
    # Your implementation
    return predictions
end
```
"""
abstract type ForecastAdapter end


"""
    model_name(adapter::ForecastAdapter) -> String

Return the model name.
"""
function model_name end


"""
    package_name(adapter::ForecastAdapter) -> String

Return the package name.
"""
function package_name end


"""
    fit_predict(adapter::ForecastAdapter, train_values, test_size, horizon) -> Vector{Float64}

Fit model and generate predictions.

# Arguments
- `adapter::ForecastAdapter`: The adapter instance
- `train_values::AbstractVector{<:Real}`: Training data
- `test_size::Int`: Number of test periods to predict
- `horizon::Int`: Forecast horizon (may differ from test_size for rolling forecasts)

# Returns
Predictions vector with length matching test_size.
"""
function fit_predict end


"""
    get_params(adapter::ForecastAdapter) -> Dict{String, Any}

Get model parameters.

Default implementation returns empty dict.
"""
function get_params(adapter::ForecastAdapter)::Dict{String, Any}
    return Dict{String, Any}()
end


# =============================================================================
# Naive Adapter - Persistence Baseline
# =============================================================================

"""
    NaiveAdapter <: ForecastAdapter

Naive persistence baseline adapter.

Predicts the last observed value for all future periods.

# Example
```julia
adapter = NaiveAdapter()
train = [1.0, 2.0, 3.0, 4.0, 5.0]
predictions = fit_predict(adapter, train, 3, 1)
# Returns: [5.0, 5.0, 5.0]
```
"""
struct NaiveAdapter <: ForecastAdapter end

model_name(::NaiveAdapter)::String = "Naive"
package_name(::NaiveAdapter)::String = "TemporalValidation"

"""
    fit_predict(adapter::NaiveAdapter, train_values, test_size, horizon)

Generate naive forecasts (repeat last value).
"""
function fit_predict(
    ::NaiveAdapter,
    train_values::AbstractVector{<:Real},
    test_size::Int,
    horizon::Int
)::Vector{Float64}
    isempty(train_values) && error("train_values cannot be empty")
    test_size > 0 || error("test_size must be positive")

    last_value = Float64(train_values[end])
    return fill(last_value, test_size)
end


# =============================================================================
# Seasonal Naive Adapter - Seasonal Baseline
# =============================================================================

"""
    SeasonalNaiveAdapter <: ForecastAdapter

Seasonal naive baseline adapter.

Predicts using the value from the same period in the last season.

# Fields
- `season_length::Int`: Length of seasonal period (default: 52 for weekly with yearly seasonality)

# Example
```julia
adapter = SeasonalNaiveAdapter(4)  # Quarterly seasonality
train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 2 full seasons
predictions = fit_predict(adapter, train, 4, 1)
# Returns: [5.0, 6.0, 7.0, 8.0]  # Values from last season
```
"""
struct SeasonalNaiveAdapter <: ForecastAdapter
    season_length::Int

    function SeasonalNaiveAdapter(season_length::Int = 52)
        season_length > 0 || error("season_length must be positive, got $season_length")
        new(season_length)
    end
end

model_name(adapter::SeasonalNaiveAdapter)::String = "SeasonalNaive_$(adapter.season_length)"
package_name(::SeasonalNaiveAdapter)::String = "TemporalValidation"

function get_params(adapter::SeasonalNaiveAdapter)::Dict{String, Any}
    return Dict{String, Any}("season_length" => adapter.season_length)
end

"""
    fit_predict(adapter::SeasonalNaiveAdapter, train_values, test_size, horizon)

Generate seasonal naive forecasts.
"""
function fit_predict(
    adapter::SeasonalNaiveAdapter,
    train_values::AbstractVector{<:Real},
    test_size::Int,
    horizon::Int
)::Vector{Float64}
    isempty(train_values) && error("train_values cannot be empty")
    test_size > 0 || error("test_size must be positive")

    train = Float64.(train_values)
    n_train = length(train)
    predictions = Vector{Float64}(undef, test_size)

    for i in 1:test_size
        # Index into training data using seasonal lag
        # For forecasting period i (1-indexed), we want value from season_length periods ago
        lag_idx = n_train - adapter.season_length + ((i - 1) % adapter.season_length) + 1

        if lag_idx < 1
            # Fallback to last value if not enough history
            lag_idx = n_train
        end

        predictions[i] = train[lag_idx]
    end

    return predictions
end


# =============================================================================
# Metric Functions
# =============================================================================

"""
    compute_comparison_metrics(predictions, actuals) -> Dict{String, Float64}

Compute standard comparison metrics.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
Dictionary with:
- `mae`: Mean Absolute Error
- `rmse`: Root Mean Squared Error
- `mape`: Mean Absolute Percentage Error (NaN if any zeros in actuals)
- `direction_accuracy`: Proportion of correct direction predictions

# Example
```julia
predictions = [1.1, 2.0, 2.9]
actuals = [1.0, 2.0, 3.0]
metrics = compute_comparison_metrics(predictions, actuals)
# Dict("mae" => 0.0667, "rmse" => 0.0816, ...)
```
"""
function compute_comparison_metrics(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Dict{String, Float64}
    n = length(predictions)
    n == length(actuals) || error("Length mismatch: predictions=$n, actuals=$(length(actuals))")
    n > 0 || error("Arrays cannot be empty")

    preds = Float64.(predictions)
    acts = Float64.(actuals)
    errors = preds .- acts
    abs_errors = abs.(errors)

    metrics = Dict{String, Float64}()

    # MAE
    metrics["mae"] = mean(abs_errors)

    # RMSE
    metrics["rmse"] = sqrt(mean(errors .^ 2))

    # MAPE (handle zeros)
    valid_mask = acts .!= 0
    if any(valid_mask)
        pct_errors = abs.(errors[valid_mask] ./ acts[valid_mask])
        pct_errors = filter(isfinite, pct_errors)
        if !isempty(pct_errors)
            metrics["mape"] = mean(pct_errors) * 100.0
        else
            metrics["mape"] = NaN
        end
    else
        metrics["mape"] = NaN
    end

    # Direction accuracy (for changes)
    if n > 1
        pred_direction = sign.(diff(preds))
        actual_direction = sign.(diff(acts))
        metrics["direction_accuracy"] = mean(pred_direction .== actual_direction)
    else
        metrics["direction_accuracy"] = NaN
    end

    return metrics
end
