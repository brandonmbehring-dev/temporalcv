# =============================================================================
# Sensitivity Analysis
# =============================================================================

"""
Compute sensitivity of a metric to a single parameter.

# Arguments
- `base_value::Real`: Original parameter value
- `metric_fn::Function`: Function that takes parameter value and returns metric
- `perturbation::Float64=0.1`: Fractional perturbation (0.1 = ±10%)
- `n_points::Int=5`: Number of perturbation points

# Returns
`SensitivityResult` containing sensitivity analysis.

# Example
```julia
# Sensitivity of MAE to window size
result = compute_parameter_sensitivity(
    13,  # base window size
    w -> compute_rolling_mae(predictions, actuals; window=Int(w)),
    perturbation=0.2
)
```
"""
function compute_parameter_sensitivity(
    base_value::Real,
    metric_fn::Function;
    perturbation::Float64 = 0.1,
    n_points::Int = 5,
    parameter_name::Symbol = :parameter
)::SensitivityResult
    base_value = Float64(base_value)

    if n_points < 2
        error("n_points must be >= 2, got $n_points")
    end

    # Generate perturbation range
    min_val = base_value * (1 - perturbation)
    max_val = base_value * (1 + perturbation)
    perturbed_values = collect(range(min_val, max_val; length=n_points))

    # Compute base metric
    base_metric = Float64(metric_fn(base_value))

    # Compute perturbed metrics
    perturbed_metrics = Float64[]
    for val in perturbed_values
        try
            m = Float64(metric_fn(val))
            push!(perturbed_metrics, m)
        catch
            push!(perturbed_metrics, NaN)
        end
    end

    # Estimate sensitivity (finite difference)
    # Use linear regression for robustness
    valid_mask = .!isnan.(perturbed_metrics)
    valid_values = perturbed_values[valid_mask]
    valid_metrics = perturbed_metrics[valid_mask]

    sensitivity = if length(valid_values) >= 2
        # Simple linear regression slope
        x_mean = mean(valid_values)
        y_mean = mean(valid_metrics)
        numerator = sum((valid_values .- x_mean) .* (valid_metrics .- y_mean))
        denominator = sum((valid_values .- x_mean) .^ 2)
        denominator > 0 ? numerator / denominator : 0.0
    else
        0.0
    end

    return SensitivityResult(
        parameter_name,
        base_value,
        perturbed_values,
        base_metric,
        perturbed_metrics,
        sensitivity
    )
end


"""
Compute stability report for multiple parameters.

# Arguments
- `params::Dict{Symbol, Real}`: Parameter names and base values
- `metric_fn::Function`: Function that takes Dict{Symbol, Real} and returns metric
- `perturbation::Float64=0.1`: Fractional perturbation for each parameter

# Returns
`StabilityReport` with per-parameter sensitivity and overall stability score.

# Example
```julia
params = Dict(:window => 13, :gap => 4, :horizon => 1)

report = compute_stability_report(
    params,
    p -> compute_cv_mae(model, X, y; window=p[:window], gap=p[:gap])
)

println("Most sensitive to: \$(report.most_sensitive)")
println("Stability score: \$(report.stability_score)")
```
"""
function compute_stability_report(
    params::Dict{Symbol, <:Real},
    metric_fn::Function;
    perturbation::Float64 = 0.1,
    n_points::Int = 5
)::StabilityReport
    sensitivities = Dict{Symbol, SensitivityResult}()

    for (name, base_value) in params
        # Create a function that only varies this parameter
        param_fn = function(val)
            modified_params = copy(params)
            modified_params[name] = val
            metric_fn(modified_params)
        end

        result = compute_parameter_sensitivity(
            base_value,
            param_fn;
            perturbation=perturbation,
            n_points=n_points,
            parameter_name=name
        )
        sensitivities[name] = result
    end

    # Find most sensitive parameter
    max_sensitivity = -Inf
    most_sensitive = first(keys(params))

    for (name, result) in sensitivities
        abs_sens = abs(result.sensitivity)
        if abs_sens > max_sensitivity
            max_sensitivity = abs_sens
            most_sensitive = name
        end
    end

    # Compute total variation and stability score
    total_variation = sum(abs(r.sensitivity) for r in values(sensitivities))

    # Stability score: 1 / (1 + total_variation)
    # Higher when sensitivities are low
    stability_score = 1.0 / (1.0 + total_variation)

    return StabilityReport(
        sensitivities,
        most_sensitive,
        stability_score,
        total_variation
    )
end


"""
Estimate metric variance using bootstrap.

# Arguments
- `errors::AbstractVector{<:Real}`: Forecast errors
- `metric_fn::Function`: Function that computes metric from errors
- `n_bootstrap::Int=1000`: Number of bootstrap samples
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator
- `block_size::Int=1`: Block size for block bootstrap (>1 for time series)

# Returns
Tuple of (point_estimate, std_error, confidence_interval)

# Example
```julia
errors = predictions .- actuals
estimate, se, ci = bootstrap_metric_variance(
    errors,
    e -> mean(abs.(e));  # MAE
    n_bootstrap=1000
)
```
"""
function bootstrap_metric_variance(
    errors::AbstractVector{<:Real},
    metric_fn::Function;
    n_bootstrap::Int = 1000,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    block_size::Int = 1,
    confidence_level::Float64 = 0.95
)
    errors = Float64.(errors)
    n = length(errors)

    if n < 2
        error("Need at least 2 observations, got $n")
    end

    bootstrap_metrics = Float64[]

    for _ in 1:n_bootstrap
        # Block bootstrap if block_size > 1
        if block_size > 1
            n_blocks = ceil(Int, n / block_size)
            indices = Int[]
            for _ in 1:n_blocks
                start = rand(rng, 1:max(1, n - block_size + 1))
                append!(indices, start:min(start + block_size - 1, n))
            end
            indices = indices[1:n]  # Trim to original length
            sample = errors[indices]
        else
            # Simple bootstrap
            indices = rand(rng, 1:n, n)
            sample = errors[indices]
        end

        try
            m = Float64(metric_fn(sample))
            if isfinite(m)
                push!(bootstrap_metrics, m)
            end
        catch
            # Skip failed samples
        end
    end

    if length(bootstrap_metrics) < 10
        error("Too few valid bootstrap samples")
    end

    point_estimate = mean(bootstrap_metrics)
    std_error = std(bootstrap_metrics)

    # Percentile confidence interval
    alpha = 1 - confidence_level
    lower = quantile(bootstrap_metrics, alpha / 2)
    upper = quantile(bootstrap_metrics, 1 - alpha / 2)

    return (point_estimate, std_error, (lower, upper))
end
