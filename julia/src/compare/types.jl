# =============================================================================
# Compare Types - Result Dataclasses for Model Comparison
# =============================================================================

"""
    ModelResult

Result from a single model run.

# Fields
- `model_name::String`: Name of the model (e.g., "AutoARIMA", "ETS")
- `package::String`: Package that provided the model (e.g., "StatsModels")
- `metrics::Dict{String, Float64}`: Dictionary of metric name -> value
- `predictions::Vector{Float64}`: Predicted values
- `runtime_seconds::Float64`: Time taken for fit + predict
- `model_params::Dict{String, Any}`: Model hyperparameters used

# Example
```julia
result = ModelResult(
    model_name = "ARIMA",
    package = "StatsModels",
    metrics = Dict("mae" => 0.5, "rmse" => 0.7),
    predictions = [1.0, 2.0, 3.0],
    runtime_seconds = 1.5,
    model_params = Dict{String, Any}()
)
```
"""
struct ModelResult
    model_name::String
    package::String
    metrics::Dict{String, Float64}
    predictions::Vector{Float64}
    runtime_seconds::Float64
    model_params::Dict{String, Any}

    function ModelResult(;
        model_name::String,
        package::String,
        metrics::Dict{String, Float64},
        predictions::Vector{Float64},
        runtime_seconds::Float64,
        model_params::Dict{String, Any} = Dict{String, Any}()
    )
        isempty(model_name) && error("model_name cannot be empty")
        isempty(package) && error("package cannot be empty")
        runtime_seconds < 0 && error("runtime_seconds cannot be negative")

        new(model_name, package, metrics, predictions, runtime_seconds, model_params)
    end
end


"""
    get_metric(result::ModelResult, name::String) -> Float64

Get metric by name (case-insensitive).

# Arguments
- `result::ModelResult`: Model result
- `name::String`: Metric name

# Returns
Metric value.

# Throws
`KeyError` if metric not found.
"""
function get_metric(result::ModelResult, name::String)::Float64
    # Try exact match first
    if haskey(result.metrics, name)
        return result.metrics[name]
    end

    # Try case-insensitive
    name_lower = lowercase(name)
    for (key, value) in result.metrics
        if lowercase(key) == name_lower
            return value
        end
    end

    available = collect(keys(result.metrics))
    error("Metric '$name' not found. Available: $available")
end


"""
    to_dict(result::ModelResult) -> Dict{String, Any}

Convert ModelResult to dictionary.
"""
function to_dict(result::ModelResult)::Dict{String, Any}
    return Dict{String, Any}(
        "model_name" => result.model_name,
        "package" => result.package,
        "metrics" => result.metrics,
        "runtime_seconds" => result.runtime_seconds,
        "model_params" => result.model_params
    )
end


"""
    ComparisonResult

Result from comparing multiple models on a single dataset.

# Fields
- `dataset_name::String`: Name of the dataset used
- `models::Vector{ModelResult}`: Results from each model
- `primary_metric::String`: Metric used for ranking (e.g., "mae")
- `best_model::String`: Name of the best model by primary metric
- `statistical_tests::Dict{String, Any}`: Results of statistical tests (DM test, etc.)
"""
struct ComparisonResult
    dataset_name::String
    models::Vector{ModelResult}
    primary_metric::String
    best_model::String
    statistical_tests::Dict{String, Any}

    function ComparisonResult(;
        dataset_name::String,
        models::Vector{ModelResult},
        primary_metric::String,
        statistical_tests::Dict{String, Any} = Dict{String, Any}()
    )
        isempty(models) && error("models list cannot be empty")
        isempty(primary_metric) && error("primary_metric cannot be empty")

        # Compute best model (lowest metric value)
        best_value = Inf
        best_name = ""

        for model in models
            try
                value = get_metric(model, primary_metric)
                if value < best_value
                    best_value = value
                    best_name = model.model_name
                end
            catch
                continue  # Skip models without this metric
            end
        end

        if isempty(best_name)
            all_metrics = _get_all_metrics(models)
            error("No model has metric '$primary_metric'. Available metrics: $all_metrics")
        end

        new(dataset_name, models, primary_metric, best_name, statistical_tests)
    end
end


"""Get all unique metric names across models."""
function _get_all_metrics(models::Vector{ModelResult})::Vector{String}
    metrics = Set{String}()
    for model in models
        union!(metrics, keys(model.metrics))
    end
    return sort(collect(metrics))
end


"""
    get_ranking(result::ComparisonResult; metric=nothing) -> Vector{Tuple{String, Float64}}

Get models ranked by metric (ascending, best first).

# Arguments
- `result::ComparisonResult`: Comparison result
- `metric::Union{String, Nothing}=nothing`: Metric to rank by (default: primary_metric)

# Returns
List of (model_name, metric_value) tuples sorted ascending.
"""
function get_ranking(
    result::ComparisonResult;
    metric::Union{String, Nothing} = nothing
)::Vector{Tuple{String, Float64}}
    metric_name = isnothing(metric) ? result.primary_metric : metric

    rankings = Tuple{String, Float64}[]

    for model in result.models
        try
            value = get_metric(model, metric_name)
            push!(rankings, (model.model_name, value))
        catch
            continue  # Skip models without this metric
        end
    end

    # Sort ascending (best = lowest)
    sort!(rankings, by = x -> x[2])
    return rankings
end


"""
    to_dict(result::ComparisonResult) -> Dict{String, Any}

Convert ComparisonResult to dictionary.
"""
function to_dict(result::ComparisonResult)::Dict{String, Any}
    return Dict{String, Any}(
        "dataset_name" => result.dataset_name,
        "models" => [to_dict(m) for m in result.models],
        "primary_metric" => result.primary_metric,
        "best_model" => result.best_model,
        "statistical_tests" => result.statistical_tests
    )
end


"""
    ComparisonReport

Report from comparing models across multiple datasets.

# Fields
- `results::Vector{ComparisonResult}`: Results from each dataset
- `summary::Dict{String, Any}`: Aggregate summary statistics
"""
struct ComparisonReport
    results::Vector{ComparisonResult}
    summary::Dict{String, Any}

    function ComparisonReport(;
        results::Vector{ComparisonResult},
        summary::Dict{String, Any} = Dict{String, Any}()
    )
        # Compute summary if not provided
        actual_summary = if isempty(summary) && !isempty(results)
            _compute_summary(results)
        else
            summary
        end

        new(results, actual_summary)
    end
end


"""Compute aggregate summary from results."""
function _compute_summary(results::Vector{ComparisonResult})::Dict{String, Any}
    # Count wins per model
    wins = Dict{String, Int}()
    for result in results
        wins[result.best_model] = get(wins, result.best_model, 0) + 1
    end

    # Average metrics per model
    avg_metrics = Dict{String, Dict{String, Vector{Float64}}}()

    for result in results
        for model in result.models
            if !haskey(avg_metrics, model.model_name)
                avg_metrics[model.model_name] = Dict{String, Vector{Float64}}()
            end

            for (metric_name, value) in model.metrics
                if !haskey(avg_metrics[model.model_name], metric_name)
                    avg_metrics[model.model_name][metric_name] = Float64[]
                end
                push!(avg_metrics[model.model_name][metric_name], value)
            end
        end
    end

    # Convert to means
    mean_metrics = Dict{String, Dict{String, Float64}}()
    for (model_name, metrics) in avg_metrics
        mean_metrics[model_name] = Dict{String, Float64}()
        for (metric_name, values) in metrics
            mean_metrics[model_name][metric_name] = mean(values)
        end
    end

    return Dict{String, Any}(
        "n_datasets" => length(results),
        "wins_by_model" => wins,
        "mean_metrics_by_model" => mean_metrics
    )
end


"""
    to_markdown(report::ComparisonReport) -> String

Generate markdown report.
"""
function to_markdown(report::ComparisonReport)::String
    lines = String[]
    push!(lines, "# Model Comparison Report\n")

    # Summary
    push!(lines, "## Summary\n")
    push!(lines, "- Datasets evaluated: $(get(report.summary, "n_datasets", 0))")

    if haskey(report.summary, "wins_by_model")
        push!(lines, "\n### Model Wins\n")
        push!(lines, "| Model | Wins |")
        push!(lines, "|-------|------|")

        wins = report.summary["wins_by_model"]
        sorted_wins = sort(collect(wins), by = x -> -x[2])
        for (model, count) in sorted_wins
            push!(lines, "| $model | $count |")
        end
    end

    # Per-dataset results
    push!(lines, "\n## Per-Dataset Results\n")

    for result in report.results
        push!(lines, "### $(result.dataset_name)\n")
        push!(lines, "Best model: **$(result.best_model)**\n")

        # Ranking table
        ranking = get_ranking(result)
        if !isempty(ranking)
            push!(lines, "| Model | $(uppercase(result.primary_metric)) |")
            push!(lines, "|-------|------|")
            for (model_name, value) in ranking
                push!(lines, "| $model_name | $(round(value, digits=4)) |")
            end
            push!(lines, "")
        end
    end

    return join(lines, "\n")
end
