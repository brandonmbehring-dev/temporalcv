# =============================================================================
# Compare Runner - Model Comparison Execution
# =============================================================================
#
# Provides functions to run model comparisons and benchmark suites.
# Integrates with StatisticalTests for DM test significance.

"""
    DatasetLike

Expected interface for datasets used in comparison.

Required methods:
- `get_values(dataset) -> Vector{Float64}`: Return dataset values
- `get_name(dataset) -> String`: Return dataset name
- `get_train_test_split(dataset) -> Tuple{Vector{Float64}, Vector{Float64}}`: Return train/test
- `get_horizon(dataset) -> Int`: Return forecast horizon
"""


"""
    run_comparison(train, test; adapters, primary_metric="mae", include_dm_test=true,
                   dataset_name="dataset", horizon=1) -> ComparisonResult

Compare multiple models on a single dataset.

# Arguments
- `train::AbstractVector{<:Real}`: Training data
- `test::AbstractVector{<:Real}`: Test data (actuals)
- `adapters::Vector{<:ForecastAdapter}`: Model adapters to compare
- `primary_metric::String="mae"`: Metric to use for ranking models
- `include_dm_test::Bool=true`: Whether to run Diebold-Mariano test
- `dataset_name::String="dataset"`: Name for the dataset
- `horizon::Int=1`: Forecast horizon (for HAC variance in DM test)

# Returns
`ComparisonResult` with model rankings.

# Example
```julia
train = collect(1.0:100.0)
test = collect(101.0:110.0)

adapters = [NaiveAdapter(), SeasonalNaiveAdapter(12)]
result = run_comparison(train, test; adapters=adapters)
println("Best model: \$(result.best_model)")
```
"""
function run_comparison(
    train::AbstractVector{<:Real},
    test::AbstractVector{<:Real};
    adapters::Vector{<:ForecastAdapter},
    primary_metric::String = "mae",
    include_dm_test::Bool = true,
    dataset_name::String = "dataset",
    horizon::Int = 1
)::ComparisonResult
    isempty(adapters) && error("adapters list cannot be empty")
    isempty(train) && error("train cannot be empty")
    isempty(test) && error("test cannot be empty")

    train_vec = Float64.(train)
    test_vec = Float64.(test)
    test_size = length(test_vec)

    # Run each adapter
    model_results = ModelResult[]

    for adapter in adapters
        start_time = time()

        local predictions::Vector{Float64}
        try
            predictions = fit_predict(adapter, train_vec, test_size, horizon)
        catch e
            @warn "$(model_name(adapter)) failed: $e"
            continue
        end

        elapsed = time() - start_time

        # Validate prediction length
        if length(predictions) != test_size
            @warn "$(model_name(adapter)) returned $(length(predictions)) predictions, expected $test_size"
            continue
        end

        # Compute metrics
        metrics = compute_comparison_metrics(predictions, test_vec)

        push!(model_results, ModelResult(
            model_name = model_name(adapter),
            package = package_name(adapter),
            metrics = metrics,
            predictions = predictions,
            runtime_seconds = elapsed,
            model_params = get_params(adapter)
        ))
    end

    isempty(model_results) && error("All adapters failed to produce results")

    # Build comparison result
    result = ComparisonResult(
        dataset_name = dataset_name,
        models = model_results,
        primary_metric = primary_metric
    )

    # Add DM test results if requested
    if include_dm_test && length(model_results) > 1
        dm_results = _run_dm_tests(model_results, test_vec, result.best_model, horizon)

        # Create new result with statistical tests
        # (ComparisonResult is immutable, so we reconstruct)
        result = ComparisonResult(
            dataset_name = dataset_name,
            models = model_results,
            primary_metric = primary_metric,
            statistical_tests = dm_results
        )
    end

    return result
end


"""Run DM tests comparing best model to others."""
function _run_dm_tests(
    model_results::Vector{ModelResult},
    test::Vector{Float64},
    best_model::String,
    horizon::Int
)::Dict{String, Any}
    dm_results = Dict{String, Any}()

    # Find best model predictions
    best_preds = nothing
    for result in model_results
        if result.model_name == best_model
            best_preds = result.predictions
            break
        end
    end

    isnothing(best_preds) && return Dict{String, Any}("error" => "Best model $best_model not found")

    # Try to get dm_test from parent module
    dm_test_fn = nothing
    try
        # Late binding: dm_test should be available via parent module
        dm_test_fn = Main.TemporalValidation.dm_test
    catch
        # Fallback: try to find it
        try
            dm_test_fn = getfield(parentmodule(@__MODULE__), :dm_test)
        catch
            return Dict{String, Any}("error" => "dm_test not available")
        end
    end

    for result in model_results
        result.model_name == best_model && continue

        other_preds = result.predictions

        # Compute errors (actual - predicted)
        errors_best = test .- best_preds
        errors_other = test .- other_preds

        try
            dm_result = dm_test_fn(errors_best, errors_other; h=horizon, loss=:absolute)

            dm_results[result.model_name] = Dict{String, Any}(
                "statistic" => dm_result.statistic,
                "p_value" => dm_result.pvalue,
                "significant" => dm_result.pvalue < 0.05
            )
        catch e
            if e isa ErrorException || e isa ArgumentError
                dm_results[result.model_name] = Dict{String, Any}("error" => string(e))
            else
                @warn "Unexpected error in DM test for $(result.model_name): $e"
                dm_results[result.model_name] = Dict{String, Any}("error" => "Unexpected: $(typeof(e)): $e")
            end
        end
    end

    return dm_results
end


"""
    run_benchmark_suite(datasets; adapters, primary_metric="mae", include_dm_test=true)
        -> ComparisonReport

Run model comparison across multiple datasets.

# Arguments
- `datasets::Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}`: List of (name, train, test) tuples
- `adapters::Vector{<:ForecastAdapter}`: Model adapters to compare
- `primary_metric::String="mae"`: Metric to use for ranking
- `include_dm_test::Bool=true`: Whether to run statistical tests

# Returns
`ComparisonReport` with full results and summary.

# Example
```julia
datasets = [
    ("series_1", collect(1.0:100.0), collect(101.0:110.0)),
    ("series_2", sin.(1:100), sin.(101:110))
]
report = run_benchmark_suite(datasets; adapters=[NaiveAdapter()])
println(to_markdown(report))
```
"""
function run_benchmark_suite(
    datasets::Vector{Tuple{String, Vector{Float64}, Vector{Float64}}};
    adapters::Vector{<:ForecastAdapter},
    primary_metric::String = "mae",
    include_dm_test::Bool = true
)::ComparisonReport
    isempty(datasets) && error("datasets list cannot be empty")
    isempty(adapters) && error("adapters list cannot be empty")

    results = ComparisonResult[]

    for (name, train, test) in datasets
        try
            result = run_comparison(
                train, test;
                adapters = adapters,
                primary_metric = primary_metric,
                include_dm_test = include_dm_test,
                dataset_name = name
            )
            push!(results, result)
        catch e
            @warn "Failed on dataset $name: $e"
            continue
        end
    end

    isempty(results) && error("All datasets failed to produce results")

    return ComparisonReport(results = results)
end


"""
    compare_to_baseline(train, test, adapter; baseline_adapter=nothing,
                        primary_metric="mae", dataset_name="dataset") -> Dict{String, Any}

Compare a single model to a baseline.

# Arguments
- `train::AbstractVector{<:Real}`: Training data
- `test::AbstractVector{<:Real}`: Test data
- `adapter::ForecastAdapter`: Model adapter to evaluate
- `baseline_adapter::Union{ForecastAdapter, Nothing}=nothing`: Baseline (default: NaiveAdapter)
- `primary_metric::String="mae"`: Metric to use for comparison
- `dataset_name::String="dataset"`: Name for the dataset

# Returns
Dictionary with:
- `model_name`: Name of the model
- `baseline_name`: Name of the baseline
- `model_<metric>`: Model's metric value
- `baseline_<metric>`: Baseline's metric value
- `improvement_pct`: Percentage improvement over baseline
- `model_is_better`: Whether model beats baseline
- `statistical_tests`: DM test results

# Example
```julia
result = compare_to_baseline(train, test, my_adapter)
println("Improvement: \$(result["improvement_pct"])%")
```
"""
function compare_to_baseline(
    train::AbstractVector{<:Real},
    test::AbstractVector{<:Real},
    adapter::ForecastAdapter;
    baseline_adapter::Union{ForecastAdapter, Nothing} = nothing,
    primary_metric::String = "mae",
    dataset_name::String = "dataset"
)::Dict{String, Any}
    # Default to NaiveAdapter
    baseline = isnothing(baseline_adapter) ? NaiveAdapter() : baseline_adapter

    comparison = run_comparison(
        train, test;
        adapters = [baseline, adapter],
        primary_metric = primary_metric,
        include_dm_test = true,
        dataset_name = dataset_name
    )

    # Extract metrics
    baseline_result = nothing
    model_result = nothing

    for result in comparison.models
        if result.model_name == model_name(baseline)
            baseline_result = result
        elseif result.model_name == model_name(adapter)
            model_result = result
        end
    end

    (isnothing(baseline_result) || isnothing(model_result)) &&
        error("Could not find both baseline and model results")

    baseline_metric = get_metric(baseline_result, primary_metric)
    model_metric = get_metric(model_result, primary_metric)

    # Compute improvement (positive = better, since lower error is better)
    improvement_pct = if baseline_metric != 0
        (baseline_metric - model_metric) / baseline_metric * 100
    else
        model_metric == 0 ? 0.0 : -Inf
    end

    return Dict{String, Any}(
        "model_name" => model_name(adapter),
        "baseline_name" => model_name(baseline),
        "model_$primary_metric" => model_metric,
        "baseline_$primary_metric" => baseline_metric,
        "improvement_pct" => improvement_pct,
        "model_is_better" => model_metric < baseline_metric,
        "statistical_tests" => comparison.statistical_tests
    )
end
