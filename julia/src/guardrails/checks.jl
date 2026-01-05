# =============================================================================
# Guardrail Checks Implementation
# =============================================================================

"""
    check_suspicious_improvement(model_metric, baseline_metric; threshold=0.20) -> GuardrailResult

Check if model improvement over baseline is suspiciously large.

# Arguments
- `model_metric::Real`: Model performance (lower is better, e.g., MAE)
- `baseline_metric::Real`: Baseline performance
- `threshold::Float64=0.20`: Maximum allowed improvement fraction [T3]

# Returns
`GuardrailResult` - FAIL if improvement > threshold, WARN if > threshold/2.

# Example
```julia
result = check_suspicious_improvement(0.75, 1.0)  # 25% improvement
# result.passed == false (exceeds 20% threshold)
```

# Thresholds [T3]
- >20% improvement: HALT (likely leakage)
- 10-20% improvement: WARN (verify carefully)
- <10% improvement: PASS
"""
function check_suspicious_improvement(
    model_metric::Real,
    baseline_metric::Real;
    threshold::Float64 = 0.20,
    lower_is_better::Bool = true
)::GuardrailResult
    baseline_metric != 0 || return skip_result("Baseline metric is zero")

    if lower_is_better
        improvement = (baseline_metric - model_metric) / abs(baseline_metric)
    else
        improvement = (model_metric - baseline_metric) / abs(baseline_metric)
    end

    details = Dict{Symbol, Any}(
        :model_metric => Float64(model_metric),
        :baseline_metric => Float64(baseline_metric),
        :improvement_pct => improvement * 100,
        :threshold_pct => threshold * 100
    )

    if improvement > threshold
        return fail_result(
            "Improvement of $(round(improvement * 100, digits=1))% exceeds $(threshold * 100)% threshold",
            details=details,
            recommendations=[
                "Check for data leakage",
                "Verify temporal boundaries",
                "Run shuffled target test"
            ]
        )
    elseif improvement > threshold / 2
        return warn_result(
            "Improvement of $(round(improvement * 100, digits=1))% is elevated",
            details=details,
            recommendations=["Verify with external holdout"]
        )
    else
        return pass_result(details=details)
    end
end


"""
    check_minimum_sample_size(n; min_samples=50) -> GuardrailResult

Check if sample size meets minimum requirements.

# Arguments
- `n::Int`: Sample size
- `min_samples::Int=50`: Minimum required samples [T3]

# Returns
`GuardrailResult` - FAIL if n < min_samples.

# Thresholds [T3]
- n >= 50: PASS (adequate for most statistical tests)
- n < 50: FAIL (insufficient for reliable inference)
"""
function check_minimum_sample_size(
    n::Int;
    min_samples::Int = 50
)::GuardrailResult
    details = Dict{Symbol, Any}(
        :n => n,
        :min_samples => min_samples
    )

    if n < min_samples
        return fail_result(
            "Sample size $n is below minimum $min_samples",
            details=details,
            recommendations=[
                "Collect more data",
                "Consider pooling with similar datasets",
                "Use bootstrap methods for uncertainty"
            ]
        )
    else
        return pass_result(details=details)
    end
end


"""
    check_stratified_sample_size(strata_sizes; min_per_stratum=10) -> GuardrailResult

Check if each stratum has sufficient samples.

# Arguments
- `strata_sizes::Vector{Int}`: Sample sizes per stratum
- `min_per_stratum::Int=10`: Minimum samples per stratum [T3]

# Returns
`GuardrailResult` - FAIL if any stratum has < min_per_stratum samples.

# Example
```julia
sizes = [100, 50, 8, 30]
result = check_stratified_sample_size(sizes; min_per_stratum=10)
# result.passed == false (stratum 3 has only 8 samples)
```
"""
function check_stratified_sample_size(
    strata_sizes::Vector{Int};
    min_per_stratum::Int = 10
)::GuardrailResult
    isempty(strata_sizes) && return skip_result("No strata provided")

    n_strata = length(strata_sizes)
    min_size = minimum(strata_sizes)
    failing_strata = findall(s -> s < min_per_stratum, strata_sizes)

    details = Dict{Symbol, Any}(
        :n_strata => n_strata,
        :strata_sizes => strata_sizes,
        :min_size => min_size,
        :min_per_stratum => min_per_stratum,
        :failing_strata => failing_strata
    )

    if !isempty(failing_strata)
        return fail_result(
            "$(length(failing_strata)) strata have fewer than $min_per_stratum samples",
            details=details,
            recommendations=[
                "Combine small strata with similar ones",
                "Use pooled estimates for small strata",
                "Consider hierarchical modeling"
            ]
        )
    else
        return pass_result(details=details)
    end
end


"""
    check_forecast_horizon_consistency(horizon_metrics; max_ratio=2.0) -> GuardrailResult

Check if performance degrades reasonably with forecast horizon.

When h=1 is much better than h=2,3,4, it often indicates gap enforcement failure.

# Arguments
- `horizon_metrics::Vector{<:Real}`: Metrics for horizons [1, 2, 3, ...]
- `max_ratio::Float64=2.0`: Maximum allowed h1/h_mean ratio [T2]

# Returns
`GuardrailResult` - WARN if h=1 >> other horizons.

# Example
```julia
# If h=1 MAE is 0.5 but h=2,3,4 are all around 1.5, something's wrong
metrics = [0.5, 1.4, 1.5, 1.6]
result = check_forecast_horizon_consistency(metrics)
# result.passed with warning (ratio too high)
```
"""
function check_forecast_horizon_consistency(
    horizon_metrics::Vector{<:Real};
    max_ratio::Float64 = 2.0,
    lower_is_better::Bool = true
)::GuardrailResult
    length(horizon_metrics) >= 2 || return skip_result("Need at least 2 horizons")
    all(isfinite, horizon_metrics) || return skip_result("Non-finite metrics")

    h1_metric = horizon_metrics[1]
    other_metrics = horizon_metrics[2:end]
    mean_other = mean(other_metrics)

    if lower_is_better
        # For MAE/RMSE, h=1 should be lowest but not dramatically so
        ratio = mean_other / max(h1_metric, 1e-10)
    else
        # For R², h=1 should be highest
        ratio = h1_metric / max(mean_other, 1e-10)
    end

    details = Dict{Symbol, Any}(
        :h1_metric => Float64(h1_metric),
        :mean_other_horizons => Float64(mean_other),
        :ratio => Float64(ratio),
        :max_ratio => max_ratio,
        :n_horizons => length(horizon_metrics)
    )

    if ratio > max_ratio
        return warn_result(
            "h=1 performance is $(round(ratio, digits=2))x better than mean of other horizons",
            details=details,
            recommendations=[
                "Verify gap >= horizon in CV setup",
                "Check for same-period feature leakage",
                "Review feature engineering pipeline"
            ]
        )
    else
        return pass_result(details=details)
    end
end


"""
    check_residual_autocorrelation(residuals; max_acf=0.2, max_lag=5) -> GuardrailResult

Check if model residuals have significant autocorrelation.

High autocorrelation in residuals suggests model is missing temporal patterns.

# Arguments
- `residuals::Vector{<:Real}`: Model residuals (actual - predicted)
- `max_acf::Float64=0.2`: Maximum allowed autocorrelation [T1]
- `max_lag::Int=5`: Number of lags to check

# Returns
`GuardrailResult` - WARN if any lag has |ACF| > max_acf.

# Example
```julia
residuals = actuals .- predictions
result = check_residual_autocorrelation(residuals)
```
"""
function check_residual_autocorrelation(
    residuals::Vector{<:Real};
    max_acf::Float64 = 0.2,
    max_lag::Int = 5
)::GuardrailResult
    n = length(residuals)
    n > max_lag + 1 || return skip_result("Insufficient samples for ACF computation")

    # Compute autocorrelation
    residuals_float = Float64.(residuals)
    mean_r = mean(residuals_float)
    centered = residuals_float .- mean_r
    var_r = sum(centered .^ 2) / n

    acf_values = Float64[]
    significant_lags = Int[]

    for lag in 1:max_lag
        cov_lag = sum(centered[1:n-lag] .* centered[lag+1:n]) / n
        acf_lag = cov_lag / max(var_r, 1e-10)
        push!(acf_values, acf_lag)

        if abs(acf_lag) > max_acf
            push!(significant_lags, lag)
        end
    end

    details = Dict{Symbol, Any}(
        :acf_values => acf_values,
        :max_acf_threshold => max_acf,
        :significant_lags => significant_lags,
        :max_observed_acf => maximum(abs.(acf_values))
    )

    if !isempty(significant_lags)
        return warn_result(
            "Significant autocorrelation at lags: $significant_lags",
            details=details,
            recommendations=[
                "Add lagged features",
                "Consider AR/ARMA residual modeling",
                "Check for regime changes"
            ]
        )
    else
        return pass_result(details=details)
    end
end


"""
    run_all_guardrails(; kwargs...) -> GuardrailSummary

Run all applicable guardrails and aggregate results.

# Keyword Arguments
- `model_metric::Union{Real, Nothing}=nothing`: Model performance
- `baseline_metric::Union{Real, Nothing}=nothing`: Baseline performance
- `n_samples::Union{Int, Nothing}=nothing`: Sample size
- `strata_sizes::Union{Vector{Int}, Nothing}=nothing`: Per-stratum sizes
- `horizon_metrics::Union{Vector{<:Real}, Nothing}=nothing`: Metrics by horizon
- `residuals::Union{Vector{<:Real}, Nothing}=nothing`: Model residuals
- `improvement_threshold::Float64=0.20`: For suspicious improvement check
- `min_samples::Int=50`: For minimum sample size check
- `min_per_stratum::Int=10`: For stratified sample size check
- `horizon_ratio::Float64=2.0`: For horizon consistency check
- `max_acf::Float64=0.2`: For residual autocorrelation check

# Returns
`GuardrailSummary` with aggregate results.

# Example
```julia
summary = run_all_guardrails(
    model_metric=0.8,
    baseline_metric=1.0,
    n_samples=200,
    residuals=model_errors
)

if !summary.passed
    println("Guardrails failed:")
    for e in summary.all_errors
        println("  - ", e)
    end
end
```
"""
function run_all_guardrails(;
    model_metric::Union{Real, Nothing} = nothing,
    baseline_metric::Union{Real, Nothing} = nothing,
    n_samples::Union{Int, Nothing} = nothing,
    strata_sizes::Union{Vector{Int}, Nothing} = nothing,
    horizon_metrics::Union{Vector{<:Real}, Nothing} = nothing,
    residuals::Union{Vector{<:Real}, Nothing} = nothing,
    improvement_threshold::Float64 = 0.20,
    min_samples::Int = 50,
    min_per_stratum::Int = 10,
    horizon_ratio::Float64 = 2.0,
    max_acf::Float64 = 0.2
)::GuardrailSummary
    results = Dict{Symbol, GuardrailResult}()

    # Suspicious improvement check
    if !isnothing(model_metric) && !isnothing(baseline_metric)
        results[:suspicious_improvement] = check_suspicious_improvement(
            model_metric, baseline_metric; threshold=improvement_threshold
        )
    end

    # Minimum sample size check
    if !isnothing(n_samples)
        results[:minimum_sample_size] = check_minimum_sample_size(
            n_samples; min_samples=min_samples
        )
    end

    # Stratified sample size check
    if !isnothing(strata_sizes)
        results[:stratified_sample_size] = check_stratified_sample_size(
            strata_sizes; min_per_stratum=min_per_stratum
        )
    end

    # Horizon consistency check
    if !isnothing(horizon_metrics)
        results[:horizon_consistency] = check_forecast_horizon_consistency(
            horizon_metrics; max_ratio=horizon_ratio
        )
    end

    # Residual autocorrelation check
    if !isnothing(residuals)
        results[:residual_autocorrelation] = check_residual_autocorrelation(
            residuals; max_acf=max_acf
        )
    end

    # Aggregate
    n_passed = count(r -> r.passed && isempty(r.skipped), values(results))
    n_failed = count(r -> !r.passed, values(results))
    n_warnings = count(r -> !isempty(r.warnings), values(results))
    n_skipped = count(r -> !isempty(r.skipped), values(results))

    all_warnings = String[]
    all_errors = String[]
    for r in values(results)
        append!(all_warnings, r.warnings)
        append!(all_errors, r.errors)
    end

    passed = n_failed == 0

    return GuardrailSummary(
        passed,
        n_passed,
        n_failed,
        n_warnings,
        n_skipped,
        results,
        all_warnings,
        all_errors
    )
end
