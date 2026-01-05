# =============================================================================
# Suspicious Improvement Gate
# =============================================================================

"""
    gate_suspicious_improvement(model_metric, baseline_metric; threshold=0.20, warn_threshold=0.10, metric_name="MAE")

Check for suspiciously large improvement over baseline.

Large improvements (e.g., >20% better than persistence) in time-series
forecasting are often indicators of data leakage rather than genuine skill.

# Arguments
- `model_metric`: Model's error metric (lower is better)
- `baseline_metric`: Baseline error metric (e.g., persistence MAE)
- `threshold`: Improvement ratio that triggers HALT (default 0.20 = 20%)
- `warn_threshold`: Improvement ratio that triggers WARN (default 0.10)
- `metric_name`: Name of metric for messages (default "MAE")

# Returns
`GateResult` with status:
- HALT if improvement exceeds threshold
- WARN if improvement exceeds warn_threshold
- PASS if improvement is reasonable
- SKIP if baseline_metric <= 0

# Notes
Experience shows that genuine forecasting improvements are modest.
If your model shows 40%+ improvement over persistence, verify with
shuffled target test before trusting the results.

# Knowledge Tier
[T3] 20% improvement threshold = "too good to be true" heuristic (empirical)

# Example
```julia
result = gate_suspicious_improvement(0.05, 0.10)  # 50% improvement
if result.status == HALT
    println("Investigation required!")
end
```
"""
function gate_suspicious_improvement(
    model_metric::Real,
    baseline_metric::Real;
    threshold::Float64 = 0.20,
    warn_threshold::Float64 = 0.10,
    metric_name::String = "MAE"
)
    if baseline_metric <= 0
        return GateResult(
            name = "suspicious_improvement",
            status = SKIP,
            message = "Baseline metric is zero or negative",
            details = Dict{String, Any}(
                "model_metric" => model_metric,
                "baseline_metric" => baseline_metric
            )
        )
    end

    # Improvement ratio: higher = model is better
    improvement = 1 - (model_metric / baseline_metric)

    details = Dict{String, Any}(
        "model_$(lowercase(metric_name))" => model_metric,
        "baseline_$(lowercase(metric_name))" => baseline_metric,
        "improvement_ratio" => improvement
    )

    if improvement > threshold
        return GateResult(
            name = "suspicious_improvement",
            status = HALT,
            message = "Model $(round(improvement * 100, digits=1))% better than baseline (max: $(round(threshold * 100, digits=0))%)",
            metric_value = improvement,
            threshold = threshold,
            details = details,
            recommendation = "Run shuffled target test. This improvement is suspicious."
        )
    end

    if improvement > warn_threshold
        return GateResult(
            name = "suspicious_improvement",
            status = WARN,
            message = "Model $(round(improvement * 100, digits=1))% better than baseline - verify carefully",
            metric_value = improvement,
            threshold = warn_threshold,
            details = details,
            recommendation = "Verify with external validation before trusting."
        )
    end

    return GateResult(
        name = "suspicious_improvement",
        status = PASS,
        message = "Improvement $(round(improvement * 100, digits=1))% is reasonable",
        metric_value = improvement,
        threshold = threshold,
        details = details
    )
end
