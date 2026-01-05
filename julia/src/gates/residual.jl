# =============================================================================
# Residual Diagnostics Gate
# =============================================================================

"""
    compute_acf(x::Vector, max_lag::Int) -> Vector{Float64}

Compute sample autocorrelation function (ACF) for lags 1 to max_lag.

# Knowledge Tier
[T1] Standard ACF formula (Box-Jenkins)
"""
function compute_acf(x::AbstractVector{<:Real}, max_lag::Int)
    n = length(x)
    x_centered = x .- mean(x)
    gamma_0 = sum(x_centered.^2) / n  # Variance (lag 0 autocovariance)

    if gamma_0 == 0
        return zeros(max_lag)
    end

    acf_values = zeros(max_lag)
    for k in 1:max_lag
        gamma_k = sum(x_centered[k+1:end] .* x_centered[1:end-k]) / n
        acf_values[k] = gamma_k / gamma_0
    end

    return acf_values
end


"""
    ljung_box_test(residuals::Vector, max_lag::Int) -> Tuple{Float64, Float64}

Ljung-Box test for autocorrelation in residuals.

Returns (Q statistic, p-value).

Under H₀ (no autocorrelation), Q ~ χ²(max_lag).

# Knowledge Tier
[T1] Ljung & Box (1978). Biometrika 65(2), 297-303.
"""
function ljung_box_test(residuals::AbstractVector{<:Real}, max_lag::Int)
    n = length(residuals)
    if n <= max_lag
        return (0.0, 1.0)
    end

    acf = compute_acf(residuals, max_lag)

    # Ljung-Box Q statistic: Q = n(n+2) Σ ρ_k² / (n-k)
    Q = 0.0
    for k in 1:max_lag
        Q += (acf[k]^2) / (n - k)
    end
    Q *= n * (n + 2)

    # P-value from chi-squared distribution
    p_value = 1.0 - cdf(Chisq(max_lag), Q)

    return (Q, p_value)
end


"""
    gate_residual_diagnostics(residuals; max_lag=10, significance=0.05, halt_on_autocorr=false, halt_on_bias=true)

Check residual quality via diagnostic tests.

This gate runs two diagnostic tests on model residuals:

1. **Ljung-Box test** [T1]: Detects residual autocorrelation
   - H₀: Residuals are white noise
   - Significant autocorrelation suggests model misspecification

2. **Mean-zero t-test** [T1]: Detects systematic bias
   - H₀: Mean(residuals) = 0
   - Non-zero mean indicates biased predictions

# Arguments
- `residuals`: Model residuals (actuals - predictions)
- `max_lag`: Maximum lag for Ljung-Box test (default 10)
- `significance`: Significance level for tests (default 0.05)
- `halt_on_autocorr`: If true, HALT on significant autocorrelation (default false)
- `halt_on_bias`: If true, HALT on significant bias (default true)

# Returns
`GateResult`:
- PASS: All tests pass
- WARN: Some tests fail but not configured to HALT
- HALT: Tests fail and configured to HALT
- SKIP: Insufficient data (n < 30)

# Knowledge Tier
[T1] Ljung-Box: Ljung & Box (1978)
[T1] t-test: Standard hypothesis testing

# Example
```julia
residuals = actuals .- predictions
result = gate_residual_diagnostics(residuals)
if result.status == WARN
    println("Check: ", result.details["failing_tests"])
end
```
"""
function gate_residual_diagnostics(
    residuals::AbstractVector{<:Real};
    max_lag::Int = 10,
    significance::Float64 = 0.05,
    halt_on_autocorr::Bool = false,
    halt_on_bias::Bool = true
)
    @assert !any(isnan, residuals) "residuals contains NaN values"

    n = length(residuals)
    MIN_SAMPLES = 30

    if n < MIN_SAMPLES
        return GateResult(
            name = "residual_diagnostics",
            status = SKIP,
            message = "Insufficient data: n=$n < $MIN_SAMPLES required for residual tests",
            details = Dict{String, Any}("n_samples" => n, "min_required" => MIN_SAMPLES),
            recommendation = "Collect more data before running residual diagnostics"
        )
    end

    # Ensure max_lag is reasonable
    max_lag = min(max_lag, n ÷ 3)
    if max_lag < 1
        max_lag = 1
    end

    test_results = Dict{String, Dict{String, Any}}()
    failing_tests = String[]

    # 1. Ljung-Box test for autocorrelation
    lb_stat, lb_pval = ljung_box_test(residuals, max_lag)
    test_results["ljung_box"] = Dict{String, Any}(
        "statistic" => lb_stat,
        "p_value" => lb_pval,
        "max_lag" => max_lag,
        "significant" => lb_pval < significance
    )
    if lb_pval < significance
        push!(failing_tests, "ljung_box")
    end

    # 2. Mean-zero t-test
    resid_mean = mean(residuals)
    resid_std = std(residuals, corrected=true)
    t_stat = resid_mean / (resid_std / sqrt(n))
    t_pval = 2 * (1 - cdf(TDist(n-1), abs(t_stat)))

    test_results["mean_zero"] = Dict{String, Any}(
        "statistic" => t_stat,
        "p_value" => t_pval,
        "mean" => resid_mean,
        "std" => resid_std,
        "significant" => t_pval < significance
    )
    if t_pval < significance
        push!(failing_tests, "mean_zero")
    end

    details = Dict{String, Any}(
        "n_samples" => n,
        "significance" => significance,
        "tests" => test_results,
        "failing_tests" => failing_tests
    )

    if isempty(failing_tests)
        return GateResult(
            name = "residual_diagnostics",
            status = PASS,
            message = "All residual diagnostics passed",
            metric_value = 0.0,
            threshold = significance,
            details = details
        )
    end

    # Check which failures warrant HALT vs WARN
    halt_reasons = String[]
    if "ljung_box" in failing_tests && halt_on_autocorr
        push!(halt_reasons, "autocorrelation")
    end
    if "mean_zero" in failing_tests && halt_on_bias
        push!(halt_reasons, "bias")
    end

    if !isempty(halt_reasons)
        return GateResult(
            name = "residual_diagnostics",
            status = HALT,
            message = "Residual diagnostics failed: $(join(halt_reasons, ", "))",
            metric_value = Float64(length(failing_tests)),
            threshold = significance,
            details = details,
            recommendation = "Investigate model specification. " *
                ("autocorrelation" in halt_reasons ? "Autocorrelation suggests missing temporal structure. " : "") *
                ("bias" in halt_reasons ? "Bias suggests systematic prediction error." : "")
        )
    end

    # WARN for failures without halt flags
    return GateResult(
        name = "residual_diagnostics",
        status = WARN,
        message = "Residual diagnostics: $(length(failing_tests)) test(s) failed",
        metric_value = Float64(length(failing_tests)),
        threshold = significance,
        details = details,
        recommendation = "Review failing tests: $(join(failing_tests, ", ")). " *
            "These may not be critical but warrant investigation."
    )
end
