# =============================================================================
# Guardrails Module - Unified Validation Checks
# =============================================================================
#
# Provides a convenience wrapper over validation gates with sensible defaults.
# Each guardrail check returns a structured result with pass/fail, warnings,
# errors, and recommendations.

module Guardrails

using Statistics
using Random

# =============================================================================
# Types
# =============================================================================

"""
Result of a guardrail check.

# Fields
- `passed::Bool`: Whether the check passed
- `warnings::Vector{String}`: Warning messages
- `errors::Vector{String}`: Error messages (when check fails)
- `details::Dict{Symbol, Any}`: Diagnostic details
- `skipped::Vector{String}`: Names of checks that were skipped
- `recommendations::Vector{String}`: Suggested actions
"""
struct GuardrailResult
    passed::Bool
    warnings::Vector{String}
    errors::Vector{String}
    details::Dict{Symbol, Any}
    skipped::Vector{String}
    recommendations::Vector{String}
end


"""
Create a passing GuardrailResult with optional details.
"""
function pass_result(; details::Dict{Symbol, Any} = Dict{Symbol, Any}())
    GuardrailResult(true, String[], String[], details, String[], String[])
end


"""
Create a failing GuardrailResult with error message.
"""
function fail_result(
    error_msg::String;
    details::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    recommendations::Vector{String} = String[]
)
    GuardrailResult(false, String[], [error_msg], details, String[], recommendations)
end


"""
Create a warning GuardrailResult.
"""
function warn_result(
    warning_msg::String;
    details::Dict{Symbol, Any} = Dict{Symbol, Any}(),
    recommendations::Vector{String} = String[]
)
    GuardrailResult(true, [warning_msg], String[], details, String[], recommendations)
end


"""
Create a skipped GuardrailResult.
"""
function skip_result(reason::String)
    GuardrailResult(true, String[], String[], Dict{Symbol, Any}(), [reason], String[])
end


"""
Aggregate results from multiple guardrails.

# Fields
- `passed::Bool`: Overall pass (all checks passed)
- `n_passed::Int`: Number of passed checks
- `n_failed::Int`: Number of failed checks
- `n_warnings::Int`: Number of checks with warnings
- `n_skipped::Int`: Number of skipped checks
- `results::Dict{Symbol, GuardrailResult}`: Individual results
- `all_warnings::Vector{String}`: All warning messages
- `all_errors::Vector{String}`: All error messages
"""
struct GuardrailSummary
    passed::Bool
    n_passed::Int
    n_failed::Int
    n_warnings::Int
    n_skipped::Int
    results::Dict{Symbol, GuardrailResult}
    all_warnings::Vector{String}
    all_errors::Vector{String}
end


# =============================================================================
# Core Functions
# =============================================================================

include("checks.jl")


# =============================================================================
# Exports
# =============================================================================

export GuardrailResult, GuardrailSummary
export pass_result, fail_result, warn_result, skip_result

export check_suspicious_improvement
export check_minimum_sample_size
export check_stratified_sample_size
export check_forecast_horizon_consistency
export check_residual_autocorrelation
export run_all_guardrails

end # module Guardrails
