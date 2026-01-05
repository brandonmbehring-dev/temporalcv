# =============================================================================
# Regimes Module - Regime Classification for Conditional Performance Analysis
# =============================================================================
#
# Classify market regimes for stratified model evaluation.
#
# CRITICAL: Volatility must be computed on CHANGES (first differences), NOT levels.
# Using levels mislabels steady drifts as "volatile" because:
# - A series drifting steadily from 3.0 to 4.0 has high std of LEVELS
# - But it has ZERO volatility of changes (constant increments)
#
# Knowledge Tiers:
# [T1] Regime-switching theory (Hamilton 1989, 1994)
# [T1] Rolling volatility as regime indicator (standard in finance)
# [T2] Volatility of CHANGES not levels (BUG-005 fix from myga-forecasting-v2)
# [T2] 3-class direction (UP/DOWN/FLAT) enables fair persistence comparison
# [T3] 13-week window assumes quarterly seasonality
# [T3] 33rd/67th percentiles for regime boundaries

module Regimes

using Statistics

# =============================================================================
# Types - Hybrid Enum + String Approach
# =============================================================================

"""
Volatility regime classification.

- `VOL_LOW`: Low volatility (below 33rd percentile)
- `VOL_MED`: Medium volatility (33rd to 67th percentile)
- `VOL_HIGH`: High volatility (above 67th percentile)
"""
@enum VolatilityRegime VOL_LOW VOL_MED VOL_HIGH

"""
Direction regime classification.

- `DIR_UP`: Significant upward move (above threshold)
- `DIR_DOWN`: Significant downward move (below -threshold)
- `DIR_FLAT`: No significant move (within threshold)
"""
@enum DirectionRegime DIR_UP DIR_DOWN DIR_FLAT

# String conversion for clean output
function regime_string(r::VolatilityRegime)
    r == VOL_LOW && return "LOW"
    r == VOL_MED && return "MED"
    r == VOL_HIGH && return "HIGH"
    return "UNKNOWN"
end

function regime_string(r::DirectionRegime)
    r == DIR_UP && return "UP"
    r == DIR_DOWN && return "DOWN"
    r == DIR_FLAT && return "FLAT"
    return "UNKNOWN"
end

# Allow Base.string to work
Base.string(r::VolatilityRegime) = regime_string(r)
Base.string(r::DirectionRegime) = regime_string(r)


"""
Result of stratified metrics computation.

# Fields
- `overall_mae::Float64`: Mean Absolute Error across all samples
- `overall_rmse::Float64`: Root Mean Squared Error across all samples
- `n_total::Int`: Total number of samples
- `by_regime::Dict{String, Dict{Symbol, Float64}}`: Per-regime metrics
- `masked_regimes::Vector{String}`: Regimes with n < min_n that were excluded
"""
struct StratifiedMetricsResult
    overall_mae::Float64
    overall_rmse::Float64
    n_total::Int
    by_regime::Dict{String, Dict{Symbol, Float64}}
    masked_regimes::Vector{String}
end


"""
Generate human-readable summary of stratified metrics.
"""
function summary(result::StratifiedMetricsResult)::String
    lines = [
        "Overall: MAE=$(round(result.overall_mae, digits=4)), " *
        "RMSE=$(round(result.overall_rmse, digits=4)), n=$(result.n_total)",
        "",
        "By Regime:"
    ]

    # Sort by n descending
    sorted_regimes = sort(collect(result.by_regime), by=x -> -x.second[:n])

    for (regime, metrics) in sorted_regimes
        push!(lines,
            "  $regime: MAE=$(round(metrics[:mae], digits=4)), " *
            "RMSE=$(round(metrics[:rmse], digits=4)), " *
            "n=$(Int(metrics[:n])) ($(round(metrics[:pct], digits=1))%)"
        )
    end

    if !isempty(result.masked_regimes)
        push!(lines, "")
        push!(lines, "Masked (n < min_n): $(join(result.masked_regimes, ", "))")
    end

    return join(lines, "\n")
end


# =============================================================================
# Core Functions
# =============================================================================

include("classification.jl")


# =============================================================================
# Exports
# =============================================================================

export VolatilityRegime, VOL_LOW, VOL_MED, VOL_HIGH
export DirectionRegime, DIR_UP, DIR_DOWN, DIR_FLAT
export regime_string

export StratifiedMetricsResult, summary

export classify_volatility_regime, classify_direction_regime
export get_combined_regimes, get_regime_counts, mask_low_n_regimes
export compute_stratified_metrics

end # module Regimes
