# =============================================================================
# Changepoint Detection for Time Series
# =============================================================================

"""
Changepoint detection module for identifying structural breaks in time series.

Useful for:
1. Identifying regime boundaries (e.g., LOW → HIGH volatility)
2. Training models on post-changepoint data only
3. Creating regime indicators as features
4. Understanding when series behavior changed

# Knowledge Tiers
- [T1] PELT algorithm (Killick, Fearnhead & Eckley, 2012)
- [T2] Variance-based detection is a heuristic

# Example
```julia
using TemporalValidation

# Series with level shift
series = vcat(fill(3.0, 30), fill(5.0, 30))
result = detect_changepoints(series)
result.changepoints  # Vector of Changepoint

# Classify regimes
regimes = classify_regimes(series, result)
```

# References
- Killick, Fearnhead & Eckley (2012). "Optimal Detection of Changepoints
  with a Linear Computational Cost." JASA 107(500), 1590-1598.
- Truong, Oudre & Vayer (2020). "Selective review of offline change point
  detection methods." Signal Processing 167, 107299.
"""

using Statistics

# =============================================================================
# Types
# =============================================================================

"""
    Changepoint

Detected changepoint in time series.

# Fields
- `index::Int`: Index position in series
- `cost_reduction::Float64`: Cost reduction from adding this changepoint
- `regime_before::Union{String, Nothing}`: Regime classification before
- `regime_after::Union{String, Nothing}`: Regime classification after
"""
struct Changepoint
    index::Int
    cost_reduction::Float64
    regime_before::Union{String, Nothing}
    regime_after::Union{String, Nothing}
end

# Convenience constructor
Changepoint(index::Int, cost_reduction::Float64) = Changepoint(index, cost_reduction, nothing, nothing)

function Base.show(io::IO, cp::Changepoint)
    print(io, "Changepoint(index=$(cp.index), cost=$(round(cp.cost_reduction, digits=3)))")
end

"""
    ChangepointResult

Result of changepoint detection.

# Fields
- `changepoints::Vector{Changepoint}`: Detected changepoints
- `n_segments::Int`: Number of segments (changepoints + 1)
- `method::Symbol`: Detection method used
- `penalty::Float64`: Penalty parameter used
"""
struct ChangepointResult
    changepoints::Vector{Changepoint}
    n_segments::Int
    method::Symbol
    penalty::Float64
end

function Base.show(io::IO, r::ChangepointResult)
    n_cp = length(r.changepoints)
    print(io, "ChangepointResult(n_changepoints=$n_cp, n_segments=$(r.n_segments), method=$(r.method))")
end

# =============================================================================
# Detection Functions
# =============================================================================

"""
    detect_changepoints_variance(series; penalty=3.0, min_segment_length=4, window=8)

Detect changepoints using rolling variance threshold.

Detects points where the level difference between adjacent windows
exceeds a threshold relative to local volatility.

# Arguments
- `series::AbstractVector`: Time series data
- `penalty::Float64=3.0`: Threshold multiplier for detecting changes.
  Higher = fewer changepoints. Default 3.0 means detect changes > 3x baseline volatility.
- `min_segment_length::Int=4`: Minimum observations between changepoints
- `window::Int=8`: Rolling window size for computing statistics

# Returns
- `ChangepointResult`: Detected changepoints with metadata

# Example
```julia
# Series with level shift
series = vcat(fill(3.0, 30), fill(5.0, 30))
result = detect_changepoints_variance(series, penalty=2.0)
length(result.changepoints)  # 1
result.changepoints[1].index  # ~30
```

# Notes
[T2] Variance-based detection is a heuristic.
"""
function detect_changepoints_variance(
    series::AbstractVector{<:Real};
    penalty::Float64 = 3.0,
    min_segment_length::Int = 4,
    window::Int = 8
)::ChangepointResult
    arr = collect(Float64, series)
    n = length(arr)

    min_required = 2 * window + min_segment_length
    @assert n >= min_required "Series too short for changepoint detection: n=$n, need >= $min_required"

    # Use robust baseline: median absolute deviation of first differences
    diffs = diff(arr)
    median_diff = median(diffs)
    mad = median(abs.(diffs .- median_diff))
    # Convert MAD to approximate std: std ≈ MAD * 1.4826
    baseline_std = mad * 1.4826

    if baseline_std < 1e-10
        baseline_std = std(arr)  # Fallback for constant series
    end

    if baseline_std < 1e-10
        # Truly constant series - no changepoints
        return ChangepointResult(
            Changepoint[],
            1,
            :variance,
            penalty
        )
    end

    threshold = penalty * baseline_std

    # Compute level difference between adjacent non-overlapping windows
    level_changes = zeros(n)
    for i in (window+1):(n - window + 1)
        left_mean = mean(arr[(i-window):(i-1)])
        right_mean = mean(arr[i:(i+window-1)])
        level_changes[i] = abs(right_mean - left_mean)
    end

    # Find potential changepoints
    changepoints = Changepoint[]

    for i in (window+1):(n - window + 1)
        if level_changes[i] > threshold
            # Check if this is a local maximum (peak of the change)
            is_peak = true

            # Check left neighbors
            for j in max(window + 1, i - min_segment_length):(i-1)
                if level_changes[j] >= level_changes[i]
                    is_peak = false
                    break
                end
            end

            # Check right neighbors
            if is_peak
                for j in (i+1):min(n - window + 1, i + min_segment_length - 1)
                    if level_changes[j] > level_changes[i]
                        is_peak = false
                        break
                    end
                end
            end

            if is_peak
                # Check minimum segment length constraint
                if isempty(changepoints) || (i - changepoints[end].index) >= min_segment_length
                    cp = Changepoint(i, level_changes[i])
                    push!(changepoints, cp)
                end
            end
        end
    end

    return ChangepointResult(
        changepoints,
        length(changepoints) + 1,
        :variance,
        penalty
    )
end

"""
    detect_changepoints(series; method=:variance, penalty=3.0, min_segment_length=4, kwargs...)

Unified changepoint detection interface.

# Arguments
- `series::AbstractVector`: Time series data
- `method::Symbol=:variance`: Detection method (:variance)
- `penalty::Float64=3.0`: Penalty parameter
- `min_segment_length::Int=4`: Minimum observations between changepoints
- `kwargs...`: Additional arguments passed to detection function

# Returns
- `ChangepointResult`: Detected changepoints

# Example
```julia
series = vcat(fill(1.0, 30), fill(5.0, 30))
result = detect_changepoints(series)
```
"""
function detect_changepoints(
    series::AbstractVector{<:Real};
    method::Symbol = :variance,
    penalty::Float64 = 3.0,
    min_segment_length::Int = 4,
    window::Int = 8
)::ChangepointResult
    if method == :variance
        return detect_changepoints_variance(
            series;
            penalty=penalty,
            min_segment_length=min_segment_length,
            window=window
        )
    else
        error("Unknown method: $method. Use :variance.")
    end
end

# =============================================================================
# Regime Classification
# =============================================================================

"""
    classify_regimes(series, changepoints; method=:volatility, thresholds=nothing)

Assign regime labels to segments between changepoints.

# Arguments
- `series::AbstractVector`: Original time series
- `changepoints::Union{ChangepointResult, Vector{Changepoint}}`: Detected changepoints
- `method::Symbol=:volatility`: Classification method
  - `:volatility`: Classify by segment volatility (diff std)
  - `:level`: Classify by segment mean level
  - `:trend`: Classify by segment trend direction
- `thresholds::Union{Tuple{Float64, Float64}, Nothing}=nothing`:
  Custom (low, high) thresholds for LOW/MEDIUM/HIGH classification.
  If nothing, uses data-driven thresholds (33rd/67th percentiles).

# Returns
- `Vector{String}`: Array of regime labels ("LOW", "MEDIUM", "HIGH")

# Example
```julia
series = vcat(fill(1.0, 30), fill(5.0, 30))
result = detect_changepoints(series)
regimes = classify_regimes(series, result)
regimes[1]   # "LOW"
regimes[35]  # "HIGH"
```
"""
function classify_regimes(
    series::AbstractVector{<:Real},
    changepoints::Union{ChangepointResult, Vector{Changepoint}};
    method::Symbol = :volatility,
    thresholds::Union{Tuple{Float64, Float64}, Nothing} = nothing
)::Vector{String}
    arr = collect(Float64, series)
    n = length(arr)

    # Extract changepoint list
    cp_list = changepoints isa ChangepointResult ? changepoints.changepoints : changepoints

    # Get segment boundaries
    cp_indices = vcat([0], [cp.index for cp in cp_list], [n])

    # Compute segment characteristics
    segment_values = Float64[]
    for i in 1:(length(cp_indices) - 1)
        start_idx = cp_indices[i] + 1
        end_idx = cp_indices[i + 1]
        segment = arr[start_idx:end_idx]

        val = if method == :volatility
            length(segment) > 1 ? std(diff(segment)) : 0.0
        elseif method == :level
            mean(segment)
        elseif method == :trend
            if length(segment) > 1
                t = 1:length(segment)
                # Simple linear regression slope
                t_centered = t .- mean(t)
                s_centered = segment .- mean(segment)
                sum(t_centered .* s_centered) / sum(t_centered.^2)
            else
                0.0
            end
        else
            error("Unknown method: $method. Use :volatility, :level, or :trend.")
        end
        push!(segment_values, val)
    end

    # Determine thresholds
    low_thresh, high_thresh = if !isnothing(thresholds)
        thresholds
    else
        # Data-driven: use 33rd and 67th percentiles
        if length(segment_values) >= 3
            (quantile(segment_values, 0.33), quantile(segment_values, 0.67))
        else
            # Not enough segments - use overall statistics
            overall_val = mean(segment_values)
            (overall_val * 0.5, overall_val * 1.5)
        end
    end

    # Assign regime labels
    regimes = fill("", n)
    for i in 1:(length(cp_indices) - 1)
        start_idx = cp_indices[i] + 1
        end_idx = cp_indices[i + 1]
        val = segment_values[i]

        regime = if val < low_thresh
            "LOW"
        elseif val > high_thresh
            "HIGH"
        else
            "MEDIUM"
        end

        regimes[start_idx:end_idx] .= regime
    end

    return regimes
end

"""
    get_segment_boundaries(n, changepoints)

Get segment start/end indices from changepoints.

# Arguments
- `n::Int`: Series length
- `changepoints::Union{ChangepointResult, Vector{Changepoint}}`: Detected changepoints

# Returns
- `Vector{Tuple{Int, Int}}`: List of (start, end) index pairs for each segment

# Example
```julia
cps = [Changepoint(30, 1.0)]
get_segment_boundaries(60, cps)  # [(1, 30), (31, 60)]
```
"""
function get_segment_boundaries(
    n::Int,
    changepoints::Union{ChangepointResult, Vector{Changepoint}}
)::Vector{Tuple{Int, Int}}
    cp_list = changepoints isa ChangepointResult ? changepoints.changepoints : changepoints
    cp_indices = vcat([0], [cp.index for cp in cp_list], [n])

    return [(cp_indices[i] + 1, cp_indices[i + 1]) for i in 1:(length(cp_indices) - 1)]
end

"""
    create_regime_indicators(series, changepoints; recent_window=4)

Create regime indicator features for modeling.

# Arguments
- `series::AbstractVector`: Time series data
- `changepoints::Union{ChangepointResult, Vector{Changepoint}}`: Detected changepoints
- `recent_window::Int=4`: Observations to consider as "recent" regime change

# Returns
- `Dict{String, Vector}`: Dictionary with indicator arrays:
  - "is_regime_change": 1 if within recent_window of a changepoint
  - "periods_since_change": Periods since last changepoint
  - "regime_labels": Regime labels ("LOW", "MEDIUM", "HIGH")

# Example
```julia
series = vcat(fill(1.0, 30), fill(5.0, 30))
result = detect_changepoints(series)
indicators = create_regime_indicators(series, result)
```
"""
function create_regime_indicators(
    series::AbstractVector{<:Real},
    changepoints::Union{ChangepointResult, Vector{Changepoint}};
    recent_window::Int = 4
)::Dict{String, Vector}
    arr = collect(Float64, series)
    n = length(arr)

    # Extract changepoint indices
    cp_list = changepoints isa ChangepointResult ? changepoints.changepoints : changepoints
    cp_indices = Set([cp.index for cp in cp_list])

    # Periods since last changepoint
    periods_since = zeros(Int, n)
    last_cp = 0
    for i in 1:n
        if i in cp_indices
            last_cp = i
        end
        periods_since[i] = last_cp > 0 ? i - last_cp : i
    end

    # Is recent regime change
    is_recent = Int.((periods_since .<= recent_window) .& (periods_since .> 0))

    # Regime labels
    regime_labels = classify_regimes(arr, cp_list)

    return Dict(
        "is_regime_change" => is_recent,
        "periods_since_change" => periods_since,
        "regime_labels" => regime_labels
    )
end
