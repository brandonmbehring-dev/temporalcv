# =============================================================================
# Regime Classification Implementation
# =============================================================================

"""
    classify_volatility_regime(values; window=13, basis=:changes, low_pct=33.0, high_pct=67.0)

Classify volatility regime for each point using rolling window.

CRITICAL: Default `basis=:changes` computes volatility on first differences,
which is the methodologically correct approach [T2].

# Arguments
- `values::AbstractVector{<:Real}`: Time series values
- `window::Int=13`: Rolling window for volatility (13 weeks ~ 1 quarter) [T3]
- `basis::Symbol=:changes`: `:changes` (correct) or `:levels` (legacy)
- `low_pct::Float64=33.0`: Percentile threshold for LOW [T3]
- `high_pct::Float64=67.0`: Percentile threshold for HIGH [T3]

# Returns
`Vector{VolatilityRegime}` with regime for each point.

# Example
```julia
values = cumsum(randn(200) .* 0.01) .+ 3.0
regimes = classify_volatility_regime(values; window=13, basis=:changes)
```

# References
[T1] Hamilton (1989). Regime-switching models.
[T2] BUG-005: Volatility of changes, not levels.
"""
function classify_volatility_regime(
    values::AbstractVector{<:Real};
    window::Int = 13,
    basis::Symbol = :changes,
    low_pct::Float64 = 33.0,
    high_pct::Float64 = 67.0
)::Vector{VolatilityRegime}
    values = Float64.(values)
    n = length(values)

    # Handle edge cases
    if n == 0
        return VolatilityRegime[]
    end

    if n < window + 1
        # Insufficient data - return all MED
        return fill(VOL_MED, n)
    end

    # Compute series for volatility calculation
    if basis == :changes
        # Compute first differences
        series_for_vol = diff(values)
        # Pad with NaN to maintain alignment
        series_for_vol = vcat(NaN, series_for_vol)
    elseif basis == :levels
        series_for_vol = copy(values)
    else
        error("basis must be :changes or :levels, got :$basis")
    end

    # Compute rolling volatility
    rolling_vol = fill(NaN, n)

    for i in window:n
        window_data = series_for_vol[(i - window + 1):i]
        # Skip if any NaN in window
        if !any(isnan, window_data)
            rolling_vol[i] = std(window_data; corrected=true)
        end
    end

    # Get valid volatility values for threshold computation
    valid_vol = filter(!isnan, rolling_vol)

    if isempty(valid_vol)
        return fill(VOL_MED, n)
    end

    # Compute thresholds using percentiles
    vol_low = quantile(valid_vol, low_pct / 100.0)
    vol_high = quantile(valid_vol, high_pct / 100.0)

    # Classify each point
    regimes = Vector{VolatilityRegime}(undef, n)
    for i in 1:n
        vol = rolling_vol[i]
        if isnan(vol)
            regimes[i] = VOL_MED  # Default for insufficient history
        elseif vol <= vol_low
            regimes[i] = VOL_LOW
        elseif vol <= vol_high
            regimes[i] = VOL_MED
        else
            regimes[i] = VOL_HIGH
        end
    end

    return regimes
end


"""
    classify_direction_regime(values, threshold)

Classify direction using thresholded signs.

This makes persistence (predicts 0) a meaningful baseline for direction accuracy.

# Arguments
- `values::AbstractVector{<:Real}`: Values to classify (typically actual changes)
- `threshold::Real`: Move threshold (typically 70th percentile of |actuals|)

# Returns
`Vector{DirectionRegime}` with direction for each point.

# Example
```julia
actuals = [0.1, -0.1, 0.02, -0.02, 0.0]
directions = classify_direction_regime(actuals, 0.05)
# Returns: [DIR_UP, DIR_DOWN, DIR_FLAT, DIR_FLAT, DIR_FLAT]
```
"""
function classify_direction_regime(
    values::AbstractVector{<:Real},
    threshold::Real
)::Vector{DirectionRegime}
    threshold >= 0 || error("threshold must be non-negative, got $threshold")

    n = length(values)
    if n == 0
        return DirectionRegime[]
    end

    regimes = Vector{DirectionRegime}(undef, n)
    for i in 1:n
        v = values[i]
        if isnan(v) || abs(v) <= threshold
            regimes[i] = DIR_FLAT
        elseif v > 0
            regimes[i] = DIR_UP
        else
            regimes[i] = DIR_DOWN
        end
    end

    return regimes
end


"""
    get_combined_regimes(vol_regimes, dir_regimes)

Combine volatility and direction into single string label.

# Arguments
- `vol_regimes::Vector{VolatilityRegime}`: Volatility regimes
- `dir_regimes::Vector{DirectionRegime}`: Direction regimes

# Returns
`Vector{String}` with combined labels like "HIGH-UP", "LOW-FLAT".

# Example
```julia
vol = [VOL_HIGH, VOL_LOW, VOL_MED]
dir = [DIR_UP, DIR_DOWN, DIR_FLAT]
combined = get_combined_regimes(vol, dir)
# Returns: ["HIGH-UP", "LOW-DOWN", "MED-FLAT"]
```
"""
function get_combined_regimes(
    vol_regimes::Vector{VolatilityRegime},
    dir_regimes::Vector{DirectionRegime}
)::Vector{String}
    length(vol_regimes) == length(dir_regimes) ||
        error("Arrays must have same length: vol=$(length(vol_regimes)), dir=$(length(dir_regimes))")

    return ["$(regime_string(v))-$(regime_string(d))" for (v, d) in zip(vol_regimes, dir_regimes)]
end

# Also support string vectors for convenience
function get_combined_regimes(
    vol_regimes::Vector{String},
    dir_regimes::Vector{String}
)::Vector{String}
    length(vol_regimes) == length(dir_regimes) ||
        error("Arrays must have same length: vol=$(length(vol_regimes)), dir=$(length(dir_regimes))")

    return ["$v-$d" for (v, d) in zip(vol_regimes, dir_regimes)]
end


"""
    get_regime_counts(regimes)

Get sample counts per regime.

# Arguments
- `regimes`: Vector of regime labels (enum or string)

# Returns
`Dict{String, Int}` with counts per regime, sorted by count descending.

# Example
```julia
regimes = [VOL_HIGH, VOL_LOW, VOL_LOW, VOL_MED, VOL_LOW]
counts = get_regime_counts(regimes)
# Returns: Dict("LOW" => 3, "HIGH" => 1, "MED" => 1)
```
"""
function get_regime_counts(regimes::AbstractVector)::Dict{String, Int}
    if isempty(regimes)
        return Dict{String, Int}()
    end

    counts = Dict{String, Int}()
    for r in regimes
        key = string(r)
        counts[key] = get(counts, key, 0) + 1
    end

    return counts
end


"""
    mask_low_n_regimes(regimes; min_n=10, mask_value="MASKED")

Mask regime labels with insufficient samples.

# Arguments
- `regimes`: Vector of regime labels
- `min_n::Int=10`: Minimum samples required per regime [T3]
- `mask_value::String="MASKED"`: Value to use for masked regimes

# Returns
`Vector{String}` with low-n regimes masked.

# Example
```julia
regimes = vcat(fill("HIGH", 5), fill("LOW", 15))
masked = mask_low_n_regimes(regimes; min_n=10)
# "HIGH" becomes "MASKED" (only 5 samples)
```
"""
function mask_low_n_regimes(
    regimes::AbstractVector;
    min_n::Int = 10,
    mask_value::String = "MASKED"
)::Vector{String}
    if isempty(regimes)
        return String[]
    end

    counts = get_regime_counts(regimes)
    low_n_regimes = Set(r for (r, c) in counts if c < min_n)

    if isempty(low_n_regimes)
        return [string(r) for r in regimes]
    end

    result = Vector{String}(undef, length(regimes))
    for (i, r) in enumerate(regimes)
        key = string(r)
        result[i] = key in low_n_regimes ? mask_value : key
    end

    return result
end


"""
    compute_stratified_metrics(predictions, actuals, regimes; min_n=10)

Compute MAE and RMSE stratified by regime.

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions
- `actuals::AbstractVector{<:Real}`: Actual values
- `regimes`: Regime labels for each point
- `min_n::Int=10`: Minimum samples per regime [T3]

# Returns
`StratifiedMetricsResult` with overall and per-regime metrics.

# Example
```julia
vol_regimes = classify_volatility_regime(actuals; window=13, basis=:changes)
result = compute_stratified_metrics(predictions, actuals, vol_regimes)
println(summary(result))
```
"""
function compute_stratified_metrics(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    regimes::AbstractVector;
    min_n::Int = 10
)::StratifiedMetricsResult
    # Validation
    n = length(predictions)
    n > 0 || error("predictions cannot be empty")
    n == length(actuals) || error("predictions and actuals must have same length")
    n == length(regimes) || error("regimes must have same length as predictions")

    predictions = Float64.(predictions)
    actuals = Float64.(actuals)

    # Compute overall metrics
    errors = predictions .- actuals
    overall_mae = mean(abs.(errors))
    overall_rmse = sqrt(mean(errors .^ 2))

    # Get regime counts
    regime_counts = get_regime_counts(regimes)

    # Identify masked regimes
    masked_regimes = [r for (r, c) in regime_counts if c < min_n]

    # Compute per-regime metrics
    by_regime = Dict{String, Dict{Symbol, Float64}}()

    for regime_str in keys(regime_counts)
        n_regime = regime_counts[regime_str]

        if n_regime < min_n
            continue  # Skip masked regimes
        end

        # Create mask for this regime
        mask = [string(r) == regime_str for r in regimes]
        regime_errors = errors[mask]

        mae = mean(abs.(regime_errors))
        rmse = sqrt(mean(regime_errors .^ 2))
        pct = 100.0 * n_regime / n

        by_regime[regime_str] = Dict{Symbol, Float64}(
            :mae => mae,
            :rmse => rmse,
            :n => Float64(n_regime),
            :pct => pct
        )
    end

    return StratifiedMetricsResult(
        overall_mae,
        overall_rmse,
        n,
        by_regime,
        masked_regimes
    )
end
