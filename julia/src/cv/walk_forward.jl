# =============================================================================
# WalkForwardCV - Walk-forward cross-validation with gap enforcement
# =============================================================================

"""
    WalkForwardCV

Walk-forward cross-validation with gap enforcement.

Provides temporal CV that ensures no data leakage between training and test sets.
Supports both expanding and sliding window modes per Tashman (2000) recommendations.

# Knowledge Tiers
- [T1] Walk-forward validation is the standard for time-series (Tashman 2000)
- [T1] Gap >= horizon prevents information leakage for h-step forecasts
- [T2] Gap enforcement: train_end + gap < test_start prevents lookahead

# Fields
- `n_splits::Int`: Number of CV folds
- `test_size::Int`: Number of samples in each test fold
- `gap::Int`: Samples to exclude between training and test
- `window_type::Symbol`: `:expanding` or `:sliding`
- `window_size::Union{Int, Nothing}`: Training window size (required for sliding)
- `horizon::Union{Int, Nothing}`: Forecast horizon for gap validation

# Example
```julia
cv = WalkForwardCV(n_splits=5, gap=2)
for (train_idx, test_idx) in split(cv, 200)
    println("Train: \$(train_idx[1])-\$(train_idx[end]), Test: \$(test_idx[1])-\$(test_idx[end])")
end
```

# References
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy.
  International Journal of Forecasting, 16(4), 437-450.
- Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for
  time series predictor evaluation. Information Sciences, 191, 192-213.
"""
struct WalkForwardCV
    n_splits::Int
    test_size::Int
    gap::Int
    window_type::Symbol
    window_size::Union{Int, Nothing}
    horizon::Union{Int, Nothing}

    function WalkForwardCV(;
        n_splits::Int = 5,
        test_size::Int = 1,
        gap::Int = 0,
        window_type::Symbol = :expanding,
        window_size::Union{Int, Nothing} = nothing,
        horizon::Union{Int, Nothing} = nothing
    )
        # Validate parameters
        @assert n_splits >= 1 "n_splits must be >= 1, got $n_splits"
        @assert test_size >= 1 "test_size must be >= 1, got $test_size"
        @assert gap >= 0 "gap must be >= 0, got $gap"
        @assert window_type in (:expanding, :sliding) "window_type must be :expanding or :sliding, got $window_type"

        if window_type == :sliding && isnothing(window_size)
            throw(ArgumentError("window_size is required for sliding window"))
        end

        if !isnothing(window_size)
            @assert window_size >= 1 "window_size must be >= 1, got $window_size"
        end

        if !isnothing(horizon)
            @assert horizon >= 1 "horizon must be >= 1, got $horizon"
        end

        # [T1] Gap >= horizon prevents target leakage for h-step forecasting
        if !isnothing(horizon) && gap < horizon
            throw(ArgumentError(
                "gap ($gap) must be >= horizon ($horizon) to prevent target leakage. " *
                "For $horizon-step forecasting, set gap >= $horizon. " *
                "See Bergmeir & Benitez (2012) for details."
            ))
        end

        new(n_splits, test_size, gap, window_type, window_size, horizon)
    end
end

"""
    split(cv::WalkForwardCV, n::Int) -> Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}

Generate train/test split indices.

# Arguments
- `cv`: WalkForwardCV configuration
- `n`: Total number of samples

# Returns
Vector of (train_indices, test_indices) tuples using 1-based Julia indexing.
"""
function split(cv::WalkForwardCV, n::Int)
    splits = Tuple{UnitRange{Int}, UnitRange{Int}}[]

    # Calculate minimum training size for first split
    if cv.window_type == :sliding
        min_train = cv.window_size::Int
    else
        # For expanding, auto-calculate initial window size
        if !isnothing(cv.window_size)
            min_train = cv.window_size
        else
            # Default: leave enough room for n_splits test sets
            total_test = cv.n_splits * cv.test_size
            available = n - cv.gap - total_test
            min_train = max(1, available)
        end
    end

    # Check if we have enough data
    min_required = min_train + cv.gap + cv.test_size
    if n < min_required
        throw(ArgumentError(
            "Not enough samples ($n) for $(cv.n_splits) splits. " *
            "Need at least $min_required samples " *
            "(min_train=$min_train, gap=$(cv.gap), test_size=$(cv.test_size))."
        ))
    end

    # Generate splits working backwards from end
    for split_idx in 0:(cv.n_splits - 1)
        # Calculate test indices (from the end)
        offset_from_end = split_idx * cv.test_size
        test_end = n - offset_from_end
        test_start = test_end - cv.test_size + 1

        # Calculate train indices
        train_end = test_start - cv.gap - 1

        if cv.window_type == :sliding
            train_start = train_end - cv.window_size::Int + 1
        else
            train_start = 1
        end

        # Check validity
        if train_start < 1 || train_end < train_start
            break
        end

        if cv.window_type == :sliding
            if train_end - train_start + 1 < cv.window_size::Int
                break
            end
        end

        push!(splits, (train_start:train_end, test_start:test_end))
    end

    # Reverse to get chronological order (earliest split first)
    reverse!(splits)

    # Trim to n_splits if we generated more
    if length(splits) > cv.n_splits
        splits = splits[end - cv.n_splits + 1:end]
    end

    return splits
end

"""
    get_n_splits(cv::WalkForwardCV, n::Int; strict::Bool=true) -> Int

Return the number of valid splits for n samples.

# Arguments
- `cv`: WalkForwardCV configuration
- `n`: Total number of samples
- `strict`: If true (default), raise on failure. If false, return 0.

# Returns
Number of valid splits.
"""
function get_n_splits(cv::WalkForwardCV, n::Int; strict::Bool = true)
    try
        return length(split(cv, n))
    catch e
        if strict
            rethrow(e)
        end
        return 0
    end
end

"""
    get_split_info(cv::WalkForwardCV, n::Int) -> Vector{SplitInfo}

Return detailed metadata for all splits.

# Arguments
- `cv`: WalkForwardCV configuration
- `n`: Total number of samples

# Returns
Vector of SplitInfo with metadata for each split.
"""
function get_split_info(cv::WalkForwardCV, n::Int)
    splits = split(cv, n)
    infos = SplitInfo[]

    for (idx, (train_range, test_range)) in enumerate(splits)
        push!(infos, SplitInfo(
            idx - 1,  # 0-based split index
            first(train_range),
            last(train_range),
            first(test_range),
            last(test_range)
        ))
    end

    return infos
end

function Base.show(io::IO, cv::WalkForwardCV)
    print(io, "WalkForwardCV(n_splits=$(cv.n_splits), ",
          "window_type=$(cv.window_type), ",
          "window_size=$(cv.window_size), ",
          "gap=$(cv.gap), ",
          "test_size=$(cv.test_size))")
end
