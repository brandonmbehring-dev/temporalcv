# =============================================================================
# CrossFitCV - Temporal Cross-Fitting for Debiased Metrics
# =============================================================================

"""
    CrossFitCV

Temporal cross-fitting for debiased out-of-sample predictions.

For each fold k:
- Train model on ALL data before fold k (forward-only)
- Predict on fold k (out-of-sample)

This eliminates regularization bias by ensuring predictions are NEVER
made on training data. Unlike standard Double ML (random KFold), this
enforces strict temporal ordering.

# Knowledge Tiers
- [T1] Cross-fitting debiasing is established (Chernozhukov et al. 2018)
- [T2] Temporal adaptation with gap enforcement

# Fields
- `n_splits::Int`: Number of temporal folds
- `gap::Int`: Samples to exclude between training and test
- `test_size::Union{Int, Nothing}`: Size of each test fold

# Notes
**Forward-only semantics**:
- Fold 0: No training data → predictions are NaN
- Fold 1: Train on fold 0, predict on fold 1
- Fold k: Train on folds 0..k-1, predict on fold k

This is stricter than bidirectional cross-fitting but guarantees
temporal safety.

# Example
```julia
cv = CrossFitCV(n_splits=5, gap=2)
for (train_idx, test_idx) in split(cv, 200)
    println("Train: 1-\$(train_idx[end]), Test: \$(test_idx[1])-\$(test_idx[end])")
end
```

# References
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.
"""
struct CrossFitCV
    n_splits::Int
    gap::Int
    test_size::Union{Int, Nothing}

    function CrossFitCV(;
        n_splits::Int = 5,
        gap::Int = 0,
        test_size::Union{Int, Nothing} = nothing
    )
        @assert n_splits >= 2 "n_splits must be >= 2, got $n_splits"
        @assert gap >= 0 "gap must be >= 0, got $gap"

        if !isnothing(test_size)
            @assert test_size >= 1 "test_size must be >= 1, got $test_size"
        end

        new(n_splits, gap, test_size)
    end
end

"""
    _calculate_fold_indices(cv::CrossFitCV, n::Int) -> Vector{Tuple{Int, Int}}

Calculate (start, end) indices for each fold (1-based, inclusive).
"""
function _calculate_fold_indices(cv::CrossFitCV, n::Int)
    fold_size = isnothing(cv.test_size) ? n ÷ cv.n_splits : cv.test_size

    if fold_size < 1
        throw(ArgumentError(
            "Not enough samples ($n) for $(cv.n_splits) splits"
        ))
    end

    folds = Tuple{Int, Int}[]
    for k in 0:(cv.n_splits - 1)
        start = k * fold_size + 1  # 1-based

        if k == cv.n_splits - 1
            # Last fold takes remaining samples
            stop = n
        else
            stop = (k + 1) * fold_size
        end

        if start <= n
            push!(folds, (start, stop))
        end
    end

    return folds
end

"""
    split(cv::CrossFitCV, n::Int) -> Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}

Generate train/test split indices.

For fold k (k >= 1):
- Train indices: all samples from folds 0 to k-1 (minus gap)
- Test indices: samples in fold k

Fold 0 is skipped since there's no training data.

# Returns
Vector of (train_indices, test_indices) tuples using 1-based Julia indexing.
"""
function split(cv::CrossFitCV, n::Int)
    folds = _calculate_fold_indices(cv, n)
    splits = Tuple{UnitRange{Int}, UnitRange{Int}}[]

    # Skip fold 0 - no training data available
    for k in 2:length(folds)
        test_start, test_end = folds[k]

        # Train on all previous folds, respecting gap
        train_end = test_start - cv.gap - 1

        if train_end <= 0
            continue
        end

        push!(splits, (1:train_end, test_start:test_end))
    end

    return splits
end

"""
    get_n_splits(cv::CrossFitCV, n::Int; strict::Bool=true) -> Int

Return the number of valid splits for n samples.
"""
function get_n_splits(cv::CrossFitCV, n::Int; strict::Bool = true)
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
    get_fold_indices(cv::CrossFitCV, n::Int) -> Vector{Tuple{Int, Int}}

Return (start, end) indices for each fold.
"""
function get_fold_indices(cv::CrossFitCV, n::Int)
    return _calculate_fold_indices(cv, n)
end

function Base.show(io::IO, cv::CrossFitCV)
    print(io, "CrossFitCV(n_splits=$(cv.n_splits), ",
          "gap=$(cv.gap), ",
          "test_size=$(cv.test_size))")
end
