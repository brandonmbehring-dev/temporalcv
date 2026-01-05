# =============================================================================
# Purged Cross-Validation Implementation
# =============================================================================

"""
    compute_label_overlap(n::Int, horizon::Int) -> BitMatrix

Compute boolean matrix indicating which samples have overlapping labels.

For forward returns with horizon h, labels at t1 and t2 overlap if |t1 - t2| < h.

# Arguments
- `n::Int`: Number of samples
- `horizon::Int`: Label horizon (e.g., 5 for 5-day forward return)

# Returns
BitMatrix of size (n, n) where entry [i,j] is true if labels overlap.

# Example
```julia
# 5-day forward returns for 100 samples
overlap = compute_label_overlap(100, 5)
# overlap[1,4] == true (4 shared days)
# overlap[1,6] == false (no overlap)
```

# References
[T1] De Prado (2018), Section 7.4.1
"""
function compute_label_overlap(n::Int, horizon::Int)::BitMatrix
    n > 0 || error("n must be positive, got $n")
    horizon >= 0 || error("horizon must be non-negative, got $horizon")

    overlap = falses(n, n)

    for i in 1:n
        for j in 1:n
            if abs(i - j) < horizon
                overlap[i, j] = true
            end
        end
    end

    return overlap
end


"""
    estimate_purge_gap(horizon::Int; decay_factor::Float64=1.0) -> Int

Estimate appropriate purge gap from forecast horizon.

# Arguments
- `horizon::Int`: Forecast horizon
- `decay_factor::Float64=1.0`: Multiplier for conservativeness

# Returns
Recommended purge gap in samples.

# Example
```julia
gap = estimate_purge_gap(5)  # Returns 5
gap = estimate_purge_gap(5; decay_factor=1.5)  # Returns 8
```
"""
function estimate_purge_gap(horizon::Int; decay_factor::Float64 = 1.0)::Int
    horizon >= 0 || error("horizon must be non-negative, got $horizon")
    decay_factor > 0 || error("decay_factor must be positive, got $decay_factor")

    return ceil(Int, horizon * decay_factor)
end


"""
    apply_purge_and_embargo(train_indices, test_indices, n; purge_gap, embargo_pct) -> PurgedSplit

Apply purging and embargo to a train/test split.

Purging removes training samples within `purge_gap` of any test sample.
Embargo removes `embargo_pct` of samples immediately after the test set.

# Arguments
- `train_indices::Vector{Int}`: Original training indices
- `test_indices::Vector{Int}`: Test indices
- `n::Int`: Total number of samples
- `purge_gap::Int=0`: Gap for purging
- `embargo_pct::Float64=0.0`: Embargo fraction

# Returns
`PurgedSplit` with cleaned training indices.

# References
[T1] De Prado (2018), Section 7.4.2
"""
function apply_purge_and_embargo(
    train_indices::Vector{Int},
    test_indices::Vector{Int},
    n::Int;
    purge_gap::Int = 0,
    embargo_pct::Float64 = 0.0
)::PurgedSplit
    isempty(test_indices) && return PurgedSplit(train_indices, test_indices, 0, 0)

    test_min = minimum(test_indices)
    test_max = maximum(test_indices)

    n_purged = 0
    n_embargoed = 0

    # Compute embargo size
    embargo_size = ceil(Int, n * embargo_pct)

    # Determine which indices to remove from training
    remove_mask = falses(length(train_indices))

    for (idx, train_idx) in enumerate(train_indices)
        # Purge: remove if within purge_gap of test set
        if purge_gap > 0
            # Before test set
            if train_idx >= test_min - purge_gap && train_idx < test_min
                remove_mask[idx] = true
                n_purged += 1
                continue
            end
            # After test set
            if train_idx > test_max && train_idx <= test_max + purge_gap
                remove_mask[idx] = true
                n_purged += 1
                continue
            end
        end

        # Embargo: remove samples after test set
        if embargo_size > 0
            if train_idx > test_max && train_idx <= test_max + embargo_size
                remove_mask[idx] = true
                n_embargoed += 1
            end
        end
    end

    cleaned_train = train_indices[.!remove_mask]

    return PurgedSplit(cleaned_train, test_indices, n_purged, n_embargoed)
end


# =============================================================================
# PurgedKFold
# =============================================================================

"""
    get_n_splits(cv::PurgedKFold) -> Int

Return number of splits for PurgedKFold.
"""
get_n_splits(cv::PurgedKFold) = cv.n_splits


"""
    split(cv::PurgedKFold, n::Int) -> Vector{PurgedSplit}

Generate purged K-fold splits.

# Arguments
- `cv::PurgedKFold`: Cross-validation configuration
- `n::Int`: Number of samples

# Returns
Vector of `PurgedSplit`, one per fold.

# Example
```julia
cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)
splits = split(cv, 1000)

for s in splits
    # Use s.train_indices, s.test_indices
end
```
"""
function split(cv::PurgedKFold, n::Int)::Vector{PurgedSplit}
    n >= cv.n_splits || error("n ($n) must be >= n_splits ($(cv.n_splits))")

    # Create fold assignments
    fold_size = n ÷ cv.n_splits
    remainder = n % cv.n_splits

    fold_indices = Vector{Vector{Int}}(undef, cv.n_splits)
    start_idx = 1

    for fold in 1:cv.n_splits
        # Distribute remainder across first folds
        size = fold_size + (fold <= remainder ? 1 : 0)
        fold_indices[fold] = collect(start_idx:(start_idx + size - 1))
        start_idx += size
    end

    # Generate splits
    splits = PurgedSplit[]

    for test_fold in 1:cv.n_splits
        test_indices = fold_indices[test_fold]

        # All other folds are training
        train_indices = Int[]
        for fold in 1:cv.n_splits
            if fold != test_fold
                append!(train_indices, fold_indices[fold])
            end
        end
        sort!(train_indices)

        # Apply purging and embargo
        purged = apply_purge_and_embargo(
            train_indices, test_indices, n;
            purge_gap=cv.purge_gap, embargo_pct=cv.embargo_pct
        )
        push!(splits, purged)
    end

    return splits
end


# =============================================================================
# CombinatorialPurgedCV
# =============================================================================

"""
    get_n_splits(cv::CombinatorialPurgedCV) -> Int

Return number of splits for CombinatorialPurgedCV.

This is binomial(n_splits, n_test_splits).
"""
function get_n_splits(cv::CombinatorialPurgedCV)
    return binomial(cv.n_splits, cv.n_test_splits)
end


"""
    split(cv::CombinatorialPurgedCV, n::Int) -> Vector{PurgedSplit}

Generate combinatorial purged CV splits.

Uses all combinations of n_test_splits folds as test set.

# Arguments
- `cv::CombinatorialPurgedCV`: Cross-validation configuration
- `n::Int`: Number of samples

# Returns
Vector of `PurgedSplit`, one per combination.

# Example
```julia
cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)
splits = split(cv, 1000)  # 10 splits (5 choose 2)
```

# References
[T1] De Prado (2018), Section 12.3
"""
function split(cv::CombinatorialPurgedCV, n::Int)::Vector{PurgedSplit}
    n >= cv.n_splits || error("n ($n) must be >= n_splits ($(cv.n_splits))")

    # Create fold assignments
    fold_size = n ÷ cv.n_splits
    remainder = n % cv.n_splits

    fold_indices = Vector{Vector{Int}}(undef, cv.n_splits)
    start_idx = 1

    for fold in 1:cv.n_splits
        size = fold_size + (fold <= remainder ? 1 : 0)
        fold_indices[fold] = collect(start_idx:(start_idx + size - 1))
        start_idx += size
    end

    # Generate all combinations
    splits = PurgedSplit[]

    for test_folds in combinations(1:cv.n_splits, cv.n_test_splits)
        # Combine test fold indices
        test_indices = Int[]
        for fold in test_folds
            append!(test_indices, fold_indices[fold])
        end
        sort!(test_indices)

        # Training is everything else
        train_folds = setdiff(1:cv.n_splits, test_folds)
        train_indices = Int[]
        for fold in train_folds
            append!(train_indices, fold_indices[fold])
        end
        sort!(train_indices)

        # Apply purging and embargo
        purged = apply_purge_and_embargo(
            train_indices, test_indices, n;
            purge_gap=cv.purge_gap, embargo_pct=cv.embargo_pct
        )
        push!(splits, purged)
    end

    return splits
end


# =============================================================================
# PurgedWalkForward
# =============================================================================

"""
    get_n_splits(cv::PurgedWalkForward) -> Int

Return number of splits for PurgedWalkForward.
"""
get_n_splits(cv::PurgedWalkForward) = cv.n_splits


"""
    split(cv::PurgedWalkForward, n::Int) -> Vector{PurgedSplit}

Generate purged walk-forward splits.

# Arguments
- `cv::PurgedWalkForward`: Cross-validation configuration
- `n::Int`: Number of samples

# Returns
Vector of `PurgedSplit`, one per walk-forward window.

# Example
```julia
cv = PurgedWalkForward(n_splits=5, train_size=200, purge_gap=5)
splits = split(cv, 1000)
```
"""
function split(cv::PurgedWalkForward, n::Int)::Vector{PurgedSplit}
    cv.n_splits >= 1 || error("n_splits must be >= 1")

    # Compute test fold size
    test_size = n ÷ (cv.n_splits + 1)
    test_size >= 1 || error("Not enough samples for $(cv.n_splits) splits")

    splits = PurgedSplit[]

    for split_idx in 1:cv.n_splits
        # Test window
        test_start = (split_idx * test_size) + 1
        test_end = min(test_start + test_size - 1, n)

        # Skip if test window goes beyond data
        test_end <= n || continue

        test_indices = collect(test_start:test_end)

        # Training window
        if cv.train_size > 0
            # Fixed window
            train_end = test_start - cv.purge_gap - 1
            train_start = max(1, train_end - cv.train_size + 1)
        else
            # Expanding window
            train_start = 1
            train_end = test_start - cv.purge_gap - 1
        end

        train_end >= train_start || continue

        train_indices = collect(train_start:train_end)

        # Apply additional purging and embargo
        purged = apply_purge_and_embargo(
            train_indices, test_indices, n;
            purge_gap=cv.purge_gap, embargo_pct=cv.embargo_pct
        )
        push!(splits, purged)
    end

    return splits
end


# =============================================================================
# Utility Functions
# =============================================================================

"""
    get_train_test_indices(split::PurgedSplit) -> Tuple{Vector{Int}, Vector{Int}}

Extract train and test indices from a PurgedSplit.
"""
function get_train_test_indices(s::PurgedSplit)
    return (s.train_indices, s.test_indices)
end


"""
    total_purged_samples(splits::Vector{PurgedSplit}) -> Int

Sum of all purged samples across splits.
"""
function total_purged_samples(splits::Vector{PurgedSplit})
    return sum(s.n_purged for s in splits)
end


"""
    total_embargoed_samples(splits::Vector{PurgedSplit}) -> Int

Sum of all embargoed samples across splits.
"""
function total_embargoed_samples(splits::Vector{PurgedSplit})
    return sum(s.n_embargoed for s in splits)
end


# Additional exports
export get_train_test_indices, total_purged_samples, total_embargoed_samples
