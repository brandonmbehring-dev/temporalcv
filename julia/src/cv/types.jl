# =============================================================================
# CV Types - SplitInfo, SplitResult, WalkForwardResults
# =============================================================================

"""
    SplitInfo

Metadata for a single CV split.

Useful for debugging and visualizing the split structure.

# Fields
- `split_idx::Int`: Zero-based split index
- `train_start::Int`: First training index (inclusive, 1-based)
- `train_end::Int`: Last training index (inclusive, 1-based)
- `test_start::Int`: First test index (inclusive, 1-based)
- `test_end::Int`: Last test index (inclusive, 1-based)

# Example
```julia
info = SplitInfo(0, 1, 100, 103, 112)
println("Gap: \$(gap(info)), Train size: \$(train_size(info))")
```
"""
struct SplitInfo
    split_idx::Int
    train_start::Int
    train_end::Int
    test_start::Int
    test_end::Int

    function SplitInfo(split_idx, train_start, train_end, test_start, test_end)
        # Validate temporal ordering - train_end must be before test_start
        if train_end >= test_start
            throw(ArgumentError(
                "Temporal leakage: train_end ($train_end) >= test_start ($test_start)"
            ))
        end
        new(split_idx, train_start, train_end, test_start, test_end)
    end
end

"""Number of training samples."""
train_size(info::SplitInfo) = info.train_end - info.train_start + 1

"""Number of test samples."""
test_size(info::SplitInfo) = info.test_end - info.test_start + 1

"""Actual gap between train end and test start."""
gap(info::SplitInfo) = info.test_start - info.train_end - 1


"""
    SplitResult

Result from a single walk-forward split.

Contains predictions, actuals, and metadata for one CV split.

# Fields
- `split_idx::Int`: Zero-based split index
- `train_start::Int`: First training index (inclusive, 1-based)
- `train_end::Int`: Last training index (inclusive, 1-based)
- `test_start::Int`: First test index (inclusive, 1-based)
- `test_end::Int`: Last test index (inclusive, 1-based)
- `predictions::Vector{Float64}`: Model predictions for this split's test set
- `actuals::Vector{Float64}`: Actual values for this split's test set
"""
struct SplitResult
    split_idx::Int
    train_start::Int
    train_end::Int
    test_start::Int
    test_end::Int
    predictions::Vector{Float64}
    actuals::Vector{Float64}

    function SplitResult(split_idx, train_start, train_end, test_start, test_end,
                         predictions, actuals)
        if train_end >= test_start
            throw(ArgumentError(
                "Temporal leakage: train_end ($train_end) >= test_start ($test_start)"
            ))
        end
        if length(predictions) != length(actuals)
            throw(ArgumentError(
                "predictions length ($(length(predictions))) != actuals length ($(length(actuals)))"
            ))
        end
        new(split_idx, train_start, train_end, test_start, test_end,
            Vector{Float64}(predictions), Vector{Float64}(actuals))
    end
end

"""Number of training samples."""
train_size(r::SplitResult) = r.train_end - r.train_start + 1

"""Number of test samples."""
test_size(r::SplitResult) = r.test_end - r.test_start + 1

"""Actual gap between train end and test start."""
gap(r::SplitResult) = r.test_start - r.train_end - 1

"""Prediction errors (predictions - actuals)."""
errors(r::SplitResult) = r.predictions .- r.actuals

"""Absolute prediction errors."""
absolute_errors(r::SplitResult) = abs.(errors(r))

"""Mean Absolute Error for this split."""
mae(r::SplitResult) = mean(absolute_errors(r))

"""Root Mean Squared Error for this split."""
rmse(r::SplitResult) = sqrt(mean(errors(r).^2))

"""Mean signed error (positive = over-prediction)."""
bias(r::SplitResult) = mean(errors(r))

"""Convert to SplitInfo (metadata without predictions/actuals)."""
function to_split_info(r::SplitResult)
    SplitInfo(r.split_idx, r.train_start, r.train_end, r.test_start, r.test_end)
end


"""
    WalkForwardResults

Aggregated walk-forward cross-validation results.

Collects results from all splits and provides aggregate metrics.

# Fields
- `splits::Vector{SplitResult}`: Results from each CV split

# Example
```julia
results = walk_forward_evaluate(model, X, y, n_splits=5)
println("Overall MAE: \$(mae(results))")
for split in results.splits
    println("  Split \$(split.split_idx): MAE=\$(mae(split))")
end
```
"""
struct WalkForwardResults
    splits::Vector{SplitResult}

    function WalkForwardResults(splits::Vector{SplitResult})
        if isempty(splits)
            throw(ArgumentError("WalkForwardResults requires at least one split"))
        end
        new(splits)
    end
end

"""Number of CV splits."""
n_splits(r::WalkForwardResults) = length(r.splits)

"""All predictions concatenated across splits."""
function predictions(r::WalkForwardResults)
    vcat([s.predictions for s in r.splits]...)
end

"""All actuals concatenated across splits."""
function actuals(r::WalkForwardResults)
    vcat([s.actuals for s in r.splits]...)
end

"""All errors concatenated across splits."""
errors(r::WalkForwardResults) = predictions(r) .- actuals(r)

"""All absolute errors concatenated."""
absolute_errors(r::WalkForwardResults) = abs.(errors(r))

"""Overall Mean Absolute Error."""
mae(r::WalkForwardResults) = mean(absolute_errors(r))

"""Overall Root Mean Squared Error."""
rmse(r::WalkForwardResults) = sqrt(mean(errors(r).^2))

"""Overall mean signed error (positive = over-prediction)."""
bias(r::WalkForwardResults) = mean(errors(r))

"""Overall Mean Squared Error."""
mse(r::WalkForwardResults) = mean(errors(r).^2)

"""Total number of test samples across all splits."""
total_samples(r::WalkForwardResults) = sum(test_size(s) for s in r.splits)
