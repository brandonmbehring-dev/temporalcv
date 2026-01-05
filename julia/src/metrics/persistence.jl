# =============================================================================
# High-Persistence Series Metrics
# =============================================================================

"""
High-persistence series metrics module.

Specialized tools for evaluating forecasts on high-persistence time series
where the persistence baseline (predict no change) is trivially good.

Key concepts:
- **Move threshold**: Separates "significant" moves from noise (FLAT)
- **MC-SS**: Move-Conditional Skill Score relative to persistence
- **3-class direction**: UP/DOWN/FLAT makes persistence a fair baseline

# Knowledge Tiers
- [T1] Persistence baseline = predict no change (standard in forecasting literature)
- [T1] Skill score formula: SS = 1 - (model_error / baseline_error) (Murphy 1988)
- [T1] Directional accuracy testing framework (Pesaran & Timmermann 1992)
- [T2] MC-SS = skill score computed on moves only (myga-forecasting-v2 Phase 11)
- [T2] 70th percentile threshold defines "significant" moves (v2 empirical finding)
- [T2] Threshold MUST come from training data only (BUG-003 fix in v2)
- [T3] 10 samples per direction for reliability (rule of thumb, not validated)
- [T3] Scale-aware epsilon for numerical stability (implementation choice)

# References
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy.
- Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of
  predictive performance.
- Murphy, A.H. (1988). Skill scores based on the mean square error.
"""

using Statistics

# Import types from current module context
# (types.jl is included before this file)

# =============================================================================
# Target Mode Type
# =============================================================================

"""
Target mode for persistence metrics: `:change` or `:level`.

- `:change`: Data represents changes/returns (recommended)
- `:level`: Raw level data (NOT supported - convert first)

Persistence metrics assume baseline predicts "no change" (zero).
This only makes sense for change/return data.
"""
const TargetMode = Symbol

# =============================================================================
# Threshold Computation
# =============================================================================

"""
    compute_move_threshold(actuals; percentile=70.0, target_mode=:change)

Compute move threshold from historical changes.

Default: 70th percentile of |actuals|.

# Arguments
- `actuals::AbstractVector`: Historical actual values (from training data).
  Should be *changes* (returns/differences), not raw levels.
- `percentile::Float64=70.0`: Percentile of |actuals| to use as threshold
- `target_mode::Symbol=:change`: Whether actuals are changes or levels.
  - `:change`: Data already represents differences (recommended)
  - `:level`: Will raise error; convert to changes first

# Returns
- `Float64`: Move threshold

# Notes
CRITICAL: The threshold MUST be computed from training data only
to prevent regime threshold leakage (BUG-003 in myga-forecasting).

Using 70th percentile means ~30% of historical changes are "moves"
and ~70% are "flat". This provides a meaningful signal-to-noise ratio.

# Knowledge Tier
[T2] 70th percentile threshold from myga-forecasting-v2.
"""
function compute_move_threshold(
    actuals::AbstractVector{<:Real};
    percentile::Float64 = 70.0,
    target_mode::Symbol = :change
)::Float64
    if target_mode == :level
        error("target_mode=:level not supported for persistence metrics. " *
              "Persistence baseline assumes data represents changes/returns. " *
              "Convert levels to changes first: changes = diff(levels)")
    end

    arr = collect(Float64, actuals)

    if isempty(arr)
        error("Cannot compute threshold from empty array")
    end

    if !(0 < percentile <= 100)
        error("percentile must be in (0, 100], got $percentile")
    end

    # Julia quantile uses fraction, not percentage
    return quantile(abs.(arr), percentile / 100.0)
end


# =============================================================================
# Move Classification
# =============================================================================

"""
    classify_moves(values, threshold)

Classify values into UP, DOWN, FLAT categories.

# Arguments
- `values::AbstractVector`: Values to classify (typically actuals or predictions)
- `threshold::Float64`: Threshold for flat classification

# Returns
- `Vector{MoveDirection}`: Array of MoveDirection enums

# Example
```julia
values = [0.1, -0.1, 0.02, -0.02, 0.0]
moves = classify_moves(values, 0.05)
# [UP, DOWN, FLAT, FLAT, FLAT]
```
"""
function classify_moves(
    values::AbstractVector{<:Real},
    threshold::Float64
)::Vector{MoveDirection}
    arr = collect(Float64, values)

    if threshold < 0
        error("threshold must be non-negative, got $threshold")
    end

    return map(arr) do v
        if v > threshold
            UP
        elseif v < -threshold
            DOWN
        else
            FLAT
        end
    end
end


# =============================================================================
# Scale-Aware Epsilon
# =============================================================================

"""
    _get_scale_aware_epsilon(values)

Compute scale-aware epsilon for division safety.

Uses median absolute value to determine appropriate epsilon,
avoiding issues with very small magnitude data.
"""
function _get_scale_aware_epsilon(values::Vector{Float64})::Float64
    nonzero = filter(!iszero, values)
    if !isempty(nonzero)
        scale = median(abs.(nonzero))
        return max(1e-10, scale * 1e-8)
    end
    return 1e-10
end


# =============================================================================
# Move-Conditional Metrics
# =============================================================================

"""
    compute_move_conditional_metrics(predictions, actuals; threshold=nothing,
                                      threshold_percentile=70.0, target_mode=:change)

Compute move-conditional evaluation metrics.

Evaluates model performance separately for:
- UP moves: actual > threshold
- DOWN moves: actual < -threshold
- FLAT periods: |actual| <= threshold

# Arguments
- `predictions::AbstractVector`: Model predictions (predicted changes)
- `actuals::AbstractVector`: Actual values (actual changes)
- `threshold::Union{Float64, Nothing}=nothing`: Move threshold.
  If nothing, computed from actuals (NOT recommended for walk-forward).
- `threshold_percentile::Float64=70.0`: Percentile for threshold if computed
- `target_mode::Symbol=:change`: Data type (`:change` or `:level`)

# Returns
- `MoveConditionalResult`: Move-conditional metrics including MC-SS

# Notes
MC-SS (Move-Conditional Skill Score) formula:
    MC-SS = 1 - (model_mae_on_moves / persistence_mae_on_moves)

Where:
- model_mae_on_moves = MAE of predictions on UP and DOWN only
- persistence_mae_on_moves = mean(|actual|) on UP and DOWN only
  (because persistence predicts 0, its error equals |actual|)

CRITICAL: For walk-forward evaluation, `threshold` should be computed
from training data only to prevent leakage.

# Knowledge Tier
[T2] MC-SS formulation from myga-forecasting-v2 Phase 11.
"""
function compute_move_conditional_metrics(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    threshold::Union{Float64, Nothing} = nothing,
    threshold_percentile::Float64 = 70.0,
    target_mode::Symbol = :change
)::MoveConditionalResult
    if target_mode == :level
        error("target_mode=:level not supported for persistence metrics. " *
              "Convert levels to changes first: changes = diff(levels)")
    end

    preds = collect(Float64, predictions)
    acts = collect(Float64, actuals)

    # Validate no NaN values
    if any(isnan, preds)
        error("predictions contains NaN values. Clean data before processing.")
    end
    if any(isnan, acts)
        error("actuals contains NaN values. Clean data before processing.")
    end

    if length(preds) != length(acts)
        error("Arrays must have same length. " *
              "predictions: $(length(preds)), actuals: $(length(acts))")
    end

    if isempty(preds)
        return MoveConditionalResult(
            NaN, NaN, NaN,
            0, 0, 0,
            NaN, 0.0
        )
    end

    # Compute or use provided threshold
    actual_threshold = if isnothing(threshold)
        compute_move_threshold(acts; percentile=threshold_percentile)
    else
        threshold
    end

    # Classify moves based on ACTUALS
    classifications = classify_moves(acts, actual_threshold)

    # Create masks
    up_mask = classifications .== UP
    down_mask = classifications .== DOWN
    flat_mask = classifications .== FLAT

    # Counts
    n_up = sum(up_mask)
    n_down = sum(down_mask)
    n_flat = sum(flat_mask)

    # Conditional MAEs
    mae_up = if n_up > 0
        mean(abs.(preds[up_mask] .- acts[up_mask]))
    else
        @warn "No UP moves in sample. mae_up will be NaN."
        NaN
    end

    mae_down = if n_down > 0
        mean(abs.(preds[down_mask] .- acts[down_mask]))
    else
        @warn "No DOWN moves in sample. mae_down will be NaN."
        NaN
    end

    mae_flat = if n_flat > 0
        mean(abs.(preds[flat_mask] .- acts[flat_mask]))
    else
        @warn "No FLAT periods in sample. mae_flat will be NaN."
        NaN
    end

    # Compute MC-SS on moves only (UP + DOWN)
    move_mask = up_mask .| down_mask
    n_moves_count = n_up + n_down

    skill_score = if n_moves_count > 0
        # Model MAE on moves
        model_mae_moves = mean(abs.(preds[move_mask] .- acts[move_mask]))

        # Persistence MAE on moves (persistence predicts 0, so error = |actual|)
        persistence_mae_moves = mean(abs.(acts[move_mask]))

        # Guard against division by zero
        epsilon = _get_scale_aware_epsilon(acts[move_mask])
        if persistence_mae_moves > epsilon
            1.0 - (model_mae_moves / persistence_mae_moves)
        else
            @warn "Persistence MAE on moves is near zero. skill_score will be NaN."
            NaN
        end
    else
        @warn "No moves (UP or DOWN) in sample. skill_score will be NaN."
        NaN
    end

    return MoveConditionalResult(
        mae_up, mae_down, mae_flat,
        n_up, n_down, n_flat,
        skill_score, actual_threshold
    )
end


# =============================================================================
# Direction Accuracy
# =============================================================================

"""
    compute_direction_accuracy(predictions, actuals; move_threshold=nothing)

Compute directional accuracy.

# Arguments
- `predictions::AbstractVector`: Model predictions
- `actuals::AbstractVector`: Actual values
- `move_threshold::Union{Float64, Nothing}=nothing`:
  If provided, uses 3-class (UP/DOWN/FLAT) comparison.
  If nothing, uses 2-class (positive/negative sign) comparison.

# Returns
- `Float64`: Direction accuracy as fraction (0-1)

# Notes
**Without threshold (2-class)**:
- Compares signs: both positive OR both negative = correct
- Zero actuals are excluded
- Persistence (predicts 0) gets 0% accuracy

**With threshold (3-class)**:
- UP: value > threshold
- DOWN: value < -threshold
- FLAT: |value| <= threshold
- Correct if both have same class (including both FLAT)
- Persistence (predicts 0 = FLAT) gets credit when actual is also FLAT

The 3-class version provides a meaningful baseline for persistence model
comparison.

# Knowledge Tier
[T1] Directional accuracy from Pesaran & Timmermann (1992).
"""
function compute_direction_accuracy(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    move_threshold::Union{Float64, Nothing} = nothing
)::Float64
    preds = collect(Float64, predictions)
    acts = collect(Float64, actuals)

    if length(preds) != length(acts)
        error("Arrays must have same length. " *
              "predictions: $(length(preds)), actuals: $(length(acts))")
    end

    if isempty(preds)
        return 0.0
    end

    if !isnothing(move_threshold)
        # 3-class comparison
        pred_dirs = classify_moves(preds, move_threshold)
        actual_dirs = classify_moves(acts, move_threshold)

        correct = pred_dirs .== actual_dirs
        return mean(correct)
    end

    # 2-class (sign) comparison
    epsilon = 1e-10
    nonzero_mask = abs.(acts) .> epsilon

    if sum(nonzero_mask) == 0
        return 0.0
    end

    pred_signs = sign.(preds[nonzero_mask])
    actual_signs = sign.(acts[nonzero_mask])

    return mean(pred_signs .== actual_signs)
end


# =============================================================================
# Move-Only Metrics
# =============================================================================

"""
    compute_move_only_mae(predictions, actuals, threshold)

Compute MAE only on moves (excluding FLAT).

# Arguments
- `predictions::AbstractVector`: Model predictions
- `actuals::AbstractVector`: Actual values
- `threshold::Float64`: Move threshold

# Returns
- `Tuple{Float64, Int}`: (mae, n_moves) - MAE on moves and count of moves

# Notes
This isolates model performance on "significant" moves,
excluding periods where nothing happened (FLAT).
"""
function compute_move_only_mae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    threshold::Float64
)::Tuple{Float64, Int}
    preds = collect(Float64, predictions)
    acts = collect(Float64, actuals)

    if threshold < 0
        error("threshold must be non-negative, got $threshold")
    end

    if length(preds) != length(acts)
        error("Arrays must have same length.")
    end

    # Identify moves (UP or DOWN, not FLAT)
    move_mask = abs.(acts) .> threshold
    n_moves = sum(move_mask)

    if n_moves == 0
        return (NaN, 0)
    end

    mae = mean(abs.(preds[move_mask] .- acts[move_mask]))
    return (mae, n_moves)
end


"""
    compute_persistence_mae(actuals; threshold=nothing)

Compute MAE of persistence baseline.

Persistence predicts 0 (no change), so MAE = mean(|actual|).

# Arguments
- `actuals::AbstractVector`: Actual values
- `threshold::Union{Float64, Nothing}=nothing`:
  If provided, computes MAE only on moves

# Returns
- `Float64`: Persistence baseline MAE
"""
function compute_persistence_mae(
    actuals::AbstractVector{<:Real};
    threshold::Union{Float64, Nothing} = nothing
)::Float64
    acts = collect(Float64, actuals)

    if isempty(acts)
        return NaN
    end

    if !isnothing(threshold)
        move_mask = abs.(acts) .> threshold
        if sum(move_mask) == 0
            return NaN
        end
        return mean(abs.(acts[move_mask]))
    end

    return mean(abs.(acts))
end
