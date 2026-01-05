# =============================================================================
# Event-Aware Metrics (Brier Score, PR-AUC)
# =============================================================================

"""
Event-aware metrics for direction prediction with proper calibration.

Key concepts:
- **Brier score**: Probabilistic direction accuracy (2 and 3-class)
- **Murphy decomposition**: Reliability, resolution, uncertainty components

# Knowledge Tiers
- [T1] Brier (1950). Verification of forecasts expressed in terms of probability
- [T1] Murphy (1973). A new vector partition of the probability score
- [T2] 3-class direction classification with FLAT from myga-forecasting-v2
"""

using Statistics

# =============================================================================
# Helper: Murphy Decomposition
# =============================================================================

"""
    _compute_brier_decomposition(pred_probs, actual_binary; n_bins=10)

Compute Brier score decomposition (Murphy 1973).

Returns (reliability, resolution, uncertainty).

Murphy (1973) decomposition:
    BS = Reliability - Resolution + Uncertainty

Where:
- Reliability = (1/N) × sum_k n_k × (f_k - o_k)²  (calibration error)
- Resolution = (1/N) × sum_k n_k × (o_k - o_bar)²  (refinement)
- Uncertainty = o_bar × (1 - o_bar)  (inherent uncertainty)
"""
function _compute_brier_decomposition(
    pred_probs::Vector{Float64},
    actual_binary::Vector{Float64};
    n_bins::Int = 10
)::Tuple{Float64, Float64, Float64}
    n = length(pred_probs)
    if n == 0
        return (NaN, NaN, NaN)
    end

    # Overall positive rate
    o_bar = mean(actual_binary)

    # Uncertainty is base rate variance
    uncertainty = o_bar * (1 - o_bar)

    # Create bins
    bin_edges = range(0, 1, length=n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    for b in 1:n_bins
        # Get samples in this bin
        if b < n_bins
            mask = (pred_probs .>= bin_edges[b]) .& (pred_probs .< bin_edges[b+1])
        else
            mask = (pred_probs .>= bin_edges[b]) .& (pred_probs .<= bin_edges[b+1])
        end

        n_b = sum(mask)
        if n_b == 0
            continue
        end

        # Mean forecast probability in bin
        f_b = mean(pred_probs[mask])
        # Observed positive rate in bin
        o_b = mean(actual_binary[mask])

        # Reliability contribution: weighted squared calibration error
        reliability += n_b * (f_b - o_b)^2
        # Resolution contribution: weighted squared deviation from climatology
        resolution += n_b * (o_b - o_bar)^2
    end

    # Normalize by total samples
    reliability /= n
    resolution /= n

    return (reliability, resolution, uncertainty)
end


# =============================================================================
# Brier Score
# =============================================================================

"""
    compute_direction_brier(pred_probs, actual_directions; n_classes=2)

Compute Brier score for direction prediction.

# Arguments
- `pred_probs::AbstractVector` or `AbstractMatrix`: Predicted probabilities
  - n_classes=2: (n_samples,) probability of positive direction
  - n_classes=3: (n_samples, 3) probabilities for [DOWN, FLAT, UP]
- `actual_directions::AbstractVector{<:Integer}`: Actual directions
  - n_classes=2: 0 = negative, 1 = positive
  - n_classes=3: 0 = DOWN, 1 = FLAT, 2 = UP
- `n_classes::Int=2`: Number of classes (2 or 3)

# Returns
- `BrierScoreResult`: Brier score with Murphy decomposition

# Notes
Brier score = (1/N) × sum((p_i - o_i)²)

For 3-class, we compute multiclass Brier:
BS = (1/N) × sum_i sum_k (p_ik - o_ik)²

# Knowledge Tier
[T1] Brier (1950); Murphy (1973).
"""
function compute_direction_brier(
    pred_probs::AbstractVector{<:Real},
    actual_directions::AbstractVector{<:Integer};
    n_classes::Int = 2
)::BrierScoreResult
    if n_classes != 2
        error("For n_classes=3, use the matrix form of compute_direction_brier")
    end

    probs = collect(Float64, pred_probs)
    dirs = collect(Int, actual_directions)
    n = length(dirs)

    if n == 0
        return BrierScoreResult(NaN, NaN, NaN, NaN, 0, n_classes)
    end

    if length(probs) != n
        error("Length mismatch: pred_probs has $(length(probs)), " *
              "actual_directions has $n")
    end

    # Validate probability range
    if any(probs .< 0) || any(probs .> 1)
        error("Probabilities must be in [0, 1]")
    end

    # Convert to float for Brier calculation
    actual_onehot = Float64.(dirs)

    # Brier score
    brier = mean((probs .- actual_onehot).^2)

    # Base rate
    base_rate = mean(actual_onehot)
    uncertainty = base_rate * (1 - base_rate)

    # Murphy decomposition
    reliability, resolution, _ = _compute_brier_decomposition(probs, actual_onehot)

    return BrierScoreResult(brier, reliability, resolution, uncertainty, n, n_classes)
end


"""
    compute_direction_brier(pred_probs, actual_directions; n_classes=3)

Compute 3-class Brier score for direction prediction.

# Arguments
- `pred_probs::AbstractMatrix`: Shape (n_samples, 3) probabilities for [DOWN, FLAT, UP]
- `actual_directions::AbstractVector{<:Integer}`: 0=DOWN, 1=FLAT, 2=UP
- `n_classes::Int=3`: Must be 3 for this method
"""
function compute_direction_brier(
    pred_probs::AbstractMatrix{<:Real},
    actual_directions::AbstractVector{<:Integer};
    n_classes::Int = 3
)::BrierScoreResult
    if n_classes != 3
        error("Matrix form requires n_classes=3")
    end

    probs = collect(Float64, pred_probs)
    dirs = collect(Int, actual_directions)
    n = length(dirs)

    if n == 0
        return BrierScoreResult(NaN, NaN, NaN, NaN, 0, n_classes)
    end

    if size(probs, 1) != n || size(probs, 2) != 3
        error("For n_classes=3, pred_probs should be (n_samples, 3). " *
              "Got size $(size(probs))")
    end

    # Validate probability sums
    prob_sums = sum(probs, dims=2)
    if !all(isapprox.(prob_sums, 1.0, atol=1e-6))
        error("Probability vectors must sum to 1.0. " *
              "Got sums ranging from $(minimum(prob_sums)) to $(maximum(prob_sums))")
    end

    # One-hot encode actuals
    actual_onehot = zeros(n, 3)
    for i in 1:n
        a = dirs[i]
        if !(0 <= a <= 2)
            error("For n_classes=3, actual_directions must be 0, 1, or 2. " *
                  "Got $a at index $i")
        end
        actual_onehot[i, a + 1] = 1.0
    end

    # Multiclass Brier score
    brier = mean(sum((probs .- actual_onehot).^2, dims=2))

    # Base rates per class
    class_rates = mean(actual_onehot, dims=1)[:]
    uncertainty = sum(class_rates .* (1 .- class_rates))

    # Murphy decomposition per class (one-vs-rest), then aggregate
    reliability = 0.0
    resolution = 0.0
    for k in 1:3
        rel_k, res_k, _ = _compute_brier_decomposition(
            probs[:, k], actual_onehot[:, k]
        )
        reliability += rel_k
        resolution += res_k
    end

    return BrierScoreResult(brier, reliability, resolution, uncertainty, n, n_classes)
end


# =============================================================================
# Brier Skill Score
# =============================================================================

"""
    skill_score(result::BrierScoreResult)

Compute Brier skill score relative to climatology.

BSS = 1 - (BS / uncertainty)

Positive values indicate skill over climatology.
"""
function skill_score(result::BrierScoreResult)::Float64
    if result.uncertainty == 0 || isnan(result.uncertainty)
        return 0.0
    end
    return 1.0 - (result.brier_score / result.uncertainty)
end


# =============================================================================
# Calibrated Direction Brier
# =============================================================================

"""
    compute_calibrated_direction_brier(pred_probs, actual_directions; n_bins=10)

Compute Brier score with reliability diagram data.

Returns Brier score along with binned calibration data
for plotting reliability diagrams.

# Arguments
- `pred_probs::AbstractVector`: Predicted probabilities (1D, for positive class)
- `actual_directions::AbstractVector{<:Integer}`: Actual binary outcomes (0 or 1)
- `n_bins::Int=10`: Number of bins for calibration

# Returns
- `Tuple{Float64, Vector{Float64}, Vector{Float64}}`:
  (brier_score, bin_means, bin_true_fractions)
  NaN values indicate empty bins.

# Notes
Reliability diagrams show calibration:
- X-axis: Mean predicted probability in each bin
- Y-axis: Fraction of positives in each bin
- Perfect calibration: diagonal line
"""
function compute_calibrated_direction_brier(
    pred_probs::AbstractVector{<:Real},
    actual_directions::AbstractVector{<:Integer};
    n_bins::Int = 10
)::Tuple{Float64, Vector{Float64}, Vector{Float64}}
    probs = collect(Float64, pred_probs)
    dirs = collect(Float64, actual_directions)

    if length(probs) != length(dirs)
        error("Length mismatch: pred_probs has $(length(probs)), " *
              "actual_directions has $(length(dirs))")
    end

    if n_bins < 1
        error("n_bins must be >= 1, got $n_bins")
    end

    if isempty(probs)
        return (NaN, Float64[], Float64[])
    end

    # Compute Brier score
    brier = mean((probs .- dirs).^2)

    # Bin predictions
    bin_edges = range(0, 1, length=n_bins + 1)
    bin_means = fill(NaN, n_bins)
    bin_true_fractions = fill(NaN, n_bins)

    for i in 1:n_bins
        if i < n_bins
            mask = (probs .>= bin_edges[i]) .& (probs .< bin_edges[i + 1])
        else
            mask = (probs .>= bin_edges[i]) .& (probs .<= bin_edges[i + 1])
        end

        if sum(mask) > 0
            bin_means[i] = mean(probs[mask])
            bin_true_fractions[i] = mean(dirs[mask])
        end
    end

    return (brier, bin_means, bin_true_fractions)
end
