# =============================================================================
# Pesaran-Timmermann Test for Directional Accuracy
# =============================================================================

"""
    pt_test(actual, predicted; move_threshold=nothing)

Pesaran-Timmermann test for directional accuracy.

Tests whether the model's ability to predict direction (sign)
is significantly better than random guessing.

# Arguments
- `actual`: Actual values (typically changes/returns)
- `predicted`: Predicted values (typically changes/returns)
- `move_threshold`: If provided, uses 3-class classification (UP/DOWN/FLAT):
  - UP: value > threshold
  - DOWN: value < -threshold
  - FLAT: |value| <= threshold
  If nothing, uses 2-class (positive/negative sign).

# Returns
`PTTestResult` with accuracy, expected, and significance.

# Notes
H0: Direction predictions are no better than random (independence)
H1: Direction predictions have skill (one-sided test)

The test accounts for marginal probabilities of directions in both
actual and predicted series.

# References
[T1] Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test
     of predictive performance. Journal of Business & Economic Statistics,
     10(4), 461-465.

# Example
```julia
result = pt_test(actual_changes, pred_changes, move_threshold=0.01)
println("Accuracy: \$(round(result.accuracy * 100, digits=1))%, Skill: \$(round(skill(result) * 100, digits=1))%")
```
"""
function pt_test(
    actual::AbstractVector{<:Real},
    predicted::AbstractVector{<:Real};
    move_threshold::Union{Float64, Nothing} = nothing
)
    # Validate inputs
    @assert length(actual) == length(predicted) "Arrays must have same length. Got actual=$(length(actual)), predicted=$(length(predicted))"
    @assert !any(isnan, actual) "actual contains NaN values"
    @assert !any(isnan, predicted) "predicted contains NaN values"

    n = length(actual)
    @assert n >= 30 "Insufficient samples for PT test. Need >= 30, got $n"

    # Classify directions
    if !isnothing(move_threshold)
        # 3-class: UP=1, DOWN=-1, FLAT=0
        function classify(values::AbstractVector, threshold::Float64)
            classes = zeros(Int8, length(values))
            classes[values .> threshold] .= 1   # UP
            classes[values .< -threshold] .= -1 # DOWN
            return classes
        end

        actual_class = classify(actual, move_threshold)
        pred_class = classify(predicted, move_threshold)
        n_classes = 3

        # Compute directional accuracy
        correct = actual_class .== pred_class
        n_effective = n
        p_hat = mean(correct)

        # Marginal probabilities for each class
        p_y_up = mean(actual_class .== 1)
        p_y_down = mean(actual_class .== -1)
        p_y_flat = mean(actual_class .== 0)

        p_x_up = mean(pred_class .== 1)
        p_x_down = mean(pred_class .== -1)
        p_x_flat = mean(pred_class .== 0)

        # Expected accuracy under independence (null)
        p_star = p_y_up * p_x_up + p_y_down * p_x_down + p_y_flat * p_x_flat

        # Variance estimates (simplified for 3-class) [T3]
        var_p_hat = p_star * (1 - p_star) / n_effective
        var_p_star = p_star * (1 - p_star) / n_effective * 4

    else
        # 2-class: sign comparison (exclude zeros)
        actual_class = sign.(actual)
        pred_class = sign.(predicted)
        n_classes = 2

        nonzero_mask = actual_class .!= 0
        n_effective = sum(nonzero_mask)

        if n_effective == 0
            @warn "PT test has no non-zero observations for 2-class mode. Returning pvalue=1.0."
            return PTTestResult(
                NaN,  # statistic
                1.0,  # pvalue
                0.0,  # accuracy
                0.5,  # expected
                n,
                2
            )
        end

        correct = actual_class .== pred_class
        p_hat = mean(correct[nonzero_mask])

        # Marginal probabilities (on non-zero samples)
        p_y_pos = mean(actual[nonzero_mask] .> 0)
        p_x_pos = mean(predicted[nonzero_mask] .> 0)

        # Expected accuracy under independence
        p_star = p_y_pos * p_x_pos + (1 - p_y_pos) * (1 - p_x_pos)

        # Variance estimates (2-class formula from PT 1992, equation 8) [T1]
        var_p_hat = p_star * (1 - p_star) / n_effective
        term1 = (2 * p_y_pos - 1)^2 * p_x_pos * (1 - p_x_pos) / n_effective
        term2 = (2 * p_x_pos - 1)^2 * p_y_pos * (1 - p_y_pos) / n_effective
        term3 = 4 * p_y_pos * p_x_pos * (1 - p_y_pos) * (1 - p_x_pos) / n_effective
        var_p_star = term1 + term2 + term3
    end

    # Total variance under null
    var_total = var_p_hat + var_p_star

    if var_total <= 0
        @warn "PT test total variance is non-positive (var_total=$var_total). Returning pvalue=1.0."
        return PTTestResult(
            NaN,      # statistic
            1.0,      # pvalue
            p_hat,    # accuracy
            p_star,   # expected
            n_effective,
            n_classes
        )
    end

    # PT statistic (z-score)
    pt_stat = (p_hat - p_star) / sqrt(var_total)

    # One-sided p-value (testing if better than random)
    pvalue = 1 - cdf(Normal(), pt_stat)

    return PTTestResult(
        pt_stat,
        pvalue,
        p_hat,
        p_star,
        n_effective,
        n_classes
    )
end
