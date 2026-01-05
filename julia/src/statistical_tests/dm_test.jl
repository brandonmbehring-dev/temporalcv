# =============================================================================
# Diebold-Mariano Test for Equal Predictive Accuracy
# =============================================================================

"""
    dm_test(errors_1, errors_2; h=1, loss=:squared, alternative=:two_sided, harvey_correction=true)

Diebold-Mariano test for equal predictive accuracy.

Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t) is the loss differential.

# Arguments
- `errors_1`: Forecast errors from model 1 (actual - prediction)
- `errors_2`: Forecast errors from model 2 (baseline)
- `h`: Forecast horizon. Used for HAC bandwidth (h-1) and Harvey adjustment.
- `loss`: Loss function (:squared or :absolute), default :squared
- `alternative`: Alternative hypothesis (:two_sided, :less, :greater)
  - :two_sided: Models have different accuracy
  - :less: Model 1 more accurate (lower loss)
  - :greater: Model 2 more accurate (model 1 has higher loss)
- `harvey_correction`: Apply Harvey et al. (1997) small-sample adjustment.

# Returns
`DMTestResult` with test statistic, p-value, and diagnostics.

# Notes
Harvey adjustment: DM_adj = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)

For h>1 step forecasts, errors are MA(h-1) and HAC variance is required.
The Harvey adjustment corrects for small-sample bias.

# References
[T1] Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy.
     Journal of Business & Economic Statistics, 13(3), 253-263.
[T1] Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality
     of prediction mean squared errors. International Journal of Forecasting,
     13(2), 281-291.

# Example
```julia
result = dm_test(model_errors, baseline_errors, h=2, alternative=:less)
if significant_at_05(result)
    println("Model significantly better than baseline")
end
```
"""
function dm_test(
    errors_1::AbstractVector{<:Real},
    errors_2::AbstractVector{<:Real};
    h::Int = 1,
    loss::Symbol = :squared,
    alternative::Symbol = :two_sided,
    harvey_correction::Bool = true
)
    # Validate inputs
    @assert length(errors_1) == length(errors_2) "Error arrays must have same length. Got $(length(errors_1)) and $(length(errors_2))"
    @assert !any(isnan, errors_1) "errors_1 contains NaN values"
    @assert !any(isnan, errors_2) "errors_2 contains NaN values"

    n = length(errors_1)
    @assert n >= 30 "Insufficient samples for reliable DM test. Need >= 30, got $n"
    @assert h >= 1 "Horizon h must be >= 1, got $h"
    @assert loss in (:squared, :absolute) "Unknown loss function: $loss. Use :squared or :absolute"
    @assert alternative in (:two_sided, :less, :greater) "Unknown alternative: $alternative"

    # Compute loss differential
    if loss == :squared
        loss_1 = errors_1.^2
        loss_2 = errors_2.^2
    else  # :absolute
        loss_1 = abs.(errors_1)
        loss_2 = abs.(errors_2)
    end

    d = loss_1 .- loss_2  # Positive = model 1 has higher loss (worse)
    d_bar = mean(d)

    # HAC variance with h-1 bandwidth for h-step forecasts
    bandwidth = max(0, h - 1)
    var_d = compute_hac_variance(d, bandwidth=bandwidth)

    # Handle degenerate case
    if var_d <= 0
        @warn "DM test variance is non-positive (var_d=$var_d). Returning pvalue=1.0."
        return DMTestResult(
            NaN,      # statistic
            1.0,      # pvalue
            h,
            n,
            loss,
            alternative,
            harvey_correction,
            d_bar
        )
    end

    # DM statistic
    dm_stat = d_bar / sqrt(var_d)

    # Harvey et al. (1997) small-sample adjustment
    if harvey_correction
        adjustment = sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        dm_stat = dm_stat * adjustment
    end

    # Compute p-value using t-distribution for Harvey, normal otherwise
    if harvey_correction
        dist = TDist(n - 1)
        if alternative == :two_sided
            pvalue = 2 * (1 - cdf(dist, abs(dm_stat)))
        elseif alternative == :less
            pvalue = cdf(dist, dm_stat)
        else  # :greater
            pvalue = 1 - cdf(dist, dm_stat)
        end
    else
        dist = Normal()
        if alternative == :two_sided
            pvalue = 2 * (1 - cdf(dist, abs(dm_stat)))
        elseif alternative == :less
            pvalue = cdf(dist, dm_stat)
        else  # :greater
            pvalue = 1 - cdf(dist, dm_stat)
        end
    end

    return DMTestResult(
        dm_stat,
        pvalue,
        h,
        n,
        loss,
        alternative,
        harvey_correction,
        d_bar
    )
end
