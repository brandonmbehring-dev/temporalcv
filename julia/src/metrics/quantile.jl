# =============================================================================
# Quantile and Interval Metrics
# =============================================================================

"""
Quantile and interval metrics for probabilistic forecasts.

Implements proper scoring rules:
- **Pinball loss**: Quantile regression loss (asymmetric around quantile)
- **CRPS**: Continuous Ranked Probability Score (proper scoring rule)
- **Interval score**: Proper scoring rule for prediction intervals
- **Quantile coverage**: Empirical coverage of prediction intervals

# Knowledge Tiers
- [T1] Pinball loss formula and properties (Koenker & Bassett 1978)
- [T1] CRPS as proper scoring rule (Gneiting & Raftery 2007)
- [T1] Interval score (Gneiting & Raftery 2007, equation 43)
- [T2] CRPS sample approximation (standard practice)

# References
- Koenker, R. & Bassett, G. (1978). Regression quantiles. Econometrica.
- Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules.
  JASA, 102(477), 359-378.
"""

using Statistics

# =============================================================================
# Pinball Loss
# =============================================================================

"""
    compute_pinball_loss(actuals, quantile_preds, tau)

Compute pinball loss (quantile loss) for quantile regression.

The pinball loss is asymmetric around the quantile, penalizing
under-predictions more heavily for high quantiles and over-predictions
more heavily for low quantiles.

# Arguments
- `actuals::AbstractVector`: Actual observed values
- `quantile_preds::AbstractVector`: Predicted values at quantile tau
- `tau::Float64`: Quantile level in (0, 1). E.g., tau=0.9 for 90th percentile

# Returns
- `Float64`: Mean pinball loss

# Notes
The pinball loss is defined as:

    L(y, q; τ) = τ × max(y - q, 0) + (1 - τ) × max(q - y, 0)

Lower values indicate better quantile predictions.

# Knowledge Tier
[T1] Koenker & Bassett (1978).
"""
function compute_pinball_loss(
    actuals::AbstractVector{<:Real},
    quantile_preds::AbstractVector{<:Real},
    tau::Float64
)::Float64
    if !(0 < tau < 1)
        error("tau must be in (0, 1), got $tau")
    end

    if length(actuals) != length(quantile_preds)
        error("Array lengths must match. " *
              "Got actuals=$(length(actuals)), quantile_preds=$(length(quantile_preds))")
    end

    if isempty(actuals)
        error("Arrays cannot be empty")
    end

    acts = collect(Float64, actuals)
    preds = collect(Float64, quantile_preds)

    # Compute pinball loss
    errors = acts .- preds
    loss = map(errors) do e
        e >= 0 ? tau * e : (tau - 1) * e
    end

    return mean(loss)
end


# =============================================================================
# CRPS (Continuous Ranked Probability Score)
# =============================================================================

"""
    compute_crps(actuals, forecast_samples)

Compute Continuous Ranked Probability Score (CRPS).

CRPS is a proper scoring rule for probabilistic forecasts. It measures
the compatibility between the forecast distribution and the observation.

# Arguments
- `actuals::AbstractVector`: Actual observed values, shape (n,)
- `forecast_samples::AbstractMatrix`: Samples from forecast distribution,
  shape (n, n_samples). Each row contains samples for one observation.

# Returns
- `Float64`: Mean CRPS across all observations

# Notes
CRPS is computed as:

    CRPS = E|X - y| - 0.5 × E|X - X'|

where X and X' are independent draws from the forecast distribution
and y is the observation.

Lower CRPS indicates better probabilistic calibration. CRPS has the
same units as the observations.

# Knowledge Tier
[T1] Gneiting & Raftery (2007).
"""
function compute_crps(
    actuals::AbstractVector{<:Real},
    forecast_samples::AbstractMatrix{<:Real}
)::Float64
    acts = collect(Float64, actuals)
    samples = collect(Float64, forecast_samples)

    n_obs = length(acts)
    if size(samples, 1) != n_obs
        error("Number of observations must match. " *
              "Got $(n_obs) actuals but $(size(samples, 1)) sample rows")
    end

    if n_obs == 0
        error("Arrays cannot be empty")
    end

    # Sample-based CRPS approximation
    # CRPS = E|X - y| - 0.5 * E|X - X'|
    crps_values = zeros(n_obs)

    for i in 1:n_obs
        sample_row = samples[i, :]
        y = acts[i]

        # E|X - y|
        term1 = mean(abs.(sample_row .- y))

        # E|X - X'| using sorted samples
        n_samples = length(sample_row)
        if n_samples > 1
            sorted_samples = sort(sample_row)
            # E|X - X'| = 2 * sum_i (2*i - n - 1) * x_{(i)} / n^2
            indices = collect(1:n_samples)
            term2 = 2 * sum((2 .* indices .- n_samples .- 1) .* sorted_samples) / (n_samples^2)
        else
            term2 = 0.0
        end

        crps_values[i] = term1 - 0.5 * abs(term2)
    end

    return mean(crps_values)
end


# =============================================================================
# Interval Score
# =============================================================================

"""
    compute_interval_score(actuals, lower, upper, alpha)

Compute interval score for prediction intervals.

The interval score is a proper scoring rule for prediction intervals,
penalizing both interval width and coverage failures.

# Arguments
- `actuals::AbstractVector`: Actual observed values
- `lower::AbstractVector`: Lower bounds of prediction intervals
- `upper::AbstractVector`: Upper bounds of prediction intervals
- `alpha::Float64`: Nominal non-coverage rate in (0, 1).
  E.g., alpha=0.05 for 95% intervals.

# Returns
- `Float64`: Mean interval score

# Notes
The interval score is defined as (Gneiting & Raftery 2007, equation 43):

    IS(l, u; y) = (u - l) + (2/α) × (l - y) × I(y < l) + (2/α) × (y - u) × I(y > u)

Components:
- (u - l): Penalizes wide intervals
- (2/α) × (l - y) × I(y < l): Penalizes under-coverage (actual below lower)
- (2/α) × (y - u) × I(y > u): Penalizes under-coverage (actual above upper)

Lower interval scores indicate better interval forecasts.

# Knowledge Tier
[T1] Gneiting & Raftery (2007), equation 43.
"""
function compute_interval_score(
    actuals::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    alpha::Float64
)::Float64
    if !(0 < alpha < 1)
        error("alpha must be in (0, 1), got $alpha")
    end

    n = length(actuals)
    if length(lower) != n || length(upper) != n
        error("Array lengths must match. " *
              "Got actuals=$n, lower=$(length(lower)), upper=$(length(upper))")
    end

    if n == 0
        error("Arrays cannot be empty")
    end

    acts = collect(Float64, actuals)
    lo = collect(Float64, lower)
    up = collect(Float64, upper)

    # Check lower <= upper
    violations = sum(lo .> up)
    if violations > 0
        error("lower must be <= upper for all observations. Found $violations violations")
    end

    # Compute interval score
    width = up .- lo
    penalty_factor = 2.0 / alpha

    # Penalty for actuals below lower bound
    below = acts .< lo
    penalty_below = penalty_factor .* (lo .- acts) .* below

    # Penalty for actuals above upper bound
    above = acts .> up
    penalty_above = penalty_factor .* (acts .- up) .* above

    scores = width .+ penalty_below .+ penalty_above

    return mean(scores)
end


"""
    compute_winkler_score(actuals, lower, upper, alpha)

Compute Winkler score for prediction intervals.

Alias for `compute_interval_score`. The Winkler score (Winkler 1972)
is the original formulation.

# Knowledge Tier
[T1] Winkler (1972); Gneiting & Raftery (2007).
"""
function compute_winkler_score(
    actuals::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real},
    alpha::Float64
)::Float64
    return compute_interval_score(actuals, lower, upper, alpha)
end


# =============================================================================
# Quantile Coverage
# =============================================================================

"""
    compute_quantile_coverage(actuals, lower, upper)

Compute empirical coverage of prediction intervals.

Calculates the fraction of observations that fall within their
prediction intervals.

# Arguments
- `actuals::AbstractVector`: Actual observed values
- `lower::AbstractVector`: Lower bounds of prediction intervals
- `upper::AbstractVector`: Upper bounds of prediction intervals

# Returns
- `Float64`: Empirical coverage rate in [0, 1]

# Notes
Coverage is computed as:

    coverage = mean(I(lower <= actual <= upper))

For a well-calibrated (1-α) prediction interval, empirical coverage
should be approximately (1-α).
"""
function compute_quantile_coverage(
    actuals::AbstractVector{<:Real},
    lower::AbstractVector{<:Real},
    upper::AbstractVector{<:Real}
)::Float64
    n = length(actuals)
    if length(lower) != n || length(upper) != n
        error("Array lengths must match. " *
              "Got actuals=$n, lower=$(length(lower)), upper=$(length(upper))")
    end

    if n == 0
        error("Arrays cannot be empty")
    end

    acts = collect(Float64, actuals)
    lo = collect(Float64, lower)
    up = collect(Float64, upper)

    # Compute coverage
    covered = (acts .>= lo) .& (acts .<= up)

    return mean(covered)
end
