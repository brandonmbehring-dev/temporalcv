# =============================================================================
# Asymmetric Loss Functions
# =============================================================================

"""
Asymmetric loss functions that penalize over- and under-predictions differently.

# Knowledge Tiers
- [T1] LinEx loss (Varian 1975, Zellner 1986)
- [T1] Huber loss (Huber 1964)
- [T2] Asymmetric MAPE (common practice in forecasting)
- [T2] Directional loss (common in trading/financial applications)

# References
- Varian, H.R. (1975). A Bayesian approach to real estate assessment.
- Zellner, A. (1986). Bayesian estimation and prediction using asymmetric
  loss functions. JASA, 81(394), 446-451.
- Huber, P.J. (1964). Robust estimation of a location parameter.
  Annals of Mathematical Statistics, 35(1), 73-101.
"""

# =============================================================================
# LinEx Loss
# =============================================================================

"""
    compute_linex_loss(predictions, actuals; a=1.0, b=1.0) -> Float64

Compute LinEx (linear-exponential) asymmetric loss.

The LinEx loss function penalizes errors asymmetrically: one direction
receives exponential penalties while the other receives linear penalties.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual observed values
- `a::Float64=1.0`: Asymmetry parameter
  - a > 0: under-predictions (pred < actual) penalized exponentially
  - a < 0: over-predictions (pred > actual) penalized exponentially
  - |a| controls the degree of asymmetry (larger = more asymmetric)
- `b::Float64=1.0`: Scaling parameter (must be > 0)

# Returns
Mean LinEx loss.

# Formula
```
L(e) = b × (exp(a × e) - a × e - 1)
```
where e = actual - prediction (error).

# Properties
- Always non-negative (L(0) = 0)
- Convex and asymmetric around zero
- Approximately quadratic near zero
- Exponential growth on one side, linear on the other

# Example
```julia
actuals = [10.0, 20.0, 30.0]
predictions = [12.0, 18.0, 28.0]
# Under-predictions costly (a > 0)
loss = compute_linex_loss(predictions, actuals; a=0.5)
```
"""
function compute_linex_loss(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    a::Float64 = 1.0,
    b::Float64 = 1.0
)::Float64
    if a == 0.0
        error("a cannot be 0 (would make loss symmetric)")
    end

    if b <= 0.0
        error("b must be > 0, got $b")
    end

    if length(predictions) != length(actuals)
        error("Array lengths must match. Got predictions=$(length(predictions)), actuals=$(length(actuals))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    # Error: actual - prediction
    # Positive error = under-prediction (pred < actual)
    # Negative error = over-prediction (pred > actual)
    errors = actuals .- predictions

    # LinEx: b × (exp(a × e) - a × e - 1)
    # Clip to prevent overflow for large |a × e|
    a_e = clamp.(a .* errors, -700.0, 700.0)  # exp(700) is near float max
    loss = b .* (exp.(a_e) .- a .* errors .- 1.0)

    return mean(loss)
end


# =============================================================================
# Asymmetric MAPE
# =============================================================================

"""
    compute_asymmetric_mape(predictions, actuals; alpha=0.5, epsilon=1e-8) -> Float64

Compute asymmetric MAPE with different over/under penalties.

Asymmetric MAPE allows different weights for over-predictions vs
under-predictions, useful when the costs are asymmetric.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual observed values
- `alpha::Float64=0.5`: Weight for under-predictions in [0, 1]
  - alpha = 0.5: symmetric (standard MAPE behavior)
  - alpha > 0.5: under-predictions penalized more
  - alpha < 0.5: over-predictions penalized more
- `epsilon::Float64=1e-8`: Small constant to prevent division by zero

# Returns
Asymmetric MAPE as a fraction (multiply by 100 for percentage).

# Formula
```
AMAPE = mean( w(e) × |e| / |actual| )
```
where:
- e = actual - prediction
- w(e) = alpha if e > 0 (under-prediction), (1-alpha) if e < 0

# Example
```julia
actuals = [100.0, 200.0, 300.0]
predictions = [110.0, 180.0, 280.0]
# Penalize under-predictions more (alpha=0.7)
amape = compute_asymmetric_mape(predictions, actuals; alpha=0.7)
```
"""
function compute_asymmetric_mape(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    alpha::Float64 = 0.5,
    epsilon::Float64 = 1e-8
)::Float64
    if !(0.0 <= alpha <= 1.0)
        error("alpha must be in [0, 1], got $alpha")
    end

    if length(predictions) != length(actuals)
        error("Array lengths must match. Got predictions=$(length(predictions)), actuals=$(length(actuals))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    errors = actuals .- predictions
    abs_errors = abs.(errors)
    abs_actuals = abs.(actuals) .+ epsilon

    # Weight based on error direction
    weights = [e > 0 ? alpha : (1.0 - alpha) for e in errors]

    # Weighted percentage error
    weighted_ape = weights .* abs_errors ./ abs_actuals

    return mean(weighted_ape)
end


# =============================================================================
# Directional Loss
# =============================================================================

"""
    compute_directional_loss(predictions, actuals; up_miss_weight=1.0,
                             down_miss_weight=1.0, previous_actuals=nothing) -> Float64

Compute directional loss with custom weights for missing UP vs DOWN moves.

This loss function penalizes directional prediction errors, with
different penalties for missing an upward move vs a downward move.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values (or predicted changes)
- `actuals::AbstractVector{<:Real}`: Actual observed values (or actual changes)
- `up_miss_weight::Float64=1.0`: Weight for missing an UP move
- `down_miss_weight::Float64=1.0`: Weight for missing a DOWN move
- `previous_actuals::Union{AbstractVector{<:Real}, Nothing}=nothing`:
  Previous actual values for computing changes. If provided, predictions
  and actuals are treated as levels.

# Returns
Mean directional loss.

# Formula
```
L = up_miss_weight × I(miss_up) + down_miss_weight × I(miss_down)
```
where:
- miss_up: predicted direction is non-positive but actual is positive
- miss_down: predicted direction is non-negative but actual is negative

# Example
```julia
pred_changes = [1.0, -1.0, 1.0, -1.0]
actual_changes = [0.5, 0.3, -0.2, -0.1]
# Missing UP costs 2x more than missing DOWN
loss = compute_directional_loss(pred_changes, actual_changes;
                                up_miss_weight=2.0, down_miss_weight=1.0)
```
"""
function compute_directional_loss(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    up_miss_weight::Float64 = 1.0,
    down_miss_weight::Float64 = 1.0,
    previous_actuals::Union{AbstractVector{<:Real}, Nothing} = nothing
)::Float64
    if up_miss_weight < 0.0 || down_miss_weight < 0.0
        error("Weights must be non-negative")
    end

    if length(predictions) != length(actuals)
        error("Array lengths must match. Got predictions=$(length(predictions)), actuals=$(length(actuals))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    # Compute changes if previous_actuals provided
    if previous_actuals !== nothing
        if length(previous_actuals) != length(predictions)
            error("previous_actuals length must match. Got $(length(previous_actuals)), expected $(length(predictions))")
        end
        pred_changes = predictions .- previous_actuals
        actual_changes = actuals .- previous_actuals
    else
        pred_changes = predictions
        actual_changes = actuals
    end

    pred_sign = sign.(pred_changes)
    actual_sign = sign.(actual_changes)

    # Miss UP: actual is positive, prediction is not positive
    miss_up = (actual_sign .> 0) .& (pred_sign .<= 0)

    # Miss DOWN: actual is negative, prediction is not negative
    miss_down = (actual_sign .< 0) .& (pred_sign .>= 0)

    # Compute weighted loss
    loss = up_miss_weight .* miss_up .+ down_miss_weight .* miss_down

    return mean(loss)
end


# =============================================================================
# Squared Log Error
# =============================================================================

"""
    compute_squared_log_error(predictions, actuals; epsilon=1e-8) -> Float64

Compute mean squared logarithmic error (MSLE).

MSLE is useful when targets span several orders of magnitude and
relative errors are more important than absolute errors. It also
naturally penalizes under-predictions more than over-predictions.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values (must be non-negative)
- `actuals::AbstractVector{<:Real}`: Actual values (must be non-negative)
- `epsilon::Float64=1e-8`: Small constant added before log to handle zeros

# Returns
Mean squared logarithmic error.

# Formula
```
MSLE = mean( (log(1 + actual) - log(1 + pred))² )
```

# Properties
- Scale-invariant (relative errors)
- Naturally asymmetric: penalizes under-predictions more for same |error|
- Appropriate for strictly positive targets

# Example
```julia
actuals = [100.0, 1000.0, 10000.0]
predictions = [110.0, 900.0, 11000.0]
msle = compute_squared_log_error(predictions, actuals)
```
"""
function compute_squared_log_error(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    epsilon::Float64 = 1e-8
)::Float64
    if length(predictions) != length(actuals)
        error("Array lengths must match. Got predictions=$(length(predictions)), actuals=$(length(actuals))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    if any(predictions .< 0) || any(actuals .< 0)
        error("MSLE requires non-negative values")
    end

    log_pred = log.(predictions .+ 1.0 .+ epsilon)
    log_actual = log.(actuals .+ 1.0 .+ epsilon)

    msle = mean((log_actual .- log_pred).^2)

    return msle
end


# =============================================================================
# Huber Loss
# =============================================================================

"""
    compute_huber_loss(predictions, actuals; delta=1.0) -> Float64

Compute Huber loss (smooth approximation to MAE).

Huber loss is quadratic for small errors and linear for large errors,
providing robustness to outliers while maintaining differentiability.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values
- `delta::Float64=1.0`: Threshold where loss transitions from quadratic to linear

# Returns
Mean Huber loss.

# Formula
```
L(e) = 0.5 × e²                 if |e| ≤ δ
     = δ × (|e| - 0.5 × δ)      if |e| > δ
```
where e = actual - prediction.

# Properties
- Quadratic near zero (like MSE)
- Linear in tails (like MAE, robust to outliers)
- Continuously differentiable (unlike MAE)
- Symmetric (unlike LinEx)

# Example
```julia
actuals = [1.0, 2.0, 100.0]  # One outlier
predictions = [1.1, 1.9, 10.0]  # Misses outlier badly
huber = compute_huber_loss(predictions, actuals; delta=1.0)
# Huber will be much less affected by the outlier than MSE
```
"""
function compute_huber_loss(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    delta::Float64 = 1.0
)::Float64
    if delta <= 0.0
        error("delta must be > 0, got $delta")
    end

    if length(predictions) != length(actuals)
        error("Array lengths must match. Got predictions=$(length(predictions)), actuals=$(length(actuals))")
    end

    if isempty(predictions)
        error("Arrays cannot be empty")
    end

    errors = actuals .- predictions
    abs_errors = abs.(errors)

    # Quadratic for |e| <= delta, linear for |e| > delta
    loss = [
        abs_e <= delta ? 0.5 * e^2 : delta * (abs_e - 0.5 * delta)
        for (e, abs_e) in zip(errors, abs_errors)
    ]

    return mean(loss)
end
