# =============================================================================
# Theoretical AR Bounds
# =============================================================================

"""
Theoretical minimum MSE for AR(1) process at horizon h.

# Arguments
- `phi::Float64`: AR(1) coefficient. Must satisfy |phi| < 1 for stationarity.
- `sigma_sq::Float64`: Innovation variance sigma^2.
- `horizon::Int=1`: Forecast horizon.

# Returns
Theoretical minimum MSE for h-step ahead forecast.

# Formula [T1]
```
MSE(h) = sigma^2 * (1 + phi^2 + phi^4 + ... + phi^(2(h-1)))
       = sigma^2 * (1 - phi^(2h)) / (1 - phi^2)
```

For h=1: MSE = sigma^2 (just the innovation variance)
For h->inf: MSE -> sigma^2 / (1 - phi^2) = Var(y)

# References
Hamilton, J.D. (1994). Time Series Analysis. Ch. 4.

# Example
```julia
# 1-step MSE for AR(1) with phi=0.9, sigma^2=1
mse = compute_ar1_mse_bound(0.9, 1.0)  # Returns 1.0

# 5-step MSE
mse5 = compute_ar1_mse_bound(0.9, 1.0; horizon=5)  # Returns ~4.1
```
"""
function compute_ar1_mse_bound(
    phi::Real,
    sigma_sq::Real;
    horizon::Int = 1
)::Float64
    phi = Float64(phi)
    sigma_sq = Float64(sigma_sq)

    # Validation
    if abs(phi) >= 1.0
        error("phi must satisfy |phi| < 1 for stationarity, got $phi")
    end
    if sigma_sq <= 0
        error("sigma_sq must be positive, got $sigma_sq")
    end
    if horizon < 1
        error("horizon must be >= 1, got $horizon")
    end

    # Special case: phi = 0 (white noise)
    if phi == 0
        return sigma_sq * horizon
    end

    # General case: geometric sum
    # MSE = sigma^2 * sum(phi^(2i) for i=0..h-1) = sigma^2 * (1 - phi^(2h)) / (1 - phi^2)
    phi_sq = phi ^ 2
    mse = sigma_sq * (1.0 - phi_sq ^ horizon) / (1.0 - phi_sq)
    return mse
end


"""
Theoretical minimum MAE for AR(1) process at horizon h.

For Gaussian innovations, MAE = sqrt(2/pi) * RMSE ≈ 0.798 * RMSE.

# Arguments
- `phi::Float64`: AR(1) coefficient. Must satisfy |phi| < 1.
- `sigma::Float64`: Innovation standard deviation (not variance).
- `horizon::Int=1`: Forecast horizon.

# Returns
Theoretical minimum MAE for h-step ahead forecast.

# Formula [T1]
For X ~ N(0, sigma^2), E[|X|] = sigma * sqrt(2/pi) (half-normal mean).

# Example
```julia
# For h=1, MAE = sigma * sqrt(2/pi) ≈ 0.798
mae = compute_ar1_mae_bound(0.0, 1.0; horizon=1)  # ≈ 0.798
```
"""
function compute_ar1_mae_bound(
    phi::Real,
    sigma::Real;
    horizon::Int = 1
)::Float64
    sigma = Float64(sigma)
    if sigma <= 0
        error("sigma must be positive, got $sigma")
    end

    mse = compute_ar1_mse_bound(phi, sigma^2; horizon=horizon)
    rmse = sqrt(mse)
    # E[|X|] = sigma * sqrt(2/pi) for X ~ N(0, sigma^2)
    mae = rmse * sqrt(2.0 / pi)
    return mae
end


"""
Theoretical minimum RMSE for AR(1) process at horizon h.

Simply sqrt(MSE).

# Arguments
- `phi::Float64`: AR(1) coefficient.
- `sigma::Float64`: Innovation standard deviation.
- `horizon::Int=1`: Forecast horizon.

# Returns
Theoretical minimum RMSE.
"""
function compute_ar1_rmse_bound(
    phi::Real,
    sigma::Real;
    horizon::Int = 1
)::Float64
    sigma = Float64(sigma)
    if sigma <= 0
        error("sigma must be positive, got $sigma")
    end
    mse = compute_ar1_mse_bound(phi, sigma^2; horizon=horizon)
    return sqrt(mse)
end


"""
Theoretical minimum MSE for AR(2) process at horizon h.

For AR(2): y_t = phi1*y_{t-1} + phi2*y_{t-2} + eps_t

# Arguments
- `phi1::Float64`: First AR coefficient.
- `phi2::Float64`: Second AR coefficient.
- `sigma_sq::Float64`: Innovation variance.
- `horizon::Int=1`: Forecast horizon.

# Stationarity Conditions
- phi1 + phi2 < 1
- phi2 - phi1 < 1
- |phi2| < 1

# Formula [T1]
Compute MSE via psi-weight recursion:
y_t = sum(psi_j * eps_{t-j}) where psi_0=1, psi_1=phi1, psi_j=phi1*psi_{j-1}+phi2*psi_{j-2}
MSE(h) = sigma^2 * sum(psi_j^2 for j=0..h-1)

# References
Hamilton (1994), Chapter 4.
"""
function compute_ar2_mse_bound(
    phi1::Real,
    phi2::Real,
    sigma_sq::Real;
    horizon::Int = 1
)::Float64
    phi1 = Float64(phi1)
    phi2 = Float64(phi2)
    sigma_sq = Float64(sigma_sq)

    if sigma_sq <= 0
        error("sigma_sq must be positive, got $sigma_sq")
    end
    if horizon < 1
        error("horizon must be >= 1, got $horizon")
    end

    # Check stationarity conditions
    if phi1 + phi2 >= 1 || phi2 - phi1 >= 1 || abs(phi2) >= 1
        error("AR(2) coefficients violate stationarity: " *
              "phi1=$phi1, phi2=$phi2. " *
              "Need: phi1+phi2<1, phi2-phi1<1, |phi2|<1")
    end

    # Compute MSE via recursion for psi weights
    psi = zeros(horizon)
    psi[1] = 1.0
    if horizon > 1
        psi[2] = phi1
    end
    for j in 3:horizon
        psi[j] = phi1 * psi[j-1] + phi2 * psi[j-2]
    end

    mse = sigma_sq * sum(psi .^ 2)
    return mse
end


"""
Generate a synthetic AR(1) time series.

y_t = phi*y_{t-1} + eps_t, where eps_t ~ N(0, sigma^2)

# Arguments
- `phi::Float64`: AR(1) coefficient. Must satisfy |phi| < 1.
- `sigma::Float64`: Innovation standard deviation.
- `n::Int`: Number of observations to generate.
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator.

# Returns
Generated AR(1) series of length n.

# Example
```julia
using Random
rng = MersenneTwister(42)
series = generate_ar1_series(0.9, 1.0, 100; rng=rng)
```
"""
function generate_ar1_series(
    phi::Real,
    sigma::Real,
    n::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Vector{Float64}
    phi = Float64(phi)
    sigma = Float64(sigma)

    if abs(phi) >= 1.0
        error("phi must satisfy |phi| < 1 for stationarity, got $phi")
    end
    if sigma <= 0
        error("sigma must be positive, got $sigma")
    end
    if n < 1
        error("n must be >= 1, got $n")
    end

    # Initialize from stationary distribution
    # Var(y) = sigma^2 / (1 - phi^2) for stationary AR(1)
    y0_std = phi != 0 ? sigma / sqrt(1.0 - phi^2) : sigma
    y = zeros(n)
    y[1] = randn(rng) * y0_std

    # Generate innovations
    innovations = randn(rng, n) .* sigma

    # AR(1) recursion
    for t in 2:n
        y[t] = phi * y[t-1] + innovations[t]
    end

    return y
end


"""
Generate a synthetic AR(2) time series.

y_t = phi1*y_{t-1} + phi2*y_{t-2} + eps_t, where eps_t ~ N(0, sigma^2)

# Arguments
- `phi1::Float64`: First AR coefficient.
- `phi2::Float64`: Second AR coefficient.
- `sigma::Float64`: Innovation standard deviation.
- `n::Int`: Number of observations to generate.
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator.

# Returns
Generated AR(2) series of length n.
"""
function generate_ar2_series(
    phi1::Real,
    phi2::Real,
    sigma::Real,
    n::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG
)::Vector{Float64}
    phi1 = Float64(phi1)
    phi2 = Float64(phi2)
    sigma = Float64(sigma)

    if sigma <= 0
        error("sigma must be positive, got $sigma")
    end
    if n < 2
        error("n must be >= 2 for AR(2), got $n")
    end

    # Check stationarity conditions
    if phi1 + phi2 >= 1 || phi2 - phi1 >= 1 || abs(phi2) >= 1
        error("AR(2) coefficients violate stationarity: " *
              "phi1=$phi1, phi2=$phi2. " *
              "Need: phi1+phi2<1, phi2-phi1<1, |phi2|<1")
    end

    # Compute unconditional variance for initialization
    # gamma_0 = sigma^2 / ((1 - phi2) * ((1 + phi2)^2 - phi1^2))
    denom = (1 - phi2) * ((1 + phi2)^2 - phi1^2)
    y0_std = if denom > 0
        gamma0 = sigma^2 / denom
        sqrt(max(gamma0, sigma^2))
    else
        sigma
    end

    y = zeros(n)
    y[1] = randn(rng) * y0_std
    y[2] = randn(rng) * y0_std

    # Generate innovations
    innovations = randn(rng, n) .* sigma

    # AR(2) recursion
    for t in 3:n
        y[t] = phi1 * y[t-1] + phi2 * y[t-2] + innovations[t]
    end

    return y
end


"""
Estimate AR(1) parameters from a time series using OLS.

# Arguments
- `values::AbstractVector{<:Real}`: Time series data.

# Returns
Tuple of (phi, sigma) where:
- phi: Estimated AR(1) coefficient
- sigma: Estimated innovation standard deviation

# Method
OLS regression of y_t on y_{t-1}.
sigma estimated from residual variance.
"""
function estimate_ar1_params(values::AbstractVector{<:Real})::Tuple{Float64, Float64}
    values = Float64.(values)
    n = length(values)

    if n < 3
        error("Need at least 3 observations for AR(1) estimation, got $n")
    end

    # y_t = phi * y_{t-1} + eps_t
    # OLS: phi = sum(y_t * y_{t-1}) / sum(y_{t-1}^2)
    y = values[2:end]
    y_lag = values[1:end-1]

    # Demean for better estimation
    y_mean = mean(values)
    y_dm = y .- y_mean
    y_lag_dm = y_lag .- y_mean

    # OLS estimate
    numerator = sum(y_dm .* y_lag_dm)
    denominator = sum(y_lag_dm .^ 2)

    if denominator == 0
        return (0.0, std(values))
    end

    phi = numerator / denominator

    # Clamp to stationary region
    phi = clamp(phi, -0.9999, 0.9999)

    # Estimate sigma from residuals
    residuals = y_dm .- phi .* y_lag_dm
    sigma = std(residuals; corrected=true)

    return (phi, sigma)
end


"""
Check if model MSE beats theoretical AR(1) minimum.

If model_mse < theoretical_mse / tolerance, this indicates
the model is "impossibly good" - likely due to data leakage.

# Arguments
- `model_mse::Float64`: Observed model MSE.
- `phi::Float64`: Estimated AR(1) coefficient.
- `sigma_sq::Float64`: Estimated innovation variance.
- `horizon::Int=1`: Forecast horizon.
- `tolerance::Float64=1.5`: Factor for finite-sample variation.
  HALT if model_mse < theoretical_mse / tolerance.
- `metric_name::String="MSE"`: Name for the metric in messages.

# Returns
BoundsCheckResult with status:
- :HALT if model beats bounds (leakage likely)
- :WARN if model is suspiciously close to bounds (ratio < 1.2)
- :PASS if model is within expected range

# Knowledge Tier
[T3] Tolerance factor 1.5 is an empirical heuristic.

# Example
```julia
# Model with MSE = 0.5 vs theoretical minimum of 1.0
result = check_against_ar1_bounds(model_mse=0.5, phi=0.9, sigma_sq=1.0)
result.status  # :HALT

# Model with MSE = 1.2 vs theoretical minimum of 1.0
result = check_against_ar1_bounds(model_mse=1.2, phi=0.9, sigma_sq=1.0)
result.status  # :PASS
```
"""
function check_against_ar1_bounds(;
    model_mse::Real,
    phi::Real,
    sigma_sq::Real,
    horizon::Int = 1,
    tolerance::Float64 = 1.5,
    metric_name::String = "MSE"
)::BoundsCheckResult
    model_mse = Float64(model_mse)
    phi = Float64(phi)
    sigma_sq = Float64(sigma_sq)

    # Try to compute theoretical bound
    theoretical_mse = try
        compute_ar1_mse_bound(phi, sigma_sq; horizon=horizon)
    catch e
        return BoundsCheckResult(
            :SKIP,
            model_mse,
            NaN,
            NaN,
            "Cannot compute bounds: $e",
            "Verify AR(1) coefficient and variance estimates"
        )
    end

    ratio = model_mse / theoretical_mse
    threshold = 1.0 / tolerance  # e.g., 0.667 for tolerance=1.5

    # HALT: Beating theoretical bounds (impossible without leakage)
    if ratio < threshold
        return BoundsCheckResult(
            :HALT,
            model_mse,
            theoretical_mse,
            ratio,
            "Model $metric_name ($model_mse) is $(round(ratio * 100, digits=1))% of theoretical " *
            "minimum ($theoretical_mse). This is below the $(round(threshold * 100, digits=1))% " *
            "threshold, indicating likely data leakage.",
            "Investigate for lookahead bias: " *
            "1) Check feature computation for future leakage " *
            "2) Verify train/test split respects temporal order " *
            "3) Examine threshold/parameter computations"
        )
    end

    # WARN: Suspiciously close to theoretical bounds (ratio < 1.2)
    if ratio < 1.2
        return BoundsCheckResult(
            :WARN,
            model_mse,
            theoretical_mse,
            ratio,
            "Model $metric_name ($model_mse) is only $(round(ratio * 100, digits=1))% of " *
            "theoretical minimum ($theoretical_mse). This is unusually " *
            "good - verify no subtle leakage.",
            "Verify feature engineering does not use future information"
        )
    end

    # PASS: Model is within expected range
    return BoundsCheckResult(
        :PASS,
        model_mse,
        theoretical_mse,
        ratio,
        "Model $metric_name ($model_mse) is $(round(ratio * 100, digits=1))% of theoretical " *
        "minimum ($theoretical_mse). Within expected range.",
        ""
    )
end


"""
Check model against theoretical AR(2) bounds.

Similar to check_against_ar1_bounds but for AR(2) processes.

# Arguments
- `model_mse::Float64`: Observed model MSE.
- `phi1::Float64`: First AR coefficient.
- `phi2::Float64`: Second AR coefficient.
- `sigma_sq::Float64`: Estimated innovation variance.
- `horizon::Int=1`: Forecast horizon.
- `tolerance::Float64=1.5`: Factor for finite-sample variation.
- `metric_name::String="MSE"`: Name for the metric in messages.

# Returns
BoundsCheckResult with status :HALT, :WARN, :PASS, or :SKIP.
"""
function check_against_ar2_bounds(;
    model_mse::Real,
    phi1::Real,
    phi2::Real,
    sigma_sq::Real,
    horizon::Int = 1,
    tolerance::Float64 = 1.5,
    metric_name::String = "MSE"
)::BoundsCheckResult
    model_mse = Float64(model_mse)
    phi1 = Float64(phi1)
    phi2 = Float64(phi2)
    sigma_sq = Float64(sigma_sq)

    # Try to compute theoretical bound
    theoretical_mse = try
        compute_ar2_mse_bound(phi1, phi2, sigma_sq; horizon=horizon)
    catch e
        return BoundsCheckResult(
            :SKIP,
            model_mse,
            NaN,
            NaN,
            "Cannot compute bounds: $e",
            "Verify AR(2) coefficients and variance estimates"
        )
    end

    ratio = model_mse / theoretical_mse
    threshold = 1.0 / tolerance

    if ratio < threshold
        return BoundsCheckResult(
            :HALT,
            model_mse,
            theoretical_mse,
            ratio,
            "Model $metric_name ($model_mse) is $(round(ratio * 100, digits=1))% of theoretical " *
            "minimum ($theoretical_mse). Likely data leakage.",
            "Investigate for lookahead bias"
        )
    end

    if ratio < 1.2
        return BoundsCheckResult(
            :WARN,
            model_mse,
            theoretical_mse,
            ratio,
            "Model $metric_name ($model_mse) is unusually close to theoretical minimum.",
            "Verify no subtle leakage"
        )
    end

    return BoundsCheckResult(
        :PASS,
        model_mse,
        theoretical_mse,
        ratio,
        "Model within expected range.",
        ""
    )
end


"""
Compute all AR(1) bounds for a given set of parameters.

# Arguments
- `phi::Float64`: AR(1) coefficient.
- `sigma::Float64`: Innovation standard deviation.
- `horizon::Int=1`: Forecast horizon.

# Returns
AR1Bounds struct with MSE, MAE, RMSE bounds.

# Example
```julia
bounds = compute_ar1_bounds(0.9, 1.0; horizon=1)
bounds.mse   # 1.0
bounds.mae   # ≈ 0.798
bounds.rmse  # 1.0
```
"""
function compute_ar1_bounds(
    phi::Real,
    sigma::Real;
    horizon::Int = 1
)::AR1Bounds
    phi = Float64(phi)
    sigma = Float64(sigma)

    mse = compute_ar1_mse_bound(phi, sigma^2; horizon=horizon)
    rmse = sqrt(mse)
    mae = rmse * sqrt(2.0 / pi)

    return AR1Bounds(mse, mae, rmse, phi, sigma, horizon)
end


"""
Compute all AR(2) bounds for a given set of parameters.

# Arguments
- `phi1::Float64`: First AR coefficient.
- `phi2::Float64`: Second AR coefficient.
- `sigma::Float64`: Innovation standard deviation.
- `horizon::Int=1`: Forecast horizon.

# Returns
AR2Bounds struct with MSE, MAE, RMSE bounds.
"""
function compute_ar2_bounds(
    phi1::Real,
    phi2::Real,
    sigma::Real;
    horizon::Int = 1
)::AR2Bounds
    phi1 = Float64(phi1)
    phi2 = Float64(phi2)
    sigma = Float64(sigma)

    mse = compute_ar2_mse_bound(phi1, phi2, sigma^2; horizon=horizon)
    rmse = sqrt(mse)
    mae = rmse * sqrt(2.0 / pi)

    return AR2Bounds(mse, mae, rmse, phi1, phi2, sigma, horizon)
end
