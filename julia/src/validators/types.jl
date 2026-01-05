# =============================================================================
# Validators Types
# =============================================================================

"""
Theoretical bounds for AR(1) process at a given horizon.

# Fields
- `mse::Float64`: Theoretical minimum MSE
- `mae::Float64`: Theoretical minimum MAE
- `rmse::Float64`: Theoretical minimum RMSE
- `phi::Float64`: AR(1) coefficient
- `sigma::Float64`: Innovation standard deviation
- `horizon::Int`: Forecast horizon
"""
struct AR1Bounds
    mse::Float64
    mae::Float64
    rmse::Float64
    phi::Float64
    sigma::Float64
    horizon::Int
end


"""
Theoretical bounds for AR(2) process at a given horizon.

# Fields
- `mse::Float64`: Theoretical minimum MSE
- `mae::Float64`: Theoretical minimum MAE
- `rmse::Float64`: Theoretical minimum RMSE
- `phi1::Float64`: First AR coefficient
- `phi2::Float64`: Second AR coefficient
- `sigma::Float64`: Innovation standard deviation
- `horizon::Int`: Forecast horizon
"""
struct AR2Bounds
    mse::Float64
    mae::Float64
    rmse::Float64
    phi1::Float64
    phi2::Float64
    sigma::Float64
    horizon::Int
end


"""
Result of checking model performance against theoretical bounds.

# Fields
- `status::Symbol`: :PASS, :WARN, :HALT, or :SKIP
- `model_metric::Float64`: The model's observed metric
- `theoretical_bound::Float64`: The theoretical minimum
- `ratio::Float64`: model_metric / theoretical_bound
- `message::String`: Human-readable description
- `recommendation::String`: Suggested action if not PASS
"""
struct BoundsCheckResult
    status::Symbol
    model_metric::Float64
    theoretical_bound::Float64
    ratio::Float64
    message::String
    recommendation::String
end

# Convenience constructors
function BoundsCheckResult(status::Symbol, model_metric::Float64, theoretical_bound::Float64,
                           ratio::Float64, message::String)
    return BoundsCheckResult(status, model_metric, theoretical_bound, ratio, message, "")
end
