# =============================================================================
# HAC Variance Estimation (Newey-West with Bartlett kernel)
# =============================================================================

"""
    bartlett_kernel(j::Int, bandwidth::Int) -> Float64

Bartlett kernel weight for lag j.

# Arguments
- `j`: Lag index (non-negative)
- `bandwidth`: Kernel bandwidth

# Returns
Kernel weight in [0, 1].

# Reference
[T1] Newey, W.K. & West, K.D. (1987). A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix.
"""
function bartlett_kernel(j::Int, bandwidth::Int)
    if abs(j) <= bandwidth
        return 1.0 - abs(j) / (bandwidth + 1)
    end
    return 0.0
end


"""
    compute_hac_variance(d::Vector; bandwidth::Union{Int, Nothing}=nothing) -> Float64

Compute HAC (Heteroskedasticity and Autocorrelation Consistent) variance.

Uses Newey-West estimator with Bartlett kernel.

# Arguments
- `d`: Series (typically loss differential for DM test)
- `bandwidth`: Kernel bandwidth. If nothing, uses automatic selection:
  `floor(4 * (n/100)^(2/9))` per Andrews (1991).

# Returns
HAC variance estimate.

# Notes
For h-step forecasts, errors are MA(h-1), so bandwidth = h-1 is appropriate.
The automatic bandwidth is a general-purpose choice when h is unknown.

Complexity: O(n × bandwidth)

# Reference
[T1] Newey, W.K. & West, K.D. (1987). Econometrica, 55(3), 703-708.
[T1] Andrews, D.W.K. (1991). Econometrica, 59(3), 817-858.

# Example
```julia
# For 2-step ahead forecast
d = loss1 .- loss2
variance = compute_hac_variance(d, bandwidth=1)  # h-1 = 1
```
"""
function compute_hac_variance(d::AbstractVector{<:Real}; bandwidth::Union{Int, Nothing}=nothing)
    n = length(d)
    d_demeaned = d .- mean(d)

    # Automatic bandwidth: Andrews (1991) rule
    if isnothing(bandwidth)
        bandwidth = max(1, floor(Int, 4 * (n / 100)^(2/9)))
    end

    # Compute autocovariances
    gamma = zeros(bandwidth + 1)
    for j in 0:bandwidth
        if j == 0
            gamma[j+1] = mean(d_demeaned.^2)
        else
            gamma[j+1] = mean(d_demeaned[j+1:end] .* d_demeaned[1:end-j])
        end
    end

    # Apply Bartlett kernel weights
    variance = gamma[1]
    for j in 1:bandwidth
        weight = bartlett_kernel(j, bandwidth)
        variance += 2 * weight * gamma[j+1]
    end

    return variance / n
end
