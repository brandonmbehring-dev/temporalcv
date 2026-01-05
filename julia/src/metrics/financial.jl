# =============================================================================
# Financial and Trading Metrics
# =============================================================================

"""
Risk-adjusted and trading performance metrics for evaluating forecasting
systems in financial contexts.

# Knowledge Tiers
- [T1] Sharpe ratio (Sharpe 1966, 1994)
- [T1] Maximum drawdown (standard risk metric)
- [T1] Information ratio (Goodwin 1998)
- [T2] Hit rate for directional accuracy (common practice)
- [T2] Profit factor for trading evaluation (common practice)

# References
- Sharpe, W.F. (1966). Mutual fund performance. Journal of Business, 39(1), 119-138.
- Sharpe, W.F. (1994). The Sharpe ratio. Journal of Portfolio Management, 21(1), 49-58.
- Goodwin, T.H. (1998). The information ratio. Financial Analysts Journal, 54(4), 34-43.
"""

# =============================================================================
# Sharpe Ratio
# =============================================================================

"""
    compute_sharpe_ratio(returns; risk_free_rate=0.0, annualization=252.0) -> Float64

Compute annualized Sharpe ratio.

The Sharpe ratio measures risk-adjusted excess return: the mean return
in excess of the risk-free rate, divided by return volatility.

# Arguments
- `returns::AbstractVector{<:Real}`: Period returns (e.g., daily log returns)
- `risk_free_rate::Float64=0.0`: Risk-free rate per period
- `annualization::Float64=252.0`: Number of periods per year
  (252 for daily, 52 for weekly, 12 for monthly)

# Returns
Annualized Sharpe ratio.

# Formula
```
SR = √(annualization) × mean(r - rf) / std(r)
```

A higher Sharpe ratio indicates better risk-adjusted performance.
Values above 1.0 are generally considered good, above 2.0 excellent.

# Example
```julia
daily_returns = [0.01, -0.005, 0.008, 0.003, -0.002]
sharpe = compute_sharpe_ratio(daily_returns)
```
"""
function compute_sharpe_ratio(
    returns::AbstractVector{<:Real};
    risk_free_rate::Float64 = 0.0,
    annualization::Float64 = 252.0
)::Float64
    if isempty(returns)
        error("Returns array cannot be empty")
    end

    if length(returns) < 2
        error("Need at least 2 return observations for Sharpe ratio")
    end

    excess_returns = returns .- risk_free_rate
    mean_excess = mean(excess_returns)
    std_returns = std(returns; corrected=true)

    if std_returns == 0.0
        # Zero volatility: undefined ratio
        if mean_excess > 0
            return Inf
        elseif mean_excess < 0
            return -Inf
        else
            return 0.0
        end
    end

    sharpe = sqrt(annualization) * mean_excess / std_returns

    return sharpe
end


# =============================================================================
# Maximum Drawdown
# =============================================================================

"""
    compute_max_drawdown(; cumulative_returns=nothing, returns=nothing) -> Float64

Compute maximum drawdown from peak to trough.

Maximum drawdown measures the largest decline from a historical peak
in cumulative returns, representing the worst-case loss.

# Arguments
- `cumulative_returns::Union{AbstractVector{<:Real}, Nothing}=nothing`:
  Cumulative returns (or price/equity curve). If provided, `returns` is ignored.
- `returns::Union{AbstractVector{<:Real}, Nothing}=nothing`:
  Period returns. Used to compute cumulative returns if `cumulative_returns`
  is not provided.

# Returns
Maximum drawdown as a positive fraction (e.g., 0.20 = 20% drawdown).

# Formula
```
MDD = max_t [ (peak_t - trough_t) / peak_t ]
```
where peak_t is the running maximum up to time t.

# Example
```julia
cumulative = [100.0, 110.0, 105.0, 120.0, 108.0, 125.0]
mdd = compute_max_drawdown(cumulative_returns=cumulative)
# From 120 to 108 = 10%
```
"""
function compute_max_drawdown(;
    cumulative_returns::Union{AbstractVector{<:Real}, Nothing} = nothing,
    returns::Union{AbstractVector{<:Real}, Nothing} = nothing
)::Float64
    if cumulative_returns === nothing && returns === nothing
        error("Must provide either cumulative_returns or returns")
    end

    if cumulative_returns !== nothing
        curve = Float64.(cumulative_returns)
    else
        if isempty(returns)
            error("Returns array cannot be empty")
        end
        # Convert to cumulative returns (1 + r cumulative product)
        curve = cumprod(1.0 .+ returns)
    end

    if isempty(curve)
        error("Input array cannot be empty")
    end

    # Running maximum
    running_max = accumulate(max, curve)

    # Drawdown at each point
    drawdowns = (running_max .- curve) ./ running_max

    # Handle case where running_max is zero (shouldn't happen with valid data)
    drawdowns = replace(drawdowns, NaN => 0.0)

    return maximum(drawdowns)
end


# =============================================================================
# Cumulative Return
# =============================================================================

"""
    compute_cumulative_return(returns; method=:geometric) -> Float64

Compute cumulative return over the period.

# Arguments
- `returns::AbstractVector{<:Real}`: Period returns
- `method::Symbol=:geometric`: Compounding method
  - `:geometric`: (1+r1) × (1+r2) × ... - 1 (correct for compounding)
  - `:arithmetic`: sum of returns (simple addition)

# Returns
Cumulative return as a fraction (e.g., 0.25 = 25% total return).

# Example
```julia
returns = [0.05, 0.03, -0.02, 0.04]
cum_ret = compute_cumulative_return(returns)
```
"""
function compute_cumulative_return(
    returns::AbstractVector{<:Real};
    method::Symbol = :geometric
)::Float64
    if isempty(returns)
        error("Returns array cannot be empty")
    end

    if method == :geometric
        cumulative = prod(1.0 .+ returns) - 1.0
    elseif method == :arithmetic
        cumulative = sum(returns)
    else
        error("Invalid method '$method'. Use :geometric or :arithmetic")
    end

    return cumulative
end


# =============================================================================
# Information Ratio
# =============================================================================

"""
    compute_information_ratio(portfolio_returns, benchmark_returns;
                              annualization=252.0) -> Float64

Compute information ratio (active return per unit tracking error).

The information ratio measures how much excess return a portfolio
generates relative to a benchmark, per unit of tracking error.

# Arguments
- `portfolio_returns::AbstractVector{<:Real}`: Portfolio period returns
- `benchmark_returns::AbstractVector{<:Real}`: Benchmark period returns
- `annualization::Float64=252.0`: Number of periods per year

# Returns
Annualized information ratio.

# Formula
```
IR = √(annualization) × mean(r_p - r_b) / std(r_p - r_b)
```

A higher IR indicates better risk-adjusted active performance.
Values above 0.5 are generally considered good, above 1.0 excellent.

# Example
```julia
portfolio = [0.02, 0.01, -0.01, 0.03, 0.00]
benchmark = [0.01, 0.01, 0.00, 0.02, 0.01]
ir = compute_information_ratio(portfolio, benchmark)
```
"""
function compute_information_ratio(
    portfolio_returns::AbstractVector{<:Real},
    benchmark_returns::AbstractVector{<:Real};
    annualization::Float64 = 252.0
)::Float64
    if isempty(portfolio_returns) || isempty(benchmark_returns)
        error("Return arrays cannot be empty")
    end

    if length(portfolio_returns) != length(benchmark_returns)
        error("Array lengths must match. Got portfolio=$(length(portfolio_returns)), benchmark=$(length(benchmark_returns))")
    end

    if length(portfolio_returns) < 2
        error("Need at least 2 observations for information ratio")
    end

    active_returns = portfolio_returns .- benchmark_returns
    mean_active = mean(active_returns)
    tracking_error = std(active_returns; corrected=true)

    if tracking_error == 0.0
        if mean_active > 0
            return Inf
        elseif mean_active < 0
            return -Inf
        else
            return 0.0
        end
    end

    ir = sqrt(annualization) * mean_active / tracking_error

    return ir
end


# =============================================================================
# Hit Rate
# =============================================================================

"""
    compute_hit_rate(predicted_changes, actual_changes) -> Float64

Compute directional hit rate (fraction of correct direction predictions).

Hit rate measures the percentage of times the predicted direction
(up/down) matches the actual direction.

# Arguments
- `predicted_changes::AbstractVector{<:Real}`: Predicted changes
- `actual_changes::AbstractVector{<:Real}`: Actual changes

# Returns
Hit rate in [0, 1]. E.g., 0.60 means 60% of predictions had correct direction.

# Formula
```
hit_rate = mean( sign(pred) == sign(actual) )
```
where sign(0) = 0 is treated as matching any sign (conservative).

A hit rate above 0.5 indicates directional skill (better than random).
For financial applications, hit rates of 0.52-0.55 can be valuable.

# Example
```julia
predicted = [0.01, -0.02, 0.01, 0.02, -0.01]
actual = [0.02, -0.01, -0.01, 0.03, -0.02]
hr = compute_hit_rate(predicted, actual)  # 4/5 = 80%
```
"""
function compute_hit_rate(
    predicted_changes::AbstractVector{<:Real},
    actual_changes::AbstractVector{<:Real}
)::Float64
    if isempty(predicted_changes) || isempty(actual_changes)
        error("Arrays cannot be empty")
    end

    if length(predicted_changes) != length(actual_changes)
        error("Array lengths must match. Got predicted=$(length(predicted_changes)), actual=$(length(actual_changes))")
    end

    # Compare signs: both positive, both negative, or either is zero
    pred_sign = sign.(predicted_changes)
    actual_sign = sign.(actual_changes)

    # Match if: same sign, or either is zero (no clear direction)
    hits = (pred_sign .== actual_sign) .| (pred_sign .== 0) .| (actual_sign .== 0)

    return mean(hits)
end


# =============================================================================
# Profit Factor
# =============================================================================

"""
    compute_profit_factor(predicted_changes, actual_changes;
                          returns=nothing) -> Float64

Compute profit factor (gross profit / gross loss ratio).

Profit factor measures the ratio of total profits to total losses
from trading on directional predictions.

# Arguments
- `predicted_changes::AbstractVector{<:Real}`: Predicted changes (sign = direction)
- `actual_changes::AbstractVector{<:Real}`: Actual changes
- `returns::Union{AbstractVector{<:Real}, Nothing}=nothing`:
  Actual returns to use for P&L. If not provided, uses `actual_changes`.

# Returns
Profit factor. > 1.0 indicates profitable strategy.
Returns Inf if no losses, 0.0 if no profits.

# Trading Logic
- If pred > 0: go long, P&L = return
- If pred < 0: go short, P&L = -return
- If pred == 0: no trade, P&L = 0

A profit factor of 1.5 means for every \$1 lost, \$1.50 is gained.
Values above 1.0 indicate a profitable strategy, above 2.0 is excellent.

# Example
```julia
predicted = [1.0, -1.0, 1.0, -1.0]  # Buy, sell, buy, sell
actual = [0.02, -0.01, -0.01, -0.02]
pf = compute_profit_factor(predicted, actual)
```
"""
function compute_profit_factor(
    predicted_changes::AbstractVector{<:Real},
    actual_changes::AbstractVector{<:Real};
    returns::Union{AbstractVector{<:Real}, Nothing} = nothing
)::Float64
    if returns !== nothing
        rets = Float64.(returns)
    else
        rets = Float64.(actual_changes)
    end

    if isempty(predicted_changes) || isempty(actual_changes)
        error("Arrays cannot be empty")
    end

    if length(predicted_changes) != length(rets)
        error("Array lengths must match. Got predicted=$(length(predicted_changes)), returns=$(length(rets))")
    end

    # Compute P&L based on predicted direction
    pred_sign = sign.(predicted_changes)
    pnl = pred_sign .* rets

    gross_profit = sum(pnl[pnl .> 0])
    gross_loss = abs(sum(pnl[pnl .< 0]))

    if gross_loss == 0.0
        if gross_profit > 0.0
            return Inf
        else
            return 0.0
        end
    end

    return gross_profit / gross_loss
end


# =============================================================================
# Calmar Ratio
# =============================================================================

"""
    compute_calmar_ratio(returns; annualization=252.0) -> Float64

Compute Calmar ratio (annualized return / max drawdown).

The Calmar ratio measures return relative to the worst historical
decline, focusing on tail risk.

# Arguments
- `returns::AbstractVector{<:Real}`: Period returns
- `annualization::Float64=252.0`: Number of periods per year

# Returns
Calmar ratio. Higher is better.

# Formula
```
Calmar = Annualized Return / Max Drawdown
```

It measures how much return is generated per unit of worst-case risk.
Useful for evaluating strategies where drawdowns are a key concern.

# Example
```julia
returns = [0.01, -0.02, 0.03, -0.01, 0.02]
calmar = compute_calmar_ratio(returns)
```
"""
function compute_calmar_ratio(
    returns::AbstractVector{<:Real};
    annualization::Float64 = 252.0
)::Float64
    if isempty(returns)
        error("Returns array cannot be empty")
    end

    # Annualized return
    cumulative = prod(1.0 .+ returns)
    n_periods = length(returns)
    annualized_return = cumulative^(annualization / n_periods) - 1.0

    # Max drawdown
    max_dd = compute_max_drawdown(returns=returns)

    if max_dd == 0.0
        if annualized_return > 0.0
            return Inf
        elseif annualized_return < 0.0
            return -Inf
        else
            return 0.0
        end
    end

    return annualized_return / max_dd
end
