"""
Financial and Trading Metrics Module.

Risk-adjusted and trading performance metrics for evaluating forecasting
systems in financial contexts:

- **Sharpe ratio**: Risk-adjusted excess return (annualized)
- **Max drawdown**: Largest peak-to-trough decline
- **Cumulative return**: Total return over period
- **Information ratio**: Active return per unit tracking error
- **Hit rate**: Fraction of correct directional predictions
- **Profit factor**: Ratio of gross profit to gross loss

Knowledge Tiers
---------------
[T1] Sharpe ratio (Sharpe 1966, 1994)
[T1] Maximum drawdown (standard risk metric)
[T1] Information ratio (Goodwin 1998)
[T2] Hit rate for directional accuracy (common practice)
[T2] Profit factor for trading evaluation (common practice)

Example
-------
>>> from temporalcv.metrics.financial import (
...     compute_sharpe_ratio,
...     compute_max_drawdown,
...     compute_hit_rate,
...     compute_profit_factor,
... )
>>>
>>> # Risk-adjusted return
>>> sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.02/252)
>>>
>>> # Trading metrics
>>> hit_rate = compute_hit_rate(predicted_changes, actual_changes)
>>> profit = compute_profit_factor(predicted_changes, actual_changes, returns)

References
----------
[T1] Sharpe, W.F. (1966). Mutual fund performance. Journal of Business, 39(1), 119-138.
[T1] Sharpe, W.F. (1994). The Sharpe ratio. Journal of Portfolio Management, 21(1), 49-58.
[T1] Goodwin, T.H. (1998). The information ratio. Financial Analysts Journal, 54(4), 34-43.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_sharpe_ratio(
    returns: ArrayLike,
    risk_free_rate: float = 0.0,
    annualization: float = 252.0,
) -> float:
    """
    Compute annualized Sharpe ratio.

    The Sharpe ratio measures risk-adjusted excess return: the mean return
    in excess of the risk-free rate, divided by return volatility.

    Parameters
    ----------
    returns : array-like
        Period returns (e.g., daily log returns or simple returns).
    risk_free_rate : float, default 0.0
        Risk-free rate per period (same frequency as returns).
        E.g., for daily returns with 2% annual risk-free rate: 0.02/252.
    annualization : float, default 252.0
        Number of periods per year for annualization.
        Use 252 for daily, 52 for weekly, 12 for monthly.

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Raises
    ------
    ValueError
        If returns array is empty or has insufficient data.

    Notes
    -----
    The Sharpe ratio is computed as:

        SR = sqrt(annualization) * mean(r - rf) / std(r)

    Where:
    - r: period returns
    - rf: risk-free rate per period
    - std: standard deviation of returns

    A higher Sharpe ratio indicates better risk-adjusted performance.
    Values above 1.0 are generally considered good, above 2.0 excellent.

    Examples
    --------
    >>> daily_returns = np.array([0.01, -0.005, 0.008, 0.003, -0.002])
    >>> sharpe = compute_sharpe_ratio(daily_returns)

    See Also
    --------
    compute_information_ratio : Measures active return relative to tracking error.
    compute_max_drawdown : Risk metric measuring largest decline.
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    if len(returns) < 2:
        raise ValueError("Need at least 2 return observations for Sharpe ratio")

    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)

    if std_returns == 0:
        # Zero volatility: undefined ratio (infinite if positive mean, neg if negative)
        if mean_excess > 0:
            return np.inf
        elif mean_excess < 0:
            return -np.inf
        else:
            return 0.0

    sharpe = np.sqrt(annualization) * mean_excess / std_returns

    return float(sharpe)


def compute_max_drawdown(
    cumulative_returns: Optional[ArrayLike] = None,
    returns: Optional[ArrayLike] = None,
) -> float:
    """
    Compute maximum drawdown from peak to trough.

    Maximum drawdown measures the largest decline from a historical peak
    in cumulative returns, representing the worst-case loss.

    Parameters
    ----------
    cumulative_returns : array-like, optional
        Cumulative returns (or price/equity curve). If provided, `returns`
        is ignored.
    returns : array-like, optional
        Period returns. Used to compute cumulative returns if
        `cumulative_returns` is not provided.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g., 0.20 = 20% drawdown).

    Raises
    ------
    ValueError
        If neither input is provided or arrays are empty.

    Notes
    -----
    The maximum drawdown is computed as:

        MDD = max_t [ (peak_t - trough_t) / peak_t ]

    where peak_t is the running maximum up to time t.

    Drawdown is always reported as a positive number (the magnitude of
    the decline). A 30% drawdown means the strategy lost 30% from its peak.

    Examples
    --------
    >>> cumulative = np.array([100, 110, 105, 120, 108, 125])
    >>> mdd = compute_max_drawdown(cumulative_returns=cumulative)
    >>> print(f"Max drawdown: {mdd:.1%}")  # From 120 to 108 = 10%

    >>> returns = np.array([0.10, -0.05, 0.15, -0.10, 0.16])
    >>> mdd = compute_max_drawdown(returns=returns)

    See Also
    --------
    compute_sharpe_ratio : Risk-adjusted return metric.
    compute_cumulative_return : Total return over period.
    """
    if cumulative_returns is None and returns is None:
        raise ValueError("Must provide either cumulative_returns or returns")

    if cumulative_returns is not None:
        curve = np.asarray(cumulative_returns, dtype=np.float64)
    else:
        returns = np.asarray(returns, dtype=np.float64)
        if len(returns) == 0:
            raise ValueError("Returns array cannot be empty")
        # Convert to cumulative returns (1 + r cumulative product)
        curve = np.cumprod(1 + returns)

    if len(curve) == 0:
        raise ValueError("Input array cannot be empty")

    # Running maximum
    running_max = np.maximum.accumulate(curve)

    # Drawdown at each point
    drawdowns = (running_max - curve) / running_max

    # Handle case where running_max is zero (shouldn't happen with valid data)
    drawdowns = np.nan_to_num(drawdowns, nan=0.0)

    return float(np.max(drawdowns))


def compute_cumulative_return(
    returns: ArrayLike,
    method: str = "geometric",
) -> float:
    """
    Compute cumulative return over the period.

    Parameters
    ----------
    returns : array-like
        Period returns (simple returns, not log returns for geometric method).
    method : {"geometric", "arithmetic"}, default "geometric"
        - "geometric": (1+r1) * (1+r2) * ... - 1 (correct for compounding)
        - "arithmetic": sum of returns (simple addition, ignores compounding)

    Returns
    -------
    float
        Cumulative return as a fraction (e.g., 0.25 = 25% total return).

    Raises
    ------
    ValueError
        If returns array is empty or invalid method specified.

    Notes
    -----
    Geometric compounding is the mathematically correct method for computing
    returns over multiple periods. Arithmetic is an approximation that works
    reasonably well for small returns.

    Examples
    --------
    >>> returns = np.array([0.05, 0.03, -0.02, 0.04])
    >>> cum_ret = compute_cumulative_return(returns)
    >>> print(f"Total return: {cum_ret:.2%}")

    See Also
    --------
    compute_max_drawdown : Largest decline from peak.
    compute_sharpe_ratio : Risk-adjusted return metric.
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    if method == "geometric":
        cumulative = np.prod(1 + returns) - 1
    elif method == "arithmetic":
        cumulative = np.sum(returns)
    else:
        raise ValueError(f"Invalid method '{method}'. Use 'geometric' or 'arithmetic'")

    return float(cumulative)


def compute_information_ratio(
    portfolio_returns: ArrayLike,
    benchmark_returns: ArrayLike,
    annualization: float = 252.0,
) -> float:
    """
    Compute information ratio (active return per unit tracking error).

    The information ratio measures how much excess return a portfolio
    generates relative to a benchmark, per unit of tracking error (the
    volatility of the excess returns).

    Parameters
    ----------
    portfolio_returns : array-like
        Portfolio period returns.
    benchmark_returns : array-like
        Benchmark period returns.
    annualization : float, default 252.0
        Number of periods per year for annualization.

    Returns
    -------
    float
        Annualized information ratio.

    Raises
    ------
    ValueError
        If arrays are empty, have different lengths, or insufficient data.

    Notes
    -----
    The information ratio is computed as:

        IR = sqrt(annualization) * mean(r_p - r_b) / std(r_p - r_b)

    Where:
    - r_p: portfolio returns
    - r_b: benchmark returns
    - std: standard deviation of active returns (tracking error)

    A higher IR indicates better risk-adjusted active performance.
    Values above 0.5 are generally considered good, above 1.0 excellent.

    Examples
    --------
    >>> portfolio = np.array([0.02, 0.01, -0.01, 0.03, 0.00])
    >>> benchmark = np.array([0.01, 0.01, 0.00, 0.02, 0.01])
    >>> ir = compute_information_ratio(portfolio, benchmark)

    See Also
    --------
    compute_sharpe_ratio : Similar metric for absolute returns.
    """
    portfolio_returns = np.asarray(portfolio_returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        raise ValueError("Return arrays cannot be empty")

    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError(
            f"Array lengths must match. "
            f"Got portfolio={len(portfolio_returns)}, benchmark={len(benchmark_returns)}"
        )

    if len(portfolio_returns) < 2:
        raise ValueError("Need at least 2 observations for information ratio")

    active_returns = portfolio_returns - benchmark_returns
    mean_active = np.mean(active_returns)
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        if mean_active > 0:
            return np.inf
        elif mean_active < 0:
            return -np.inf
        else:
            return 0.0

    ir = np.sqrt(annualization) * mean_active / tracking_error

    return float(ir)


def compute_hit_rate(
    predicted_changes: ArrayLike,
    actual_changes: ArrayLike,
) -> float:
    """
    Compute directional hit rate (fraction of correct direction predictions).

    Hit rate measures the percentage of times the predicted direction
    (up/down) matches the actual direction.

    Parameters
    ----------
    predicted_changes : array-like
        Predicted changes (differences, returns, or any signed values).
        Sign determines predicted direction.
    actual_changes : array-like
        Actual changes. Sign determines actual direction.

    Returns
    -------
    float
        Hit rate in [0, 1]. E.g., 0.60 means 60% of predictions had correct direction.

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.

    Notes
    -----
    Hit rate is computed as:

        hit_rate = mean( sign(pred) == sign(actual) )

    where sign(0) = 0 is treated as matching any sign (conservative).

    A hit rate above 0.5 indicates directional skill (better than random).
    For financial applications, hit rates of 0.52-0.55 can be valuable.

    Examples
    --------
    >>> predicted = np.array([0.01, -0.02, 0.01, 0.02, -0.01])
    >>> actual = np.array([0.02, -0.01, -0.01, 0.03, -0.02])
    >>> hr = compute_hit_rate(predicted, actual)
    >>> print(f"Hit rate: {hr:.1%}")  # 4/5 = 80%

    See Also
    --------
    compute_profit_factor : Weights directional accuracy by magnitude.
    """
    predicted = np.asarray(predicted_changes, dtype=np.float64)
    actual = np.asarray(actual_changes, dtype=np.float64)

    if len(predicted) == 0 or len(actual) == 0:
        raise ValueError("Arrays cannot be empty")

    if len(predicted) != len(actual):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predicted={len(predicted)}, actual={len(actual)}"
        )

    # Compare signs: both positive, both negative, or either is zero
    pred_sign = np.sign(predicted)
    actual_sign = np.sign(actual)

    # Match if: same sign, or either is zero (no clear direction)
    hits = (pred_sign == actual_sign) | (pred_sign == 0) | (actual_sign == 0)

    return float(np.mean(hits))


def compute_profit_factor(
    predicted_changes: ArrayLike,
    actual_changes: ArrayLike,
    returns: Optional[ArrayLike] = None,
) -> float:
    """
    Compute profit factor (gross profit / gross loss ratio).

    Profit factor measures the ratio of total profits to total losses
    from trading on directional predictions.

    Parameters
    ----------
    predicted_changes : array-like
        Predicted changes (sign determines trade direction).
    actual_changes : array-like
        Actual changes (used if returns not provided).
    returns : array-like, optional
        Actual returns to use for profit/loss calculation.
        If not provided, uses `actual_changes`.

    Returns
    -------
    float
        Profit factor. > 1.0 indicates profitable strategy.
        Returns np.inf if no losses, 0.0 if no profits.

    Raises
    ------
    ValueError
        If arrays are empty or have different lengths.

    Notes
    -----
    Trading logic:
    - If pred > 0: go long, P&L = return
    - If pred < 0: go short, P&L = -return
    - If pred == 0: no trade, P&L = 0

    Profit factor is then:
        PF = sum(positive P&L) / abs(sum(negative P&L))

    A profit factor of 1.5 means for every $1 lost, $1.50 is gained.
    Values above 1.0 indicate a profitable strategy, above 2.0 is excellent.

    Examples
    --------
    >>> predicted = np.array([1.0, -1.0, 1.0, -1.0])  # Buy, sell, buy, sell
    >>> actual = np.array([0.02, -0.01, -0.01, -0.02])
    >>> pf = compute_profit_factor(predicted, actual)
    >>> # Long on +2%, short on -1% (gain), long on -1% (loss), short on -2% (gain)
    >>> # Profits: 0.02 + 0.01 + 0.02 = 0.05
    >>> # Losses: 0.01
    >>> # PF = 0.05 / 0.01 = 5.0

    See Also
    --------
    compute_hit_rate : Simple directional accuracy (unweighted).
    compute_cumulative_return : Total return without trading logic.
    """
    predicted = np.asarray(predicted_changes, dtype=np.float64)
    actual = np.asarray(actual_changes, dtype=np.float64)

    if returns is not None:
        returns = np.asarray(returns, dtype=np.float64)
    else:
        returns = actual

    if len(predicted) == 0 or len(actual) == 0:
        raise ValueError("Arrays cannot be empty")

    if len(predicted) != len(returns):
        raise ValueError(
            f"Array lengths must match. "
            f"Got predicted={len(predicted)}, returns={len(returns)}"
        )

    # Compute P&L based on predicted direction
    pred_sign = np.sign(predicted)
    pnl = pred_sign * returns

    gross_profit: float = float(np.sum(pnl[pnl > 0]))
    gross_loss: float = float(np.abs(np.sum(pnl[pnl < 0])))

    if gross_loss == 0:
        if gross_profit > 0:
            return np.inf
        else:
            return 0.0

    return float(gross_profit / gross_loss)


def compute_calmar_ratio(
    returns: ArrayLike,
    annualization: float = 252.0,
) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).

    The Calmar ratio measures return relative to the worst historical
    decline, focusing on tail risk.

    Parameters
    ----------
    returns : array-like
        Period returns.
    annualization : float, default 252.0
        Number of periods per year for annualization.

    Returns
    -------
    float
        Calmar ratio. Higher is better.

    Raises
    ------
    ValueError
        If returns array is empty.

    Notes
    -----
    The Calmar ratio is computed as:

        Calmar = Annualized Return / Max Drawdown

    It measures how much return is generated per unit of worst-case risk.
    Useful for evaluating strategies where drawdowns are a key concern.

    Examples
    --------
    >>> returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    >>> calmar = compute_calmar_ratio(returns)

    See Also
    --------
    compute_sharpe_ratio : Uses volatility instead of drawdown.
    compute_max_drawdown : The denominator in Calmar ratio.
    """
    returns = np.asarray(returns, dtype=np.float64)

    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    # Annualized return
    cumulative = np.prod(1 + returns)
    n_periods = len(returns)
    annualized_return = cumulative ** (annualization / n_periods) - 1

    # Max drawdown
    max_dd = compute_max_drawdown(returns=returns)

    if max_dd == 0:
        if annualized_return > 0:
            return np.inf
        elif annualized_return < 0:
            return -np.inf
        else:
            return 0.0

    return float(annualized_return / max_dd)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "compute_cumulative_return",
    "compute_information_ratio",
    "compute_hit_rate",
    "compute_profit_factor",
    "compute_calmar_ratio",
]
