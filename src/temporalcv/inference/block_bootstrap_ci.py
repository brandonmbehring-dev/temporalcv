"""
Block Bootstrap Confidence Intervals.

Moving Block Bootstrap (MBB) for computing confidence intervals on statistics
from time series data while preserving temporal dependence structure.

Knowledge Tiers
---------------
[T1] Moving Block Bootstrap theory (Kunsch 1989, Liu & Singh 1992)
[T1] Optimal block length n^(1/3) (Politis & Romano 1994)
[T1] Percentile CI method (Efron & Tibshirani 1993)
[T3] Gate-specific adaptations are implementation choices

References
----------
[T1] Kunsch, H.R. (1989). The jackknife and the bootstrap for general
     stationary observations. Annals of Statistics, 17(3), 1217-1241.
[T1] Liu, R.Y. & Singh, K. (1992). Moving blocks jackknife and bootstrap
     capture weak dependence. Exploring the limits of bootstrap.
[T1] Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
     Journal of the American Statistical Association, 89(428), 1303-1313.
[T1] Lahiri, S.N. (2003). Resampling Methods for Dependent Data. Springer.

Example
-------
>>> from temporalcv.inference import moving_block_bootstrap
>>> import numpy as np
>>>
>>> # Time series data
>>> data = np.random.randn(100)
>>> result = moving_block_bootstrap(
...     data,
...     statistic_fn=np.mean,
...     n_bootstrap=200,
...     alpha=0.05,
... )
>>> print(f"Estimate: {result.estimate:.3f}")
>>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlockBootstrapResult:
    """
    Result of block bootstrap confidence interval estimation.

    Attributes
    ----------
    estimate : float
        Point estimate of the statistic on original data
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float
        Upper confidence interval bound
    alpha : float
        Significance level used (default 0.05 for 95% CI)
    std_error : float
        Bootstrap standard error
    n_bootstrap : int
        Number of bootstrap replications performed
    block_length : int
        Block length used for resampling
    bootstrap_distribution : np.ndarray
        Full bootstrap sample distribution (for diagnostics)

    Notes
    -----
    The CI is computed using the percentile method, which is appropriate
    for most statistics. For statistics with known bias, consider
    bias-corrected accelerated (BCa) intervals (not implemented here).
    """

    estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    std_error: float
    n_bootstrap: int
    block_length: int
    bootstrap_distribution: NDArray[np.float64]


def compute_block_length(n: int) -> int:
    """
    Compute optimal block length using n^(1/3) rule.

    This is the asymptotically optimal rate for the moving block bootstrap
    when estimating the variance of sample means from stationary time series.

    Knowledge Tier: [T1] - Established asymptotic result.

    Parameters
    ----------
    n : int
        Sample size (length of time series)

    Returns
    -------
    int
        Optimal block length, minimum 1

    References
    ----------
    Kunsch, H.R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics, 17(3), 1217-1241.

    Politis, D.N. & Romano, J.P. (1994). The stationary bootstrap.
    Journal of the American Statistical Association, 89(428), 1303-1313.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return max(1, int(np.floor(n ** (1 / 3))))


def _create_block_indices(
    n: int,
    block_length: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """
    Create bootstrap sample indices using moving block bootstrap.

    Samples overlapping blocks with replacement and concatenates them
    to form a bootstrap sample of approximately length n.

    Parameters
    ----------
    n : int
        Original sample size
    block_length : int
        Length of each block
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Indices for bootstrap sample (length may be slightly > n)
    """
    # Number of blocks we can form
    n_blocks = n - block_length + 1
    if n_blocks < 1:
        # Block length >= n, use entire series
        return np.arange(n)

    # Number of blocks needed to cover n observations
    n_blocks_needed = int(np.ceil(n / block_length))

    # Sample block start positions
    block_starts = rng.integers(0, n_blocks, size=n_blocks_needed)

    # Create indices by concatenating blocks
    indices = []
    for start in block_starts:
        indices.extend(range(start, start + block_length))

    # Truncate to exactly n observations
    return np.array(indices[:n], dtype=np.intp)


def moving_block_bootstrap(
    data: NDArray[np.floating],
    statistic_fn: Callable[[NDArray[np.floating]], float],
    n_bootstrap: int = 100,
    block_length: Union[int, str] = "auto",
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> BlockBootstrapResult:
    """
    Compute block bootstrap confidence interval for a statistic.

    Uses the Moving Block Bootstrap (MBB) to preserve temporal dependence
    when computing confidence intervals. Blocks of consecutive observations
    are sampled with replacement and concatenated.

    Knowledge Tier: [T1] - MBB is established for stationary time series.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape: (n,) for 1D or (n, d) for multivariate.
        For 2D data, blocks are sampled along axis 0.
    statistic_fn : callable
        Function that computes a scalar statistic from data.
        Signature: fn(data) -> float
    n_bootstrap : int, default=100
        Number of bootstrap replications.
    block_length : int or "auto", default="auto"
        Block length for MBB. If "auto", uses n^(1/3) rule.
    alpha : float, default=0.05
        Significance level for confidence interval (1 - alpha CI).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BlockBootstrapResult
        Bootstrap CI result including estimate, bounds, and diagnostics.

    Raises
    ------
    ValueError
        If data has fewer than 2 elements
        If n_bootstrap < 1
        If alpha not in (0, 1)
        If block_length < 1 (when int)

    Notes
    -----
    **Algorithm** (Kunsch 1989):

    1. Compute original statistic: theta_hat = statistic_fn(data)
    2. Determine block length l (default: n^(1/3))
    3. Form overlapping blocks: B_i = (data[i], ..., data[i+l-1])
    4. For b = 1 to n_bootstrap:
       - Sample k = ceil(n/l) blocks with replacement
       - Concatenate to form bootstrap sample
       - Compute theta_b* = statistic_fn(bootstrap_sample)
    5. CI from percentiles: [percentile(alpha/2), percentile(1-alpha/2)]

    **When to use**:

    - Time series with temporal dependence
    - Statistics computed on full series (mean, variance, quantiles)
    - Gate metrics where blocks preserve local structure

    **Caveats**:

    - Assumes approximate stationarity
    - Block length selection affects performance
    - For non-stationary data, consider tapered block bootstrap

    Example
    -------
    >>> import numpy as np
    >>> # AR(1) process
    >>> np.random.seed(42)
    >>> n = 200
    >>> phi = 0.7
    >>> data = np.zeros(n)
    >>> for t in range(1, n):
    ...     data[t] = phi * data[t-1] + np.random.randn()
    >>> result = moving_block_bootstrap(data, np.mean, n_bootstrap=500)
    >>> print(f"Mean: {result.estimate:.3f}")
    >>> print(f"SE: {result.std_error:.3f}")
    >>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")

    References
    ----------
    Kunsch, H.R. (1989). The jackknife and the bootstrap for general
    stationary observations. Annals of Statistics, 17(3), 1217-1241.

    Complexity: O(n_bootstrap * n * statistic_fn_complexity)

    See Also
    --------
    wild_cluster_bootstrap : For CV fold-level inference.
    """
    # Ensure numpy array
    data = np.asarray(data, dtype=np.float64)

    # Validate data shape
    if data.ndim == 0:
        raise ValueError("data must be at least 1-dimensional")

    n = data.shape[0]

    if n < 2:
        raise ValueError(f"data must have at least 2 elements, got {n}")

    # Validate n_bootstrap
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    # Validate alpha
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    # Determine block length
    if block_length == "auto":
        bl = compute_block_length(n)
    else:
        if not isinstance(block_length, int):
            raise TypeError(
                f"block_length must be int or 'auto', got {type(block_length).__name__}"
            )
        if block_length < 1:
            raise ValueError(f"block_length must be >= 1, got {block_length}")
        bl = block_length

    # Warn if block length is very large relative to n
    if bl > n // 2:
        import warnings

        warnings.warn(
            f"Block length ({bl}) is more than half the sample size ({n}). "
            f"Consider using a smaller block length for more bootstrap variability.",
            UserWarning,
            stacklevel=2,
        )

    # Initialize RNG
    rng = np.random.default_rng(random_state)

    # Compute original estimate
    try:
        original_estimate = float(statistic_fn(data))
    except Exception as e:
        raise ValueError(f"statistic_fn failed on original data: {e}") from e

    # Bootstrap loop
    bootstrap_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Create bootstrap indices
        indices = _create_block_indices(n, bl, rng)

        # Get bootstrap sample
        if data.ndim == 1:
            bootstrap_data = data[indices]
        else:
            bootstrap_data = data[indices, :]

        # Compute statistic
        try:
            bootstrap_samples[b] = statistic_fn(bootstrap_data)
        except Exception:
            # If statistic fails on bootstrap sample, use NaN
            bootstrap_samples[b] = np.nan

    # Remove any failed bootstrap samples
    valid_samples = bootstrap_samples[~np.isnan(bootstrap_samples)]

    if len(valid_samples) < n_bootstrap * 0.5:
        import warnings

        warnings.warn(
            f"More than 50% of bootstrap samples failed. "
            f"Only {len(valid_samples)}/{n_bootstrap} valid samples.",
            UserWarning,
            stacklevel=2,
        )

    if len(valid_samples) < 2:
        raise ValueError(
            "Fewer than 2 valid bootstrap samples. "
            "Check that statistic_fn works on resampled data."
        )

    # Compute standard error
    std_error = float(np.std(valid_samples, ddof=1))

    # Compute confidence interval (percentile method)
    ci_lower = float(np.percentile(valid_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(valid_samples, 100 * (1 - alpha / 2)))

    return BlockBootstrapResult(
        estimate=original_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        std_error=std_error,
        n_bootstrap=n_bootstrap,
        block_length=bl,
        bootstrap_distribution=valid_samples,
    )


# =============================================================================
# Convenience functions for common statistics
# =============================================================================


def bootstrap_ci_mean(
    data: NDArray[np.floating],
    n_bootstrap: int = 100,
    block_length: Union[int, str] = "auto",
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> BlockBootstrapResult:
    """
    Compute block bootstrap CI for the mean.

    Convenience wrapper around moving_block_bootstrap for the sample mean.

    Parameters
    ----------
    data : np.ndarray
        Time series data. Shape: (n,) for 1D.
    n_bootstrap : int, default=100
        Number of bootstrap replications.
    block_length : int or "auto", default="auto"
        Block length for MBB.
    alpha : float, default=0.05
        Significance level (1 - alpha CI).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BlockBootstrapResult
        Bootstrap CI result for the mean.
    """
    return moving_block_bootstrap(
        data=data,
        statistic_fn=lambda x: float(np.mean(x)),
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        alpha=alpha,
        random_state=random_state,
    )


def bootstrap_ci_mae(
    errors: NDArray[np.floating],
    n_bootstrap: int = 100,
    block_length: Union[int, str] = "auto",
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> BlockBootstrapResult:
    """
    Compute block bootstrap CI for Mean Absolute Error.

    Convenience wrapper for MAE, commonly used in forecast evaluation.

    Parameters
    ----------
    errors : np.ndarray
        Forecast errors (actual - predicted). Shape: (n,)
    n_bootstrap : int, default=100
        Number of bootstrap replications.
    block_length : int or "auto", default="auto"
        Block length for MBB.
    alpha : float, default=0.05
        Significance level (1 - alpha CI).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    BlockBootstrapResult
        Bootstrap CI result for MAE.
    """
    return moving_block_bootstrap(
        data=errors,
        statistic_fn=lambda x: float(np.mean(np.abs(x))),
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        alpha=alpha,
        random_state=random_state,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "BlockBootstrapResult",
    "compute_block_length",
    "moving_block_bootstrap",
    "bootstrap_ci_mean",
    "bootstrap_ci_mae",
]
