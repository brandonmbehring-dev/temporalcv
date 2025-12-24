"""
Wild Cluster Bootstrap for CV Fold Inference.

Bootstrap inference for test statistics across CV folds when standard
cluster standard errors are unreliable (few folds).

Knowledge Tiers
---------------
[T1] Wild bootstrap theory (Wu 1986, Liu 1988, Mammen 1993)
[T1] Webb 6-point distribution for few clusters (Webb 2023)
[T2] Application to CV folds assumes approximate independence
[T3] Threshold of 13 clusters for Rademacher vs Webb (empirical heuristic)

References
----------
[T1] Cameron, A.C., Gelbach, J.B. & Miller, D.L. (2008). Bootstrap-based
     improvements for inference with clustered errors. Review of Economics
     and Statistics, 90(3), 414-427.
[T1] Webb, M.D. (2023). Reworking wild bootstrap based inference for
     clustered errors. Canadian Journal of Economics, 56(3), 839-867.
[T1] MacKinnon, J.G. & Webb, M.D. (2017). Wild bootstrap inference for
     wildly different cluster sizes. Journal of Applied Econometrics,
     32(2), 233-254.

Example
-------
>>> from temporalcv.inference import wild_cluster_bootstrap
>>>
>>> # Per-fold DM test statistics
>>> fold_statistics = np.array([0.15, 0.22, -0.05, 0.18, 0.12])
>>> result = wild_cluster_bootstrap(fold_statistics, n_bootstrap=999)
>>> print(f"Estimate: {result.estimate:.3f}, SE: {result.se:.3f}")
>>> print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
>>> print(f"p-value: {result.p_value:.3f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class WildBootstrapResult:
    """
    Result of wild cluster bootstrap inference.

    Attributes
    ----------
    estimate : float
        Original test statistic (mean of fold_statistics)
    se : float
        Bootstrap standard error
    ci_lower : float
        Lower confidence interval bound
    ci_upper : float
        Upper confidence interval bound
    p_value : float
        Two-tailed p-value for H0: estimate = 0
    n_bootstrap : int
        Number of bootstrap replications performed
    n_clusters : int
        Number of folds/clusters
    weight_type : str
        Weight distribution used ("rademacher" or "webb")
    bootstrap_distribution : np.ndarray
        Full bootstrap sample distribution (for diagnostics)

    Notes
    -----
    The p-value tests H0: true statistic = 0 using the percentile method.
    For one-sided tests, divide p_value by 2.
    """

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_bootstrap: int
    n_clusters: int
    weight_type: str
    bootstrap_distribution: np.ndarray


def _rademacher_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Rademacher weights: {-1, +1} with p=0.5 each.

    Standard choice for many clusters (n >= 13).

    Parameters
    ----------
    n : int
        Number of weights to generate
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Array of {-1, +1} values
    """
    return cast(np.ndarray, rng.choice([-1, 1], size=n).astype(np.float64))


def _webb_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate Webb 6-point weights: {-1.5, -1.0, -0.5, +0.5, +1.0, +1.5}.

    Each value has probability 1/6. This distribution provides more
    bootstrap samples than Rademacher when clusters are few (n < 13),
    since 6^n > 2^n gives more unique combinations.

    Parameters
    ----------
    n : int
        Number of weights to generate
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Array of Webb 6-point weights

    References
    ----------
    Webb, M.D. (2023). Reworking wild bootstrap based inference for
    clustered errors. Canadian Journal of Economics, 56(3), 839-867.
    """
    webb_values = np.array([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
    return cast(np.ndarray, rng.choice(webb_values, size=n))


def wild_cluster_bootstrap(
    fold_statistics: ArrayLike,
    n_bootstrap: int = 999,
    weight_type: Literal["auto", "rademacher", "webb"] = "auto",
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> WildBootstrapResult:
    """
    Wild cluster bootstrap for CV fold-level inference.

    Computes bootstrap standard errors and confidence intervals for
    test statistics computed across CV folds. Useful when the number
    of folds is small (typical: 5-10) and standard asymptotic inference
    is unreliable.

    Knowledge Tier: [T2] - Wild bootstrap well-established, but CV fold
    independence assumption is domain-specific.

    Parameters
    ----------
    fold_statistics : ArrayLike
        Test statistic value for each fold (e.g., per-fold DM statistic,
        mean squared error, or any fold-level metric). Shape: (n_folds,)
    n_bootstrap : int, default=999
        Number of bootstrap replications. Use odd number for symmetric
        percentile CI computation.
    weight_type : {"auto", "rademacher", "webb"}, default="auto"
        Weight distribution:
        - "auto": Webb if n_folds < 13, Rademacher otherwise
        - "rademacher": {-1, +1} with p=0.5 each
        - "webb": {-1.5, -1.0, -0.5, +0.5, +1.0, +1.5} with p=1/6 each
    alpha : float, default=0.05
        Significance level for confidence interval (1 - alpha CI)
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    WildBootstrapResult
        Bootstrap inference results including SE, CI, and p-value

    Raises
    ------
    ValueError
        If fold_statistics has fewer than 2 elements
        If n_bootstrap < 1
        If alpha not in (0, 1)
        If weight_type not in {"auto", "rademacher", "webb"}

    Notes
    -----
    **Algorithm**:
    1. Compute original estimate as mean of fold_statistics
    2. For b = 1 to n_bootstrap:
       - Generate cluster-level weights (one weight per fold)
       - Compute weighted mean: sum(w_k * d_k) / n_folds
       - Store bootstrap sample
    3. SE = std(bootstrap_distribution)
    4. CI from percentiles of bootstrap distribution
    5. P-value: proportion of |bootstrap| >= |original|

    **⚠️ CAVEAT**: CV folds with overlapping training windows are NOT
    fully independent clusters. This method assumes approximate independence
    which holds better for:
    - Sliding window CV (non-overlapping training)
    - Large datasets where overlap is small relative to fold size
    - Weak temporal dependence in the data

    For expanding window CV with strong dependence, results may be
    anti-conservative. Consider fold-residual bootstrap as alternative
    [Cameron et al. 2008].

    Example
    -------
    >>> # Per-fold error differences from DM test
    >>> fold_diffs = np.array([0.15, 0.22, -0.05, 0.18, 0.12])
    >>> result = wild_cluster_bootstrap(fold_diffs)
    >>> if result.p_value < 0.05:
    ...     print("Significant difference between models")

    References
    ----------
    Cameron, A.C., Gelbach, J.B. & Miller, D.L. (2008). Bootstrap-based
    improvements for inference with clustered errors.
    Webb, M.D. (2023). Reworking wild bootstrap based inference for
    clustered errors.

    Complexity: O(n_bootstrap × n_folds)

    See Also
    --------
    dm_test : Test that produces fold-level statistics.
    BootstrapUncertainty : Alternative bootstrap for prediction intervals.
    WalkForwardCV : CV strategy that produces fold-level results.
    """
    # Input validation
    fold_statistics = np.asarray(fold_statistics, dtype=np.float64)

    if fold_statistics.ndim != 1:
        raise ValueError(
            f"fold_statistics must be 1-dimensional, got shape {fold_statistics.shape}"
        )

    n_folds = len(fold_statistics)

    if n_folds < 2:
        raise ValueError(
            f"fold_statistics must have at least 2 elements, got {n_folds}"
        )

    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if weight_type not in ("auto", "rademacher", "webb"):
        raise ValueError(
            f"weight_type must be 'auto', 'rademacher', or 'webb', got {weight_type!r}"
        )

    # Initialize RNG
    rng = np.random.default_rng(random_state)

    # Select weight distribution
    if weight_type == "auto":
        # Webb for few clusters, Rademacher for many
        # Threshold of 13: 6^13 ≈ 13 billion vs 2^13 = 8192
        actual_weight_type = "webb" if n_folds < 13 else "rademacher"
    else:
        actual_weight_type = weight_type

    weight_fn = _webb_weights if actual_weight_type == "webb" else _rademacher_weights

    # Warn for very few clusters
    if n_folds < 6:
        import warnings
        warnings.warn(
            f"Only {n_folds} folds/clusters. Wild bootstrap inference may be "
            f"unreliable. Consider using more folds or reporting results with caution.",
            UserWarning,
            stacklevel=2,
        )

    # Original estimate
    original_estimate = float(np.mean(fold_statistics))

    # Bootstrap loop
    bootstrap_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Generate weights (one per fold)
        weights = weight_fn(n_folds, rng)

        # Compute weighted mean
        # This is the wild bootstrap analog of resampling the fold statistics
        bootstrap_samples[b] = np.mean(weights * fold_statistics)

    # Compute standard error
    se = float(np.std(bootstrap_samples, ddof=1))

    # Compute confidence interval (percentile method)
    ci_lower = float(np.percentile(bootstrap_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_samples, 100 * (1 - alpha / 2)))

    # Compute p-value (two-tailed)
    # How often does the bootstrap distribution exceed the original estimate
    # in absolute value? This tests H0: true statistic = 0
    centered_bootstrap = bootstrap_samples - np.mean(bootstrap_samples)
    p_value = float(np.mean(np.abs(centered_bootstrap) >= np.abs(original_estimate)))

    # Handle edge case: if original is exactly 0
    if np.abs(original_estimate) < 1e-15:
        p_value = 1.0

    return WildBootstrapResult(
        estimate=original_estimate,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
        n_clusters=n_folds,
        weight_type=actual_weight_type,
        bootstrap_distribution=bootstrap_samples,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
]
