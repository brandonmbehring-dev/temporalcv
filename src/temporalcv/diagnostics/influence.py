"""
Influence diagnostics for statistical tests.

Identifies which observations disproportionately affect test statistics,
helping diagnose whether results are driven by a few influential points.

Knowledge Tier: [T2] - Influence functions are well-established in statistics,
but HAC-adjusted variants for time series are empirical best practice.

References
----------
[T1] Cook, R.D. (1977). Detection of Influential Observation in Linear Regression.
     Technometrics, 19(1), 15-18.
[T2] Künsch, H.R. (1989). The Jackknife and the Bootstrap for General Stationary
     Observations. Annals of Statistics, 17(3), 1217-1241.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np

from temporalcv.statistical_tests import compute_hac_variance


@dataclass(frozen=True)
class InfluenceDiagnostic:
    """
    Result of influence analysis on DM test statistic.

    Provides two views of influence:
    1. observation_influence: Per-observation scores (granular, for exploration)
    2. block_influence: Per-block scores (robust, for decisions)

    Attributes
    ----------
    observation_influence : np.ndarray
        Per-observation influence scores using HAC-adjusted formula.
        ψ_i = (d_i - d̄) / √(HAC_var * n)
    observation_high_mask : np.ndarray
        Boolean mask: True where |observation_influence| > threshold * std
    block_influence : np.ndarray
        Per-block influence scores using block jackknife.
        More robust for autocorrelated data.
    block_high_mask : np.ndarray
        Boolean mask: True where |block_influence| > threshold * std
    block_indices : List[Tuple[int, int]]
        (start, end) indices for each block (end is exclusive)
    n_high_influence_obs : int
        Count of high-influence observations
    n_high_influence_blocks : int
        Count of high-influence blocks
    influence_threshold : float
        Threshold multiplier used (e.g., 2.0 means 2σ)

    Notes
    -----
    **Recommendation**: Use block_influence for decisions as it properly
    accounts for serial correlation in time series. Use observation_influence
    for exploratory analysis to identify specific problematic points.
    """

    observation_influence: np.ndarray
    observation_high_mask: np.ndarray
    block_influence: np.ndarray
    block_high_mask: np.ndarray
    block_indices: List[Tuple[int, int]]
    n_high_influence_obs: int
    n_high_influence_blocks: int
    influence_threshold: float


def compute_dm_influence(
    errors1: np.ndarray,
    errors2: np.ndarray,
    h: int = 1,
    loss: Literal["squared", "absolute"] = "squared",
    influence_threshold: float = 2.0,
) -> InfluenceDiagnostic:
    """
    Compute influence of each observation on DM test statistic.

    Provides TWO complementary views:

    1. **Observation-level** (HAC-adjusted influence function):
       - Granular per-observation scores
       - Formula: ψ_i = (d_i - d̄) / √(HAC_var * n)
       - Best for: Exploratory analysis, identifying specific outliers

    2. **Block-level** (Block jackknife):
       - Leave-block-out influence scores
       - Block size = max(h, 1) to account for forecast horizon
       - Best for: Decision-making, robust to autocorrelation

    Knowledge Tier: [T2] - Combines standard influence theory with
    HAC variance for time series robustness.

    Parameters
    ----------
    errors1 : np.ndarray
        Forecast errors from model 1 (actual - prediction)
    errors2 : np.ndarray
        Forecast errors from model 2 (baseline)
    h : int, default=1
        Forecast horizon. Used for HAC bandwidth and block size.
    loss : {"squared", "absolute"}, default="squared"
        Loss function: "squared" for MSE comparison, "absolute" for MAE.
    influence_threshold : float, default=2.0
        Multiplier for standard deviation to flag high-influence points.
        2.0 means flag if |influence| > 2 * std(influences).

    Returns
    -------
    InfluenceDiagnostic
        Contains both observation-level and block-level influence diagnostics.

    Notes
    -----
    **Caveat**: For time series with autocorrelated errors, observations are
    not independent. The block-level influence accounts for this by using
    non-overlapping blocks of size h. The observation-level influence is
    HAC-adjusted but still treats observations individually.

    Example
    -------
    >>> errors1 = model1_predictions - actuals
    >>> errors2 = baseline_predictions - actuals
    >>> diag = compute_dm_influence(errors1, errors2, h=4)
    >>> print(f"High-influence blocks: {diag.n_high_influence_blocks}")
    >>> # Check which blocks are driving the DM result
    >>> for i, (start, end) in enumerate(diag.block_indices):
    ...     if diag.block_high_mask[i]:
    ...         print(f"Block {i} (indices {start}:{end}): influence={diag.block_influence[i]:.3f}")

    See Also
    --------
    dm_test : The test statistic being analyzed.
    compute_hac_variance : HAC variance used for observation-level influence.
    gap_sensitivity_analysis : Complementary sensitivity diagnostic.
    """
    errors1 = np.asarray(errors1, dtype=np.float64)
    errors2 = np.asarray(errors2, dtype=np.float64)

    if len(errors1) != len(errors2):
        raise ValueError(
            f"Error arrays must have same length: {len(errors1)} vs {len(errors2)}"
        )

    n = len(errors1)
    if n < 10:
        raise ValueError(f"Need at least 10 observations for influence analysis, got {n}")

    # Compute loss differentials
    if loss == "squared":
        d = errors1**2 - errors2**2
    else:  # absolute
        d = np.abs(errors1) - np.abs(errors2)

    d_mean = np.mean(d)

    # ==========================================================================
    # Observation-level influence (HAC-adjusted)
    # ==========================================================================
    # HAC variance with bandwidth = h - 1 (appropriate for h-step forecasts)
    bandwidth = max(h - 1, 0) if h > 1 else None
    hac_var = compute_hac_variance(d, bandwidth=bandwidth)

    # Influence function: ψ_i = (d_i - d̄) / √(HAC_var * n)
    # This measures how much each observation "pulls" the mean
    if hac_var > 0:
        obs_influence = (d - d_mean) / np.sqrt(hac_var * n)
    else:
        # Degenerate case: all d_i are identical
        obs_influence = np.zeros(n)

    # Flag high-influence observations
    obs_std = np.std(obs_influence)
    if obs_std > 0:
        obs_high_mask = np.abs(obs_influence) > influence_threshold * obs_std
    else:
        obs_high_mask = np.zeros(n, dtype=bool)

    # ==========================================================================
    # Block-level influence (block jackknife)
    # ==========================================================================
    block_size = max(h, 1)
    n_blocks = n // block_size

    if n_blocks < 2:
        # Not enough for block jackknife, use single-observation blocks
        block_size = 1
        n_blocks = n

    block_indices: List[Tuple[int, int]] = []
    block_influence_list: List[float] = []

    # Compute full DM statistic (mean of d)
    full_dm = d_mean

    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, n)
        block_indices.append((start, end))

        # Leave-block-out: compute DM without this block
        mask: np.ndarray = np.ones(n, dtype=bool)
        mask[start:end] = False
        d_without = d[mask]

        if len(d_without) > 0:
            dm_without = np.mean(d_without)
            # Influence = (full - leave_out) * sqrt(n)
            # Positive influence means removing block decreases DM stat
            influence_b = (full_dm - dm_without) * np.sqrt(n)
        else:
            influence_b = 0.0

        block_influence_list.append(influence_b)

    block_influence = np.array(block_influence_list)

    # Flag high-influence blocks
    block_std = np.std(block_influence)
    if block_std > 0:
        block_high_mask = np.abs(block_influence) > influence_threshold * block_std
    else:
        block_high_mask = np.zeros(n_blocks, dtype=bool)

    return InfluenceDiagnostic(
        observation_influence=obs_influence,
        observation_high_mask=obs_high_mask,
        block_influence=block_influence,
        block_high_mask=block_high_mask,
        block_indices=block_indices,
        n_high_influence_obs=int(np.sum(obs_high_mask)),
        n_high_influence_blocks=int(np.sum(block_high_mask)),
        influence_threshold=influence_threshold,
    )
