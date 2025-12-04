"""
Moving Block Bootstrap Strategy.

Implements the Moving Block Bootstrap (MBB) method from Kunsch (1989).
Samples overlapping blocks of fixed length to preserve local autocorrelation.

Example
-------
>>> from temporalcv.bagging import MovingBlockBootstrap
>>>
>>> strategy = MovingBlockBootstrap(block_length=10)
>>> samples = strategy.generate_samples(X, y, n_samples=20, rng=rng)

References
----------
- Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General
  Stationary Observations." Annals of Statistics, 17(3), 1217-1241.
- Lahiri (1999). "Theoretical comparisons of block bootstrap methods"
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

from temporalcv.bagging.base import BootstrapStrategy


class MovingBlockBootstrap(BootstrapStrategy):
    """
    Moving Block Bootstrap (Kunsch, 1989).

    Samples overlapping blocks of fixed length to preserve local
    autocorrelation structure. The resampled series has the same
    length as the original.

    Parameters
    ----------
    block_length : int or None, default=None
        Length of each block. If None, auto-compute as int(n^(1/3))
        which is asymptotically optimal per Kunsch (1989).

    Attributes
    ----------
    block_length : int or None
        Configured block length (may be None for auto-detect)

    Notes
    -----
    - Block length O(n^1/3) is optimal for bias-variance tradeoff
    - Blocks are sampled WITH replacement from overlapping positions
    - Uses vectorized NumPy operations for performance

    Warning
    -------
    Block bootstrap assumes temporal ordering and weakly-dependent data.
    For highly persistent series (ACF(1) > 0.95), consider larger block_length
    to better preserve the dependence structure. The asymptotic theory
    assumes stationarity and mixing conditions that may not hold for
    unit-root or near-unit-root processes.

    Examples
    --------
    >>> strategy = MovingBlockBootstrap(block_length=10)
    >>> # Or auto-detect block length:
    >>> strategy = MovingBlockBootstrap()

    References
    ----------
    Kunsch (1989), Ann. Stat. 17(3): Block length O(n^{1/3}) optimal
    """

    def __init__(self, block_length: Optional[int] = None):
        self.block_length = block_length

    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap samples using Moving Block Bootstrap.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_obs, n_features)
        y : np.ndarray
            Target series (n_obs,)
        n_samples : int
            Number of bootstrap samples
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            Bootstrap samples [(X_boot, y_boot), ...]
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)

        # Auto block length: O(n^1/3), clamped to [1, n]
        if self.block_length is None:
            block_len = max(1, min(n, int(n ** (1 / 3))))
        else:
            block_len = max(1, min(n, self.block_length))
            if self.block_length > n:
                warnings.warn(
                    f"block_length={self.block_length} > n={n}, clamped to {block_len}"
                )

        # Number of blocks needed to cover n observations
        n_blocks = max(1, (n + block_len - 1) // block_len)

        # VECTORIZED: Generate all block starts for all samples at once
        # Shape: (n_samples, n_blocks)
        max_start = max(1, n - block_len + 1)
        all_block_starts = rng.integers(0, max_start, size=(n_samples, n_blocks))

        # VECTORIZED: Create block offset array once
        block_offsets = np.arange(block_len)

        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_samples):
            # VECTORIZED: Compute all indices using broadcasting
            # block_starts[:, np.newaxis] + block_offsets -> (n_blocks, block_len)
            indices = (all_block_starts[i, :, np.newaxis] + block_offsets).ravel()[:n]

            # Use numpy fancy indexing
            X_boot = X[indices]
            y_boot = y[indices]
            samples.append((X_boot, y_boot))

        return samples

    def __repr__(self) -> str:
        return f"MovingBlockBootstrap(block_length={self.block_length})"


# =============================================================================
# Public API
# =============================================================================

__all__ = ["MovingBlockBootstrap"]
