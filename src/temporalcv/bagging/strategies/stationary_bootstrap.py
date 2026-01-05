"""
Stationary Bootstrap Strategy.

Implements the Stationary Bootstrap method from Politis & Romano (1994).
Uses geometric distribution for block lengths, producing stationary
resampled series.

Example
-------
>>> from temporalcv.bagging import StationaryBootstrap
>>>
>>> strategy = StationaryBootstrap(expected_block_length=10.0)
>>> samples = strategy.generate_samples(X, y, n_samples=20, rng=rng)

References
----------
- Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap."
  JASA, 89(428), 1303-1313.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from temporalcv.bagging.base import BootstrapStrategy


def _stationary_bootstrap_indices(
    n: int,
    p: float,
    uniforms: np.ndarray,
    jump_targets: np.ndarray,
    start_idx: int,
) -> np.ndarray:
    """
    Generate stationary bootstrap indices using Markov chain.

    Parameters
    ----------
    n : int
        Number of indices to generate
    p : float
        Probability of jumping to random position
    uniforms : np.ndarray
        Pre-generated uniform random values (length n)
    jump_targets : np.ndarray
        Pre-generated jump target indices (length n)
    start_idx : int
        Starting index

    Returns
    -------
    np.ndarray
        Bootstrap indices of length n
    """
    indices: np.ndarray = np.empty(n, dtype=np.int64)
    i: int = start_idx

    for j in range(n):
        indices[j] = i
        if uniforms[j] < p:
            i = int(jump_targets[j])
        else:
            i = (i + 1) % n

    result: np.ndarray = indices
    return result


class StationaryBootstrap(BootstrapStrategy):
    """
    Stationary Bootstrap (Politis & Romano, 1994).

    Uses geometric distribution for block lengths, producing stationary
    resampled series. More robust to block length choice than MBB.

    Parameters
    ----------
    expected_block_length : float or None, default=None
        Expected block length (1/p where p is the geometric parameter).
        If None, auto-compute as n^(1/3).

    Attributes
    ----------
    expected_block_length : float or None
        Configured expected block length

    Notes
    -----
    - At each step, with probability p=1/expected_block_length, jump to
      a random position; otherwise continue to the next observation
    - Uses circular wrap: after position n-1 comes position 0
    - Per-sample random generation avoids shared index exhaustion

    Warning
    -------
    Stationary bootstrap assumes temporal ordering and weakly-dependent data.
    For highly persistent series (ACF(1) > 0.95), consider larger
    expected_block_length to better preserve the dependence structure.
    The asymptotic theory assumes stationarity and mixing conditions
    that may not hold for unit-root or near-unit-root processes.

    Examples
    --------
    >>> strategy = StationaryBootstrap(expected_block_length=10.0)
    >>> # Or auto-detect:
    >>> strategy = StationaryBootstrap()

    References
    ----------
    Politis & Romano (1994), JASA 89(428): Geometric block lengths
    """

    def __init__(self, expected_block_length: Optional[float] = None):
        self.expected_block_length = expected_block_length

    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap samples using Stationary Bootstrap.

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

        # Auto block length: O(n^1/3), clamped
        if self.expected_block_length is None:
            exp_len = max(1.0, n ** (1 / 3))
        else:
            exp_len = max(1.0, self.expected_block_length)

        # Geometric parameter: probability of jumping
        p = 1.0 / exp_len

        samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for _ in range(n_samples):
            # PER-SAMPLE random generation (avoids shared index exhaustion)
            uniforms = rng.random(n)
            jump_targets = rng.integers(0, n, size=n)
            start_idx = int(rng.integers(0, n))

            # Build indices using Markov chain
            indices = _stationary_bootstrap_indices(n, p, uniforms, jump_targets, start_idx)

            # Numpy fancy indexing
            X_boot = X[indices]
            y_boot = y[indices]
            samples.append((X_boot, y_boot))

        return samples

    def __repr__(self) -> str:
        return f"StationaryBootstrap(expected_block_length={self.expected_block_length})"


# =============================================================================
# Public API
# =============================================================================

__all__ = ["StationaryBootstrap"]
