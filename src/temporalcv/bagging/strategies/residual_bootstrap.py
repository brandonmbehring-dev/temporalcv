"""
Residual Bootstrap Strategy with STL Decomposition.

Implements model-based residual bootstrap for nonstationary time series.
Uses STL (Seasonal-Trend decomposition using LOESS) to extract components,
then bootstraps residuals while preserving trend and seasonality.

Algorithm (per Bergmeir, Hyndman & Benitez 2016)
------------------------------------------------
1. Decompose y using STL: y = trend + seasonal + residual
2. Bootstrap residuals using block bootstrap (preserves autocorrelation)
3. Reconstruct: y_boot = trend + seasonal + residual_boot
4. X stays unchanged (preserves deterministic X→y relationship)

References
----------
- Bergmeir, Hyndman & Benitez (2016). "Bagging Exponential Smoothing Methods."
  IJF 32(2), 303-312.
- Cleveland et al. (1990). "STL: A Seasonal-Trend Decomposition."
  J. Official Statistics, 6(1), 3-73.
- Hyndman & Athanasopoulos (2021). FPP3 Section 12.5: STL + residual bootstrap
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from temporalcv.bagging.base import BootstrapStrategy


# Optional statsmodels import for STL
try:
    from statsmodels.tsa.seasonal import STL

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    STL = None  # type: ignore[misc, assignment]


class ResidualBootstrap(BootstrapStrategy):
    """Residual Bootstrap with STL Decomposition.

    For nonstationary series with trend and/or seasonality, decomposes
    the target into components, bootstraps only the residuals (using
    block bootstrap to preserve autocorrelation), then reconstructs.

    Parameters
    ----------
    seasonal_period : int | None
        Seasonal period for STL decomposition.
        - If None, auto-detect based on series length (default 52 for weekly)
        - Common values: 7 (daily→weekly), 12 (monthly→yearly), 52 (weekly→yearly)
    block_length : int | None
        Block length for residual MBB. If None, auto-compute as n^(1/3).
    robust : bool
        Use robust STL fitting (default True). More resistant to outliers.
    stl_kwargs : dict | None
        Additional keyword arguments passed to statsmodels STL.

    Attributes
    ----------
    decomposition_ : STL result | None
        Stored decomposition from most recent generate_samples call.
        Contains trend, seasonal, resid components.

    Examples
    --------
    >>> import numpy as np
    >>> from temporalcv.bagging import TimeSeriesBagger, ResidualBootstrap
    >>> from sklearn.linear_model import Ridge
    >>>
    >>> # Weekly data with yearly seasonality
    >>> strategy = ResidualBootstrap(seasonal_period=52, block_length=8)
    >>> bagger = TimeSeriesBagger(Ridge(), strategy, n_estimators=20)
    >>> bagger.fit(X_train, y_train)

    Notes
    -----
    [T1] Bergmeir et al. (2016), IJF 32(2): Bagging with STL decomposition.
    The residual bootstrap preserves the deterministic structure (trend +
    seasonal) while randomizing the stochastic component. Block bootstrap
    is used for residuals to preserve any remaining autocorrelation.
    """

    def __init__(
        self,
        seasonal_period: int | None = None,
        block_length: int | None = None,
        robust: bool = True,
        stl_kwargs: dict[str, Any] | None = None,
    ):
        if not HAS_STATSMODELS:
            raise ImportError(
                "statsmodels required for ResidualBootstrap. "
                "Install with: pip install statsmodels"
            )

        if seasonal_period is not None and seasonal_period < 2:
            raise ValueError(f"seasonal_period must be >= 2, got {seasonal_period}")
        if block_length is not None and block_length < 1:
            raise ValueError(f"block_length must be >= 1, got {block_length}")

        self.seasonal_period = seasonal_period
        self.block_length = block_length
        self.robust = robust
        self.stl_kwargs = stl_kwargs or {}
        self.decomposition_: Any = None

    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate bootstrap samples using STL + residual bootstrap.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_observations, n_features)
        y : np.ndarray
            Target series (n_observations,)
        n_samples : int
            Number of bootstrap samples to generate
        rng : np.random.Generator
            Random number generator for reproducibility

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (X_boot, y_boot) tuples. X is unchanged, y is bootstrapped.

        Raises
        ------
        ValueError
            If series is too short for STL decomposition.
        RuntimeError
            If STL decomposition fails.

        Notes
        -----
        1. Decompose y into trend + seasonal + residual
        2. Block bootstrap the residuals
        3. Reconstruct y_boot = trend + seasonal + residual_boot
        4. X stays unchanged (preserves deterministic relationship)
        """
        y_arr = np.asarray(y).ravel()
        X_arr = np.asarray(X)
        n = len(y_arr)

        if n < 4:
            raise ValueError(f"Series too short for STL decomposition: n={n}, need >= 4")

        # Determine seasonal period
        period = self.seasonal_period
        if period is None:
            # Auto-detect: default to reasonable value based on series length
            # Use n // 2 - 1 as upper bound to ensure STL works
            period = min(52, max(2, n // 2 - 1))

        # Ensure period doesn't exceed data requirements
        # STL needs at least 2 full cycles ideally
        if period >= n // 2:
            warnings.warn(
                f"seasonal_period={period} >= n//2={n // 2}, "
                f"reducing to {max(2, n // 3)}",
                UserWarning,
                stacklevel=2,
            )
            period = max(2, n // 3)

        # STL decomposition
        try:
            stl = STL(
                y_arr,
                period=period,
                robust=self.robust,
                **self.stl_kwargs,
            )
            decomposition = stl.fit()
            self.decomposition_ = decomposition

            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid

        except Exception as e:
            raise RuntimeError(
                f"STL decomposition failed: {e}. "
                f"Try adjusting seasonal_period (current={period}) or "
                f"series length (n={n})"
            ) from e

        # Block length for residual bootstrap (moving block bootstrap)
        if self.block_length is None:
            block_len = max(1, min(n, int(n ** (1 / 3))))
        else:
            block_len = max(1, min(n, self.block_length))

        n_blocks = max(1, (n + block_len - 1) // block_len)  # Ceiling division

        # Pre-generate all block starts for all samples
        max_start = max(1, n - block_len + 1)
        all_block_starts = rng.integers(0, max_start, size=(n_samples, n_blocks))

        # Block offset array for residual resampling
        block_offsets = np.arange(block_len)

        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_samples):
            # Compute bootstrap indices using MBB on residuals
            indices = (all_block_starts[i, :, np.newaxis] + block_offsets).ravel()
            # Ensure exactly n indices (truncate if needed)
            if len(indices) < n:
                indices = np.resize(indices, n)
            else:
                indices = indices[:n]

            # Bootstrap residuals only
            residual_boot = residual[indices]

            # Reconstruct y: trend + seasonal + bootstrapped residual
            # Note: trend and seasonal stay fixed, only residuals are resampled
            y_boot = trend + seasonal + residual_boot

            # X stays unchanged - residual bootstrap only bootstraps y
            # This preserves the deterministic X→y relationship
            samples.append((X_arr.copy(), y_boot))

        return samples

    def __repr__(self) -> str:
        return (
            f"ResidualBootstrap("
            f"seasonal_period={self.seasonal_period}, "
            f"block_length={self.block_length}, "
            f"robust={self.robust})"
        )


def create_residual_bagger(
    base_model: Any,
    seasonal_period: int | None = None,
    block_length: int | None = None,
    robust: bool = True,
    n_estimators: int = 50,
    aggregation: str = "mean",
    random_state: int | None = None,
) -> Any:
    """Create a TimeSeriesBagger with ResidualBootstrap strategy.

    Convenience function for creating bagged models with STL-based
    residual bootstrap - appropriate for nonstationary series with
    trend and/or seasonality.

    Parameters
    ----------
    base_model : estimator
        Model with fit/predict interface.
    seasonal_period : int | None
        Seasonal period for STL decomposition.
    block_length : int | None
        Block length for residual bootstrap.
    robust : bool
        Use robust STL fitting.
    n_estimators : int
        Number of bootstrap samples/estimators.
    aggregation : {'mean', 'median'}
        How to combine predictions.
    random_state : int | None
        Random seed for reproducibility.

    Returns
    -------
    TimeSeriesBagger
        Configured bagger with residual bootstrap strategy.

    Examples
    --------
    >>> from temporalcv.bagging import create_residual_bagger
    >>> from sklearn.linear_model import Ridge
    >>>
    >>> bagger = create_residual_bagger(
    ...     Ridge(),
    ...     seasonal_period=52,
    ...     n_estimators=100,
    ...     random_state=42
    ... )
    >>> bagger.fit(X_train, y_train)
    >>> predictions = bagger.predict(X_test)
    """
    from temporalcv.bagging.base import TimeSeriesBagger

    strategy = ResidualBootstrap(
        seasonal_period=seasonal_period,
        block_length=block_length,
        robust=robust,
    )

    return TimeSeriesBagger(
        base_model=base_model,
        strategy=strategy,
        n_estimators=n_estimators,
        aggregation=aggregation,  # type: ignore[arg-type]
        random_state=random_state,
    )
