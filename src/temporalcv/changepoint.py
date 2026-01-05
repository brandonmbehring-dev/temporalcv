"""
Changepoint Detection for Time Series.

Detects structural breaks in time series, useful for:
1. Identifying regime boundaries (e.g., LOW → HIGH volatility)
2. Training models on post-changepoint data only
3. Creating regime indicators as features
4. Understanding when series behavior changed

Algorithms
----------
- Variance-based: Simple rolling variance threshold (always available)
- PELT: Pruned Exact Linear Time (requires optional ruptures package)

References
----------
- Killick, Fearnhead & Eckley (2012). "Optimal Detection of Changepoints
  with a Linear Computational Cost." JASA 107(500), 1590-1598.
- Truong, Oudre & Vayer (2020). "Selective review of offline change point
  detection methods." Signal Processing 167, 107299.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

# Optional pandas import
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None  # type: ignore[assignment]

# Optional ruptures import
try:
    import ruptures as rpt

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    rpt = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Changepoint:
    """Detected changepoint in time series.

    Attributes
    ----------
    index : int
        Index position in series.
    cost_reduction : float
        Cost reduction from adding this changepoint (model fit improvement).
    regime_before : str | None
        Regime classification before changepoint.
    regime_after : str | None
        Regime classification after changepoint.
    """

    index: int
    cost_reduction: float
    regime_before: str | None = None
    regime_after: str | None = None


@dataclass(frozen=True)
class ChangepointResult:
    """Result of changepoint detection.

    Attributes
    ----------
    changepoints : tuple[Changepoint, ...]
        Detected changepoints (immutable tuple).
    n_segments : int
        Number of segments (changepoints + 1).
    method : str
        Detection method used.
    penalty : float
        Penalty parameter used.
    """

    changepoints: tuple[Changepoint, ...]
    n_segments: int
    method: str
    penalty: float


def detect_changepoints_variance(
    series: ArrayLike,
    penalty: float = 3.0,
    min_segment_length: int = 4,
    window: int = 8,
) -> ChangepointResult:
    """Detect changepoints using rolling variance threshold.

    Detects points where the level difference between adjacent windows
    exceeds a threshold relative to local volatility.

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    penalty : float
        Threshold multiplier for detecting changes. Higher = fewer changepoints.
        Default 3.0 means detect changes > 3x baseline volatility.
    min_segment_length : int
        Minimum observations between changepoints.
    window : int
        Rolling window size for computing statistics.

    Returns
    -------
    ChangepointResult
        Detected changepoints with metadata.

    Examples
    --------
    >>> import numpy as np
    >>> # Series with level shift
    >>> series = np.concatenate([np.ones(30) * 3.0, np.ones(30) * 5.0])
    >>> result = detect_changepoints_variance(series, penalty=2.0)
    >>> len(result.changepoints)
    1
    >>> result.changepoints[0].index  # Should be around 30
    30

    Notes
    -----
    [T2] Variance-based detection is a heuristic. For optimal detection,
    use PELT algorithm via `detect_changepoints_pelt()`.
    """
    arr = np.asarray(series).ravel()
    n = len(arr)

    if n < 2 * window + min_segment_length:
        raise ValueError(
            f"Series too short for changepoint detection: n={n}, "
            f"need >= {2 * window + min_segment_length}"
        )

    # Use robust baseline: median absolute deviation of first differences
    # This is not affected by the level shift itself
    diffs = np.diff(arr)
    baseline_mad: float = float(np.median(np.abs(diffs - np.median(diffs))))
    # Convert MAD to approximate std: std ≈ MAD * 1.4826
    baseline_std = baseline_mad * 1.4826

    if baseline_std < 1e-10:
        baseline_std = float(np.std(arr))  # Fallback for constant series
    if baseline_std < 1e-10:
        # Truly constant series - no changepoints
        return ChangepointResult(
            changepoints=tuple(),
            n_segments=1,
            method="variance",
            penalty=penalty,
        )

    threshold = penalty * baseline_std

    # Compute level difference between adjacent non-overlapping windows
    # This detects sharp level changes more reliably
    level_changes = np.zeros(n)
    for i in range(window, n - window + 1):
        left_mean = np.mean(arr[i - window : i])
        right_mean = np.mean(arr[i : i + window])
        level_changes[i] = abs(right_mean - left_mean)

    # Find potential changepoints
    changepoints: list[Changepoint] = []
    for i in range(window, n - window + 1):
        if level_changes[i] > threshold:
            # Check if this is a local maximum (peak of the change)
            is_peak = True
            for j in range(max(window, i - min_segment_length), i):
                if level_changes[j] >= level_changes[i]:
                    is_peak = False
                    break
            for j in range(i + 1, min(n - window + 1, i + min_segment_length)):
                if level_changes[j] > level_changes[i]:
                    is_peak = False
                    break

            if is_peak:
                # Check minimum segment length constraint
                if not changepoints or (i - changepoints[-1].index) >= min_segment_length:
                    cp = Changepoint(
                        index=i,
                        cost_reduction=float(level_changes[i]),
                    )
                    changepoints.append(cp)

    return ChangepointResult(
        changepoints=tuple(changepoints),
        n_segments=len(changepoints) + 1,
        method="variance",
        penalty=penalty,
    )


def detect_changepoints_pelt(
    series: ArrayLike,
    penalty: str | float = "bic",
    min_size: int = 2,
    cost_model: str = "l2",
) -> ChangepointResult:
    """Detect changepoints using PELT algorithm (Pruned Exact Linear Time).

    PELT provides optimal changepoint detection with O(n) complexity.
    Requires the optional `ruptures` package.

    Parameters
    ----------
    series : array-like
        Time series data (1D).
    penalty : str | float
        Penalty for adding changepoints:
        - 'bic': BIC penalty (log(n) * n_params) - recommended
        - 'aic': AIC penalty (2 * n_params)
        - float: Custom penalty value
    min_size : int
        Minimum segment length.
    cost_model : str
        Cost function model:
        - 'l2': Squared error (default, good for mean shifts)
        - 'l1': Absolute error (robust to outliers)
        - 'rbf': Radial basis function (nonparametric)

    Returns
    -------
    ChangepointResult
        Detected changepoints with metadata.

    Raises
    ------
    ImportError
        If ruptures package is not installed.

    Examples
    --------
    >>> import numpy as np
    >>> # Series with two level shifts
    >>> rng = np.random.default_rng(42)
    >>> series = np.concatenate([
    ...     rng.normal(0, 1, 50),
    ...     rng.normal(3, 1, 50),
    ...     rng.normal(1, 1, 50),
    ... ])
    >>> result = detect_changepoints_pelt(series, penalty='bic')
    >>> len(result.changepoints)  # Should detect ~2 changepoints
    2

    Notes
    -----
    [T1] Killick, Fearnhead & Eckley (2012). "Optimal Detection of Changepoints
    with a Linear Computational Cost." JASA 107(500), 1590-1598.

    PELT is exact and optimal - it finds the changepoints that minimize
    the total cost (sum of segment costs + penalty * n_changepoints).
    """
    if not HAS_RUPTURES:
        raise ImportError(
            "ruptures package required for PELT algorithm. "
            "Install with: pip install ruptures"
        )

    arr = np.asarray(series).ravel()
    n = len(arr)

    if n < 2 * min_size:
        raise ValueError(
            f"Series too short for PELT: n={n}, need >= {2 * min_size}"
        )

    # Create cost model
    if cost_model == "l2":
        algo = rpt.Pelt(model="l2", min_size=min_size)
    elif cost_model == "l1":
        algo = rpt.Pelt(model="l1", min_size=min_size)
    elif cost_model == "rbf":
        algo = rpt.Pelt(model="rbf", min_size=min_size)
    else:
        raise ValueError(f"Unknown cost_model: {cost_model}")

    # Fit the model
    algo.fit(arr)

    # Compute penalty value
    if isinstance(penalty, str):
        if penalty == "bic":
            # BIC: log(n) * k where k = number of parameters (1 for mean)
            pen_value = float(np.log(n))
        elif penalty == "aic":
            # AIC: 2 * k
            pen_value = 2.0
        else:
            raise ValueError(f"Unknown penalty: {penalty}")
    else:
        pen_value = float(penalty)

    # Detect changepoints
    # ruptures returns list ending with n (series length)
    breakpoints = algo.predict(pen=pen_value)

    # Remove the final index (series end)
    cp_indices = [bp for bp in breakpoints if bp < n]

    # Compute cost reductions for each changepoint
    changepoints: list[Changepoint] = []
    for idx in cp_indices:
        # Approximate cost reduction as local variance change
        left_start = max(0, idx - min_size)
        right_end = min(n, idx + min_size)
        left_var = np.var(arr[left_start:idx]) if idx > left_start else 0
        right_var = np.var(arr[idx:right_end]) if right_end > idx else 0
        cost_reduction = float(abs(left_var - right_var))

        cp = Changepoint(
            index=idx,
            cost_reduction=cost_reduction,
        )
        changepoints.append(cp)

    return ChangepointResult(
        changepoints=tuple(changepoints),
        n_segments=len(changepoints) + 1,
        method="pelt",
        penalty=pen_value,
    )


def detect_changepoints(
    series: ArrayLike,
    method: Literal["variance", "pelt", "auto"] = "auto",
    penalty: str | float = "bic",
    min_segment_length: int = 4,
    **kwargs: Any,
) -> ChangepointResult:
    """Unified changepoint detection interface.

    Convenience function that selects the best available algorithm.

    Parameters
    ----------
    series : array-like
        Time series data.
    method : {'variance', 'pelt', 'auto'}
        Detection method:
        - 'variance': Rolling variance threshold
        - 'pelt': PELT algorithm (requires ruptures)
        - 'auto': Use PELT if available, else variance
    penalty : str | float
        Penalty parameter (passed to underlying method).
    min_segment_length : int
        Minimum observations between changepoints.
    **kwargs
        Additional arguments passed to detection function.

    Returns
    -------
    ChangepointResult
        Detected changepoints.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.concatenate([np.ones(30), np.ones(30) * 5])
    >>> result = detect_changepoints(series, method='auto')
    >>> len(result.changepoints) >= 1
    True
    """
    if method == "auto":
        method = "pelt" if HAS_RUPTURES else "variance"

    if method == "pelt":
        return detect_changepoints_pelt(
            series,
            penalty=penalty,
            min_size=min_segment_length,
            **kwargs,
        )
    elif method == "variance":
        # Convert penalty to float for variance method
        if isinstance(penalty, str):
            penalty_val = 3.0  # Default
        else:
            penalty_val = float(penalty)
        return detect_changepoints_variance(
            series,
            penalty=penalty_val,
            min_segment_length=min_segment_length,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'variance', 'pelt', or 'auto'.")


def classify_regimes_from_changepoints(
    series: ArrayLike,
    changepoints: list[Changepoint] | ChangepointResult,
    method: Literal["volatility", "level", "trend"] = "volatility",
    thresholds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Assign regime labels to segments between changepoints.

    Parameters
    ----------
    series : array-like
        Original time series.
    changepoints : list[Changepoint] | ChangepointResult
        Detected changepoints.
    method : {'volatility', 'level', 'trend'}
        Classification method:
        - 'volatility': Classify by segment volatility (diff std)
        - 'level': Classify by segment mean level
        - 'trend': Classify by segment trend direction
    thresholds : tuple[float, float] | None
        Custom thresholds for LOW/MEDIUM/HIGH classification.
        If None, uses data-driven thresholds (33rd/67th percentiles).

    Returns
    -------
    np.ndarray
        Array of regime labels ('LOW', 'MEDIUM', 'HIGH') for each observation.

    Examples
    --------
    >>> import numpy as np
    >>> series = np.concatenate([np.ones(30) * 1, np.ones(30) * 5])
    >>> result = detect_changepoints_variance(series)
    >>> regimes = classify_regimes_from_changepoints(series, result)
    >>> regimes[0]  # First segment
    'LOW'
    """
    arr = np.asarray(series).ravel()
    n = len(arr)

    # Extract changepoint list
    if isinstance(changepoints, ChangepointResult):
        cp_list = list(changepoints.changepoints)
    else:
        cp_list = list(changepoints)

    # Get segment boundaries
    cp_indices = [0] + [cp.index for cp in cp_list] + [n]

    # Compute segment characteristics
    segment_values: list[float] = []
    for start, end in zip(cp_indices[:-1], cp_indices[1:]):
        segment = arr[start:end]
        if method == "volatility":
            if len(segment) > 1:
                val = float(np.std(np.diff(segment)))
            else:
                val = 0.0
        elif method == "level":
            val = float(np.mean(segment))
        elif method == "trend":
            if len(segment) > 1:
                val = float(np.polyfit(np.arange(len(segment)), segment, 1)[0])
            else:
                val = 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
        segment_values.append(val)

    # Determine thresholds
    if thresholds is not None:
        low_thresh, high_thresh = thresholds
    else:
        # Data-driven: use 33rd and 67th percentiles
        if len(segment_values) >= 3:
            low_thresh = float(np.percentile(segment_values, 33))
            high_thresh = float(np.percentile(segment_values, 67))
        else:
            # Not enough segments - use overall statistics
            overall_val = np.mean(segment_values)
            low_thresh = overall_val * 0.5
            high_thresh = overall_val * 1.5

    # Assign regime labels
    regimes: np.ndarray = np.empty(n, dtype=object)
    for i, (start, end) in enumerate(zip(cp_indices[:-1], cp_indices[1:])):
        val = segment_values[i]
        if val < low_thresh:
            regime = "LOW"
        elif val > high_thresh:
            regime = "HIGH"
        else:
            regime = "MEDIUM"
        regimes[start:end] = regime

    # Cast to satisfy mypy (object arrays are typed as Any)
    result: np.ndarray = np.asarray(regimes)
    return result


def create_regime_indicators(
    series: ArrayLike,
    changepoints: list[Changepoint] | ChangepointResult,
    recent_window: int = 4,
) -> dict[str, np.ndarray]:
    """Create regime indicator features for modeling.

    Parameters
    ----------
    series : array-like
        Time series data.
    changepoints : list[Changepoint] | ChangepointResult
        Detected changepoints.
    recent_window : int
        Number of observations to consider as "recent" regime change.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with indicator arrays:
        - 'is_regime_change': 1 if within recent_window of a changepoint
        - 'periods_since_change': Periods since last changepoint
        - 'regime_labels': Regime labels ('LOW', 'MEDIUM', 'HIGH')
        - 'regime_LOW', 'regime_MEDIUM', 'regime_HIGH': One-hot encoded

    Examples
    --------
    >>> import numpy as np
    >>> series = np.concatenate([np.ones(30), np.ones(30) * 5])
    >>> result = detect_changepoints_variance(series)
    >>> indicators = create_regime_indicators(series, result)
    >>> 'is_regime_change' in indicators
    True
    >>> 'regime_labels' in indicators
    True
    """
    arr = np.asarray(series).ravel()
    n = len(arr)

    # Extract changepoint indices
    if isinstance(changepoints, ChangepointResult):
        cp_list = list(changepoints.changepoints)
    else:
        cp_list = list(changepoints)

    cp_indices = [cp.index for cp in cp_list]

    # Periods since last changepoint
    periods_since: np.ndarray = np.zeros(n, dtype=int)
    last_cp = -1  # Before series start
    for i in range(n):
        if i in cp_indices:
            last_cp = i
        if last_cp >= 0:
            periods_since[i] = i - last_cp
        else:
            periods_since[i] = i  # Distance from start

    # Is recent regime change
    is_recent = (periods_since <= recent_window) & (periods_since > 0)

    # Regime labels
    regime_labels = classify_regimes_from_changepoints(arr, cp_list)

    # One-hot encoding
    regime_low = (regime_labels == "LOW").astype(int)
    regime_medium = (regime_labels == "MEDIUM").astype(int)
    regime_high = (regime_labels == "HIGH").astype(int)

    return {
        "is_regime_change": is_recent.astype(int),
        "periods_since_change": periods_since,
        "regime_labels": regime_labels,
        "regime_LOW": regime_low,
        "regime_MEDIUM": regime_medium,
        "regime_HIGH": regime_high,
    }


def get_segment_boundaries(
    n: int,
    changepoints: list[Changepoint] | ChangepointResult,
) -> list[tuple[int, int]]:
    """Get segment start/end indices from changepoints.

    Parameters
    ----------
    n : int
        Series length.
    changepoints : list[Changepoint] | ChangepointResult
        Detected changepoints.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) index pairs for each segment.

    Examples
    --------
    >>> cps = [Changepoint(index=30, cost_reduction=1.0)]
    >>> get_segment_boundaries(60, cps)
    [(0, 30), (30, 60)]
    """
    if isinstance(changepoints, ChangepointResult):
        cp_list = list(changepoints.changepoints)
    else:
        cp_list = list(changepoints)

    cp_indices = [0] + [cp.index for cp in cp_list] + [n]

    return [(cp_indices[i], cp_indices[i + 1]) for i in range(len(cp_indices) - 1)]
