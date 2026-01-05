"""
Monash Forecasting Repository Loaders.

Provides loaders for M3, M4 datasets with documented train/test splits.

Requires: datasetsforecast (optional dependency)

Example
-------
>>> from temporalcv.benchmarks import load_m3, load_m4
>>> m3 = load_m3(subset="monthly")
>>> m4 = load_m4(subset="weekly", sample_size=100)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np

from temporalcv.benchmarks.base import (
    DatasetMetadata,
    TimeSeriesDataset,
    validate_dataset,
)


MONASH_URL = "https://forecastingdata.org/"

M3_HORIZONS: Dict[str, int] = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "other": 8,
}

M4_HORIZONS: Dict[str, int] = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "weekly": 13,
    "daily": 14,
    "hourly": 48,
}

# Frequency codes for metadata
M3_FREQUENCY: Dict[str, str] = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "other": "D",
}

M4_FREQUENCY: Dict[str, str] = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
    "hourly": "H",
}


def _check_datasetsforecast() -> None:
    """Check datasetsforecast is installed."""
    try:
        from datasetsforecast.m3 import M3  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "datasetsforecast required for M3/M4 loading.\n"
            "Install with: pip install temporalcv[monash]\n"
            "Or: pip install datasetsforecast"
        ) from e


def load_m3(
    subset: Literal["yearly", "quarterly", "monthly", "other"] = "monthly",
    sample_size: Optional[int] = None,
) -> TimeSeriesDataset:
    """
    Load M3 Competition dataset from Monash repository.

    Parameters
    ----------
    subset : {'yearly', 'quarterly', 'monthly', 'other'}, default='monthly'
        Frequency subset to load
    sample_size : int, optional
        Number of series to sample (for testing)

    Returns
    -------
    TimeSeriesDataset
        M3 dataset with competition split

    Raises
    ------
    ImportError
        If datasetsforecast not installed

    Notes
    -----
    M3 Competition (2000):
    - 3003 time series total
    - Monthly: 1428, Quarterly: 756, Yearly: 645, Other: 174
    - Standard horizons per frequency
    """
    _check_datasetsforecast()

    from datasetsforecast.m3 import M3

    # Load via datasetsforecast
    df, _, _ = M3.load(directory=None, group=subset)

    # Convert to array format
    unique_ids = df["unique_id"].unique()
    if sample_size is not None and sample_size < len(unique_ids):
        rng = np.random.default_rng(42)
        unique_ids = rng.choice(unique_ids, size=sample_size, replace=False)

    series_list = []
    original_lengths: list[int] = []
    for uid in unique_ids:
        vals = df[df["unique_id"] == uid]["y"].values.astype(np.float64)
        series_list.append(vals)
        original_lengths.append(len(vals))

    # Truncate to common length (required for array format)
    min_len_int = min(original_lengths)
    max_len_int = max(original_lengths)
    was_truncated = min_len_int != max_len_int
    values = np.array([s[:min_len_int] for s in series_list], dtype=np.float64)

    # M3 Competition protocol: test = last `horizon` observations
    horizon = M3_HORIZONS[subset]
    train_end_idx = min_len_int - horizon

    metadata = DatasetMetadata(
        name=f"m3_{subset}",
        frequency=M3_FREQUENCY[subset],
        horizon=horizon,
        n_series=len(values),
        total_observations=values.size,
        train_end_idx=train_end_idx,
        characteristics={
            "competition": "M3",
            "documented_splits": True,
        },
        license="open_access",
        source_url=MONASH_URL,
        official_split=not was_truncated,  # Only true if data unchanged from competition
        truncated=was_truncated,
        original_series_lengths=original_lengths if was_truncated else None,
        split_source="M3 Competition (Makridakis & Hibon, 2000)",
    )

    dataset = TimeSeriesDataset(
        metadata=metadata,
        values=values,
    )

    validate_dataset(dataset)
    return dataset


def load_m4(
    subset: Literal[
        "yearly", "quarterly", "monthly", "weekly", "daily", "hourly"
    ] = "monthly",
    sample_size: Optional[int] = 100,
) -> TimeSeriesDataset:
    """
    Load M4 Competition dataset from Monash repository.

    Parameters
    ----------
    subset : {'yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly'}
        Frequency subset to load
    sample_size : int, default=100
        Number of series to sample (M4 is very large)

    Returns
    -------
    TimeSeriesDataset
        M4 dataset with competition split

    Raises
    ------
    ImportError
        If datasetsforecast not installed

    Notes
    -----
    M4 Competition (2018):
    - 100,000 time series total
    - Diverse domains and frequencies
    """
    _check_datasetsforecast()

    from datasetsforecast.m4 import M4

    # datasetsforecast expects capitalized group names: 'Monthly' not 'monthly'
    df, _, _ = M4.load(directory=None, group=subset.title())

    unique_ids = df["unique_id"].unique()
    if sample_size is not None and sample_size < len(unique_ids):
        rng = np.random.default_rng(42)
        unique_ids = rng.choice(unique_ids, size=sample_size, replace=False)

    series_list = []
    original_lengths: list[int] = []
    for uid in unique_ids:
        vals = df[df["unique_id"] == uid]["y"].values.astype(np.float64)
        series_list.append(vals)
        original_lengths.append(len(vals))

    # Truncate to common length (required for array format)
    min_len_int = min(original_lengths)
    max_len_int = max(original_lengths)
    was_truncated = min_len_int != max_len_int
    values = np.array([s[:min_len_int] for s in series_list], dtype=np.float64)

    # M4 Competition protocol: test = last `horizon` observations
    horizon = M4_HORIZONS[subset]
    train_end_idx = min_len_int - horizon

    metadata = DatasetMetadata(
        name=f"m4_{subset}",
        frequency=M4_FREQUENCY[subset],
        horizon=horizon,
        n_series=len(values),
        total_observations=values.size,
        train_end_idx=train_end_idx,
        characteristics={
            "competition": "M4",
            "documented_splits": True,
        },
        license="open_access",
        source_url=MONASH_URL,
        official_split=not was_truncated,  # Only true if data unchanged from competition
        truncated=was_truncated,
        original_series_lengths=original_lengths if was_truncated else None,
        split_source="M4 Competition (Makridakis et al., 2018)",
    )

    dataset = TimeSeriesDataset(
        metadata=metadata,
        values=values,
    )

    validate_dataset(dataset)
    return dataset


# =============================================================================
# Public API
# =============================================================================

__all__ = ["load_m3", "load_m4", "MONASH_URL", "M3_HORIZONS", "M4_HORIZONS"]
