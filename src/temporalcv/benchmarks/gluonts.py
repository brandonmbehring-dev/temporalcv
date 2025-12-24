"""
GluonTS Dataset Loaders.

Provides loaders for electricity and traffic datasets from GluonTS.

Requires: gluonts (optional dependency)

Example
-------
>>> from temporalcv.benchmarks import load_electricity, load_traffic
>>> elec = load_electricity()
>>> traffic = load_traffic()
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from temporalcv.benchmarks.base import (
    DatasetMetadata,
    TimeSeriesDataset,
    validate_dataset,
)


def _check_gluonts() -> None:
    """Check gluonts is installed."""
    try:
        from gluonts.dataset.repository import get_dataset  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "gluonts required for electricity/traffic datasets.\n"
            "Install with: pip install temporalcv[gluonts]\n"
            "Or: pip install gluonts"
        ) from e


def load_electricity(
    subset: Optional[int] = None,
) -> TimeSeriesDataset:
    """
    Load UCI Electricity dataset via GluonTS.

    Parameters
    ----------
    subset : int, optional
        Number of series to include. None for all 370 series.

    Returns
    -------
    TimeSeriesDataset
        Electricity consumption dataset

    Raises
    ------
    ImportError
        If gluonts not installed

    Notes
    -----
    - 370 time series (clients)
    - Hourly data (2011-2014)
    - Standard horizon: 24 hours
    """
    _check_gluonts()

    from gluonts.dataset.repository import get_dataset

    gluon_ds = get_dataset("electricity", regenerate=False)

    # Extract values from ListDataset
    all_values = []
    original_lengths: list[int] = []
    for entry in gluon_ds.train:
        vals = np.asarray(entry["target"], dtype=np.float64)
        all_values.append(vals)
        original_lengths.append(len(vals))

    if subset is not None and subset < len(all_values):
        all_values = all_values[:subset]
        original_lengths = original_lengths[:subset]

    # Stack into array (requires common length)
    min_len = min(original_lengths)
    max_len = max(original_lengths)
    was_truncated = min_len != max_len
    values = np.array([v[:min_len] for v in all_values], dtype=np.float64)

    # Split: last 7 days (168 hours) for test
    # NOTE: This is a convenience split, NOT GluonTS official protocol
    train_end_idx = min_len - 168

    metadata = DatasetMetadata(
        name="electricity",
        frequency="H",
        horizon=24,
        n_series=len(values),
        total_observations=values.size,
        train_end_idx=train_end_idx,
        characteristics={
            "seasonality": [24, 168],  # Daily, weekly
            "probabilistic_benchmark": True,
        },
        license="open_access",
        source_url="https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014",
        official_split=False,  # Convenience split, not GluonTS protocol
        truncated=was_truncated,
        original_series_lengths=original_lengths if was_truncated else None,
        split_source="temporalcv convenience (last 7 days)",
    )

    dataset = TimeSeriesDataset(
        metadata=metadata,
        values=values,
    )

    validate_dataset(dataset)
    return dataset


def load_traffic(
    subset: Optional[int] = None,
) -> TimeSeriesDataset:
    """
    Load California Traffic dataset via GluonTS.

    Parameters
    ----------
    subset : int, optional
        Number of series to include. None for all 862 series.

    Returns
    -------
    TimeSeriesDataset
        Traffic occupancy dataset

    Raises
    ------
    ImportError
        If gluonts not installed

    Notes
    -----
    - 862 time series (sensors)
    - Hourly data (2 years)
    - Standard horizon: 24 hours
    """
    _check_gluonts()

    from gluonts.dataset.repository import get_dataset

    gluon_ds = get_dataset("traffic", regenerate=False)

    all_values = []
    original_lengths: list[int] = []
    for entry in gluon_ds.train:
        vals = np.asarray(entry["target"], dtype=np.float64)
        all_values.append(vals)
        original_lengths.append(len(vals))

    if subset is not None and subset < len(all_values):
        all_values = all_values[:subset]
        original_lengths = original_lengths[:subset]

    min_len = min(original_lengths)
    max_len = max(original_lengths)
    was_truncated = min_len != max_len
    values = np.array([v[:min_len] for v in all_values], dtype=np.float64)

    # Split: last 7 days (168 hours) for test
    # NOTE: This is a convenience split, NOT GluonTS official protocol
    train_end_idx = min_len - 168

    metadata = DatasetMetadata(
        name="traffic",
        frequency="H",
        horizon=24,
        n_series=len(values),
        total_observations=values.size,
        train_end_idx=train_end_idx,
        characteristics={
            "seasonality": [24, 168],
            "spatial_correlation": True,
        },
        license="open_access",
        source_url="https://pems.dot.ca.gov/",
        official_split=False,  # Convenience split, not GluonTS protocol
        truncated=was_truncated,
        original_series_lengths=original_lengths if was_truncated else None,
        split_source="temporalcv convenience (last 7 days)",
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

__all__ = ["load_electricity", "load_traffic"]
