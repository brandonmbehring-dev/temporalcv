"""
FRED Economic Data Loader.

Loads interest rate and yield curve data from FRED (public domain).

Requires: fredapi (optional dependency)

Example
-------
>>> from temporalcv.benchmarks import load_fred_rates
>>> dataset = load_fred_rates(series="DGS10", start="2010-01-01")
>>> train, test = dataset.get_train_test_split()

Notes
-----
FRED data is public domain (US government data).
Get API key from: https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional, Union

import numpy as np

from temporalcv.benchmarks.base import (
    DatasetMetadata,
    DatasetNotFoundError,
    TimeSeriesDataset,
    validate_dataset,
)


# Standard FRED series for rate forecasting
FRED_RATE_SERIES: Dict[str, str] = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "DGS5": "5-Year Treasury Constant Maturity Rate",
    "DGS30": "30-Year Treasury Constant Maturity Rate",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "AAA": "Moody's Seasoned Aaa Corporate Bond Yield",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
}


def _check_fredapi() -> None:
    """Check fredapi is installed."""
    try:
        import fredapi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "fredapi required for FRED data loading.\n"
            "Install with: pip install temporalcv[fred]\n"
            "Or: pip install fredapi"
        ) from e


def load_fred_rates(
    series: Union[str, List[str]] = "DGS10",
    start: str = "2000-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
    frequency: Literal["D", "W", "M"] = "W",
    train_fraction: float = 0.8,
) -> TimeSeriesDataset:
    """
    Load interest rate data from FRED.

    Parameters
    ----------
    series : str or list[str], default="DGS10"
        FRED series ID(s) to load. Common choices:
        - DGS10: 10-Year Treasury
        - DGS2: 2-Year Treasury
        - FEDFUNDS: Fed Funds Rate
    start : str, default="2000-01-01"
        Start date (YYYY-MM-DD)
    end : str, optional
        End date (YYYY-MM-DD). Default: today.
        **IMPORTANT**: For reproducible results, always set an explicit end date.
        Running with end=None on different days produces different datasets.
    api_key : str, optional
        FRED API key. If None, uses FRED_API_KEY env var.
    frequency : {'D', 'W', 'M'}, default='W'
        Resampling frequency
    train_fraction : float, default=0.8
        Fraction of data for training (for standard split)

    Returns
    -------
    TimeSeriesDataset
        Dataset with values and standard split

    Raises
    ------
    ImportError
        If fredapi not installed
    DatasetNotFoundError
        If API key not configured

    Notes
    -----
    FRED data is public domain (US government).
    Get API key from: https://fred.stlouisfed.org/docs/api/api_key.html

    **Date Pinning for Reproducibility:**
    Without an explicit `end` date, this function fetches data up to today,
    making results non-reproducible across different runs. For reproducible
    research, always specify both `start` and `end`:

        dataset = load_fred_rates("DGS10", start="2010-01-01", end="2023-12-31")

    Examples
    --------
    >>> # Reproducible (date-pinned)
    >>> dataset = load_fred_rates("DGS10", start="2010-01-01", end="2023-12-31")
    >>> train, test = dataset.get_train_test_split()
    >>>
    >>> # Non-reproducible (changes daily)
    >>> dataset = load_fred_rates("DGS10", frequency="W")  # end defaults to today
    """
    _check_fredapi()

    from fredapi import Fred

    # Get API key
    key = api_key or os.environ.get("FRED_API_KEY")
    if key is None:
        raise DatasetNotFoundError(
            dataset_name="FRED",
            download_url="https://fred.stlouisfed.org/docs/api/api_key.html",
            instructions=(
                "1. Create free account at FRED\n"
                "2. Request API key\n"
                "3. Set FRED_API_KEY environment variable or pass api_key parameter"
            ),
        )

    fred = Fred(api_key=key)

    # Handle single series or list
    if isinstance(series, str):
        series_list = [series]
    else:
        series_list = list(series)

    # Load data
    all_data: List[np.ndarray] = []
    for s in series_list:
        data = fred.get_series(s, observation_start=start, observation_end=end)
        # Resample if needed
        if frequency == "W":
            data = data.resample("W-FRI").last()
        elif frequency == "M":
            data = data.resample("ME").last()
        all_data.append(data.values)

    # Stack if multiple series
    if len(all_data) == 1:
        values = all_data[0]
    else:
        values = np.column_stack(all_data)

    # Remove NaN (FRED has missing observations)
    if values.ndim > 1:
        mask = ~np.isnan(values).any(axis=-1)
    else:
        mask = ~np.isnan(values)
    values = values[mask]

    # Compute train/test split
    train_end_idx = int(len(values) * train_fraction)

    # Standard horizon for rates: 2 weeks (per MYGA research)
    horizon = 2 if frequency == "W" else 1

    # Track date-pinning for reproducibility
    is_date_pinned = end is not None

    metadata = DatasetMetadata(
        name=f"fred_{series_list[0]}" if len(series_list) == 1 else "fred_multi",
        frequency=frequency,
        horizon=horizon,
        n_series=len(series_list),
        total_observations=len(values),
        train_end_idx=train_end_idx,
        characteristics={
            "series_ids": series_list,
            "high_persistence": True,  # Rates are highly persistent
            "acf1_typical": 0.99,
            "date_pinned": is_date_pinned,
            "start_date": start,
            "end_date": end,  # None if not pinned
        },
        license="public_domain",
        source_url="https://fred.stlouisfed.org/",
        official_split=False,  # No official split; train_fraction is invented
        truncated=False,
        original_series_lengths=None,
        split_source=f"temporalcv convenience ({train_fraction:.0%} train)",
    )

    dataset = TimeSeriesDataset(
        metadata=metadata,
        values=values,
    )

    validate_dataset(dataset)
    return dataset


def list_available_series() -> Dict[str, str]:
    """
    Return dictionary of commonly-used FRED rate series.

    Returns
    -------
    dict[str, str]
        Series ID -> Description mapping
    """
    return FRED_RATE_SERIES.copy()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "load_fred_rates",
    "list_available_series",
    "FRED_RATE_SERIES",
]
