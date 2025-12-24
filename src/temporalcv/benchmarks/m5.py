"""
M5 Walmart Sales Dataset Loader.

NOTE: Due to Kaggle TOS, data cannot be bundled with temporalcv.
Users must download from Kaggle and provide path.

Example
-------
>>> from temporalcv.benchmarks import load_m5
>>> dataset = load_m5(path="~/data/m5/")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from temporalcv.benchmarks.base import (
    DatasetMetadata,
    DatasetNotFoundError,
    TimeSeriesDataset,
    validate_dataset,
)


M5_DOWNLOAD_URL = "https://www.kaggle.com/competitions/m5-forecasting-accuracy/data"

M5_INSTRUCTIONS = """
1. Create Kaggle account at https://www.kaggle.com/
2. Navigate to M5 competition: https://www.kaggle.com/competitions/m5-forecasting-accuracy
3. Accept competition rules
4. Download: sales_train_evaluation.csv, calendar.csv, sell_prices.csv
5. Place files in a directory and pass path to load_m5()

Required files:
- sales_train_evaluation.csv
- calendar.csv (optional, for dates)
- sell_prices.csv (optional, for exogenous)
"""


def load_m5(
    path: Optional[str] = None,
    sample_size: Optional[int] = 100,
    aggregate: bool = True,
) -> TimeSeriesDataset:
    """
    Load M5 Walmart sales dataset.

    Due to Kaggle TOS, data must be downloaded manually.

    Parameters
    ----------
    path : str, optional
        Directory containing M5 files. If None, raises DatasetNotFoundError
        with download instructions.
    sample_size : int, optional
        Number of series to sample (for faster testing). None for all.
    aggregate : bool, default=True
        If True, returns aggregate daily sales. If False, returns item-level.

    Returns
    -------
    TimeSeriesDataset
        M5 dataset with standard competition split

    Raises
    ------
    DatasetNotFoundError
        If path not provided or files not found
    ImportError
        If pandas not installed

    Notes
    -----
    M5 Competition details:
    - 30,490 time series (item-store combinations)
    - 1,941 days of data (2011-01-29 to 2016-06-19)
    - Hierarchical structure: item -> category -> department -> store -> state
    - Standard horizon: 28 days
    """
    if path is None:
        raise DatasetNotFoundError(
            dataset_name="M5 Walmart",
            download_url=M5_DOWNLOAD_URL,
            instructions=M5_INSTRUCTIONS,
        )

    data_path = Path(path).expanduser()
    sales_file = data_path / "sales_train_evaluation.csv"

    if not sales_file.exists():
        raise DatasetNotFoundError(
            dataset_name="M5 Walmart",
            download_url=M5_DOWNLOAD_URL,
            instructions=(
                f"sales_train_evaluation.csv not found in {data_path}\n\n"
                f"{M5_INSTRUCTIONS}"
            ),
        )

    # Load data (pandas required for M5)
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas required for M5 loading: pip install pandas"
        ) from e

    df = pd.read_csv(sales_file)

    # Extract time series columns (d_1, d_2, ..., d_1941)
    ts_cols = [c for c in df.columns if c.startswith("d_")]
    values = df[ts_cols].values.astype(np.float64)

    if sample_size is not None and sample_size < len(values):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(values), size=sample_size, replace=False)
        values = values[indices]

    if aggregate:
        values = values.sum(axis=0)  # Aggregate across series
        n_series = 1
    else:
        n_series = len(values)

    # Standard M5 split: last 28 days are test
    train_end_idx = len(ts_cols) - 28

    metadata = DatasetMetadata(
        name="m5_walmart",
        frequency="D",
        horizon=28,
        n_series=n_series,
        total_observations=values.size,
        train_end_idx=train_end_idx,
        characteristics={
            "hierarchical": True,
            "intermittent": True,
            "competition": "M5",
        },
        license="kaggle_tos",
        source_url=M5_DOWNLOAD_URL,
        official_split=True,  # M5 Competition protocol
        truncated=False,  # M5 has fixed-length series
        original_series_lengths=None,
        split_source="M5 Competition (Kaggle, 2020)",
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

__all__ = ["load_m5", "M5_DOWNLOAD_URL", "M5_INSTRUCTIONS"]
