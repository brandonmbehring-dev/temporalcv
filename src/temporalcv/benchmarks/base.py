"""
Benchmark Dataset Infrastructure.

Provides Dataset protocol and DatasetNotFoundError for consistent
interface across all benchmark loaders.

Example
-------
>>> from temporalcv.benchmarks import TimeSeriesDataset, DatasetMetadata
>>>
>>> metadata = DatasetMetadata(
...     name="my_dataset",
...     frequency="W",
...     horizon=2,
...     n_series=1,
...     total_observations=100,
... )
>>> dataset = TimeSeriesDataset(metadata=metadata, values=np.random.randn(100))
>>> train, test = dataset.get_train_test_split()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np


class DatasetNotFoundError(FileNotFoundError):
    """
    Raised when a dataset is not available locally.

    Provides download instructions for datasets that cannot be bundled
    due to licensing restrictions.

    Attributes
    ----------
    dataset_name : str
        Name of the missing dataset
    download_url : str
        URL where dataset can be obtained
    instructions : str
        Human-readable download instructions

    Examples
    --------
    >>> raise DatasetNotFoundError(
    ...     dataset_name="M5",
    ...     download_url="https://kaggle.com/...",
    ...     instructions="Download from Kaggle and extract to ~/data/m5/"
    ... )
    """

    def __init__(
        self,
        dataset_name: str,
        download_url: str,
        instructions: str,
    ):
        self.dataset_name = dataset_name
        self.download_url = download_url
        self.instructions = instructions
        message = (
            f"Dataset '{dataset_name}' not found.\n\n"
            f"Download from: {download_url}\n\n"
            f"Instructions:\n{instructions}"
        )
        super().__init__(message)


@dataclass
class DatasetMetadata:
    """
    Metadata for a benchmark dataset.

    Attributes
    ----------
    name : str
        Dataset identifier
    frequency : str
        Temporal frequency ('D', 'W', 'M', 'H', 'Y')
    horizon : int
        Standard forecast horizon for this dataset
    n_series : int
        Number of time series
    total_observations : int
        Total observations across all series
    train_end_idx : int, optional
        Index of last training observation (for standard split)
    characteristics : dict
        Additional properties (seasonality, persistence, etc.)
    license : str
        License type ('public_domain', 'open_access', 'kaggle_tos', etc.)
    source_url : str
        Original source URL
    official_split : bool
        Whether the split is an official competition/benchmark split.
        True = split matches competition protocol (reproducible comparisons).
        False = split invented for convenience (NOT comparable to published results).
    truncated : bool
        Whether series have been truncated from their original lengths.
        If True, results may not be comparable to original benchmark.
    original_series_lengths : list[int], optional
        Original lengths of each series before truncation (if applicable).
        Used to document what was lost due to truncation.
    split_source : str
        Source of the split definition (e.g., "M4 Competition", "GluonTS",
        "Monash Forecasting Repository", "invented"). Empty = unknown.
    """

    name: str
    frequency: str
    horizon: int
    n_series: int
    total_observations: int
    train_end_idx: Optional[int] = None
    characteristics: Dict[str, Any] = field(default_factory=dict)
    license: str = "unknown"
    source_url: str = ""
    official_split: bool = False
    truncated: bool = False
    original_series_lengths: Optional[list[int]] = None
    split_source: str = ""

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.n_series < 1:
            raise ValueError(f"n_series must be >= 1, got {self.n_series}")
        if self.total_observations < 1:
            raise ValueError(
                f"total_observations must be >= 1, got {self.total_observations}"
            )
        if self.train_end_idx is not None and self.train_end_idx < 1:
            raise ValueError(f"train_end_idx must be >= 1, got {self.train_end_idx}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "frequency": self.frequency,
            "horizon": self.horizon,
            "n_series": self.n_series,
            "total_observations": self.total_observations,
            "train_end_idx": self.train_end_idx,
            "characteristics": self.characteristics,
            "license": self.license,
            "source_url": self.source_url,
            "official_split": self.official_split,
            "truncated": self.truncated,
            "original_series_lengths": self.original_series_lengths,
            "split_source": self.split_source,
        }


@runtime_checkable
class Dataset(Protocol):
    """
    Protocol for benchmark datasets.

    All dataset loaders must return objects satisfying this protocol.
    """

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata."""
        ...

    @property
    def values(self) -> np.ndarray:
        """Time series values as array."""
        ...

    @property
    def timestamps(self) -> Optional[np.ndarray]:
        """Timestamps if available."""
        ...

    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard train/test split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (train_values, test_values)
        """
        ...


@dataclass
class TimeSeriesDataset:
    """
    Concrete implementation of Dataset protocol.

    Generic container for benchmark time series data.

    Attributes
    ----------
    metadata : DatasetMetadata
        Dataset metadata
    values : np.ndarray
        Time series values. Shape: (n_obs,) for single series or
        (n_series, n_obs) for multiple series.
    timestamps : np.ndarray, optional
        Timestamps for observations
    exogenous : np.ndarray, optional
        Exogenous features if available

    Examples
    --------
    >>> metadata = DatasetMetadata(
    ...     name="test",
    ...     frequency="W",
    ...     horizon=2,
    ...     n_series=1,
    ...     total_observations=100,
    ...     train_end_idx=80,
    ... )
    >>> dataset = TimeSeriesDataset(metadata=metadata, values=np.random.randn(100))
    >>> train, test = dataset.get_train_test_split()
    >>> print(f"Train: {len(train)}, Test: {len(test)}")
    Train: 80, Test: 20
    """

    metadata: DatasetMetadata
    values: np.ndarray
    timestamps: Optional[np.ndarray] = None
    exogenous: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Validate dataset."""
        self.values = np.asarray(self.values)
        if len(self.values) == 0:
            raise ValueError("Dataset values cannot be empty")

        # Validate train/test split leaves room for horizon
        if self.metadata.train_end_idx is not None:
            idx = self.metadata.train_end_idx
            horizon = self.metadata.horizon
            n_obs = self.values.shape[-1] if self.values.ndim > 1 else len(self.values)

            if idx <= horizon:
                raise ValueError(
                    f"train_end_idx ({idx}) must be > horizon ({horizon}) "
                    f"to allow sufficient training data"
                )
            if idx > n_obs - horizon:
                raise ValueError(
                    f"train_end_idx ({idx}) leaves only {n_obs - idx} test observations, "
                    f"but need at least {horizon} for horizon-step forecasting"
                )

    def get_train_test_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return standard train/test split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (train_values, test_values)

        Raises
        ------
        ValueError
            If no standard split is defined
        """
        if self.metadata.train_end_idx is None:
            raise ValueError(
                f"Dataset '{self.metadata.name}' has no standard train/test split. "
                f"Set train_end_idx in metadata."
            )

        idx = self.metadata.train_end_idx

        if self.values.ndim == 1:
            return self.values[:idx], self.values[idx:]
        else:
            # Multi-series: shape (n_series, n_obs)
            return self.values[:, :idx], self.values[:, idx:]

    @property
    def has_exogenous(self) -> bool:
        """Check if exogenous features are available."""
        return self.exogenous is not None

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        if self.values.ndim == 1:
            return len(self.values)
        return int(self.values.shape[1])


def validate_dataset(dataset: Dataset) -> None:
    """
    Validate dataset satisfies protocol requirements.

    Parameters
    ----------
    dataset : Dataset
        Dataset to validate

    Raises
    ------
    ValueError
        If dataset is invalid
    """
    if len(dataset.values) == 0:
        raise ValueError("Dataset values cannot be empty")

    if dataset.metadata.n_series < 1:
        raise ValueError("Dataset must have at least 1 series")

    if dataset.metadata.horizon < 1:
        raise ValueError("Horizon must be >= 1")


def create_synthetic_dataset(
    n_obs: int = 200,
    n_series: int = 1,
    frequency: str = "W",
    horizon: int = 2,
    train_fraction: float = 0.8,
    ar_coef: float = 0.9,
    noise_std: float = 0.1,
    seed: int = 42,
) -> TimeSeriesDataset:
    """
    Create synthetic AR(1) dataset for testing.

    Parameters
    ----------
    n_obs : int, default=200
        Number of observations per series
    n_series : int, default=1
        Number of time series
    frequency : str, default="W"
        Temporal frequency
    horizon : int, default=2
        Forecast horizon
    train_fraction : float, default=0.8
        Fraction for training data
    ar_coef : float, default=0.9
        AR(1) coefficient (persistence)
    noise_std : float, default=0.1
        Noise standard deviation
    seed : int, default=42
        Random seed

    Returns
    -------
    TimeSeriesDataset
        Synthetic dataset

    Examples
    --------
    >>> dataset = create_synthetic_dataset(n_obs=100, ar_coef=0.95)
    >>> print(f"Shape: {dataset.values.shape}")
    """
    rng = np.random.default_rng(seed)

    if n_series == 1:
        # Single series: shape (n_obs,)
        values = np.zeros(n_obs)
        values[0] = rng.normal(0, noise_std)
        for t in range(1, n_obs):
            values[t] = ar_coef * values[t - 1] + rng.normal(0, noise_std)
    else:
        # Multiple series: shape (n_series, n_obs)
        values = np.zeros((n_series, n_obs))
        values[:, 0] = rng.normal(0, noise_std, size=n_series)
        for t in range(1, n_obs):
            values[:, t] = ar_coef * values[:, t - 1] + rng.normal(
                0, noise_std, size=n_series
            )

    train_end_idx = int(n_obs * train_fraction)

    metadata = DatasetMetadata(
        name="synthetic_ar1",
        frequency=frequency,
        horizon=horizon,
        n_series=n_series,
        total_observations=n_obs * n_series if n_series > 1 else n_obs,
        train_end_idx=train_end_idx,
        characteristics={
            "ar_coef": ar_coef,
            "noise_std": noise_std,
            "synthetic": True,
        },
        license="synthetic",
        source_url="",
    )

    return TimeSeriesDataset(metadata=metadata, values=values)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "DatasetNotFoundError",
    "DatasetMetadata",
    "Dataset",
    "TimeSeriesDataset",
    "validate_dataset",
    "create_synthetic_dataset",
]
