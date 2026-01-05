"""
Financial Cross-Validation with Purging and Embargo.

Implements cross-validation techniques for financial machine learning where
labels often overlap (e.g., 5-day forward returns share 4 days of data).
Standard CV leaks information through this overlap.

Key Concepts
------------
- **Purging**: Remove training samples within `purge_gap` of any test sample
- **Embargo**: Additional percentage of samples removed after test set
- **Label overlap**: When labels use future data (e.g., forward returns)

Classes
-------
- PurgedKFold: K-fold with purging and embargo
- CombinatorialPurgedCV: All (n choose k) combinations with purging
- PurgedWalkForward: Walk-forward with purging

References
----------
- De Prado (2018). "Advances in Financial Machine Learning." Wiley.
  Chapter 7: Cross-Validation in Finance.
- Lopez de Prado & Lewis (2019). "Detection of False Investment Strategies
  Using Unsupervised Learning Methods."
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PurgedSplit:
    """A single train/test split with purging information.

    Attributes
    ----------
    train_indices : np.ndarray
        Indices for training set (after purging).
    test_indices : np.ndarray
        Indices for test set.
    n_purged : int
        Number of samples purged from training.
    n_embargoed : int
        Number of samples embargoed after test set.
    """

    train_indices: np.ndarray
    test_indices: np.ndarray
    n_purged: int
    n_embargoed: int


def compute_label_overlap(
    n_samples: int,
    horizon: int,
) -> np.ndarray:
    """Compute overlap matrix for labels with given horizon.

    For financial labels like forward returns, sample i and j share
    data if abs(i - j) < horizon.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    horizon : int
        Label horizon (e.g., 5 for 5-day forward returns).

    Returns
    -------
    np.ndarray
        Boolean matrix (n_samples, n_samples) where entry (i,j) is True
        if labels at indices i and j share any underlying data points.

    Examples
    --------
    >>> overlap = compute_label_overlap(10, horizon=3)
    >>> overlap[0, 2]  # Samples 0 and 2 share data (within horizon)
    True
    >>> overlap[0, 5]  # Samples 0 and 5 don't share data
    False

    Notes
    -----
    [T1] De Prado (2018), Chapter 7.
    For h-day forward returns: label_t uses data from t to t+h,
    so labels t1 and t2 overlap if |t1 - t2| < h.
    """
    indices = np.arange(n_samples)
    # Compute pairwise distances
    dist_matrix = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
    result: np.ndarray = dist_matrix < horizon
    return result


def estimate_purge_gap(
    horizon: int,
    decay_factor: float = 1.0,
) -> int:
    """Estimate appropriate purge gap given label horizon.

    Parameters
    ----------
    horizon : int
        Label horizon (e.g., 5 for 5-day forward returns).
    decay_factor : float
        Multiplier for horizon. Default 1.0 means purge_gap = horizon.
        Use >1.0 for conservative purging.

    Returns
    -------
    int
        Suggested purge gap.

    Examples
    --------
    >>> estimate_purge_gap(horizon=5)
    5
    >>> estimate_purge_gap(horizon=5, decay_factor=1.5)
    8

    Notes
    -----
    [T2] Rule of thumb: purge_gap >= horizon to prevent any overlap.
    The decay_factor allows for more aggressive (>1) or relaxed (<1) purging.
    """
    return max(1, int(horizon * decay_factor))


def _apply_purge_and_embargo(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
    purge_gap: int,
    embargo_pct: float,
) -> tuple[np.ndarray, int, int]:
    """Apply purging and embargo to training indices.

    Parameters
    ----------
    train_indices : np.ndarray
        Original training indices.
    test_indices : np.ndarray
        Test indices.
    n_samples : int
        Total number of samples.
    purge_gap : int
        Remove training samples within purge_gap of test samples.
    embargo_pct : float
        Remove additional embargo_pct * n_samples after test set.

    Returns
    -------
    purged_train : np.ndarray
        Training indices after purging and embargo.
    n_purged : int
        Number of samples removed by purging.
    n_embargoed : int
        Number of samples removed by embargo.
    """
    train_set = set(train_indices)
    test_min = int(np.min(test_indices))
    test_max = int(np.max(test_indices))

    # Purging: remove training samples within purge_gap of any test sample
    purge_indices = set()
    for t_idx in test_indices:
        for offset in range(-purge_gap, purge_gap + 1):
            idx = t_idx + offset
            if idx in train_set:
                purge_indices.add(idx)

    # Embargo: remove additional samples after test set
    n_embargo = int(embargo_pct * n_samples)
    embargo_indices = set()
    for i in range(test_max + 1, min(test_max + 1 + n_embargo, n_samples)):
        if i in train_set:
            embargo_indices.add(i)

    # Also remove embargo samples from before test set (symmetric)
    for i in range(max(0, test_min - n_embargo), test_min):
        if i in train_set:
            embargo_indices.add(i)

    # Combine and remove
    remove_indices = purge_indices | embargo_indices
    purged_train = np.array([i for i in train_indices if i not in remove_indices])

    return (
        purged_train,
        len(purge_indices - embargo_indices),  # Count purged only (not in embargo)
        len(embargo_indices),
    )


class PurgedKFold:
    """Purged K-Fold cross-validation for overlapping labels.

    Removes samples from training set that are within purge_gap of any
    test sample. Prevents information leakage when labels use future
    data (e.g., 5-day forward returns share 4 days).

    Parameters
    ----------
    n_splits : int
        Number of folds.
    purge_gap : int
        Remove training samples within this distance of test samples.
    embargo_pct : float
        Additional percentage of samples to remove after test set.
    shuffle : bool
        Whether to shuffle before splitting. Default False for time series.

    Examples
    --------
    >>> cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     score = model.score(X[test_idx], y[test_idx])

    Notes
    -----
    [T1] De Prado (2018), Chapter 7.3.
    Standard K-fold leaks information when labels overlap. Purging removes
    the overlapping samples from training.

    See Also
    --------
    sklearn.model_selection.KFold : Standard K-fold without purging.
    PurgedWalkForward : Time-ordered CV with purging.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        shuffle: bool = False,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        if not 0 <= embargo_pct < 1:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")

        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.shuffle = shuffle

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with purging.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target (ignored, for sklearn compatibility).
        groups : array-like, optional
            Group labels (ignored, for sklearn compatibility).

        Yields
        ------
        train : np.ndarray
            Training indices (after purging).
        test : np.ndarray
            Test indices.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_indices = indices[current : current + fold_size]
            train_indices = np.concatenate(
                [indices[:current], indices[current + fold_size :]]
            )

            purged_train, _, _ = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            yield purged_train, test_indices
            current += fold_size

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits

    def split_detailed(
        self,
        X: ArrayLike,
    ) -> Iterator[PurgedSplit]:
        """Generate detailed split information including purge/embargo counts.

        Parameters
        ----------
        X : array-like
            Training data.

        Yields
        ------
        PurgedSplit
            Detailed split with train/test indices and purge/embargo counts.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            test_indices = indices[current : current + fold_size]
            train_indices = np.concatenate(
                [indices[:current], indices[current + fold_size :]]
            )

            purged_train, n_purged, n_embargoed = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            yield PurgedSplit(
                train_indices=purged_train,
                test_indices=test_indices,
                n_purged=n_purged,
                n_embargoed=n_embargoed,
            )
            current += fold_size


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (CPCV).

    Generates all (n choose k) combinations of groups for test sets,
    applying purging and embargo to each. Ensures every sample is tested
    exactly once across all paths.

    Parameters
    ----------
    n_splits : int
        Number of groups to divide data into.
    n_test_splits : int
        Number of groups to use for each test set.
    purge_gap : int
        Remove training samples within this distance of test samples.
    embargo_pct : float
        Additional percentage of samples to remove after test set.

    Examples
    --------
    >>> cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)
    >>> n_paths = cv.get_n_splits(X)  # C(5,2) = 10 paths
    >>> for train_idx, test_idx in cv.split(X):
    ...     # Each sample tested in exactly 2 paths

    Notes
    -----
    [T1] De Prado (2018), Chapter 7.4.
    CPCV provides more reliable backtests by testing each sample multiple
    times (via different paths). The number of paths is C(n_splits, n_test_splits).

    See Also
    --------
    PurgedKFold : Standard purged K-fold (each sample tested once).
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits must be in [1, n_splits), got {n_test_splits}"
            )
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        if not 0 <= embargo_pct < 1:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for all combinatorial paths.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target (ignored).
        groups : array-like, optional
            Group labels (ignored).

        Yields
        ------
        train : np.ndarray
            Training indices (after purging).
        test : np.ndarray
            Test indices.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        # Divide into groups
        indices = np.arange(n_samples)
        group_indices = np.array_split(indices, self.n_splits)

        # Generate all combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            # Test indices = union of selected groups
            test_indices = np.concatenate([group_indices[g] for g in test_groups])

            # Train indices = all other groups
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            train_indices = np.concatenate([group_indices[g] for g in train_groups])

            # Apply purging and embargo
            purged_train, _, _ = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            yield purged_train, test_indices

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return number of combinatorial paths."""
        from math import comb

        return comb(self.n_splits, self.n_test_splits)


class PurgedWalkForward:
    """Walk-forward cross-validation with purging for overlapping labels.

    Extends standard walk-forward CV with purge_gap and embargo to handle
    financial labels that overlap.

    Parameters
    ----------
    n_splits : int
        Number of test periods.
    train_size : int | None
        Fixed training window size. If None, uses expanding window.
    test_size : int | None
        Fixed test window size. If None, auto-computed.
    purge_gap : int
        Remove training samples within this distance of test samples.
    embargo_pct : float
        Additional percentage of samples to remove after test set.
    extra_gap : int
        Additional separation between train and test (on top of purge_gap).

    Examples
    --------
    >>> cv = PurgedWalkForward(
    ...     n_splits=5,
    ...     train_size=100,
    ...     test_size=20,
    ...     purge_gap=5
    ... )
    >>> for train_idx, test_idx in cv.split(X):
    ...     model.fit(X[train_idx], y[train_idx])
    ...     predictions = model.predict(X[test_idx])

    Notes
    -----
    [T1] De Prado (2018), Chapter 7.
    Walk-forward is preferred for time series because it respects temporal
    order. Adding purging prevents leakage from overlapping labels.

    See Also
    --------
    temporalcv.cv.WalkForwardCV : Standard walk-forward without purging.
    PurgedKFold : Purged K-fold (not time-ordered).
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        extra_gap: int = 0,
    ):
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if train_size is not None and train_size < 1:
            raise ValueError(f"train_size must be >= 1, got {train_size}")
        if test_size is not None and test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        if purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
        if not 0 <= embargo_pct < 1:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")
        if extra_gap < 0:
            raise ValueError(f"extra_gap must be >= 0, got {extra_gap}")

        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.extra_gap = extra_gap

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test indices with purging.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target (ignored).
        groups : array-like, optional
            Group labels (ignored).

        Yields
        ------
        train : np.ndarray
            Training indices (after purging).
        test : np.ndarray
            Test indices.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        # Compute test size if not specified
        test_size = self.test_size
        if test_size is None:
            # Reserve space for train, extra_gap, and splits
            min_train = self.train_size or (n_samples // (self.n_splits + 1))
            available = n_samples - min_train - self.extra_gap
            test_size = max(1, available // self.n_splits)

        # Compute starting positions for each split
        total_gap = self.extra_gap + self.purge_gap
        for split_idx in range(self.n_splits):
            # Test window position
            test_end = n_samples - (self.n_splits - split_idx - 1) * test_size
            test_start = test_end - test_size

            # Train window
            if self.train_size is not None:
                # Fixed window
                train_end = test_start - total_gap
                train_start = max(0, train_end - self.train_size)
            else:
                # Expanding window
                train_start = 0
                train_end = test_start - total_gap

            if train_end <= train_start:
                # Not enough data for this split
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            # Apply additional purging and embargo
            purged_train, _, _ = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            if len(purged_train) > 0:
                yield purged_train, test_indices

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return number of splits."""
        return self.n_splits

    def split_detailed(
        self,
        X: ArrayLike,
    ) -> Iterator[PurgedSplit]:
        """Generate detailed split information.

        Parameters
        ----------
        X : array-like
            Training data.

        Yields
        ------
        PurgedSplit
            Detailed split with purge/embargo counts.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        test_size = self.test_size
        if test_size is None:
            min_train = self.train_size or (n_samples // (self.n_splits + 1))
            available = n_samples - min_train - self.extra_gap
            test_size = max(1, available // self.n_splits)

        total_gap = self.extra_gap + self.purge_gap
        for split_idx in range(self.n_splits):
            test_end = n_samples - (self.n_splits - split_idx - 1) * test_size
            test_start = test_end - test_size

            if self.train_size is not None:
                train_end = test_start - total_gap
                train_start = max(0, train_end - self.train_size)
            else:
                train_start = 0
                train_end = test_start - total_gap

            if train_end <= train_start:
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            purged_train, n_purged, n_embargoed = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            if len(purged_train) > 0:
                yield PurgedSplit(
                    train_indices=purged_train,
                    test_indices=test_indices,
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
