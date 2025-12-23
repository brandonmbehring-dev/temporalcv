"""
Walk-Forward Cross-Validation Module.

Provides sklearn-compatible temporal cross-validation with gap enforcement
for h-step forecasting scenarios.

Knowledge Tiers
---------------
[T1] Walk-forward validation is the standard for time-series (Tashman 2000)
[T1] Gap >= horizon prevents information leakage for h-step forecasts
[T1] Expanding window vs sliding window are both valid approaches (Tashman 2000)
[T2] Gap enforcement: train[-1] + gap < test[0] prevents lookahead
[T2] sklearn TimeSeriesSplit-compatible API for ecosystem integration
[T3] Minimum window size and test size not rigorously validated

Example
-------
>>> from temporalcv import WalkForwardCV
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.linear_model import Ridge
>>>
>>> cv = WalkForwardCV(n_splits=5, gap=2, window_type="sliding", window_size=104)
>>> scores = cross_val_score(Ridge(), X, y, cv=cv)
>>>
>>> # Manual iteration with gap verification
>>> for train_idx, test_idx in cv.split(X):
...     assert train_idx[-1] + cv.gap < test_idx[0]  # Gap enforced
...     model.fit(X[train_idx], y[train_idx])
...     preds = model.predict(X[test_idx])

References
----------
[T1] Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy:
     An analysis and review. International Journal of Forecasting, 16(4), 437-450.
     Key insight: "Rolling origin" validation respects temporal ordering.
[T1] Bergmeir, C. & Benitez, J.M. (2012). On the use of cross-validation for
     time series predictor evaluation. Information Sciences, 191, 192-213.
     Compares blocking, h-blocking, and modified CV approaches.
[T2] sklearn TimeSeriesSplit: Extended with gap parameter and sliding window.
     See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class SplitInfo:
    """
    Metadata for a single CV split.

    Useful for debugging and visualizing the split structure.

    Attributes
    ----------
    split_idx : int
        Zero-based split index
    train_start : int
        First training index (inclusive)
    train_end : int
        Last training index (inclusive)
    test_start : int
        First test index (inclusive)
    test_end : int
        Last test index (inclusive)
    """

    split_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        """Number of training samples."""
        return self.train_end - self.train_start + 1

    @property
    def test_size(self) -> int:
        """Number of test samples."""
        return self.test_end - self.test_start + 1

    @property
    def gap(self) -> int:
        """Actual gap between train end and test start."""
        return self.test_start - self.train_end - 1

    def __post_init__(self) -> None:
        """Validate temporal ordering."""
        if self.train_end >= self.test_start:
            raise ValueError(
                f"Temporal leakage: train_end ({self.train_end}) >= "
                f"test_start ({self.test_start})"
            )


class WalkForwardCV(BaseCrossValidator):  # type: ignore[misc]
    """
    Walk-forward cross-validation with gap enforcement.

    Provides sklearn-compatible temporal CV that ensures no data leakage
    between training and test sets. Supports both expanding and sliding
    window modes per Tashman (2000) recommendations.

    Parameters
    ----------
    n_splits : int, default=5
        Number of CV folds.
    window_type : {"expanding", "sliding"}, default="expanding"
        Type of training window:
        - "expanding": Training set grows from min_train_size
        - "sliding": Fixed-size training window that slides forward
    window_size : int, optional
        Training window size. Required for sliding window.
        For expanding window, this is the minimum initial training size.
        Default is None (auto-calculated for expanding).
    gap : int, default=0
        Number of samples to exclude between training and test.
        For h-step forecasting with change targets, set gap >= h
        to prevent target leakage.
    test_size : int, default=1
        Number of samples in each test fold.

    Attributes
    ----------
    n_splits : int
        Number of splits.
    window_type : str
        Window type ("expanding" or "sliding").
    window_size : int or None
        Window size parameter.
    gap : int
        Gap between train and test.
    test_size : int
        Test set size.

    Examples
    --------
    >>> cv = WalkForwardCV(n_splits=5, gap=2)
    >>> for train, test in cv.split(X):
    ...     print(f"Train: {train[0]}-{train[-1]}, Test: {test[0]}-{test[-1]}")

    >>> # With sklearn
    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(model, X, y, cv=cv)

    Notes
    -----
    For h-step change targets where y[t] = rate[t+h] - rate[t], the last
    valid training target is at index (train_end - h). Setting gap >= h
    ensures that computing the training target doesn't require test-period data.

    See Also
    --------
    CrossFitCV : Forward-only cross-fitting for debiased predictions.
    sklearn.model_selection.TimeSeriesSplit : sklearn's built-in temporal splitter.
    gate_temporal_boundary : Verify gap enforcement meets requirements.
    """

    def __init__(
        self,
        n_splits: int = 5,
        window_type: Literal["expanding", "sliding"] = "expanding",
        window_size: Optional[int] = None,
        gap: int = 0,
        test_size: int = 1,
    ) -> None:
        """Initialize WalkForwardCV."""
        # Validate parameters
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")

        if window_type not in ("expanding", "sliding"):
            raise ValueError(
                f"window_type must be 'expanding' or 'sliding', got {window_type!r}"
            )

        if window_type == "sliding" and window_size is None:
            raise ValueError("window_size is required for sliding window")

        if window_size is not None and window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")

        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")

        self.n_splits = n_splits
        self.window_type = window_type
        self.window_size = window_size
        self.gap = gap
        self.test_size = test_size

    def _get_n_samples(self, X: ArrayLike) -> int:
        """Get number of samples from array-like."""
        if hasattr(X, "shape"):
            return int(X.shape[0])
        return len(X)

    def _calculate_splits(
        self, n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate all train/test splits.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns
        -------
        list of (train_indices, test_indices) tuples
        """
        splits: List[Tuple[np.ndarray, np.ndarray]] = []

        # Calculate minimum training size for first split
        if self.window_type == "sliding":
            assert self.window_size is not None  # Validated in __init__
            min_train = self.window_size
        else:
            # For expanding, auto-calculate initial window size
            if self.window_size is not None:
                min_train = self.window_size
            else:
                # Default: leave enough room for n_splits test sets
                # min_train + gap + test_size + (n_splits - 1) * test_size <= n_samples
                total_test = self.n_splits * self.test_size
                available = n_samples - self.gap - total_test
                min_train = max(1, available)

        # Check if we have enough data
        min_required = min_train + self.gap + self.test_size
        if n_samples < min_required:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits. "
                f"Need at least {min_required} samples "
                f"(min_train={min_train}, gap={self.gap}, test_size={self.test_size})."
            )

        # Generate splits working backwards from end
        # Last test set ends at n_samples - 1
        # Work backwards to find all split positions
        for split_idx in range(self.n_splits):
            # Calculate test indices for this split (from the end)
            # Split 0 is the last (most recent) split
            # We reverse this later to get chronological order
            offset_from_end = split_idx * self.test_size
            test_end = n_samples - 1 - offset_from_end
            test_start = test_end - self.test_size + 1

            # Calculate train indices
            train_end = test_start - self.gap - 1

            if self.window_type == "sliding":
                assert self.window_size is not None
                train_start = train_end - self.window_size + 1
            else:
                train_start = 0

            # Check validity
            if train_start < 0 or train_end < train_start:
                break  # Can't create more valid splits

            if self.window_type == "sliding":
                assert self.window_size is not None
                if train_end - train_start + 1 < self.window_size:
                    break  # Not enough training data

            train_indices = np.arange(train_start, train_end + 1, dtype=np.intp)
            test_indices = np.arange(test_start, test_end + 1, dtype=np.intp)

            splits.append((train_indices, test_indices))

        # Reverse to get chronological order (earliest split first)
        splits = list(reversed(splits))

        # Trim to n_splits if we generated more
        if len(splits) > self.n_splits:
            splits = splits[-self.n_splits:]

        return splits

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target variable (not used, for API compatibility).
        groups : array-like of shape (n_samples,), optional
            Group labels (not used, for API compatibility).

        Yields
        ------
        train : np.ndarray
            Training set indices for this split.
        test : np.ndarray
            Test set indices for this split.

        Examples
        --------
        >>> cv = WalkForwardCV(n_splits=3, gap=2)
        >>> for train, test in cv.split(X):
        ...     X_train, X_test = X[train], X[test]
        ...     y_train, y_test = y[train], y[test]
        """
        n_samples = self._get_n_samples(X)
        splits = self._calculate_splits(n_samples)

        for train_indices, test_indices in splits:
            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """
        Return the number of splitting iterations.

        Parameters
        ----------
        X : array-like, optional
            Training data. If provided, returns actual number of valid splits.
            If None, returns configured n_splits.
        y : array-like, optional
            Not used, for API compatibility.
        groups : array-like, optional
            Not used, for API compatibility.

        Returns
        -------
        int
            Number of splits.
        """
        if X is not None:
            n_samples = self._get_n_samples(X)
            try:
                splits = self._calculate_splits(n_samples)
                return len(splits)
            except ValueError:
                return 0
        return self.n_splits

    def get_split_info(self, X: ArrayLike) -> List[SplitInfo]:
        """
        Return detailed metadata for all splits.

        Useful for debugging and visualizing the split structure.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to split.

        Returns
        -------
        list of SplitInfo
            Metadata for each split.

        Examples
        --------
        >>> cv = WalkForwardCV(n_splits=3, gap=2)
        >>> for info in cv.get_split_info(X):
        ...     print(f"Split {info.split_idx}: train {info.train_start}-{info.train_end}, "
        ...           f"test {info.test_start}-{info.test_end}, gap={info.gap}")
        """
        n_samples = self._get_n_samples(X)
        splits = self._calculate_splits(n_samples)

        infos: List[SplitInfo] = []
        for idx, (train, test) in enumerate(splits):
            info = SplitInfo(
                split_idx=idx,
                train_start=int(train[0]),
                train_end=int(train[-1]),
                test_start=int(test[0]),
                test_end=int(test[-1]),
            )
            infos.append(info)

        return infos

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"WalkForwardCV(n_splits={self.n_splits}, "
            f"window_type={self.window_type!r}, "
            f"window_size={self.window_size}, "
            f"gap={self.gap}, "
            f"test_size={self.test_size})"
        )


# =============================================================================
# CrossFitCV - Temporal Cross-Fitting for Debiased Metrics
# =============================================================================


class CrossFitCV(BaseCrossValidator):  # type: ignore[misc]
    """
    Temporal cross-fitting for debiased out-of-sample predictions.

    For each fold k:
    - Train model on ALL data before fold k (forward-only)
    - Predict on fold k (out-of-sample)

    This eliminates regularization bias by ensuring predictions are NEVER
    made on training data. Unlike standard Double ML (random KFold), this
    enforces strict temporal ordering.

    Knowledge Tier: [T1] - Cross-fitting debiasing is established
    (Chernozhukov et al. 2018). [T2] - Temporal adaptation with gap enforcement.

    Parameters
    ----------
    n_splits : int, default=5
        Number of temporal folds. Data is divided into n_splits consecutive
        chunks of approximately equal size.
    gap : int, default=0
        Number of samples to exclude between training and test for each fold.
        Prevents lookahead bias in h-step forecasting.
    test_size : int, optional
        Size of each test fold. If None, computed automatically as
        n_samples // n_splits.

    Attributes
    ----------
    n_splits : int
        Number of splits
    gap : int
        Gap between train and test
    test_size : int or None
        Test fold size

    Notes
    -----
    **Forward-only semantics**:
    - Fold 0: No training data → predictions are NaN
    - Fold 1: Train on fold 0, predict on fold 1
    - Fold k: Train on folds 0..k-1, predict on fold k

    This is stricter than bidirectional cross-fitting but guarantees
    temporal safety. The first fold cannot receive predictions since
    there's no historical data to train on.

    **vs WalkForwardCV**:
    - WalkForwardCV: Each split is a train/test pair for evaluation
    - CrossFitCV: Each observation gets ONE out-of-sample prediction

    Example
    -------
    >>> cv = CrossFitCV(n_splits=5, gap=2)
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(f"Train: 0-{train_idx[-1]}, Test: {test_idx[0]}-{test_idx[-1]}")

    References
    ----------
    Chernozhukov, V., et al. (2018). Double/debiased machine learning for
    treatment and structural parameters. The Econometrics Journal, 21(1), C1-C68.

    See Also
    --------
    WalkForwardCV : Standard walk-forward CV for model evaluation.
    dm_test : Uses cross-fitted predictions for forecast comparison.
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
    ) -> None:
        """Initialize CrossFitCV."""
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        if gap < 0:
            raise ValueError(f"gap must be >= 0, got {gap}")

        if test_size is not None and test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")

        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size

    def _calculate_fold_indices(
        self, n_samples: int
    ) -> List[Tuple[int, int]]:
        """
        Calculate (start, end) indices for each fold.

        Returns list of (start, end) tuples where end is exclusive.
        """
        if self.test_size is not None:
            fold_size = self.test_size
        else:
            fold_size = n_samples // self.n_splits

        if fold_size < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits"
            )

        folds: List[Tuple[int, int]] = []
        for k in range(self.n_splits):
            start = k * fold_size
            if k == self.n_splits - 1:
                # Last fold takes remaining samples
                end = n_samples
            else:
                end = (k + 1) * fold_size

            if start < n_samples:
                folds.append((start, end))

        return folds

    def split(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.

        For fold k (k >= 1):
        - Train indices: all samples from folds 0 to k-1 (minus gap)
        - Test indices: samples in fold k

        Fold 0 is skipped since there's no training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,), optional
            Target (not used, for API compatibility)
        groups : array-like of shape (n_samples,), optional
            Group labels (not used, for API compatibility)

        Yields
        ------
        train : np.ndarray
            Training set indices
        test : np.ndarray
            Test set indices
        """
        n_samples = len(X) if not hasattr(X, "shape") else X.shape[0]
        folds = self._calculate_fold_indices(n_samples)

        # Skip fold 0 - no training data available
        for k in range(1, len(folds)):
            test_start, test_end = folds[k]

            # Train on all previous folds, respecting gap
            train_end = test_start - self.gap

            if train_end <= 0:
                continue

            train_indices = np.arange(0, train_end, dtype=np.intp)
            test_indices = np.arange(test_start, test_end, dtype=np.intp)

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        groups: Optional[ArrayLike] = None,
    ) -> int:
        """Return number of splitting iterations."""
        if X is not None:
            return sum(1 for _ in self.split(X))
        return self.n_splits - 1

    def fit_predict(
        self,
        model,
        X: ArrayLike,
        y: ArrayLike,
    ) -> np.ndarray:
        """
        Return out-of-sample predictions for all observations.

        Each observation (except fold 0) appears in exactly one test fold.
        Predictions for fold 0 are NaN since there's no training data.

        Parameters
        ----------
        model : sklearn estimator
            Model with fit(X, y) and predict(X) methods.
        X : ArrayLike
            Features array of shape (n_samples, n_features)
        y : ArrayLike
            Target array of shape (n_samples,)

        Returns
        -------
        np.ndarray
            Out-of-sample predictions, shape (n_samples,).
            First fold values are NaN.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = len(y)

        predictions = np.full(n_samples, np.nan)

        for train_idx, test_idx in self.split(X):
            try:
                model_clone = clone(model)
            except TypeError:
                model_clone = model

            model_clone.fit(X[train_idx], y[train_idx])
            predictions[test_idx] = model_clone.predict(X[test_idx])

        return predictions

    def fit_predict_residuals(
        self,
        model,
        X: ArrayLike,
        y: ArrayLike,
    ) -> np.ndarray:
        """
        Return out-of-sample residuals (y - y_hat).

        Parameters
        ----------
        model : sklearn estimator
            Model with fit/predict interface
        X : ArrayLike
            Features
        y : ArrayLike
            Target

        Returns
        -------
        np.ndarray
            Out-of-sample residuals, shape (n_samples,)
        """
        predictions = self.fit_predict(model, X, y)
        return np.asarray(y) - predictions

    def get_fold_indices(self, X: ArrayLike) -> List[Tuple[int, int]]:
        """
        Return (start, end) indices for each fold.

        Parameters
        ----------
        X : array-like
            Data array

        Returns
        -------
        List[Tuple[int, int]]
            List of (start, end) for each fold
        """
        n_samples = len(X) if not hasattr(X, "shape") else X.shape[0]
        return self._calculate_fold_indices(n_samples)

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CrossFitCV(n_splits={self.n_splits}, "
            f"gap={self.gap}, "
            f"test_size={self.test_size})"
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "SplitInfo",
    "WalkForwardCV",
    "CrossFitCV",
]
