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
from typing import Any, Generator, Iterator, List, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class SplitInfo:
    """
    Metadata for a single CV split.

    Useful for debugging and visualizing the split structure.
    Supports optional datetime fields when data has DatetimeIndex.

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
    train_start_date : datetime, optional
        Training period start date (if DatetimeIndex available)
    train_end_date : datetime, optional
        Training period end date
    test_start_date : datetime, optional
        Test period start date
    test_end_date : datetime, optional
        Test period end date

    Example
    -------
    >>> info = SplitInfo(
    ...     split_idx=0,
    ...     train_start=0, train_end=99,
    ...     test_start=102, test_end=111,
    ... )
    >>> print(f"Gap: {info.gap}, Train size: {info.train_size}")
    """

    split_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    # Optional datetime fields (populated when data has DatetimeIndex)
    train_start_date: Optional[Any] = None  # datetime
    train_end_date: Optional[Any] = None
    test_start_date: Optional[Any] = None
    test_end_date: Optional[Any] = None

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

    @property
    def has_dates(self) -> bool:
        """Check if datetime information is available."""
        return self.train_start_date is not None

    def __post_init__(self) -> None:
        """Validate temporal ordering."""
        if self.train_end >= self.test_start:
            raise ValueError(
                f"Temporal leakage: train_end ({self.train_end}) >= "
                f"test_start ({self.test_start})"
            )


@dataclass
class SplitResult:
    """
    Result from a single walk-forward split.

    Contains predictions, actuals, and metadata for one CV split.
    Used as building block for WalkForwardResults.

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
    predictions : np.ndarray
        Model predictions for this split's test set
    actuals : np.ndarray
        Actual values for this split's test set
    train_start_date : datetime, optional
        Training period start date (if DatetimeIndex available)
    train_end_date : datetime, optional
        Training period end date
    test_start_date : datetime, optional
        Test period start date
    test_end_date : datetime, optional
        Test period end date

    Example
    -------
    >>> result = SplitResult(
    ...     split_idx=0,
    ...     train_start=0, train_end=99,
    ...     test_start=102, test_end=111,
    ...     predictions=np.array([1.0, 1.1, ...]),
    ...     actuals=np.array([1.05, 1.08, ...]),
    ... )
    >>> print(f"Split {result.split_idx}: MAE={result.mae:.4f}")
    """

    split_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    predictions: np.ndarray
    actuals: np.ndarray
    # Optional datetime fields (populated when X has DatetimeIndex)
    train_start_date: Optional[Any] = None  # datetime
    train_end_date: Optional[Any] = None
    test_start_date: Optional[Any] = None
    test_end_date: Optional[Any] = None

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

    @property
    def has_dates(self) -> bool:
        """Check if datetime information is available."""
        return self.train_start_date is not None

    @property
    def errors(self) -> np.ndarray:
        """Prediction errors (predictions - actuals)."""
        return cast(np.ndarray, self.predictions - self.actuals)

    @property
    def absolute_errors(self) -> np.ndarray:
        """Absolute prediction errors."""
        return cast(np.ndarray, np.abs(self.errors))

    @property
    def mae(self) -> float:
        """Mean Absolute Error for this split."""
        return float(np.mean(self.absolute_errors))

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error for this split."""
        return float(np.sqrt(np.mean(self.errors**2)))

    @property
    def bias(self) -> float:
        """Mean signed error (positive = over-prediction)."""
        return float(np.mean(self.errors))

    def to_split_info(self) -> SplitInfo:
        """Convert to SplitInfo (metadata without predictions/actuals)."""
        return SplitInfo(
            split_idx=self.split_idx,
            train_start=self.train_start,
            train_end=self.train_end,
            test_start=self.test_start,
            test_end=self.test_end,
            train_start_date=self.train_start_date,
            train_end_date=self.train_end_date,
            test_start_date=self.test_start_date,
            test_end_date=self.test_end_date,
        )


@dataclass
class WalkForwardResults:
    """
    Aggregated walk-forward cross-validation results.

    Collects results from all splits and provides lazy-computed
    aggregate metrics. Designed for the common workflow of running
    CV and then analyzing overall performance.

    Attributes
    ----------
    splits : List[SplitResult]
        Results from each CV split
    cv_config : dict, optional
        Configuration of the CV (n_splits, gap, window_type, etc.)

    Properties (lazy-computed)
    --------------------------
    mae : float
        Overall Mean Absolute Error across all splits
    rmse : float
        Overall Root Mean Squared Error
    bias : float
        Overall mean signed error
    predictions : np.ndarray
        All predictions concatenated
    actuals : np.ndarray
        All actuals concatenated

    Example
    -------
    >>> from temporalcv import walk_forward_evaluate
    >>> results = walk_forward_evaluate(model, X, y, n_splits=5)
    >>> print(f"Overall MAE: {results.mae:.4f}")
    >>> print(f"Overall RMSE: {results.rmse:.4f}")
    >>> for split in results.splits:
    ...     print(f"  Split {split.split_idx}: MAE={split.mae:.4f}")
    """

    splits: List["SplitResult"]
    cv_config: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate splits list."""
        if not self.splits:
            raise ValueError("WalkForwardResults requires at least one split")

    @property
    def n_splits(self) -> int:
        """Number of CV splits."""
        return len(self.splits)

    @property
    def predictions(self) -> np.ndarray:
        """All predictions concatenated across splits."""
        return cast(np.ndarray, np.concatenate([s.predictions for s in self.splits]))

    @property
    def actuals(self) -> np.ndarray:
        """All actuals concatenated across splits."""
        return cast(np.ndarray, np.concatenate([s.actuals for s in self.splits]))

    @property
    def errors(self) -> np.ndarray:
        """All errors concatenated across splits."""
        return cast(np.ndarray, self.predictions - self.actuals)

    @property
    def absolute_errors(self) -> np.ndarray:
        """All absolute errors concatenated."""
        return cast(np.ndarray, np.abs(self.errors))

    @property
    def mae(self) -> float:
        """Overall Mean Absolute Error."""
        return float(np.mean(self.absolute_errors))

    @property
    def rmse(self) -> float:
        """Overall Root Mean Squared Error."""
        return float(np.sqrt(np.mean(self.errors**2)))

    @property
    def bias(self) -> float:
        """Overall mean signed error (positive = over-prediction)."""
        return float(np.mean(self.errors))

    @property
    def mse(self) -> float:
        """Overall Mean Squared Error."""
        return float(np.mean(self.errors**2))

    @property
    def total_samples(self) -> int:
        """Total number of test samples across all splits."""
        return sum(s.test_size for s in self.splits)

    def per_split_metrics(self) -> List[dict[str, float]]:
        """Return metrics for each split as list of dicts."""
        return [
            {
                "split_idx": s.split_idx,
                "mae": s.mae,
                "rmse": s.rmse,
                "bias": s.bias,
                "n_samples": s.test_size,
            }
            for s in self.splits
        ]

    def to_dataframe(self) -> Any:
        """
        Export results to pandas DataFrame.

        Returns DataFrame with one row per prediction, including
        split index, prediction, actual, error, and dates if available.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: split_idx, prediction, actual, error,
            and optionally date columns

        Raises
        ------
        ImportError
            If pandas is not installed
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas required for to_dataframe(): pip install pandas"
            ) from e

        rows = []
        for split in self.splits:
            for i, (pred, actual) in enumerate(
                zip(split.predictions, split.actuals)
            ):
                row = {
                    "split_idx": split.split_idx,
                    "prediction": pred,
                    "actual": actual,
                    "error": pred - actual,
                    "abs_error": abs(pred - actual),
                }
                if split.has_dates and split.test_start_date is not None:
                    # Add date if available (requires DatetimeIndex tracking)
                    row["test_idx"] = split.test_start + i
                rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return formatted summary of results."""
        lines = [
            "WalkForwardResults Summary",
            "=" * 40,
            f"Splits: {self.n_splits}",
            f"Total samples: {self.total_samples}",
            "",
            "Overall Metrics:",
            f"  MAE:  {self.mae:.6f}",
            f"  RMSE: {self.rmse:.6f}",
            f"  Bias: {self.bias:+.6f}",
            "",
            "Per-Split MAE:",
        ]
        for s in self.splits:
            lines.append(f"  Split {s.split_idx}: {s.mae:.6f} (n={s.test_size})")

        return "\n".join(lines)


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
    horizon : int, optional
        Forecast horizon (h-step ahead). When provided, validates that
        gap >= horizon to prevent target leakage for multi-step forecasting.
        [T1] Per Bergmeir & Benitez (2012), gap must equal or exceed horizon.
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
    horizon : int or None
        Forecast horizon for gap validation.
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
        horizon: Optional[int] = None,
        window_type: Literal["expanding", "sliding"] = "expanding",
        window_size: Optional[int] = None,
        gap: int = 0,
        test_size: int = 1,
    ) -> None:
        """Initialize WalkForwardCV."""
        # Validate parameters
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")

        if horizon is not None and horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

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

        # [T1] Gap >= horizon prevents target leakage for h-step forecasting
        # Per Bergmeir & Benitez (2012): gap must equal or exceed forecast horizon
        if horizon is not None and gap < horizon:
            raise ValueError(
                f"gap ({gap}) must be >= horizon ({horizon}) to prevent target leakage. "
                f"For {horizon}-step forecasting, set gap >= {horizon}. "
                "See Bergmeir & Benitez (2012) for details on temporal CV for multi-step forecasts."
            )

        self.n_splits = n_splits
        self.horizon = horizon
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
        *,
        strict: bool = True,
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
        strict : bool, default=True
            If True (default), raise ValueError when splits cannot be computed
            (e.g., insufficient data). If False, return 0 on failure.

            .. versionadded:: 1.0.0
               The ``strict`` parameter was added to prevent silent failures.
               Previously, errors were silently swallowed and 0 was returned.

        Returns
        -------
        int
            Number of splits.

        Raises
        ------
        ValueError
            If strict=True and splits cannot be computed due to insufficient data
            or invalid configuration.
        """
        if X is not None:
            n_samples = self._get_n_samples(X)
            try:
                splits = self._calculate_splits(n_samples)
                return len(splits)
            except ValueError as e:
                if strict:
                    raise ValueError(
                        f"Cannot compute n_splits: {e}. "
                        f"Set strict=False to return 0 instead of raising."
                    ) from e
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
        *,
        strict: bool = True,
    ) -> int:
        """
        Return number of splitting iterations.

        Parameters
        ----------
        X : array-like, optional
            Training data. If provided, returns actual number of valid splits.
        y : array-like, optional
            Not used, for API compatibility.
        groups : array-like, optional
            Not used, for API compatibility.
        strict : bool, default=True
            If True (default), raise ValueError on failure.
            If False, return 0 on failure.

            .. versionadded:: 1.0.0

        Returns
        -------
        int
            Number of splits.
        """
        if X is not None:
            try:
                return sum(1 for _ in self.split(X))
            except ValueError as e:
                if strict:
                    raise ValueError(
                        f"Cannot compute n_splits: {e}. "
                        f"Set strict=False to return 0 instead of raising."
                    ) from e
                return 0
        return self.n_splits - 1

    def fit_predict(
        self,
        model: Any,
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

        return cast(np.ndarray, predictions)

    def fit_predict_residuals(
        self,
        model: Any,
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
        return cast(np.ndarray, np.asarray(y) - predictions)

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
# walk_forward_evaluate Function
# =============================================================================


def walk_forward_evaluate(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
    cv: Optional[WalkForwardCV] = None,
    n_splits: int = 5,
    horizon: Optional[int] = None,
    window_type: Literal["expanding", "sliding"] = "expanding",
    window_size: Optional[int] = None,
    gap: int = 0,
    test_size: int = 1,
    verbose: bool = False,
) -> WalkForwardResults:
    """
    Evaluate model using walk-forward cross-validation.

    Convenience function that runs walk-forward CV and returns structured
    results with per-split and aggregate metrics. Follows sklearn patterns:
    clones model for each split to ensure independence.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Model with fit(X, y) and predict(X) methods. Will be cloned
        for each split.
    X : ArrayLike
        Features array of shape (n_samples, n_features) or (n_samples,).
        If pandas DataFrame/Series with DatetimeIndex, dates are captured.
    y : ArrayLike
        Target array of shape (n_samples,)
    cv : WalkForwardCV, optional
        Pre-configured CV splitter. If provided, other CV parameters
        are ignored.
    n_splits : int, default=5
        Number of CV folds (ignored if cv is provided)
    horizon : int, optional
        Forecast horizon for gap validation (ignored if cv is provided)
    window_type : {"expanding", "sliding"}, default="expanding"
        Training window type (ignored if cv is provided)
    window_size : int, optional
        Window size for sliding window (ignored if cv is provided)
    gap : int, default=0
        Gap between train and test (ignored if cv is provided)
    test_size : int, default=1
        Number of test samples per fold (ignored if cv is provided)
    verbose : bool, default=False
        If True, print progress for each split

    Returns
    -------
    WalkForwardResults
        Aggregated results with per-split metrics and overall metrics.
        Access via .mae, .rmse, .bias properties or iterate over .splits.

    Example
    -------
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> np.random.seed(42)
    >>> X = np.random.randn(200, 5)
    >>> y = X[:, 0] * 0.5 + np.random.randn(200) * 0.1
    >>>
    >>> # Evaluate model
    >>> results = walk_forward_evaluate(Ridge(), X, y, n_splits=5, gap=2)
    >>> print(f"MAE: {results.mae:.4f}")
    >>> print(f"RMSE: {results.rmse:.4f}")
    >>>
    >>> # Access per-split details
    >>> for split in results.splits:
    ...     print(f"Split {split.split_idx}: MAE={split.mae:.4f}")

    Notes
    -----
    **Model cloning**: Each split gets a fresh clone of the model via
    sklearn.base.clone(). If clone() fails (non-sklearn model), the
    original model is reused (be aware this may cause state leakage).

    **Date extraction**: If X is a pandas DataFrame/Series with DatetimeIndex,
    the SplitResult objects will include train_start_date, train_end_date,
    test_start_date, and test_end_date fields.

    Knowledge Tier: [T2] - Convenience wrapper over established WalkForwardCV.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    # Create CV splitter if not provided
    if cv is None:
        cv = WalkForwardCV(
            n_splits=n_splits,
            horizon=horizon,
            window_type=window_type,
            window_size=window_size,
            gap=gap,
            test_size=test_size,
        )

    # Attempt to extract dates from X if it's a pandas object
    dates = None
    try:
        import pandas as pd

        if hasattr(X, "index") and isinstance(X.index, pd.DatetimeIndex):  # type: ignore[union-attr]
            dates = X.index  # type: ignore[union-attr]
    except ImportError:
        pass  # pandas not available

    # Collect split results
    split_results: List[SplitResult] = []

    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        if verbose:
            print(f"Split {split_idx}: train[{train_idx[0]}:{train_idx[-1]}], "
                  f"test[{test_idx[0]}:{test_idx[-1]}]")

        # Clone model for this split
        try:
            model_clone = clone(model)
        except TypeError:
            # clone() failed (non-sklearn model), reuse original
            model_clone = model

        # Fit and predict
        model_clone.fit(X_arr[train_idx], y_arr[train_idx])
        predictions = model_clone.predict(X_arr[test_idx])
        actuals = y_arr[test_idx]

        # Build SplitResult
        train_start_date = None
        train_end_date = None
        test_start_date = None
        test_end_date = None

        if dates is not None:
            train_start_date = dates[train_idx[0]]
            train_end_date = dates[train_idx[-1]]
            test_start_date = dates[test_idx[0]]
            test_end_date = dates[test_idx[-1]]

        split_result = SplitResult(
            split_idx=split_idx,
            train_start=int(train_idx[0]),
            train_end=int(train_idx[-1]),
            test_start=int(test_idx[0]),
            test_end=int(test_idx[-1]),
            predictions=np.asarray(predictions),
            actuals=np.asarray(actuals),
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            test_start_date=test_start_date,
            test_end_date=test_end_date,
        )
        split_results.append(split_result)

    # Build CV config for reference
    cv_config = {
        "n_splits": cv.n_splits,
        "horizon": cv.horizon,
        "window_type": cv.window_type,
        "window_size": cv.window_size,
        "gap": cv.gap,
        "test_size": cv.test_size,
    }

    return WalkForwardResults(splits=split_results, cv_config=cv_config)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "SplitInfo",
    "SplitResult",
    "WalkForwardResults",
    "WalkForwardCV",
    "CrossFitCV",
    "walk_forward_evaluate",
]
