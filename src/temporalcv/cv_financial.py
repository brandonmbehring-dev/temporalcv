"""
Financial Cross-Validation with Purging and Embargo.

Implements cross-validation techniques for financial machine learning where
labels often overlap (e.g., 5-day forward returns share 4 days of data).
Standard CV leaks information through this overlap.

Key Concepts
------------
- **Purging**: Remove training samples within `purge_gap` of any test sample
- **Embargo**: Additional samples removed immediately after each contiguous
  test run (one-sided, De Prado)
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

See Also
--------
eval-toolkit hosts an adapted purged K-fold (``eval_toolkit/splits.py``,
"Adapted from temporalcv"). Both implement label-overlap purging; they differ
in domain — classification-evaluation there, forecasting/financial time series
here — so they make different assumptions about how the overlap window is
defined and indexed. The two are deliberately kept separate per the hub
pattern *universal-vs-unique.md* ("Two toolkits, same-named concept"):
consolidate only a concept that is genuinely identical across domains (#17).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import Any, ClassVar

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from temporalcv._serialization import result_to_dict
from temporalcv._typing import ArrayLike


@dataclass(frozen=True, slots=True, eq=False)
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
        Number of samples embargoed after the contiguous test run(s).
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    train_indices: np.ndarray
    test_indices: np.ndarray
    n_purged: int
    n_embargoed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping of this split."""
        return result_to_dict(self)


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
        Remove ``ceil(embargo_pct * n_samples)`` training samples immediately
        after each contiguous run of test indices (De Prado one-sided embargo).
        Applied per run, so interior boundaries of multi-block test sets are
        embargoed too. The embargo is one-sided by design: a run's leading edge
        is purging's responsibility (label overlap), not the embargo's.

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

    # Purging: remove training samples within purge_gap of any test sample
    purge_indices = set()
    for t_idx in test_indices:
        for offset in range(-purge_gap, purge_gap + 1):
            idx = t_idx + offset
            if idx in train_set:
                purge_indices.add(idx)

    # Embargo (De Prado, one-sided): remove training samples immediately AFTER
    # each *contiguous run* of test indices. The embargo is one-sided by design
    # — a run's leading edge is purging's job (label overlap), not the
    # embargo's — so no pre-test embargo is applied. The span is taken
    # from per-run maxima, NOT a single global test_max — that is what protects
    # the interior boundaries of multi-block test sets (CombinatorialPurgedCV,
    # shuffled PurgedKFold); a global span silently skips them. ``ceil`` rounds
    # the leakage guard toward MORE protection, so a small nonzero embargo_pct
    # can never truncate to zero embargoed rows.
    n_embargo = math.ceil(embargo_pct * n_samples)
    embargo_indices: set[int] = set()
    if n_embargo > 0 and len(test_indices) > 0:
        sorted_test = np.unique(test_indices)
        # Last index of each maximal contiguous run (a gap > 1 ends a run).
        run_ends = sorted_test[np.append(np.diff(sorted_test) != 1, True)]
        for end in run_ends:
            stop = min(int(end) + 1 + n_embargo, n_samples)
            for i in range(int(end) + 1, stop):
                if i in train_set:
                    embargo_indices.add(i)

    # Combine and remove. dtype is pinned because an empty list would
    # otherwise infer float64 — unusable as an index array downstream.
    remove_indices = purge_indices | embargo_indices
    purged_train = np.array([i for i in train_indices if i not in remove_indices], dtype=np.intp)

    return (
        purged_train,
        len(purge_indices - embargo_indices),  # Count purged only (not in embargo)
        len(embargo_indices),
    )


class PurgedKFold(BaseCrossValidator):  # type: ignore[misc]
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
        Fraction of samples (``ceil``) embargoed immediately after each
        contiguous test run.
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

    An over-aggressive configuration — any fold whose train set is emptied
    by purge/embargo removal, or ``n_samples < n_splits`` (empty test
    folds) — raises ``ValueError`` at ``split``/``split_detailed`` call
    time instead of silently yielding unusable folds, so ``split`` always
    yields exactly ``n_splits`` folds with non-empty train and test sets.

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
        if shuffle:
            warnings.warn(
                "PurgedKFold(shuffle=True) is deprecated and will be removed in "
                "temporalcv 3.0. De Prado's PurgedKFold is a time-series splitter "
                "with no shuffle: shuffling scatters the test blocks and breaks the "
                "temporal ordering that purging and embargo assume. During the "
                "deprecation window the shuffle is seeded (reproducible and "
                "consistent between split() and split_detailed()); migrate to "
                "shuffle=False or PurgedWalkForward.",
                DeprecationWarning,
                stacklevel=2,
            )

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

        Returns
        -------
        Iterator[tuple[np.ndarray, np.ndarray]]
            ``(train, test)`` index pairs; train indices are post-purging.

        Raises
        ------
        ValueError
            If ``n_samples < n_splits`` (some test folds would be empty)
            or purge/embargo removal empties any fold's train set. Raised
            at call time, before any fold is produced.
        """
        return iter(
            [(detailed.train_indices, detailed.test_indices) for detailed in self.split_detailed(X)]
        )

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return number of splits.

        ``split`` either yields exactly this many folds or raises
        ``ValueError`` on an over-aggressive configuration, so the nominal
        count is always truthful.
        """
        return self.n_splits

    def split_detailed(
        self,
        X: ArrayLike,
    ) -> Iterator[PurgedSplit]:
        """Generate detailed split information including purge/embargo counts.

        All folds are validated at call time: a configuration that empties
        any train set raises before any fold is produced, so a consumer
        never does partial work on a doomed iteration.

        Parameters
        ----------
        X : array-like
            Training data.

        Returns
        -------
        Iterator[PurgedSplit]
            Detailed splits with train/test indices and purge/embargo counts.

        Raises
        ------
        ValueError
            If ``n_samples < n_splits`` (some test folds would be empty)
            or purge/embargo removal empties any fold's train set.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        if n_samples < self.n_splits:
            raise ValueError(
                f"PurgedKFold cannot split n_samples={n_samples} into "
                f"n_splits={self.n_splits} folds: some test folds would be "
                f"empty. Reduce n_splits or provide more samples."
            )

        indices: np.ndarray = np.arange(n_samples)
        if self.shuffle:
            # Deprecated path (see __init__): seed a local generator so the
            # shuffle is reproducible and identical across split()/split_detailed
            # calls, instead of consuming the unseedable global RNG.
            indices = np.random.default_rng(0).permutation(n_samples)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[: n_samples % self.n_splits] += 1

        splits: list[PurgedSplit] = []
        current = 0
        for split_idx, fold_size in enumerate(fold_sizes):
            test_indices = indices[current : current + fold_size]
            train_indices = np.concatenate([indices[:current], indices[current + fold_size :]])

            purged_train, n_purged, n_embargoed = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            if len(purged_train) == 0:
                raise ValueError(
                    f"PurgedKFold fold {split_idx}: purge/embargo removal "
                    f"emptied the train set for n_samples={n_samples} "
                    f"(n_splits={self.n_splits}, purge_gap={self.purge_gap}, "
                    f"embargo_pct={self.embargo_pct}). Reduce "
                    f"purge_gap/embargo_pct, increase n_splits (smaller test "
                    f"blocks shrink the purged band), or provide more samples."
                )

            splits.append(
                PurgedSplit(
                    train_indices=purged_train,
                    test_indices=test_indices,
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
            )
            current += fold_size

        return iter(splits)


class CombinatorialPurgedCV(BaseCrossValidator):  # type: ignore[misc]
    """Combinatorial Purged Cross-Validation (CPCV).

    Generates all (n choose k) combinations of groups for test sets,
    applying purging and embargo to each. Every sample is tested the same
    number of times — its group appears in C(n_splits-1, n_test_splits-1)
    of the paths.

    Parameters
    ----------
    n_splits : int
        Number of groups to divide data into.
    n_test_splits : int
        Number of groups to use for each test set.
    purge_gap : int
        Remove training samples within this distance of test samples.
    embargo_pct : float
        Fraction of samples (``ceil``) embargoed immediately after each
        contiguous test run.

    Examples
    --------
    >>> cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)
    >>> n_paths = cv.get_n_splits(X)  # C(5,2) = 10 paths
    >>> for train_idx, test_idx in cv.split(X):
    ...     # Each sample tested in C(4, 1) = 4 of the 10 paths

    Notes
    -----
    [T1] De Prado (2018), Chapter 7.4.
    CPCV provides more reliable backtests by testing each sample multiple
    times (via different paths). The number of paths is C(n_splits, n_test_splits).

    An over-aggressive configuration — any path whose train set is emptied
    by purge/embargo removal, or ``n_samples < n_splits`` (empty groups) —
    raises ``ValueError`` at ``split`` call time instead of silently
    yielding unusable paths, so ``split`` always yields exactly
    ``get_n_splits()`` paths with non-empty train and test sets.

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
            raise ValueError(f"n_test_splits must be in [1, n_splits), got {n_test_splits}")
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

        All paths are validated at call time: a configuration that empties
        any train set raises before any path is produced.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target (ignored).
        groups : array-like, optional
            Group labels (ignored).

        Returns
        -------
        Iterator[tuple[np.ndarray, np.ndarray]]
            ``(train, test)`` index pairs for all C(n_splits, n_test_splits)
            paths; train indices are post-purging.

        Raises
        ------
        ValueError
            If ``n_samples < n_splits`` (some groups would be empty) or
            purge/embargo removal empties any path's train set. Raised at
            call time, before any path is produced.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        if n_samples < self.n_splits:
            raise ValueError(
                f"CombinatorialPurgedCV cannot divide n_samples={n_samples} "
                f"into n_splits={self.n_splits} groups: some groups would be "
                f"empty. Reduce n_splits or provide more samples."
            )

        # Divide into groups
        indices = np.arange(n_samples)
        group_indices = np.array_split(indices, self.n_splits)

        # Generate all combinations of test groups
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for path_idx, test_groups in enumerate(
            combinations(range(self.n_splits), self.n_test_splits)
        ):
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

            if len(purged_train) == 0:
                raise ValueError(
                    f"CombinatorialPurgedCV path {path_idx} (test groups "
                    f"{test_groups}): purge/embargo removal emptied the train "
                    f"set for n_samples={n_samples} (n_splits={self.n_splits}, "
                    f"n_test_splits={self.n_test_splits}, "
                    f"purge_gap={self.purge_gap}, "
                    f"embargo_pct={self.embargo_pct}). Reduce "
                    f"purge_gap/embargo_pct/n_test_splits or provide more "
                    f"samples."
                )

            splits.append((purged_train, test_indices))

        return iter(splits)

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return number of combinatorial paths.

        ``split`` either yields exactly this many paths or raises
        ``ValueError`` on an over-aggressive configuration, so the nominal
        count is always truthful.
        """
        from math import comb

        return comb(self.n_splits, self.n_test_splits)


class PurgedWalkForward(BaseCrossValidator):  # type: ignore[misc]
    """Walk-forward cross-validation with purging for overlapping labels.

    Extends standard walk-forward CV with purge_gap and embargo to handle
    financial labels that overlap.

    Parameters
    ----------
    n_splits : int
        Number of test periods.
    train_size : int | None
        Fixed training window size. If None, uses expanding window.
        The window is never silently truncated: a configuration where it
        cannot fit raises ``ValueError`` at ``split`` call time.
    test_size : int | None
        Fixed test window size. If None, auto-computed.
    purge_gap : int
        Remove training samples within this distance of test samples.
    embargo_pct : float
        Fraction of samples (``ceil``) embargoed after the test window. In this
        forward-only geometry the train window precedes the test window, so the
        one-sided embargo removes nothing (kept for API parity); widen the
        train/test separation with purge_gap/extra_gap instead.
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

    An under-provisioned configuration — any fold whose train window is
    empty geometrically, whose fixed ``train_size`` cannot fit without
    truncation, whose train set is emptied by purge/embargo removal, or
    whose auto-sized test windows have no sample budget — raises
    ``ValueError`` at ``split``/``split_detailed`` call time instead of
    silently dropping folds or truncating the window, so ``split`` always
    yields exactly ``n_splits`` folds whose geometric train window is
    exactly ``train_size`` (when fixed). In this forward-only geometry purge
    and the one-sided (after-test) embargo remove nothing — the train window
    already sits ``purge_gap + extra_gap`` before the test window — so the
    delivered train set equals the geometric window.

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

        Returns
        -------
        Iterator[tuple[np.ndarray, np.ndarray]]
            ``(train, test)`` index pairs; train indices are post-purging.

        Raises
        ------
        ValueError
            If any fold would have an empty train window (geometrically or
            after purge/embargo removal), a fixed ``train_size`` cannot fit
            without truncation, or auto test sizing has no samples to work
            with — the configuration is under-provisioned. Raised at call
            time, before any fold is produced.
        """
        return iter(
            [(detailed.train_indices, detailed.test_indices) for detailed in self.split_detailed(X)]
        )

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Return number of splits.

        ``split`` either yields exactly this many folds or raises
        ``ValueError`` on an under-provisioned configuration, so the nominal
        count is always truthful.
        """
        return self.n_splits

    def split_detailed(
        self,
        X: ArrayLike,
    ) -> Iterator[PurgedSplit]:
        """Generate detailed split information.

        All fold geometries are validated at call time: an under-provisioned
        configuration raises before any fold is produced, so a consumer never
        does partial work on a doomed iteration.

        Parameters
        ----------
        X : array-like
            Training data.

        Returns
        -------
        Iterator[PurgedSplit]
            Detailed splits with purge/embargo counts.

        Raises
        ------
        ValueError
            If any fold would have an empty train window (geometrically or
            after purge/embargo removal), a fixed ``train_size`` cannot fit
            without truncation, or auto test sizing has no samples to work
            with for this ``n_samples``.
        """
        X_arr = np.asarray(X)
        n_samples = len(X_arr)

        test_size = self.test_size
        if test_size is None:
            min_train = self.train_size or (n_samples // (self.n_splits + 1))
            available = n_samples - min_train - self.extra_gap
            if available <= 0:
                raise ValueError(
                    f"PurgedWalkForward cannot auto-size test windows: "
                    f"n_samples={n_samples} leaves available={available} "
                    f"samples after reserving the train window and extra_gap "
                    f"(n_splits={self.n_splits}, train_size={self.train_size}, "
                    f"min_train={min_train}, extra_gap={self.extra_gap}). "
                    f"Reduce train_size (if set) or extra_gap, increase "
                    f"n_splits (expanding window reserves n_samples / "
                    f"(n_splits + 1) for training), or provide more samples."
                )
            # 0 < available < n_splits still floors to 0; keep at least one
            # test sample per fold and let the per-fold geometry guards
            # below decide whether the configuration actually fits.
            test_size = max(1, available // self.n_splits)

        total_gap = self.extra_gap + self.purge_gap
        splits: list[PurgedSplit] = []
        for split_idx in range(self.n_splits):
            test_end = n_samples - (self.n_splits - split_idx - 1) * test_size
            test_start = test_end - test_size

            if self.train_size is not None:
                train_end = test_start - total_gap
                train_start = train_end - self.train_size
                if train_start < 0:
                    raise ValueError(
                        f"PurgedWalkForward fold {split_idx}: fixed "
                        f"train_size={self.train_size} does not fit — only "
                        f"{max(train_end, 0)} samples of history precede the "
                        f"train/test gap for n_samples={n_samples} "
                        f"(n_splits={self.n_splits}, test_size={test_size}, "
                        f"purge_gap={self.purge_gap}, "
                        f"extra_gap={self.extra_gap}). Use train_size=None "
                        f"for an expanding window, reduce "
                        f"train_size/n_splits/test_size/gaps, or provide "
                        f"more samples."
                    )
            else:
                train_start = 0
                train_end = test_start - total_gap

            # Only reachable in expanding-window mode: with a fixed
            # train_size the truncation guard above already ensures
            # train_end - train_start == train_size >= 1.
            if train_end <= train_start:
                raise ValueError(
                    f"PurgedWalkForward fold {split_idx} has an empty train window "
                    f"[{train_start}, {train_end}) for n_samples={n_samples} "
                    f"(n_splits={self.n_splits}, train_size={self.train_size}, "
                    f"test_size={test_size}, purge_gap={self.purge_gap}, "
                    f"extra_gap={self.extra_gap}). Reduce n_splits/test_size/gaps "
                    f"or provide more samples."
                )

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            purged_train, n_purged, n_embargoed = _apply_purge_and_embargo(
                train_indices,
                test_indices,
                n_samples,
                self.purge_gap,
                self.embargo_pct,
            )

            if len(purged_train) == 0:
                # Defense-in-depth: in this forward-only geometry the train
                # window sits total_gap before the test window, and the embargo
                # is one-sided (after each test run), so neither purge nor
                # embargo removes a train row — an empty result would mean the
                # geometric window itself was empty (already guarded above).
                # Kept generic in case the geometry ever changes.
                raise ValueError(
                    f"PurgedWalkForward fold {split_idx}: train window "
                    f"[{train_start}, {train_end}) is unexpectedly empty after "
                    f"purge/embargo (geometry invariant violated) for "
                    f"n_samples={n_samples} (n_splits={self.n_splits}, "
                    f"test_size={test_size}, purge_gap={self.purge_gap}, "
                    f"embargo_pct={self.embargo_pct}, extra_gap={self.extra_gap}). "
                    f"Reduce n_splits/test_size or provide more samples."
                )

            splits.append(
                PurgedSplit(
                    train_indices=purged_train,
                    test_indices=test_indices,
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
            )

        return iter(splits)
