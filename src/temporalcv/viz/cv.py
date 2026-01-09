"""
Cross-validation fold visualization displays.

This module provides sklearn-style Display classes for visualizing
cross-validation fold structures, including gap enforcement for time series.

Examples
--------
>>> from temporalcv import WalkForwardCV
>>> from temporalcv.viz import CVFoldsDisplay
>>>
>>> cv = WalkForwardCV(n_splits=5, test_size=20, extra_gap=1)
>>> display = CVFoldsDisplay.from_cv(cv, X, y)
>>> display.plot()
>>> plt.show()
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from ._base import BaseDisplay
from ._style import (
    COLORS,
    TUFTE_PALETTE,
    apply_tufte_style,
    direct_label,
    set_tufte_labels,
    set_tufte_title,
)

__all__ = ["CVFoldsDisplay"]


class CVFoldsDisplay(BaseDisplay):
    """
    Visualization of cross-validation fold structure.

    Displays train/test splits as horizontal bars, with optional gap
    visualization for time series cross-validation.

    Parameters
    ----------
    train_indices : list of array-like
        Training indices for each fold.
    test_indices : list of array-like
        Test indices for each fold.
    gap_indices : list of array-like, optional
        Gap indices for each fold (for walk-forward CV with gap).
    n_samples : int, optional
        Total number of samples. Inferred from indices if not provided.

    Attributes
    ----------
    ax_ : matplotlib.axes.Axes
        The axes used for plotting.
    figure_ : matplotlib.figure.Figure
        The figure containing the plot.

    See Also
    --------
    temporalcv.WalkForwardCV : Walk-forward cross-validator with gap.
    temporalcv.cv_financial.PurgedKFold : Purged K-Fold for finance.

    Examples
    --------
    >>> from temporalcv import WalkForwardCV
    >>> from temporalcv.viz import CVFoldsDisplay
    >>> import numpy as np
    >>>
    >>> X = np.random.randn(200, 5)
    >>> y = np.random.randn(200)
    >>> cv = WalkForwardCV(n_splits=5, test_size=20, extra_gap=1)
    >>>
    >>> # From cross-validator
    >>> display = CVFoldsDisplay.from_cv(cv, X, y)
    >>> display.plot()
    >>>
    >>> # Or from pre-computed splits
    >>> splits = list(cv.split(X, y))
    >>> display = CVFoldsDisplay.from_splits(splits)
    >>> display.plot()
    """

    def __init__(
        self,
        train_indices: List[np.ndarray],
        test_indices: List[np.ndarray],
        *,
        gap_indices: Optional[List[np.ndarray]] = None,
        n_samples: Optional[int] = None,
    ):
        self.train_indices = [np.asarray(t) for t in train_indices]
        self.test_indices = [np.asarray(t) for t in test_indices]
        self.gap_indices = (
            [np.asarray(g) for g in gap_indices] if gap_indices else None
        )

        # Infer n_samples
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            all_indices = np.concatenate(
                self.train_indices + self.test_indices
            )
            self.n_samples = int(np.max(all_indices)) + 1

        self.n_splits = len(self.train_indices)

    @classmethod
    def from_cv(
        cls,
        cv: Any,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        groups: Optional[np.ndarray] = None,
    ) -> "CVFoldsDisplay":
        """
        Create display from a cross-validator object.

        Parameters
        ----------
        cv : cross-validator
            A scikit-learn compatible cross-validator with split() method.
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        groups : array-like of shape (n_samples,), optional
            Group labels for GroupKFold-like splitters.

        Returns
        -------
        CVFoldsDisplay
            The display object.

        Examples
        --------
        >>> from temporalcv import WalkForwardCV
        >>> cv = WalkForwardCV(n_splits=5, test_size=20)
        >>> display = CVFoldsDisplay.from_cv(cv, X, y)
        """
        trains = []
        tests = []
        gaps = []

        for train, test in cv.split(X, y, groups):
            trains.append(train)
            tests.append(test)

            # Detect gap (indices between train end and test start)
            if len(train) > 0 and len(test) > 0:
                gap_start = train[-1] + 1
                gap_end = test[0]
                if gap_end > gap_start:
                    gaps.append(np.arange(gap_start, gap_end))
                else:
                    gaps.append(np.array([]))
            else:
                gaps.append(np.array([]))

        # Check if any gaps exist
        has_gaps = any(len(g) > 0 for g in gaps)

        return cls(
            trains,
            tests,
            gap_indices=gaps if has_gaps else None,
            n_samples=len(X),
        )

    @classmethod
    def from_splits(
        cls,
        splits: List[Tuple[np.ndarray, np.ndarray]],
        *,
        n_samples: Optional[int] = None,
    ) -> "CVFoldsDisplay":
        """
        Create display from pre-computed splits.

        Parameters
        ----------
        splits : list of (train_indices, test_indices) tuples
            Pre-computed splits from cv.split().
        n_samples : int, optional
            Total number of samples.

        Returns
        -------
        CVFoldsDisplay
            The display object.

        Examples
        --------
        >>> splits = list(cv.split(X, y))
        >>> display = CVFoldsDisplay.from_splits(splits, n_samples=len(X))
        """
        trains = [s[0] for s in splits]
        tests = [s[1] for s in splits]

        gaps = []
        for train, test in splits:
            if len(train) > 0 and len(test) > 0:
                gap_start = int(train[-1] + 1)
                gap_end = test[0]
                if gap_end > gap_start:
                    gaps.append(np.arange(gap_start, gap_end))
                else:
                    gaps.append(np.array([]))
            else:
                gaps.append(np.array([]))

        has_gaps = any(len(g) > 0 for g in gaps)

        return cls(
            trains,
            tests,
            gap_indices=gaps if has_gaps else None,
            n_samples=n_samples,
        )

    def plot(
        self,
        *,
        ax: Optional[Axes] = None,
        tufte: bool = True,
        bar_height: float = 0.6,
        show_labels: bool = True,
        title: Optional[str] = None,
    ) -> "CVFoldsDisplay":
        """
        Plot the cross-validation fold structure.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        tufte : bool
            If True, apply Tufte styling (default).
        bar_height : float
            Height of each fold bar (0-1).
        show_labels : bool
            If True, show fold labels and sample counts.
        title : str, optional
            Plot title. If None, uses default.

        Returns
        -------
        self
            The display object for method chaining.

        Examples
        --------
        >>> display.plot(title="Walk-Forward CV Folds")
        >>> plt.tight_layout()
        >>> plt.show()
        """
        ax = self._get_ax_or_create(ax, figsize=(10, max(3, self.n_splits * 0.8)))

        if tufte:
            apply_tufte_style(ax)

        # Plot each fold
        for fold_idx in range(self.n_splits):
            y_pos = self.n_splits - 1 - fold_idx  # Reverse so fold 1 is on top

            train = self.train_indices[fold_idx]
            test = self.test_indices[fold_idx]

            # Training set
            if len(train) > 0:
                ax.barh(
                    y_pos,
                    len(train),
                    left=train[0],
                    height=bar_height,
                    color=COLORS["train"],
                    alpha=0.85,
                    edgecolor="none",
                    label="Train" if fold_idx == 0 else None,
                )

            # Gap (if exists)
            if self.gap_indices is not None and len(self.gap_indices[fold_idx]) > 0:
                gap = self.gap_indices[fold_idx]
                ax.barh(
                    y_pos,
                    len(gap),
                    left=gap[0],
                    height=bar_height,
                    color=COLORS["gap"],
                    alpha=0.5,
                    edgecolor="none",
                    label="Gap" if fold_idx == 0 else None,
                )

            # Test set
            if len(test) > 0:
                ax.barh(
                    y_pos,
                    len(test),
                    left=test[0],
                    height=bar_height,
                    color=COLORS["test"],
                    alpha=0.85,
                    edgecolor="none",
                    label="Test" if fold_idx == 0 else None,
                )

            # Direct labels (Tufte principle)
            if show_labels:
                # Fold label on left
                direct_label(
                    ax,
                    -5,
                    y_pos,
                    f"Fold {fold_idx + 1}",
                    offset=(0, 0),
                    ha="right",
                    va="center",
                    fontsize=9,
                )

                # Sample counts (right of test bar)
                if len(test) > 0:
                    direct_label(
                        ax,
                        test[-1] + 2,
                        y_pos,
                        f"n={len(train)}/{len(test)}",
                        offset=(0, 0),
                        ha="left",
                        va="center",
                        fontsize=8,
                        color=TUFTE_PALETTE["text_secondary"],
                    )

        # Styling
        ax.set_xlim(-self.n_samples * 0.15, self.n_samples * 1.1)
        ax.set_ylim(-0.5, self.n_splits - 0.5)
        ax.set_yticks([])

        set_tufte_labels(ax, xlabel="Sample Index")

        if title is None:
            gap_text = " (with gap)" if self.gap_indices is not None else ""
            title = f"Cross-Validation Folds{gap_text}"
        set_tufte_title(ax, title)

        # Minimal legend (bottom right, unobtrusive)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="lower right",
                frameon=False,
                fontsize=8,
                ncol=len(handles),
            )

        self._finalize_plot(ax)
        return self
