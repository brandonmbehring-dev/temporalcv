"""
Prediction interval visualization displays.

This module provides sklearn-style Display classes for visualizing
prediction intervals from conformal prediction.

Examples
--------
>>> from temporalcv.conformal import SplitConformalPredictor
>>> from temporalcv.viz import PredictionIntervalDisplay
>>>
>>> conformal = SplitConformalPredictor(alpha=0.10)
>>> conformal.calibrate(cal_preds, cal_actuals)
>>> intervals = conformal.predict_interval(test_preds)
>>> display = PredictionIntervalDisplay.from_conformal(intervals, test_actuals)
>>> display.plot()
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from matplotlib.axes import Axes

from ._base import BaseDisplay
from ._style import (
    COLORS,
    apply_tufte_style,
    direct_label,
    set_tufte_labels,
    set_tufte_title,
)

__all__ = ["PredictionIntervalDisplay"]


class PredictionIntervalDisplay(BaseDisplay):
    """
    Visualization of prediction intervals with actuals.

    Displays prediction intervals as a shaded region with actual values
    overlaid, highlighting coverage.

    Parameters
    ----------
    predictions : array-like
        Point predictions.
    lower : array-like
        Lower bounds of intervals.
    upper : array-like
        Upper bounds of intervals.
    actuals : array-like, optional
        Actual values for coverage visualization.
    confidence : float
        Confidence level (e.g., 0.90 for 90% intervals).

    Attributes
    ----------
    ax_ : matplotlib.axes.Axes
        The axes used for plotting.
    figure_ : matplotlib.figure.Figure
        The figure containing the plot.
    coverage_ : float or None
        Empirical coverage if actuals provided, None otherwise.

    See Also
    --------
    temporalcv.conformal.SplitConformalPredictor : Split conformal.
    temporalcv.conformal.AdaptiveConformalPredictor : Adaptive conformal.

    Examples
    --------
    >>> from temporalcv.conformal import SplitConformalPredictor
    >>> from temporalcv.viz import PredictionIntervalDisplay
    >>>
    >>> conformal = SplitConformalPredictor(alpha=0.10)
    >>> conformal.calibrate(cal_preds, cal_actuals)
    >>> intervals = conformal.predict_interval(test_preds)
    >>>
    >>> display = PredictionIntervalDisplay.from_conformal(intervals, test_actuals)
    >>> display.plot()
    """

    coverage_: Optional[float]
    _covered: Optional[np.ndarray]

    def __init__(
        self,
        predictions: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        actuals: Optional[np.ndarray] = None,
        confidence: float = 0.90,
        x: Optional[np.ndarray] = None,
    ):
        self.predictions = np.asarray(predictions)
        self.lower = np.asarray(lower)
        self.upper = np.asarray(upper)
        self.actuals = np.asarray(actuals) if actuals is not None else None
        self.confidence = confidence
        self.x = x if x is not None else np.arange(len(predictions))

        # Compute coverage if actuals provided
        if self.actuals is not None:
            covered = (self.actuals >= self.lower) & (self.actuals <= self.upper)
            self.coverage_ = float(np.mean(covered))
            self._covered = covered
        else:
            self.coverage_ = None
            self._covered = None

    @classmethod
    def from_conformal(
        cls,
        intervals: Any,
        actuals: Optional[np.ndarray] = None,
        *,
        x: Optional[np.ndarray] = None,
    ) -> PredictionIntervalDisplay:
        """
        Create display from a PredictionInterval object.

        Parameters
        ----------
        intervals : PredictionInterval
            Prediction interval object from conformal predictor.
        actuals : array-like, optional
            Actual values for coverage visualization.
        x : array-like, optional
            X-axis values (e.g., time indices).

        Returns
        -------
        PredictionIntervalDisplay
            The display object.

        Examples
        --------
        >>> intervals = conformal.predict_interval(test_preds)
        >>> display = PredictionIntervalDisplay.from_conformal(intervals, test_actuals)
        """
        return cls(
            predictions=intervals.point,
            lower=intervals.lower,
            upper=intervals.upper,
            actuals=actuals,
            confidence=intervals.confidence,
            x=x,
        )

    @classmethod
    def from_predictions(
        cls,
        predictions: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        actuals: Optional[np.ndarray] = None,
        confidence: float = 0.90,
        x: Optional[np.ndarray] = None,
    ) -> PredictionIntervalDisplay:
        """
        Create display from arrays.

        Parameters
        ----------
        predictions : array-like
            Point predictions.
        lower : array-like
            Lower bounds.
        upper : array-like
            Upper bounds.
        actuals : array-like, optional
            Actual values.
        confidence : float
            Confidence level.
        x : array-like, optional
            X-axis values.

        Returns
        -------
        PredictionIntervalDisplay
            The display object.
        """
        return cls(
            predictions=predictions,
            lower=lower,
            upper=upper,
            actuals=actuals,
            confidence=confidence,
            x=x,
        )

    def plot(
        self,
        *,
        ax: Optional[Axes] = None,
        tufte: bool = True,
        show_predictions: bool = True,
        show_actuals: bool = True,
        show_coverage: bool = True,
        title: Optional[str] = None,
    ) -> PredictionIntervalDisplay:
        """
        Plot the prediction intervals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        tufte : bool
            If True, apply Tufte styling (default).
        show_predictions : bool
            If True, show point predictions as line.
        show_actuals : bool
            If True, show actual values.
        show_coverage : bool
            If True, highlight covered/uncovered points.
        title : str, optional
            Plot title.

        Returns
        -------
        self
            The display object for method chaining.
        """
        ax = self._get_ax_or_create(ax, figsize=(10, 5))

        if tufte:
            apply_tufte_style(ax)

        # Prediction interval band
        ax.fill_between(
            self.x,
            self.lower,
            self.upper,
            alpha=0.25,
            color=COLORS["interval"],
            linewidth=0,
            label=f"{self.confidence:.0%} Interval",
        )

        # Point predictions
        if show_predictions:
            ax.plot(
                self.x,
                self.predictions,
                color=COLORS["prediction"],
                linewidth=1.5,
                label="Predictions",
            )

        # Actual values
        if show_actuals and self.actuals is not None:
            if show_coverage and self._covered is not None:
                # Covered points (green)
                covered_mask = self._covered
                ax.scatter(
                    self.x[covered_mask],
                    self.actuals[covered_mask],
                    color=COLORS["pass"],
                    s=25,
                    zorder=5,
                    label=f"Covered ({self.coverage_:.1%})",
                )
                # Uncovered points (red, larger)
                uncovered_mask = ~covered_mask
                if np.any(uncovered_mask):
                    ax.scatter(
                        self.x[uncovered_mask],
                        self.actuals[uncovered_mask],
                        color=COLORS["halt"],
                        s=40,
                        marker="x",
                        zorder=5,
                        linewidths=1.5,
                        label="Not Covered",
                    )
            else:
                ax.scatter(
                    self.x,
                    self.actuals,
                    color=COLORS["actual"],
                    s=20,
                    zorder=5,
                    label="Actuals",
                )

        # Labels
        set_tufte_labels(ax, xlabel="Index", ylabel="Value")

        # Title
        if title is None:
            coverage_text = f", {self.coverage_:.1%} coverage" if self.coverage_ else ""
            title = f"Prediction Intervals ({self.confidence:.0%} target{coverage_text})"
        set_tufte_title(ax, title)

        # Legend (Tufte-style: minimal, unobtrusive)
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=8,
        )

        self._finalize_plot(ax)
        return self

    def plot_width(
        self,
        *,
        ax: Optional[Axes] = None,
        tufte: bool = True,
        title: Optional[str] = None,
    ) -> PredictionIntervalDisplay:
        """
        Plot the interval widths over time.

        Useful for adaptive conformal where width varies.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        tufte : bool
            If True, apply Tufte styling.
        title : str, optional
            Plot title.

        Returns
        -------
        self
            The display object.
        """
        ax = self._get_ax_or_create(ax, figsize=(10, 3))

        if tufte:
            apply_tufte_style(ax)

        widths = self.upper - self.lower

        # Bar chart of widths
        ax.bar(
            self.x,
            widths,
            color=COLORS["interval"],
            alpha=0.7,
            edgecolor="none",
            width=1.0,
        )

        # Mean width line
        mean_width = float(np.mean(widths))
        ax.axhline(
            mean_width,
            color=COLORS["halt"],
            linestyle="--",
            linewidth=1.5,
        )

        # Direct label for mean
        direct_label(
            ax,
            float(self.x[-1]),
            mean_width,
            f"Mean: {mean_width:.3f}",
            offset=(5, 3),
            color=COLORS["halt"],
        )

        set_tufte_labels(ax, xlabel="Index", ylabel="Interval Width")

        if title is None:
            title = "Prediction Interval Width"
        set_tufte_title(ax, title)

        self._finalize_plot(ax)
        return self
