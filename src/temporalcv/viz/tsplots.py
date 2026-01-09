"""
Time series visualization functions (statsmodels-style API).

This module provides simple function-based API for common visualizations,
following the statsmodels pattern where ax=None creates a new figure.

Examples
--------
>>> from temporalcv.viz import plot_cv_folds, plot_prediction_intervals
>>>
>>> # Simple one-liner
>>> plot_cv_folds(cv, X)
>>> plt.show()
>>>
>>> # With custom axes
>>> fig, (ax1, ax2) = plt.subplots(1, 2)
>>> plot_cv_folds(cv, X, ax=ax1)
>>> plot_prediction_intervals(intervals, actuals, ax=ax2)
>>> plt.show()
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes

from .cv import CVFoldsDisplay
from .intervals import PredictionIntervalDisplay
from .comparison import MetricComparisonDisplay

__all__ = [
    "plot_cv_folds",
    "plot_prediction_intervals",
    "plot_interval_width",
    "plot_metric_comparison",
]


def plot_cv_folds(
    cv: Any,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    groups: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    tufte: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot cross-validation fold structure.

    Convenience function wrapping CVFoldsDisplay.

    Parameters
    ----------
    cv : cross-validator
        A scikit-learn compatible cross-validator.
    X : array-like
        Training data.
    y : array-like, optional
        Target values.
    groups : array-like, optional
        Group labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    tufte : bool
        If True, apply Tufte styling (default).
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from temporalcv import WalkForwardCV
    >>> from temporalcv.viz import plot_cv_folds
    >>>
    >>> cv = WalkForwardCV(n_splits=5, test_size=20)
    >>> plot_cv_folds(cv, X, title="Walk-Forward CV")
    >>> plt.show()

    See Also
    --------
    CVFoldsDisplay : Class-based API for more customization.
    """
    display = CVFoldsDisplay.from_cv(cv, X, y, groups=groups)
    display.plot(ax=ax, tufte=tufte, title=title)
    return display.ax_


def plot_prediction_intervals(
    intervals: Any,
    actuals: Optional[np.ndarray] = None,
    *,
    x: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    tufte: bool = True,
    show_coverage: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot prediction intervals with actuals.

    Convenience function wrapping PredictionIntervalDisplay.

    Parameters
    ----------
    intervals : PredictionInterval
        Prediction interval object from conformal predictor.
    actuals : array-like, optional
        Actual values for coverage visualization.
    x : array-like, optional
        X-axis values (e.g., time indices).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    tufte : bool
        If True, apply Tufte styling.
    show_coverage : bool
        If True, highlight covered/uncovered points.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from temporalcv.conformal import SplitConformalPredictor
    >>> from temporalcv.viz import plot_prediction_intervals
    >>>
    >>> conformal = SplitConformalPredictor(alpha=0.10)
    >>> conformal.calibrate(cal_preds, cal_actuals)
    >>> intervals = conformal.predict_interval(test_preds)
    >>>
    >>> plot_prediction_intervals(intervals, test_actuals)
    >>> plt.show()

    See Also
    --------
    PredictionIntervalDisplay : Class-based API.
    """
    display = PredictionIntervalDisplay.from_conformal(intervals, actuals, x=x)
    display.plot(ax=ax, tufte=tufte, show_coverage=show_coverage, title=title)
    return display.ax_


def plot_interval_width(
    intervals: Any,
    *,
    x: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    tufte: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot prediction interval widths over time.

    Useful for adaptive conformal where width varies.

    Parameters
    ----------
    intervals : PredictionInterval
        Prediction interval object.
    x : array-like, optional
        X-axis values.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    tufte : bool
        If True, apply Tufte styling.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from temporalcv.conformal import AdaptiveConformalPredictor
    >>> from temporalcv.viz import plot_interval_width
    >>>
    >>> adaptive = AdaptiveConformalPredictor(alpha=0.10, gamma=0.01)
    >>> # ... fit and predict ...
    >>> plot_interval_width(intervals)
    >>> plt.show()

    See Also
    --------
    PredictionIntervalDisplay.plot_width : Class-based method.
    """
    display = PredictionIntervalDisplay.from_conformal(intervals, x=x)
    display.plot_width(ax=ax, tufte=tufte, title=title)
    return display.ax_


def plot_metric_comparison(
    results: dict,
    *,
    ax: Optional[Axes] = None,
    tufte: bool = True,
    baseline: Optional[str] = None,
    show_values: bool = True,
    title: Optional[str] = None,
) -> Axes:
    """
    Plot metric comparison across models.

    Parameters
    ----------
    results : dict
        Nested dict of {model_name: {metric_name: value}}.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    tufte : bool
        If True, apply Tufte styling.
    baseline : str, optional
        Name of baseline model for highlighting.
    show_values : bool
        If True, show values on bars.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from temporalcv.viz import plot_metric_comparison
    >>>
    >>> results = {
    ...     "Model A": {"MAE": 0.15, "RMSE": 0.22},
    ...     "Model B": {"MAE": 0.12, "RMSE": 0.19},
    ... }
    >>> plot_metric_comparison(results, baseline="Model A")
    >>> plt.show()

    See Also
    --------
    MetricComparisonDisplay : Class-based API.
    """
    display = MetricComparisonDisplay.from_dict(results, baseline=baseline)
    display.plot(ax=ax, tufte=tufte, show_values=show_values, title=title)
    return display.ax_
