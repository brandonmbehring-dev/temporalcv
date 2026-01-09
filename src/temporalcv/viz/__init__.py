"""
Visualization module for temporalcv.

This module provides sklearn-style Display classes and statsmodels-style
plotting functions for visualizing cross-validation results, gate outcomes,
and prediction intervals.

All visualizations follow Edward Tufte's principles:
- Maximize data-ink ratio
- Eliminate chartjunk
- Use direct labeling instead of legends where practical
- Employ small multiples for comparison

Display Classes (sklearn-style)
-------------------------------
CVFoldsDisplay
    Cross-validation fold structure visualization.
GateResultDisplay
    Single gate result visualization (HALT/WARN/PASS).
GateComparisonDisplay
    Multiple gate comparison visualization.
PredictionIntervalDisplay
    Conformal prediction interval visualization.
MetricComparisonDisplay
    Metric comparison bar charts.

Functions (statsmodels-style)
-----------------------------
plot_cv_folds
    Plot cross-validation fold structure.
plot_gate_result
    Plot a single gate result.
plot_gate_comparison
    Plot comparison of multiple gates.
plot_prediction_intervals
    Plot prediction intervals with coverage.
plot_interval_width
    Plot interval width over time.
plot_metric_comparison
    Plot metric comparison across models.

Styling
-------
apply_tufte_style
    Apply Tufte's principles to any matplotlib axes.
TUFTE_PALETTE
    Muted color palette following Tufte's recommendations.
COLORS
    Semantic color aliases for common use cases.

Examples
--------
>>> # sklearn-style (Display classes)
>>> from temporalcv.viz import CVFoldsDisplay
>>> display = CVFoldsDisplay.from_cv(cv, X, y)
>>> display.plot()
>>>
>>> # statsmodels-style (functions)
>>> from temporalcv.viz import plot_cv_folds
>>> plot_cv_folds(cv, X)
>>> plt.show()
>>>
>>> # Custom styling
>>> from temporalcv.viz import apply_tufte_style
>>> fig, ax = plt.subplots()
>>> ax.plot(x, y)
>>> apply_tufte_style(ax)

References
----------
- Tufte, E. R. (1983). The Visual Display of Quantitative Information.
- Tufte, E. R. (2001). Envisioning Information.
- scikit-learn Visualization API: https://scikit-learn.org/stable/visualizations.html
"""

from ._style import (
    COLORS,
    TUFTE_PALETTE,
    apply_tufte_style,
    create_tufte_figure,
    direct_label,
    minimal_spines,
    muted_color,
    range_frame,
    set_tufte_labels,
    set_tufte_title,
)
from .comparison import MetricComparisonDisplay
from .cv import CVFoldsDisplay
from .gateplots import plot_gate_comparison, plot_gate_result
from .gates import GateComparisonDisplay, GateResultDisplay
from .intervals import PredictionIntervalDisplay
from .tsplots import (
    plot_cv_folds,
    plot_interval_width,
    plot_metric_comparison,
    plot_prediction_intervals,
)

__all__ = [
    # Display classes (sklearn-style)
    "CVFoldsDisplay",
    "GateResultDisplay",
    "GateComparisonDisplay",
    "PredictionIntervalDisplay",
    "MetricComparisonDisplay",
    # Functions (statsmodels-style)
    "plot_cv_folds",
    "plot_gate_result",
    "plot_gate_comparison",
    "plot_prediction_intervals",
    "plot_interval_width",
    "plot_metric_comparison",
    # Styling primitives
    "apply_tufte_style",
    "TUFTE_PALETTE",
    "COLORS",
    "direct_label",
    "minimal_spines",
    "range_frame",
    "create_tufte_figure",
    "muted_color",
    "set_tufte_labels",
    "set_tufte_title",
]
