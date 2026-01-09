"""
Metric comparison visualization displays.

This module provides sklearn-style Display classes for visualizing
metric comparisons between models or methods.

Examples
--------
>>> from temporalcv.viz import MetricComparisonDisplay
>>>
>>> results = {
...     "Model A": {"MAE": 0.15, "RMSE": 0.22},
...     "Model B": {"MAE": 0.12, "RMSE": 0.19},
... }
>>> display = MetricComparisonDisplay.from_dict(results)
>>> display.plot()
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from ._base import BaseDisplay
from ._style import (
    COLORS,
    TUFTE_PALETTE,
    apply_tufte_style,
    set_tufte_labels,
    set_tufte_title,
)

__all__ = ["MetricComparisonDisplay"]


class MetricComparisonDisplay(BaseDisplay):
    """
    Visualization comparing metrics across models.

    Displays metric comparisons as grouped bar charts with Tufte styling.

    Parameters
    ----------
    model_names : list of str
        Names of models being compared.
    metric_names : list of str
        Names of metrics being compared.
    values : array-like of shape (n_models, n_metrics)
        Metric values for each model.
    lower_is_better : dict, optional
        Map of metric name to bool indicating if lower is better.
        Default: True for all metrics.

    Attributes
    ----------
    ax_ : matplotlib.axes.Axes
        The axes used for plotting.
    figure_ : matplotlib.figure.Figure
        The figure containing the plot.

    See Also
    --------
    temporalcv.compare.compare_horizons : Compare across horizons.
    temporalcv.compare.compare_models : Compare model performance.

    Examples
    --------
    >>> from temporalcv.viz import MetricComparisonDisplay
    >>>
    >>> # From dictionary
    >>> results = {"Model A": {"MAE": 0.15}, "Model B": {"MAE": 0.12}}
    >>> display = MetricComparisonDisplay.from_dict(results)
    >>> display.plot()
    >>>
    >>> # From arrays
    >>> display = MetricComparisonDisplay.from_arrays(
    ...     model_names=["A", "B"],
    ...     metric_names=["MAE", "RMSE"],
    ...     values=[[0.15, 0.22], [0.12, 0.19]],
    ... )
    >>> display.plot()
    """

    def __init__(
        self,
        model_names: list[str],
        metric_names: list[str],
        values: np.ndarray,
        *,
        lower_is_better: dict[str, bool] | None = None,
        baseline_idx: int | None = None,
    ):
        self.model_names = list(model_names)
        self.metric_names = list(metric_names)
        self.values = np.asarray(values)
        self.lower_is_better = lower_is_better or dict.fromkeys(metric_names, True)
        self.baseline_idx = baseline_idx

        self.n_models = len(model_names)
        self.n_metrics = len(metric_names)

    @classmethod
    def from_dict(
        cls,
        results: dict[str, dict[str, float]],
        *,
        lower_is_better: dict[str, bool] | None = None,
        baseline: str | None = None,
    ) -> MetricComparisonDisplay:
        """
        Create display from a nested dictionary.

        Parameters
        ----------
        results : dict
            Nested dict of {model_name: {metric_name: value}}.
        lower_is_better : dict, optional
            Map of metric name to bool.
        baseline : str, optional
            Name of baseline model for relative comparison.

        Returns
        -------
        MetricComparisonDisplay
            The display object.

        Examples
        --------
        >>> results = {
        ...     "Baseline": {"MAE": 0.20, "RMSE": 0.28},
        ...     "Model A": {"MAE": 0.15, "RMSE": 0.22},
        ... }
        >>> display = MetricComparisonDisplay.from_dict(results, baseline="Baseline")
        """
        model_names = list(results.keys())
        metric_names = list(next(iter(results.values())).keys())

        values = np.array(
            [[results[m].get(metric, np.nan) for metric in metric_names] for m in model_names]
        )

        baseline_idx = None
        if baseline is not None and baseline in model_names:
            baseline_idx = model_names.index(baseline)

        return cls(
            model_names,
            metric_names,
            values,
            lower_is_better=lower_is_better,
            baseline_idx=baseline_idx,
        )

    @classmethod
    def from_arrays(
        cls,
        model_names: list[str],
        metric_names: list[str],
        values: np.ndarray,
        *,
        lower_is_better: dict[str, bool] | None = None,
        baseline_idx: int | None = None,
    ) -> MetricComparisonDisplay:
        """
        Create display from arrays.

        Parameters
        ----------
        model_names : list of str
            Names of models.
        metric_names : list of str
            Names of metrics.
        values : array-like of shape (n_models, n_metrics)
            Metric values.
        lower_is_better : dict, optional
            Map of metric name to bool.
        baseline_idx : int, optional
            Index of baseline model.

        Returns
        -------
        MetricComparisonDisplay
            The display object.
        """
        return cls(
            model_names,
            metric_names,
            values,
            lower_is_better=lower_is_better,
            baseline_idx=baseline_idx,
        )

    def plot(
        self,
        *,
        ax: Axes | None = None,
        tufte: bool = True,
        orientation: str = "vertical",
        show_values: bool = True,
        show_best: bool = True,
        title: str | None = None,
        metric_idx: int | None = None,
    ) -> MetricComparisonDisplay:
        """
        Plot the metric comparison.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        tufte : bool
            If True, apply Tufte styling (default).
        orientation : str
            "vertical" (bars go up) or "horizontal" (bars go right).
        show_values : bool
            If True, show metric values on bars.
        show_best : bool
            If True, highlight best model for each metric.
        title : str, optional
            Plot title.
        metric_idx : int, optional
            If provided, plot only this metric (useful for single-metric comparison).

        Returns
        -------
        self
            The display object for method chaining.
        """
        # Determine if plotting single metric or multiple
        if metric_idx is not None:
            return self._plot_single_metric(
                ax=ax,
                tufte=tufte,
                orientation=orientation,
                show_values=show_values,
                show_best=show_best,
                title=title,
                metric_idx=metric_idx,
            )
        elif self.n_metrics == 1:
            return self._plot_single_metric(
                ax=ax,
                tufte=tufte,
                orientation=orientation,
                show_values=show_values,
                show_best=show_best,
                title=title,
                metric_idx=0,
            )
        else:
            return self._plot_grouped(
                ax=ax,
                tufte=tufte,
                show_values=show_values,
                show_best=show_best,
                title=title,
            )

    def _plot_single_metric(
        self,
        *,
        ax: Axes | None,
        tufte: bool,
        orientation: str,
        show_values: bool,
        show_best: bool,
        title: str | None,
        metric_idx: int,
    ) -> MetricComparisonDisplay:
        """Plot comparison for a single metric."""
        ax = self._get_ax_or_create(ax, figsize=(8, max(3, self.n_models * 0.6)))

        if tufte:
            apply_tufte_style(ax)

        values = self.values[:, metric_idx]
        metric_name = self.metric_names[metric_idx]
        lower_better = self.lower_is_better.get(metric_name, True)

        # Determine best model
        best_idx = np.nanargmin(values) if lower_better else np.nanargmax(values)

        # Colors: best in green, others in muted blue
        colors = [
            COLORS["pass"] if i == best_idx and show_best else TUFTE_PALETTE["info"]
            for i in range(self.n_models)
        ]

        # Baseline highlighting
        if self.baseline_idx is not None and self.baseline_idx != best_idx:
            colors[self.baseline_idx] = TUFTE_PALETTE["secondary"]

        positions = np.arange(self.n_models)

        if orientation == "horizontal":
            ax.barh(
                positions,
                values,
                color=colors,
                alpha=0.85,
                edgecolor="none",
                height=0.6,
            )

            # Value labels
            if show_values:
                for _i, (pos, val) in enumerate(zip(positions, values)):
                    ha = "left" if val >= 0 else "right"
                    offset = val * 0.02 if val >= 0 else val * 0.02
                    ax.text(
                        val + offset,
                        pos,
                        f"{val:.3f}",
                        va="center",
                        ha=ha,
                        fontsize=9,
                        color=TUFTE_PALETTE["text"],
                    )

            ax.set_yticks(positions)
            ax.set_yticklabels(self.model_names)
            set_tufte_labels(ax, xlabel=metric_name)

        else:  # vertical
            ax.bar(
                positions,
                values,
                color=colors,
                alpha=0.85,
                edgecolor="none",
                width=0.6,
            )

            # Value labels
            if show_values:
                for _i, (pos, val) in enumerate(zip(positions, values)):
                    ax.text(
                        pos,
                        val + max(values) * 0.02,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        color=TUFTE_PALETTE["text"],
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels(self.model_names, rotation=0)
            set_tufte_labels(ax, ylabel=metric_name)

        if title is None:
            title = f"{metric_name} Comparison"
        set_tufte_title(ax, title)

        self._finalize_plot(ax)
        return self

    def _plot_grouped(
        self,
        *,
        ax: Axes | None,
        tufte: bool,
        show_values: bool,  # noqa: ARG002
        show_best: bool,
        title: str | None,
    ) -> MetricComparisonDisplay:
        """Plot grouped bar chart for multiple metrics."""
        figsize = (max(8, self.n_models * 2), 5)
        ax = self._get_ax_or_create(ax, figsize=figsize)

        if tufte:
            apply_tufte_style(ax)

        # Bar positioning
        bar_width = 0.8 / self.n_metrics
        positions = np.arange(self.n_models)

        # Color palette for metrics (cycle through Tufte colors)
        metric_colors = [
            TUFTE_PALETTE["info"],
            TUFTE_PALETTE["warning"],
            TUFTE_PALETTE["success"],
            TUFTE_PALETTE["accent"],
        ]

        for m_idx, metric in enumerate(self.metric_names):
            offset = (m_idx - self.n_metrics / 2 + 0.5) * bar_width
            values = self.values[:, m_idx]
            color = metric_colors[m_idx % len(metric_colors)]

            ax.bar(
                positions + offset,
                values,
                bar_width * 0.9,
                label=metric,
                color=color,
                alpha=0.85,
                edgecolor="none",
            )

            # Highlight best for each metric
            if show_best:
                lower_better = self.lower_is_better.get(metric, True)
                best_idx = np.nanargmin(values) if lower_better else np.nanargmax(values)
                # Add subtle marker for best
                ax.scatter(
                    positions[best_idx] + offset,
                    values[best_idx] + max(self.values.flat) * 0.03,
                    marker="v",
                    color=COLORS["pass"],
                    s=30,
                    zorder=5,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels(self.model_names)
        set_tufte_labels(ax, ylabel="Metric Value")

        if title is None:
            title = "Model Comparison"
        set_tufte_title(ax, title)

        # Minimal legend
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=9,
        )

        self._finalize_plot(ax)
        return self

    def plot_relative(
        self,
        *,
        ax: Axes | None = None,
        tufte: bool = True,
        title: str | None = None,
    ) -> MetricComparisonDisplay:
        """
        Plot metrics relative to baseline (percent improvement).

        Requires baseline_idx to be set.

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
        if self.baseline_idx is None:
            raise ValueError("baseline_idx must be set for relative comparison")

        ax = self._get_ax_or_create(ax, figsize=(8, max(3, self.n_models * 0.6)))

        if tufte:
            apply_tufte_style(ax)

        # Compute relative improvement
        baseline_values = self.values[self.baseline_idx]
        relative = np.zeros_like(self.values)

        for m_idx, metric in enumerate(self.metric_names):
            lower_better = self.lower_is_better.get(metric, True)
            if lower_better:
                # Improvement = (baseline - model) / baseline * 100
                relative[:, m_idx] = (baseline_values[m_idx] - self.values[:, m_idx]) / baseline_values[m_idx] * 100
            else:
                # Improvement = (model - baseline) / baseline * 100
                relative[:, m_idx] = (self.values[:, m_idx] - baseline_values[m_idx]) / baseline_values[m_idx] * 100

        # For single metric, use simple bar chart
        if self.n_metrics == 1:
            positions = np.arange(self.n_models)
            values = relative[:, 0]

            colors = [
                COLORS["pass"] if v > 0 else COLORS["halt"] if v < 0 else TUFTE_PALETTE["secondary"]
                for v in values
            ]

            ax.barh(
                positions,
                values,
                color=colors,
                alpha=0.85,
                edgecolor="none",
                height=0.6,
            )

            # Value labels
            for pos, val in zip(positions, values):
                ha = "left" if val >= 0 else "right"
                ax.text(
                    val + (1 if val >= 0 else -1),
                    pos,
                    f"{val:+.1f}%",
                    va="center",
                    ha=ha,
                    fontsize=9,
                    color=TUFTE_PALETTE["text"],
                )

            ax.axvline(0, color=TUFTE_PALETTE["spine"], linewidth=0.8, zorder=0)
            ax.set_yticks(positions)
            ax.set_yticklabels(self.model_names)
            set_tufte_labels(ax, xlabel=f"Improvement in {self.metric_names[0]} (%)")

        if title is None:
            baseline_name = self.model_names[self.baseline_idx]
            title = f"Improvement vs {baseline_name}"
        set_tufte_title(ax, title)

        self._finalize_plot(ax)
        return self
