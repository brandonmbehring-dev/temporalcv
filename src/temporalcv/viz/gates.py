"""
Gate result visualization displays.

This module provides sklearn-style Display classes for visualizing
validation gate results (HALT/WARN/PASS).

Examples
--------
>>> from temporalcv.gates import gate_signal_verification, run_gates
>>> from temporalcv.viz import GateResultDisplay
>>>
>>> result = gate_signal_verification(model, X, y)
>>> display = GateResultDisplay.from_gate(result)
>>> display.plot()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.axes import Axes

from ._base import BaseDisplay
from ._style import (
    COLORS,
    TUFTE_PALETTE,
    apply_tufte_style,
    set_tufte_title,
)

__all__ = ["GateResultDisplay", "GateComparisonDisplay"]


class GateResultDisplay(BaseDisplay):
    """
    Visualization of a single gate result.

    Displays the gate status (HALT/WARN/PASS) with metric details.

    Parameters
    ----------
    name : str
        Gate name.
    status : str
        Gate status ("HALT", "WARN", or "PASS").
    message : str
        Gate message.
    metrics : dict, optional
        Additional metrics to display.

    Attributes
    ----------
    ax_ : matplotlib.axes.Axes
        The axes used for plotting.
    figure_ : matplotlib.figure.Figure
        The figure containing the plot.

    See Also
    --------
    temporalcv.gates.gate_signal_verification : Signal verification gate.
    temporalcv.gates.gate_suspicious_improvement : Improvement gate.

    Examples
    --------
    >>> from temporalcv.gates import gate_signal_verification
    >>> from temporalcv.viz import GateResultDisplay
    >>>
    >>> result = gate_signal_verification(model, X, y, n_shuffles=100)
    >>> display = GateResultDisplay.from_gate(result)
    >>> display.plot()
    """

    def __init__(
        self,
        name: str,
        status: str,
        message: str,
        *,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.status = status.upper()
        self.message = message
        self.metrics = metrics or {}

    @classmethod
    def from_gate(cls, gate_result: Any) -> GateResultDisplay:
        """
        Create display from a GateResult object.

        Parameters
        ----------
        gate_result : GateResult
            Result from a gate function (e.g., gate_signal_verification).

        Returns
        -------
        GateResultDisplay
            The display object.

        Examples
        --------
        >>> result = gate_signal_verification(model, X, y)
        >>> display = GateResultDisplay.from_gate(result)
        """
        # Extract status string from enum
        status_str = str(gate_result.status)
        if "." in status_str:
            status_str = status_str.split(".")[-1]

        # Extract metrics if available
        metrics = {}
        if hasattr(gate_result, "details") and gate_result.details:
            metrics = gate_result.details

        return cls(
            name=gate_result.gate_name,
            status=status_str,
            message=gate_result.message,
            metrics=metrics,
        )

    def plot(
        self,
        *,
        ax: Optional[Axes] = None,
        tufte: bool = True,
        show_message: bool = True,
    ) -> GateResultDisplay:
        """
        Plot the gate result.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        tufte : bool
            If True, apply Tufte styling (default).
        show_message : bool
            If True, show the gate message.

        Returns
        -------
        self
            The display object for method chaining.
        """
        ax = self._get_ax_or_create(ax, figsize=(6, 2))

        if tufte:
            apply_tufte_style(ax)

        # Status color
        status_colors = {
            "HALT": COLORS["halt"],
            "WARN": COLORS["warn"],
            "PASS": COLORS["pass"],
        }
        color = status_colors.get(self.status, TUFTE_PALETTE["secondary"])

        # Draw status indicator
        ax.barh(0, 1, height=0.6, color=color, alpha=0.85, edgecolor="none")

        # Status text (centered, white)
        ax.text(
            0.5,
            0,
            self.status,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

        # Gate name (above)
        ax.text(
            0.5,
            0.5,
            self.name,
            ha="center",
            va="bottom",
            fontsize=10,
            color=TUFTE_PALETTE["text"],
        )

        # Message (below)
        if show_message and self.message:
            # Truncate long messages
            msg = self.message if len(self.message) < 60 else self.message[:57] + "..."
            ax.text(
                0.5,
                -0.5,
                msg,
                ha="center",
                va="top",
                fontsize=8,
                color=TUFTE_PALETTE["text_secondary"],
                style="italic",
            )

        # Remove all axes elements
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.axis("off")

        self._finalize_plot(ax)
        return self


class GateComparisonDisplay(BaseDisplay):
    """
    Visualization comparing multiple gate results.

    Displays multiple gates side by side for a comprehensive view.

    Parameters
    ----------
    gate_results : list
        List of GateResult objects or (name, status) tuples.

    Attributes
    ----------
    ax_ : matplotlib.axes.Axes
        The axes used for plotting.
    figure_ : matplotlib.figure.Figure
        The figure containing the plot.

    See Also
    --------
    temporalcv.gates.run_gates : Run multiple gates.
    GateResultDisplay : Single gate visualization.

    Examples
    --------
    >>> from temporalcv.gates import run_gates, gate_signal_verification
    >>> from temporalcv.viz import GateComparisonDisplay
    >>>
    >>> gates = [
    ...     gate_signal_verification(model, X, y),
    ...     gate_suspicious_improvement(model_mae, baseline_mae),
    ... ]
    >>> report = run_gates(gates)
    >>> display = GateComparisonDisplay.from_report(report)
    >>> display.plot()
    """

    def __init__(
        self,
        names: List[str],
        statuses: List[str],
        messages: Optional[List[str]] = None,
    ):
        self.names = names
        self.statuses = [s.upper() for s in statuses]
        self.messages = messages or [""] * len(names)
        self.n_gates = len(names)

    @classmethod
    def from_gates(cls, gate_results: List[Any]) -> GateComparisonDisplay:
        """
        Create display from a list of GateResult objects.

        Parameters
        ----------
        gate_results : list of GateResult
            Results from gate functions.

        Returns
        -------
        GateComparisonDisplay
            The display object.
        """
        names = []
        statuses = []
        messages = []

        for result in gate_results:
            names.append(result.gate_name)
            status_str = str(result.status)
            if "." in status_str:
                status_str = status_str.split(".")[-1]
            statuses.append(status_str)
            messages.append(result.message)

        return cls(names, statuses, messages)

    @classmethod
    def from_report(cls, report: Any) -> GateComparisonDisplay:
        """
        Create display from a GateReport object.

        Parameters
        ----------
        report : GateReport
            Report from run_gates().

        Returns
        -------
        GateComparisonDisplay
            The display object.
        """
        return cls.from_gates(report.results)

    def plot(
        self,
        *,
        ax: Optional[Axes] = None,
        tufte: bool = True,
        orientation: str = "horizontal",
        show_messages: bool = False,
        title: Optional[str] = None,
    ) -> GateComparisonDisplay:
        """
        Plot the gate comparison.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        tufte : bool
            If True, apply Tufte styling (default).
        orientation : str
            "horizontal" (bars left-right) or "vertical" (bars top-bottom).
        show_messages : bool
            If True, show gate messages.
        title : str, optional
            Plot title.

        Returns
        -------
        self
            The display object for method chaining.
        """
        # Determine figure size based on orientation and number of gates
        if orientation == "horizontal":
            figsize = (max(6, self.n_gates * 1.5), 2.5)
        else:
            figsize = (4, max(3, self.n_gates * 0.8))

        ax = self._get_ax_or_create(ax, figsize=figsize)

        if tufte:
            apply_tufte_style(ax)

        # Status colors
        status_colors = {
            "HALT": COLORS["halt"],
            "WARN": COLORS["warn"],
            "PASS": COLORS["pass"],
        }

        if orientation == "horizontal":
            # Horizontal bars (side by side)
            bar_width = 0.8 / self.n_gates
            positions = np.arange(self.n_gates)

            for i, (name, status) in enumerate(zip(self.names, self.statuses)):
                color = status_colors.get(status, TUFTE_PALETTE["secondary"])

                ax.bar(
                    i,
                    1,
                    width=0.7,
                    color=color,
                    alpha=0.85,
                    edgecolor="none",
                )

                # Status label (on bar)
                ax.text(
                    i,
                    0.5,
                    status,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )

                # Gate name (below)
                ax.text(
                    i,
                    -0.1,
                    name,
                    ha="center",
                    va="top",
                    fontsize=9,
                    color=TUFTE_PALETTE["text"],
                    rotation=0,
                )

            ax.set_xlim(-0.5, self.n_gates - 0.5)
            ax.set_ylim(-0.5 if not show_messages else -0.8, 1.1)
            ax.set_xticks([])
            ax.set_yticks([])

        else:
            # Vertical layout (stacked)
            for i, (name, status) in enumerate(zip(self.names, self.statuses)):
                y_pos = self.n_gates - 1 - i
                color = status_colors.get(status, TUFTE_PALETTE["secondary"])

                ax.barh(
                    y_pos,
                    1,
                    height=0.6,
                    color=color,
                    alpha=0.85,
                    edgecolor="none",
                )

                # Status label (on bar)
                ax.text(
                    0.5,
                    y_pos,
                    status,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )

                # Gate name (left of bar)
                ax.text(
                    -0.05,
                    y_pos,
                    name,
                    ha="right",
                    va="center",
                    fontsize=9,
                    color=TUFTE_PALETTE["text"],
                )

            ax.set_xlim(-0.5, 1.1)
            ax.set_ylim(-0.5, self.n_gates - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        # Title
        if title:
            set_tufte_title(ax, title)

        # Remove spines for this visualization
        for spine in ax.spines.values():
            spine.set_visible(False)

        self._finalize_plot(ax)
        return self
