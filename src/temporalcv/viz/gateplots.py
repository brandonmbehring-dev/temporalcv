"""
Gate result visualization functions (statsmodels-style API).

This module provides simple function-based API for gate visualizations,
following the statsmodels pattern where ax=None creates a new figure.

Examples
--------
>>> from temporalcv.viz import plot_gate_result, plot_gate_comparison
>>>
>>> result = gate_signal_verification(model, X, y)
>>> plot_gate_result(result)
>>> plt.show()
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from matplotlib.axes import Axes

from .gates import GateComparisonDisplay, GateResultDisplay

__all__ = [
    "plot_gate_result",
    "plot_gate_comparison",
]


def plot_gate_result(
    gate_result: Any,
    *,
    ax: Optional[Axes] = None,
    tufte: bool = True,
    show_message: bool = True,
) -> Axes:
    """
    Plot a single gate result.

    Displays the gate status (HALT/WARN/PASS) with message.

    Parameters
    ----------
    gate_result : GateResult
        Result from a gate function.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    tufte : bool
        If True, apply Tufte styling (default).
    show_message : bool
        If True, show the gate message.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from temporalcv.gates import gate_signal_verification
    >>> from temporalcv.viz import plot_gate_result
    >>>
    >>> result = gate_signal_verification(model, X, y, n_shuffles=100)
    >>> plot_gate_result(result)
    >>> plt.show()

    See Also
    --------
    GateResultDisplay : Class-based API.
    plot_gate_comparison : Compare multiple gates.
    """
    display = GateResultDisplay.from_gate(gate_result)
    display.plot(ax=ax, tufte=tufte, show_message=show_message)
    return display.ax_


def plot_gate_comparison(
    gate_results: Union[List[Any], Any],
    *,
    ax: Optional[Axes] = None,
    tufte: bool = True,
    orientation: str = "horizontal",
    title: Optional[str] = None,
) -> Axes:
    """
    Plot comparison of multiple gate results.

    Displays multiple gates side by side for a comprehensive view.

    Parameters
    ----------
    gate_results : list of GateResult or GateReport
        Results from gate functions or a GateReport from run_gates().
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    tufte : bool
        If True, apply Tufte styling.
    orientation : str
        "horizontal" (bars left-right) or "vertical" (bars top-bottom).
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.

    Examples
    --------
    >>> from temporalcv.gates import run_gates, gate_signal_verification
    >>> from temporalcv.viz import plot_gate_comparison
    >>>
    >>> gates = [
    ...     gate_signal_verification(model, X, y),
    ...     gate_suspicious_improvement(model_mae, baseline_mae),
    ... ]
    >>> report = run_gates(gates)
    >>> plot_gate_comparison(report, title="Validation Gates")
    >>> plt.show()

    See Also
    --------
    GateComparisonDisplay : Class-based API.
    plot_gate_result : Single gate visualization.
    """
    # Handle GateReport or list of GateResult
    if hasattr(gate_results, "results"):
        # It's a GateReport
        display = GateComparisonDisplay.from_report(gate_results)
    else:
        # It's a list of GateResult
        display = GateComparisonDisplay.from_gates(gate_results)

    display.plot(ax=ax, tufte=tufte, orientation=orientation, title=title)
    return display.ax_
