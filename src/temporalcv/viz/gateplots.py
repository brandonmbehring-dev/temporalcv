"""
Gate result visualization functions (statsmodels-style API).

This module provides simple function-based API for gate visualizations,
following the statsmodels pattern where ax=None creates a new figure.

Examples
--------
>>> import numpy as np
>>> from sklearn.linear_model import LinearRegression
>>> from temporalcv.gates import gate_signal_verification
>>> from temporalcv.viz import plot_gate_result
>>>
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((60, 3))
>>> y = rng.standard_normal(60)
>>> result = gate_signal_verification(
...     LinearRegression(), X, y, method="effect_size", n_shuffles=5, random_state=0
... )
>>> ax = plot_gate_result(result)
"""

from __future__ import annotations

from typing import Any

from matplotlib.axes import Axes

from .gates import GateComparisonDisplay, GateResultDisplay

__all__ = [
    "plot_gate_result",
    "plot_gate_comparison",
]


def plot_gate_result(
    gate_result: Any,
    *,
    ax: Axes | None = None,
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
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from temporalcv.gates import gate_signal_verification
    >>> from temporalcv.viz import plot_gate_result
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((60, 3))
    >>> y = rng.standard_normal(60)
    >>> result = gate_signal_verification(
    ...     LinearRegression(), X, y, method="effect_size", n_shuffles=5, random_state=0
    ... )
    >>> ax = plot_gate_result(result)

    See Also
    --------
    GateResultDisplay : Class-based API.
    plot_gate_comparison : Compare multiple gates.
    """
    display = GateResultDisplay.from_gate(gate_result)
    display.plot(ax=ax, tufte=tufte, show_message=show_message)
    return display.ax_


def plot_gate_comparison(
    gate_results: list[Any] | Any,
    *,
    ax: Axes | None = None,
    tufte: bool = True,
    orientation: str = "horizontal",
    title: str | None = None,
) -> Axes:
    """
    Plot comparison of multiple gate results.

    Displays multiple gates side by side for a comprehensive view.

    Parameters
    ----------
    gate_results : list of GateResult or ValidationReport
        Results from gate functions or a ValidationReport from run_gates().
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
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from temporalcv.gates import (
    ...     run_gates,
    ...     gate_signal_verification,
    ...     gate_suspicious_improvement,
    ... )
    >>> from temporalcv.viz import plot_gate_comparison
    >>>
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((60, 3))
    >>> y = rng.standard_normal(60)
    >>> gates = [
    ...     gate_signal_verification(
    ...         LinearRegression(), X, y, method="effect_size", n_shuffles=5, random_state=0
    ...     ),
    ...     gate_suspicious_improvement(model_metric=0.12, baseline_metric=0.20),
    ... ]
    >>> report = run_gates(gates)
    >>> ax = plot_gate_comparison(report, title="Validation Gates")

    See Also
    --------
    GateComparisonDisplay : Class-based API.
    plot_gate_result : Single gate visualization.
    """
    # Handle ValidationReport or list of GateResult. ValidationReport exposes
    # ``gates``; legacy report objects may expose ``results`` instead.
    if hasattr(gate_results, "gates") or hasattr(gate_results, "results"):
        # It's a report from run_gates()
        display = GateComparisonDisplay.from_report(gate_results)
    else:
        # It's a list of GateResult
        display = GateComparisonDisplay.from_gates(gate_results)

    display.plot(ax=ax, tufte=tufte, orientation=orientation, title=title)
    return display.ax_
