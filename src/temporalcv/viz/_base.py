"""
Base class for sklearn-style Display objects.

This module provides the foundation for all temporalcv visualization displays,
following the scikit-learn Display API pattern (from_estimator, from_predictions).

References
----------
- sklearn Visualization API: https://scikit-learn.org/stable/visualizations.html
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._style import apply_tufte_style

__all__ = ["BaseDisplay"]


class BaseDisplay(ABC):
    """
    Base class for all temporalcv Display objects.

    This class follows the scikit-learn Display pattern:
    - `from_*` class methods for construction
    - `plot()` method for rendering
    - `ax_` and `figure_` attributes after plotting

    All displays apply Tufte styling by default.

    Attributes
    ----------
    ax_ : matplotlib.axes.Axes
        The axes used for plotting. Set after `plot()` is called.
    figure_ : matplotlib.figure.Figure
        The figure containing the plot. Set after `plot()` is called.

    Examples
    --------
    Subclasses should implement:

    >>> class MyDisplay(BaseDisplay):
    ...     def __init__(self, data):
    ...         self.data = data
    ...
    ...     @classmethod
    ...     def from_model(cls, model, X, y):
    ...         data = model.predict(X)
    ...         return cls(data)
    ...
    ...     def plot(self, ax=None, tufte=True):
    ...         ax = self._validate_plot_params(ax, tufte)
    ...         ax.plot(self.data)
    ...         self._finalize_plot(ax)
    ...         return self
    """

    ax_: Axes
    figure_: Figure

    def _validate_plot_params(
        self,
        ax: Optional[Axes] = None,
        tufte: bool = True,
    ) -> Axes:
        """
        Validate and prepare axes for plotting.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes or creates new.
        tufte : bool
            If True, apply Tufte styling.

        Returns
        -------
        matplotlib.axes.Axes
            The axes to use for plotting.
        """
        if ax is None:
            ax = plt.gca()

        if tufte:
            apply_tufte_style(ax)

        return ax

    def _finalize_plot(self, ax: Axes) -> None:
        """
        Finalize the plot by setting ax_ and figure_ attributes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes used for plotting.
        """
        self.ax_ = ax
        self.figure_ = ax.figure

    @abstractmethod
    def plot(self, *, ax: Optional[Axes] = None, tufte: bool = True) -> "BaseDisplay":
        """
        Plot the visualization.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes.
        tufte : bool
            If True, apply Tufte styling (default).

        Returns
        -------
        self
            The display object for method chaining.
        """
        pass

    def _get_ax_or_create(
        self,
        ax: Optional[Axes] = None,
        figsize: tuple = (8, 5),
    ) -> Axes:
        """
        Get existing axes or create new figure with specified size.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes.
        figsize : tuple
            Figure size if creating new.

        Returns
        -------
        matplotlib.axes.Axes
            The axes to use.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax
