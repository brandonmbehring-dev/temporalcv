"""
Tufte-inspired styling primitives for temporalcv visualizations.

This module implements Edward Tufte's principles for data visualization:
1. Maximize data-ink ratio
2. Eliminate chartjunk
3. Use direct labeling instead of legends where practical
4. Employ small multiples for comparison
5. Integrate graphics and text

References
----------
- Tufte, E. R. (1983). The Visual Display of Quantitative Information.
- Tufte, E. R. (2001). Envisioning Information.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = [
    "TUFTE_PALETTE",
    "apply_tufte_style",
    "minimal_spines",
    "direct_label",
    "range_frame",
    "create_tufte_figure",
    "muted_color",
]

# =============================================================================
# Color Palette
# =============================================================================

TUFTE_PALETTE: Dict[str, str] = {
    # Primary data colors (muted, not saturated)
    "primary": "#4a4a4a",  # Dark gray for main data
    "secondary": "#8a8a8a",  # Medium gray for secondary
    "tertiary": "#b0b0b0",  # Light gray for tertiary
    # Semantic colors (muted versions)
    "accent": "#c44e52",  # Muted red for emphasis/errors
    "success": "#55a868",  # Muted green for success/pass
    "warning": "#dd8452",  # Muted orange for warnings
    "info": "#4c72b0",  # Muted blue for information
    # Structural colors
    "spine": "#cccccc",  # Very light gray for spines
    "grid": "#e5e5e5",  # Even lighter for subtle grid
    "background": "#fafafa",  # Off-white background
    "text": "#333333",  # Dark text
    "text_secondary": "#666666",  # Secondary text
}

# Semantic aliases for common use cases
COLORS = {
    "train": TUFTE_PALETTE["info"],
    "test": TUFTE_PALETTE["warning"],
    "gap": TUFTE_PALETTE["accent"],
    "pass": TUFTE_PALETTE["success"],
    "warn": TUFTE_PALETTE["warning"],
    "halt": TUFTE_PALETTE["accent"],
    "prediction": TUFTE_PALETTE["info"],
    "actual": TUFTE_PALETTE["primary"],
    "interval": TUFTE_PALETTE["info"],
    "baseline": TUFTE_PALETTE["secondary"],
}


def muted_color(color: str, saturation: float = 0.7) -> str:
    """
    Return a muted version of a color from the palette.

    Parameters
    ----------
    color : str
        Color name from TUFTE_PALETTE or a hex color.
    saturation : float
        Saturation factor (0-1). Lower = more muted.

    Returns
    -------
    str
        Hex color string.
    """
    if color in TUFTE_PALETTE:
        return TUFTE_PALETTE[color]
    if color in COLORS:
        return COLORS[color]
    return color


# =============================================================================
# Axes Styling
# =============================================================================


def apply_tufte_style(ax: Axes) -> Axes:
    """
    Apply Tufte's principles to a matplotlib axes.

    Implements:
    - Remove top and right spines (maximize data-ink)
    - Subtle spine colors
    - Remove grid (or make very subtle)
    - Refined tick styling

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.

    Returns
    -------
    matplotlib.axes.Axes
        The styled axes (for chaining).

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 2])
    >>> apply_tufte_style(ax)
    >>> plt.show()
    """
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subtle remaining spines
    ax.spines["left"].set_color(TUFTE_PALETTE["spine"])
    ax.spines["bottom"].set_color(TUFTE_PALETTE["spine"])
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Remove grid by default (high data-ink ratio)
    ax.grid(False)

    # Refined tick styling
    ax.tick_params(
        colors=TUFTE_PALETTE["text_secondary"],
        length=4,
        width=0.8,
        direction="out",
    )

    # Subtle tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(TUFTE_PALETTE["text_secondary"])
        label.set_fontsize(9)

    return ax


def minimal_spines(ax: Axes, left: bool = True, bottom: bool = True) -> Axes:
    """
    Remove all spines except specified ones.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.
    left : bool
        Keep left spine.
    bottom : bool
        Keep bottom spine.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)

    if left:
        ax.spines["left"].set_color(TUFTE_PALETTE["spine"])
        ax.spines["left"].set_linewidth(0.8)
    if bottom:
        ax.spines["bottom"].set_color(TUFTE_PALETTE["spine"])
        ax.spines["bottom"].set_linewidth(0.8)

    return ax


def range_frame(ax: Axes) -> Axes:
    """
    Create a Tufte-style range frame where spines only cover data range.

    This is a key Tufte principle: spines should indicate the range of data,
    not extend beyond it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to modify.

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    # Get data limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw range-limited spines manually
    ax.plot(
        [xlim[0], xlim[1]],
        [ylim[0], ylim[0]],
        color=TUFTE_PALETTE["spine"],
        linewidth=0.8,
        clip_on=False,
        zorder=0,
    )
    ax.plot(
        [xlim[0], xlim[0]],
        [ylim[0], ylim[1]],
        color=TUFTE_PALETTE["spine"],
        linewidth=0.8,
        clip_on=False,
        zorder=0,
    )

    return ax


# =============================================================================
# Direct Labeling
# =============================================================================


def direct_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    offset: Tuple[float, float] = (5, 5),
    **kwargs: Any,
) -> None:
    """
    Add a direct label to a data point (Tufte principle: label on the data).

    Direct labeling eliminates the need for legends, improving data-ink ratio.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    x : float
        X coordinate of the data point.
    y : float
        Y coordinate of the data point.
    text : str
        Label text.
    offset : tuple of float
        Offset in points (x, y) from the data point.
    **kwargs
        Additional arguments passed to ax.annotate().

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 2])
    >>> direct_label(ax, 2, 4, "Peak", offset=(5, 5))
    """
    defaults = {
        "fontsize": 9,
        "color": TUFTE_PALETTE["text"],
        "ha": "left",
        "va": "bottom",
    }
    defaults.update(kwargs)

    ax.annotate(
        text,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        **defaults,
    )


def direct_label_line(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    text: str,
    position: str = "end",
    **kwargs: Any,
) -> None:
    """
    Add a direct label to a line (at start, end, or max).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    x : array-like
        X data of the line.
    y : array-like
        Y data of the line.
    text : str
        Label text.
    position : str
        Where to place label: "start", "end", or "max".
    **kwargs
        Additional arguments passed to direct_label().
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if position == "end":
        idx = -1
        offset = (5, 0)
        ha = "left"
    elif position == "start":
        idx = 0
        offset = (-5, 0)
        ha = "right"
    elif position == "max":
        idx = np.argmax(y)
        offset = (0, 5)
        ha = "center"
    else:
        raise ValueError(f"position must be 'start', 'end', or 'max', got {position}")

    kwargs.setdefault("ha", ha)
    direct_label(ax, x[idx], y[idx], text, offset=offset, **kwargs)


# =============================================================================
# Figure Creation
# =============================================================================


def create_tufte_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs: Any,
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """
    Create a figure with Tufte styling applied.

    Parameters
    ----------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    figsize : tuple of float, optional
        Figure size (width, height) in inches.
    **kwargs
        Additional arguments passed to plt.subplots().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : matplotlib.axes.Axes or array of Axes
        The axes, with Tufte styling applied.
    """
    if figsize is None:
        # Tufte-inspired proportions (golden ratio-ish)
        figsize = (8 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    # Set figure background
    fig.patch.set_facecolor(TUFTE_PALETTE["background"])

    # Apply Tufte style to all axes
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            apply_tufte_style(ax)
            ax.set_facecolor(TUFTE_PALETTE["background"])
    else:
        apply_tufte_style(axes)
        axes.set_facecolor(TUFTE_PALETTE["background"])

    return fig, axes


# =============================================================================
# Convenience Functions
# =============================================================================


def add_subtle_grid(ax: Axes, axis: str = "y", **kwargs: Any) -> Axes:
    """
    Add a very subtle grid (when absolutely necessary).

    Use sparingly — Tufte prefers no grid. Only use when precise
    value reading is essential.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    axis : str
        Which axis: "x", "y", or "both".
    **kwargs
        Additional arguments passed to ax.grid().

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    """
    defaults = {
        "color": TUFTE_PALETTE["grid"],
        "linewidth": 0.5,
        "alpha": 0.5,
        "linestyle": "-",
    }
    defaults.update(kwargs)

    ax.grid(True, axis=axis, **defaults)
    return ax


def set_tufte_title(ax: Axes, title: str, **kwargs: Any) -> None:
    """
    Set a Tufte-styled title (understated, informative).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    title : str
        Title text.
    **kwargs
        Additional arguments passed to ax.set_title().
    """
    defaults = {
        "fontsize": 11,
        "fontweight": "normal",
        "color": TUFTE_PALETTE["text"],
        "loc": "left",  # Left-aligned per Tufte
        "pad": 10,
    }
    defaults.update(kwargs)
    ax.set_title(title, **defaults)


def set_tufte_labels(
    ax: Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """
    Set Tufte-styled axis labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    **kwargs
        Additional arguments passed to ax.set_xlabel/ylabel().
    """
    defaults = {
        "fontsize": 10,
        "color": TUFTE_PALETTE["text_secondary"],
    }
    defaults.update(kwargs)

    if xlabel:
        ax.set_xlabel(xlabel, **defaults)
    if ylabel:
        ax.set_ylabel(ylabel, **defaults)
