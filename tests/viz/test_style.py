"""
Tests for temporalcv.viz._style module.

Validates Tufte-style primitives for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from temporalcv.viz._style import (
    COLORS,
    TUFTE_PALETTE,
    apply_tufte_style,
    create_tufte_figure,
    direct_label,
    direct_label_line,
    minimal_spines,
    muted_color,
    range_frame,
    set_tufte_labels,
    set_tufte_title,
)


class TestTuftePalette:
    """Tests for color palette constants."""

    def test_tufte_palette_has_required_keys(self):
        """TUFTE_PALETTE contains all required color keys."""
        required = [
            "primary",
            "secondary",
            "tertiary",
            "accent",
            "success",
            "warning",
            "info",
            "spine",
            "grid",
            "background",
            "text",
            "text_secondary",
        ]
        for key in required:
            assert key in TUFTE_PALETTE, f"Missing key: {key}"

    def test_tufte_palette_values_are_hex_colors(self):
        """All palette values are valid hex color strings."""
        for key, value in TUFTE_PALETTE.items():
            assert isinstance(value, str), f"{key} is not a string"
            assert value.startswith("#"), f"{key} is not a hex color"
            assert len(value) == 7, f"{key} is not a valid hex color"

    def test_colors_alias_mapping(self):
        """COLORS aliases map to valid palette values."""
        required = ["train", "test", "gap", "pass", "warn", "halt", "prediction", "actual"]
        for key in required:
            assert key in COLORS, f"Missing COLORS key: {key}"
            assert COLORS[key] in TUFTE_PALETTE.values(), f"{key} not in palette"


class TestApplyTufteStyle:
    """Tests for apply_tufte_style function."""

    @pytest.fixture
    def ax(self):
        """Create a fresh axes for testing."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_removes_top_spine(self, ax):
        """Top spine is hidden after styling."""
        apply_tufte_style(ax)
        assert not ax.spines["top"].get_visible()

    def test_removes_right_spine(self, ax):
        """Right spine is hidden after styling."""
        apply_tufte_style(ax)
        assert not ax.spines["right"].get_visible()

    def test_keeps_left_spine(self, ax):
        """Left spine remains visible."""
        apply_tufte_style(ax)
        assert ax.spines["left"].get_visible()

    def test_keeps_bottom_spine(self, ax):
        """Bottom spine remains visible."""
        apply_tufte_style(ax)
        assert ax.spines["bottom"].get_visible()

    def test_spine_color_is_subtle(self, ax):
        """Remaining spines use subtle color."""
        apply_tufte_style(ax)
        assert ax.spines["left"].get_edgecolor()[:3] == pytest.approx(
            plt.matplotlib.colors.to_rgb(TUFTE_PALETTE["spine"]), abs=0.01
        )

    def test_grid_is_off(self, ax):
        """Grid is disabled by default."""
        apply_tufte_style(ax)
        # Check that no gridlines are visible
        assert len(ax.get_xgridlines()) == 0 or not ax.get_xgridlines()[0].get_visible()

    def test_returns_axes(self, ax):
        """Function returns the axes for chaining."""
        result = apply_tufte_style(ax)
        assert result is ax


class TestMinimalSpines:
    """Tests for minimal_spines function."""

    @pytest.fixture
    def ax(self):
        """Create a fresh axes for testing."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_removes_top_and_right(self, ax):
        """Top and right spines are always hidden."""
        minimal_spines(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

    def test_keeps_left_bottom_by_default(self, ax):
        """Left and bottom spines visible by default."""
        minimal_spines(ax)
        assert ax.spines["left"].get_visible()
        assert ax.spines["bottom"].get_visible()

    def test_can_hide_left(self, ax):
        """Left spine can be hidden."""
        minimal_spines(ax, left=False)
        assert not ax.spines["left"].get_visible()

    def test_can_hide_bottom(self, ax):
        """Bottom spine can be hidden."""
        minimal_spines(ax, bottom=False)
        assert not ax.spines["bottom"].get_visible()


class TestDirectLabel:
    """Tests for direct_label function."""

    @pytest.fixture
    def ax(self):
        """Create axes with some data."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        yield ax
        plt.close(fig)

    def test_adds_annotation(self, ax):
        """Direct label adds an annotation to the axes."""
        initial_texts = len(ax.texts)
        direct_label(ax, 2, 4, "Peak")
        assert len(ax.texts) == initial_texts + 1

    def test_annotation_text_correct(self, ax):
        """Annotation has correct text content."""
        direct_label(ax, 2, 4, "Peak")
        assert ax.texts[-1].get_text() == "Peak"

    def test_respects_kwargs(self, ax):
        """Custom kwargs are applied to annotation."""
        direct_label(ax, 2, 4, "Peak", fontsize=12, color="red")
        text = ax.texts[-1]
        assert text.get_fontsize() == 12
        assert text.get_color() == "red"


class TestDirectLabelLine:
    """Tests for direct_label_line function."""

    @pytest.fixture
    def ax(self):
        """Create axes with line data."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_labels_at_end(self, ax):
        """Label placed at end of line."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 2])
        direct_label_line(ax, x, y, "Series A", position="end")
        # Check annotation exists
        assert len(ax.texts) == 1

    def test_labels_at_start(self, ax):
        """Label placed at start of line."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 2])
        direct_label_line(ax, x, y, "Series A", position="start")
        assert len(ax.texts) == 1

    def test_labels_at_max(self, ax):
        """Label placed at maximum value."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 2])
        direct_label_line(ax, x, y, "Peak", position="max")
        assert len(ax.texts) == 1

    def test_invalid_position_raises(self, ax):
        """Invalid position raises ValueError."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 2])
        with pytest.raises(ValueError, match="position must be"):
            direct_label_line(ax, x, y, "Label", position="invalid")


class TestRangeFrame:
    """Tests for range_frame function."""

    @pytest.fixture
    def ax(self):
        """Create axes with data."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        yield ax
        plt.close(fig)

    def test_hides_all_spines(self, ax):
        """All default spines are hidden."""
        range_frame(ax)
        for spine in ax.spines.values():
            assert not spine.get_visible()

    def test_returns_axes(self, ax):
        """Function returns the axes for chaining."""
        result = range_frame(ax)
        assert result is ax


class TestCreateTufteFigure:
    """Tests for create_tufte_figure function."""

    def test_creates_figure_and_axes(self):
        """Returns figure and axes objects."""
        fig, ax = create_tufte_figure()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_creates_multiple_axes(self):
        """Creates grid of axes for nrows/ncols > 1."""
        fig, axes = create_tufte_figure(nrows=2, ncols=2)
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_applies_tufte_style(self):
        """All axes have Tufte style applied."""
        fig, ax = create_tufte_figure()
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)

    def test_respects_figsize(self):
        """Custom figsize is applied."""
        fig, ax = create_tufte_figure(figsize=(10, 5))
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 5
        plt.close(fig)


class TestSetTufteTitle:
    """Tests for set_tufte_title function."""

    @pytest.fixture
    def ax(self):
        """Create a fresh axes."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_sets_title_text(self, ax):
        """Title text is set correctly."""
        set_tufte_title(ax, "My Title")
        # Left-aligned title requires loc='left' to retrieve
        assert ax.get_title(loc="left") == "My Title"

    def test_left_aligned_by_default(self, ax):
        """Title is left-aligned per Tufte principles."""
        set_tufte_title(ax, "My Title")
        # Left title is set, center title is empty
        assert ax.get_title(loc="left") == "My Title"
        assert ax.get_title(loc="center") == ""


class TestSetTufteLabels:
    """Tests for set_tufte_labels function."""

    @pytest.fixture
    def ax(self):
        """Create a fresh axes."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_sets_xlabel(self, ax):
        """X-axis label is set correctly."""
        set_tufte_labels(ax, xlabel="Time")
        assert ax.get_xlabel() == "Time"

    def test_sets_ylabel(self, ax):
        """Y-axis label is set correctly."""
        set_tufte_labels(ax, ylabel="Value")
        assert ax.get_ylabel() == "Value"

    def test_sets_both_labels(self, ax):
        """Both labels can be set together."""
        set_tufte_labels(ax, xlabel="Time", ylabel="Value")
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Value"


class TestMutedColor:
    """Tests for muted_color function."""

    def test_returns_palette_color_by_name(self):
        """Returns color from TUFTE_PALETTE by name."""
        assert muted_color("primary") == TUFTE_PALETTE["primary"]

    def test_returns_colors_alias_by_name(self):
        """Returns color from COLORS alias by name."""
        assert muted_color("train") == COLORS["train"]

    def test_returns_raw_color_if_not_in_palette(self):
        """Returns the color as-is if not in palette."""
        assert muted_color("#123456") == "#123456"
