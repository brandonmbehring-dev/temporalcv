"""
Tests for temporalcv.viz.intervals module.

Validates PredictionIntervalDisplay class for conformal prediction visualization.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest

from temporalcv.viz import PredictionIntervalDisplay, plot_interval_width, plot_prediction_intervals


@dataclass
class MockPredictionInterval:
    """Mock PredictionInterval for testing."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float = 0.90


class TestPredictionIntervalDisplayInit:
    """Tests for PredictionIntervalDisplay initialization."""

    def test_init_basic(self):
        """Initialize with predictions and bounds."""
        predictions = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])

        display = PredictionIntervalDisplay(predictions, lower, upper)

        assert len(display.predictions) == 3
        assert display.confidence == 0.90

    def test_init_with_actuals(self):
        """Initialize with actual values for coverage."""
        predictions = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        actuals = np.array([1.0, 2.0, 4.0])  # Last one outside interval

        display = PredictionIntervalDisplay(predictions, lower, upper, actuals=actuals)

        assert display.actuals is not None
        assert display.coverage_ is not None

    def test_coverage_computation(self):
        """Coverage is correctly computed."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5])
        actuals = np.array([1.0, 2.0, 3.0, 5.0])  # 3/4 covered

        display = PredictionIntervalDisplay(predictions, lower, upper, actuals=actuals)

        assert display.coverage_ == 0.75

    def test_custom_x_axis(self):
        """Custom x-axis values are used."""
        predictions = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        x = np.array([10, 20, 30])

        display = PredictionIntervalDisplay(predictions, lower, upper, x=x)

        np.testing.assert_array_equal(display.x, x)


class TestPredictionIntervalDisplayFromConformal:
    """Tests for PredictionIntervalDisplay.from_conformal factory method."""

    def test_from_conformal_basic(self):
        """Create from PredictionInterval object."""
        intervals = MockPredictionInterval(
            point=np.array([1.0, 2.0, 3.0]),
            lower=np.array([0.5, 1.5, 2.5]),
            upper=np.array([1.5, 2.5, 3.5]),
            confidence=0.90,
        )

        display = PredictionIntervalDisplay.from_conformal(intervals)

        assert display.confidence == 0.90
        np.testing.assert_array_equal(display.predictions, intervals.point)

    def test_from_conformal_with_actuals(self):
        """Create with actual values."""
        intervals = MockPredictionInterval(
            point=np.array([1.0, 2.0]),
            lower=np.array([0.5, 1.5]),
            upper=np.array([1.5, 2.5]),
        )
        actuals = np.array([1.0, 3.0])

        display = PredictionIntervalDisplay.from_conformal(intervals, actuals)

        assert display.coverage_ == 0.5


class TestPredictionIntervalDisplayFromPredictions:
    """Tests for PredictionIntervalDisplay.from_predictions factory method."""

    def test_from_predictions(self):
        """Create from arrays."""
        display = PredictionIntervalDisplay.from_predictions(
            predictions=np.array([1.0, 2.0]),
            lower=np.array([0.5, 1.5]),
            upper=np.array([1.5, 2.5]),
            confidence=0.95,
        )

        assert display.confidence == 0.95


class TestPredictionIntervalDisplayPlot:
    """Tests for PredictionIntervalDisplay.plot method."""

    @pytest.fixture
    def display(self):
        """Create a display for testing."""
        np.random.seed(42)
        n = 50
        predictions = np.cumsum(np.random.randn(n) * 0.1) + 100
        lower = predictions - 2.0
        upper = predictions + 2.0
        actuals = predictions + np.random.randn(n) * 1.5

        return PredictionIntervalDisplay(
            predictions, lower, upper, actuals=actuals, confidence=0.90
        )

    def test_plot_creates_figure(self, display):
        """Plotting creates figure and axes."""
        display.plot()

        assert hasattr(display, "ax_")
        assert hasattr(display, "figure_")
        plt.close(display.figure_)

    def test_plot_returns_self(self, display):
        """plot() returns self for method chaining."""
        result = display.plot()

        assert result is display
        plt.close(display.figure_)

    def test_plot_with_custom_ax(self, display):
        """Plot on custom axes."""
        fig, ax = plt.subplots()
        display.plot(ax=ax)

        assert display.ax_ is ax
        plt.close(fig)

    def test_plot_with_tufte_style(self, display):
        """Tufte styling applied by default."""
        display.plot(tufte=True)

        assert not display.ax_.spines["top"].get_visible()
        plt.close(display.figure_)

    def test_plot_shows_predictions(self, display):
        """Predictions line is shown."""
        display.plot(show_predictions=True)

        lines = display.ax_.get_lines()
        assert len(lines) > 0
        plt.close(display.figure_)

    def test_plot_shows_coverage(self, display):
        """Coverage points are shown."""
        display.plot(show_coverage=True)

        # Check for scatter collections (covered/uncovered points)
        collections = display.ax_.collections
        assert len(collections) > 0
        plt.close(display.figure_)

    def test_plot_with_custom_title(self, display):
        """Custom title is applied."""
        display.plot(title="My Intervals")

        # Tufte-style uses left-aligned titles
        assert display.ax_.get_title(loc="left") == "My Intervals"
        plt.close(display.figure_)


class TestPredictionIntervalDisplayPlotWidth:
    """Tests for PredictionIntervalDisplay.plot_width method."""

    @pytest.fixture
    def display(self):
        """Create a display with varying widths (adaptive conformal)."""
        n = 30
        predictions = np.arange(n, dtype=float)
        # Varying widths (simulating adaptive conformal)
        widths = 2.0 + np.sin(np.linspace(0, 2 * np.pi, n))
        lower = predictions - widths / 2
        upper = predictions + widths / 2

        return PredictionIntervalDisplay(predictions, lower, upper)

    def test_plot_width_creates_figure(self, display):
        """Plotting width creates figure."""
        display.plot_width()

        assert hasattr(display, "ax_")
        assert hasattr(display, "figure_")
        plt.close(display.figure_)

    def test_plot_width_returns_self(self, display):
        """plot_width() returns self."""
        result = display.plot_width()

        assert result is display
        plt.close(display.figure_)

    def test_plot_width_shows_bars(self, display):
        """Width bars are shown."""
        display.plot_width()

        # Check for bar patches
        patches = display.ax_.patches
        assert len(patches) > 0
        plt.close(display.figure_)


class TestPlotIntervalFunctions:
    """Tests for function API."""

    @pytest.fixture
    def intervals(self):
        """Create mock intervals."""
        return MockPredictionInterval(
            point=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            lower=np.array([0.5, 1.5, 2.5, 3.5, 4.5]),
            upper=np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
            confidence=0.90,
        )

    def test_plot_prediction_intervals(self, intervals):
        """plot_prediction_intervals function works."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 6.0])

        ax = plot_prediction_intervals(intervals, actuals)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_plot_interval_width(self, intervals):
        """plot_interval_width function works."""
        ax = plot_interval_width(intervals)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)
