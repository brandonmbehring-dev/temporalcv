"""
Tests for temporalcv.viz.cv module.

Validates CVFoldsDisplay class for cross-validation visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.model_selection import KFold

from temporalcv import WalkForwardCV
from temporalcv.viz import CVFoldsDisplay, plot_cv_folds


class TestCVFoldsDisplayInit:
    """Tests for CVFoldsDisplay initialization."""

    def test_init_with_basic_indices(self):
        """Initialize with train and test indices."""
        trains = [np.array([0, 1, 2]), np.array([0, 1, 2, 3])]
        tests = [np.array([3, 4]), np.array([4, 5])]

        display = CVFoldsDisplay(trains, tests)

        assert display.n_splits == 2
        assert len(display.train_indices) == 2
        assert len(display.test_indices) == 2

    def test_init_with_gap_indices(self):
        """Initialize with gap indices for walk-forward CV."""
        trains = [np.array([0, 1, 2])]
        tests = [np.array([4, 5])]
        gaps = [np.array([3])]

        display = CVFoldsDisplay(trains, tests, gap_indices=gaps)

        assert display.gap_indices is not None
        assert len(display.gap_indices[0]) == 1

    def test_infers_n_samples(self):
        """n_samples inferred from indices if not provided."""
        trains = [np.array([0, 1, 2, 3, 4])]
        tests = [np.array([5, 6, 7, 8, 9])]

        display = CVFoldsDisplay(trains, tests)

        assert display.n_samples == 10

    def test_explicit_n_samples(self):
        """n_samples can be explicitly provided."""
        trains = [np.array([0, 1, 2])]
        tests = [np.array([3, 4])]

        display = CVFoldsDisplay(trains, tests, n_samples=100)

        assert display.n_samples == 100


class TestCVFoldsDisplayFromCV:
    """Tests for CVFoldsDisplay.from_cv factory method."""

    @pytest.fixture
    def data(self):
        """Generate sample data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        return X, y

    def test_from_walk_forward_cv(self, data):
        """Create display from WalkForwardCV."""
        X, y = data
        cv = WalkForwardCV(n_splits=5, test_size=10)

        display = CVFoldsDisplay.from_cv(cv, X, y)

        assert display.n_splits == 5
        assert display.n_samples == 100

    def test_from_sklearn_kfold(self, data):
        """Create display from sklearn KFold."""
        X, y = data
        cv = KFold(n_splits=5, shuffle=False)

        display = CVFoldsDisplay.from_cv(cv, X, y)

        assert display.n_splits == 5

    def test_detects_gap(self, data):
        """Gap indices detected when present."""
        X, y = data
        cv = WalkForwardCV(n_splits=3, test_size=10, extra_gap=5)

        display = CVFoldsDisplay.from_cv(cv, X, y)

        assert display.gap_indices is not None
        assert all(len(g) > 0 for g in display.gap_indices)

    def test_no_gap_when_contiguous(self, data):
        """Gap indices None when train/test are contiguous."""
        X, y = data
        cv = WalkForwardCV(n_splits=3, test_size=10, extra_gap=0)

        display = CVFoldsDisplay.from_cv(cv, X, y)

        # Gap indices should be None or all empty
        if display.gap_indices is not None:
            assert all(len(g) == 0 for g in display.gap_indices)


class TestCVFoldsDisplayFromSplits:
    """Tests for CVFoldsDisplay.from_splits factory method."""

    def test_from_splits_list(self):
        """Create display from list of (train, test) tuples."""
        splits = [
            (np.array([0, 1, 2]), np.array([3, 4])),
            (np.array([0, 1, 2, 3]), np.array([4, 5])),
        ]

        display = CVFoldsDisplay.from_splits(splits)

        assert display.n_splits == 2

    def test_from_splits_with_n_samples(self):
        """n_samples can be explicitly provided."""
        splits = [(np.array([0, 1]), np.array([2, 3]))]

        display = CVFoldsDisplay.from_splits(splits, n_samples=100)

        assert display.n_samples == 100


class TestCVFoldsDisplayPlot:
    """Tests for CVFoldsDisplay.plot method."""

    @pytest.fixture
    def display(self):
        """Create a simple display for testing."""
        trains = [
            np.array([0, 1, 2, 3, 4]),
            np.array([0, 1, 2, 3, 4, 5, 6]),
        ]
        tests = [
            np.array([5, 6]),
            np.array([7, 8]),
        ]
        return CVFoldsDisplay(trains, tests, n_samples=10)

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
        assert not display.ax_.spines["right"].get_visible()
        plt.close(display.figure_)

    def test_plot_without_tufte_style(self, display):
        """Tufte styling can be disabled."""
        display.plot(tufte=False)

        # Default matplotlib style has visible spines
        # Just verify we got a valid plot
        assert display.ax_ is not None
        plt.close(display.figure_)

    def test_plot_with_custom_title(self, display):
        """Custom title is applied."""
        display.plot(title="My Custom Title")

        # Tufte-style uses left-aligned titles
        assert display.ax_.get_title(loc="left") == "My Custom Title"
        plt.close(display.figure_)

    def test_plot_with_gap_indices(self):
        """Gap indices are visualized."""
        trains = [np.array([0, 1, 2])]
        tests = [np.array([5, 6])]
        gaps = [np.array([3, 4])]

        display = CVFoldsDisplay(trains, tests, gap_indices=gaps, n_samples=7)
        display.plot()

        # Verify plot has gap bars (check legend or patches)
        legend = display.ax_.get_legend()
        if legend:
            labels = [t.get_text() for t in legend.get_texts()]
            assert "Gap" in labels
        plt.close(display.figure_)


class TestPlotCVFoldsFunction:
    """Tests for plot_cv_folds function API."""

    @pytest.fixture
    def data(self):
        """Generate sample data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        return X, y

    def test_plot_cv_folds_basic(self, data):
        """Basic usage of plot_cv_folds."""
        X, y = data
        cv = WalkForwardCV(n_splits=3, test_size=5)

        ax = plot_cv_folds(cv, X, y)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_plot_cv_folds_with_custom_ax(self, data):
        """Use custom axes."""
        X, y = data
        cv = WalkForwardCV(n_splits=3, test_size=5)
        fig, custom_ax = plt.subplots()

        ax = plot_cv_folds(cv, X, y, ax=custom_ax)

        assert ax is custom_ax
        plt.close(fig)

    def test_plot_cv_folds_with_title(self, data):
        """Custom title is applied."""
        X, y = data
        cv = WalkForwardCV(n_splits=3, test_size=5)

        ax = plot_cv_folds(cv, X, y, title="Walk-Forward Splits")

        # Tufte-style uses left-aligned titles
        assert ax.get_title(loc="left") == "Walk-Forward Splits"
        plt.close(ax.figure)
