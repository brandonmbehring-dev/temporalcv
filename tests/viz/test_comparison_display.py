"""
Tests for temporalcv.viz.comparison module.

Validates MetricComparisonDisplay class for model comparison visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from temporalcv.viz import MetricComparisonDisplay, plot_metric_comparison


class TestMetricComparisonDisplayInit:
    """Tests for MetricComparisonDisplay initialization."""

    def test_init_basic(self):
        """Initialize with names, metrics, and values."""
        model_names = ["Model A", "Model B"]
        metric_names = ["MAE", "RMSE"]
        values = np.array([[0.15, 0.22], [0.12, 0.19]])

        display = MetricComparisonDisplay(model_names, metric_names, values)

        assert display.n_models == 2
        assert display.n_metrics == 2

    def test_init_with_lower_is_better(self):
        """Initialize with lower_is_better dict."""
        model_names = ["A", "B"]
        metric_names = ["MAE", "R2"]
        values = np.array([[0.15, 0.85], [0.12, 0.90]])
        lower_is_better = {"MAE": True, "R2": False}

        display = MetricComparisonDisplay(
            model_names, metric_names, values, lower_is_better=lower_is_better
        )

        assert display.lower_is_better["MAE"] is True
        assert display.lower_is_better["R2"] is False

    def test_init_with_baseline_idx(self):
        """Initialize with baseline model index."""
        model_names = ["Baseline", "Model A"]
        metric_names = ["MAE"]
        values = np.array([[0.20], [0.15]])

        display = MetricComparisonDisplay(model_names, metric_names, values, baseline_idx=0)

        assert display.baseline_idx == 0


class TestMetricComparisonDisplayFromDict:
    """Tests for MetricComparisonDisplay.from_dict factory method."""

    def test_from_dict_basic(self):
        """Create from nested dictionary."""
        results = {
            "Model A": {"MAE": 0.15, "RMSE": 0.22},
            "Model B": {"MAE": 0.12, "RMSE": 0.19},
        }

        display = MetricComparisonDisplay.from_dict(results)

        assert display.n_models == 2
        assert display.n_metrics == 2
        assert "Model A" in display.model_names

    def test_from_dict_with_baseline(self):
        """Create with baseline model specified."""
        results = {
            "Baseline": {"MAE": 0.20},
            "Model A": {"MAE": 0.15},
        }

        display = MetricComparisonDisplay.from_dict(results, baseline="Baseline")

        assert display.baseline_idx == 0

    def test_from_dict_preserves_order(self):
        """Model order is preserved."""
        results = {
            "First": {"MAE": 0.1},
            "Second": {"MAE": 0.2},
            "Third": {"MAE": 0.3},
        }

        display = MetricComparisonDisplay.from_dict(results)

        assert display.model_names == ["First", "Second", "Third"]


class TestMetricComparisonDisplayFromArrays:
    """Tests for MetricComparisonDisplay.from_arrays factory method."""

    def test_from_arrays(self):
        """Create from arrays."""
        display = MetricComparisonDisplay.from_arrays(
            model_names=["A", "B"],
            metric_names=["MAE", "RMSE"],
            values=np.array([[0.15, 0.22], [0.12, 0.19]]),
        )

        assert display.n_models == 2
        assert display.n_metrics == 2


class TestMetricComparisonDisplayPlot:
    """Tests for MetricComparisonDisplay.plot method."""

    @pytest.fixture
    def display_single_metric(self):
        """Create display with single metric."""
        return MetricComparisonDisplay.from_dict(
            {
                "Baseline": {"MAE": 0.20},
                "Model A": {"MAE": 0.15},
                "Model B": {"MAE": 0.12},
            }
        )

    @pytest.fixture
    def display_multi_metric(self):
        """Create display with multiple metrics."""
        return MetricComparisonDisplay.from_dict(
            {
                "Baseline": {"MAE": 0.20, "RMSE": 0.28},
                "Model A": {"MAE": 0.15, "RMSE": 0.22},
                "Model B": {"MAE": 0.12, "RMSE": 0.19},
            }
        )

    def test_plot_creates_figure(self, display_single_metric):
        """Plotting creates figure and axes."""
        display_single_metric.plot()

        assert hasattr(display_single_metric, "ax_")
        assert hasattr(display_single_metric, "figure_")
        plt.close(display_single_metric.figure_)

    def test_plot_returns_self(self, display_single_metric):
        """plot() returns self for method chaining."""
        result = display_single_metric.plot()

        assert result is display_single_metric
        plt.close(display_single_metric.figure_)

    def test_plot_single_metric_vertical(self, display_single_metric):
        """Vertical orientation for single metric."""
        display_single_metric.plot(orientation="vertical")

        assert display_single_metric.ax_ is not None
        plt.close(display_single_metric.figure_)

    def test_plot_single_metric_horizontal(self, display_single_metric):
        """Horizontal orientation for single metric."""
        display_single_metric.plot(orientation="horizontal")

        assert display_single_metric.ax_ is not None
        plt.close(display_single_metric.figure_)

    def test_plot_multi_metric_grouped(self, display_multi_metric):
        """Multiple metrics shown as grouped bars."""
        display_multi_metric.plot()

        assert display_multi_metric.ax_ is not None
        # Check for legend (needed for multi-metric)
        legend = display_multi_metric.ax_.get_legend()
        assert legend is not None
        plt.close(display_multi_metric.figure_)

    def test_plot_with_values(self, display_single_metric):
        """Values shown on bars."""
        display_single_metric.plot(show_values=True)

        texts = display_single_metric.ax_.texts
        # Should have value labels
        assert len(texts) > 0
        plt.close(display_single_metric.figure_)

    def test_plot_with_custom_title(self, display_single_metric):
        """Custom title is applied."""
        display_single_metric.plot(title="MAE Comparison")

        # Tufte-style uses left-aligned titles
        assert display_single_metric.ax_.get_title(loc="left") == "MAE Comparison"
        plt.close(display_single_metric.figure_)

    def test_plot_specific_metric(self, display_multi_metric):
        """Plot only a specific metric from multi-metric display."""
        display_multi_metric.plot(metric_idx=0)  # Plot only MAE

        assert display_multi_metric.ax_ is not None
        plt.close(display_multi_metric.figure_)


class TestMetricComparisonDisplayPlotRelative:
    """Tests for MetricComparisonDisplay.plot_relative method."""

    @pytest.fixture
    def display(self):
        """Create display with baseline."""
        return MetricComparisonDisplay.from_dict(
            {
                "Baseline": {"MAE": 0.20},
                "Model A": {"MAE": 0.15},  # 25% improvement
                "Model B": {"MAE": 0.10},  # 50% improvement
            },
            baseline="Baseline",
        )

    def test_plot_relative_creates_figure(self, display):
        """plot_relative creates figure."""
        display.plot_relative()

        assert hasattr(display, "ax_")
        plt.close(display.figure_)

    def test_plot_relative_requires_baseline(self):
        """plot_relative raises if no baseline set."""
        display = MetricComparisonDisplay.from_dict({"A": {"MAE": 0.15}, "B": {"MAE": 0.12}})

        with pytest.raises(ValueError, match="baseline_idx must be set"):
            display.plot_relative()

    def test_plot_relative_shows_improvement(self, display):
        """Relative plot shows improvement percentages."""
        display.plot_relative()

        # Check for percentage text labels
        texts = [t.get_text() for t in display.ax_.texts]
        assert any("%" in t for t in texts)
        plt.close(display.figure_)


class TestPlotMetricComparisonFunction:
    """Tests for plot_metric_comparison function API."""

    def test_plot_metric_comparison_basic(self):
        """Basic usage of plot_metric_comparison."""
        results = {
            "Model A": {"MAE": 0.15},
            "Model B": {"MAE": 0.12},
        }

        ax = plot_metric_comparison(results)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_plot_metric_comparison_with_baseline(self):
        """With baseline specified."""
        results = {
            "Baseline": {"MAE": 0.20},
            "Model A": {"MAE": 0.15},
        }

        ax = plot_metric_comparison(results, baseline="Baseline")

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_plot_metric_comparison_with_title(self):
        """Custom title applied."""
        results = {"A": {"MAE": 0.1}, "B": {"MAE": 0.2}}

        ax = plot_metric_comparison(results, title="Custom Title")

        # Tufte-style uses left-aligned titles
        assert ax.get_title(loc="left") == "Custom Title"
        plt.close(ax.figure)
