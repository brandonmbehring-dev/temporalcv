"""
Tests for temporalcv.viz.gates module.

Validates GateResultDisplay and GateComparisonDisplay classes.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytest

from temporalcv.viz import (
    GateComparisonDisplay,
    GateResultDisplay,
    plot_gate_comparison,
    plot_gate_result,
)


# Mock gate result classes for testing
class MockGateStatus(Enum):
    HALT = "HALT"
    WARN = "WARN"
    PASS = "PASS"


@dataclass
class MockGateResult:
    """Mock GateResult for testing without running actual gates."""

    gate_name: str
    status: MockGateStatus
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class MockGateReport:
    """Mock GateReport for testing."""

    results: list
    passed: bool = True
    halted: bool = False


class TestGateResultDisplayInit:
    """Tests for GateResultDisplay initialization."""

    def test_init_with_basic_params(self):
        """Initialize with name, status, message."""
        display = GateResultDisplay(
            name="signal_verification",
            status="PASS",
            message="Signal detected (p < 0.05)",
        )

        assert display.name == "signal_verification"
        assert display.status == "PASS"
        assert display.message == "Signal detected (p < 0.05)"

    def test_init_with_metrics(self):
        """Initialize with additional metrics."""
        metrics = {"p_value": 0.023, "n_shuffles": 100}
        display = GateResultDisplay(
            name="signal_verification",
            status="PASS",
            message="Signal detected",
            metrics=metrics,
        )

        assert display.metrics == metrics

    def test_status_uppercase(self):
        """Status is uppercased."""
        display = GateResultDisplay(
            name="test_gate",
            status="pass",  # lowercase
            message="Test message",
        )

        assert display.status == "PASS"


class TestGateResultDisplayFromGate:
    """Tests for GateResultDisplay.from_gate factory method."""

    def test_from_gate_pass(self):
        """Create display from PASS gate result."""
        gate_result = MockGateResult(
            gate_name="signal_verification",
            status=MockGateStatus.PASS,
            message="Signal detected",
        )

        display = GateResultDisplay.from_gate(gate_result)

        assert display.name == "signal_verification"
        assert display.status == "PASS"

    def test_from_gate_halt(self):
        """Create display from HALT gate result."""
        gate_result = MockGateResult(
            gate_name="suspicious_improvement",
            status=MockGateStatus.HALT,
            message="Improvement too large",
        )

        display = GateResultDisplay.from_gate(gate_result)

        assert display.status == "HALT"

    def test_from_gate_with_details(self):
        """Details are extracted as metrics."""
        gate_result = MockGateResult(
            gate_name="test_gate",
            status=MockGateStatus.PASS,
            message="Test",
            details={"metric": 0.5},
        )

        display = GateResultDisplay.from_gate(gate_result)

        assert display.metrics == {"metric": 0.5}


class TestGateResultDisplayPlot:
    """Tests for GateResultDisplay.plot method."""

    @pytest.fixture
    def display(self):
        """Create a simple display."""
        return GateResultDisplay(
            name="signal_verification",
            status="PASS",
            message="Signal detected (p=0.023)",
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

    def test_plot_shows_status(self, display):
        """Status text is visible in plot."""
        display.plot()

        # Check that status text exists
        texts = [t.get_text() for t in display.ax_.texts]
        assert "PASS" in texts
        plt.close(display.figure_)

    def test_plot_shows_message(self, display):
        """Message is shown when show_message=True."""
        display.plot(show_message=True)

        texts = [t.get_text() for t in display.ax_.texts]
        # Message might be truncated
        assert any("Signal" in t for t in texts)
        plt.close(display.figure_)


class TestGateComparisonDisplayInit:
    """Tests for GateComparisonDisplay initialization."""

    def test_init_with_basic_params(self):
        """Initialize with names and statuses."""
        names = ["gate1", "gate2", "gate3"]
        statuses = ["PASS", "WARN", "HALT"]

        display = GateComparisonDisplay(names, statuses)

        assert display.n_gates == 3
        assert display.names == names
        assert display.statuses == ["PASS", "WARN", "HALT"]

    def test_init_with_messages(self):
        """Initialize with optional messages."""
        names = ["gate1", "gate2"]
        statuses = ["PASS", "HALT"]
        messages = ["All good", "Problem detected"]

        display = GateComparisonDisplay(names, statuses, messages=messages)

        assert display.messages == messages


class TestGateComparisonDisplayFromGates:
    """Tests for GateComparisonDisplay.from_gates factory method."""

    def test_from_gates_list(self):
        """Create display from list of gate results."""
        gates = [
            MockGateResult("gate1", MockGateStatus.PASS, "OK"),
            MockGateResult("gate2", MockGateStatus.WARN, "Warning"),
            MockGateResult("gate3", MockGateStatus.HALT, "Stop"),
        ]

        display = GateComparisonDisplay.from_gates(gates)

        assert display.n_gates == 3
        assert display.statuses == ["PASS", "WARN", "HALT"]


class TestGateComparisonDisplayFromReport:
    """Tests for GateComparisonDisplay.from_report factory method."""

    def test_from_report(self):
        """Create display from GateReport."""
        gates = [
            MockGateResult("gate1", MockGateStatus.PASS, "OK"),
            MockGateResult("gate2", MockGateStatus.PASS, "OK"),
        ]
        report = MockGateReport(results=gates, passed=True)

        display = GateComparisonDisplay.from_report(report)

        assert display.n_gates == 2


class TestGateComparisonDisplayPlot:
    """Tests for GateComparisonDisplay.plot method."""

    @pytest.fixture
    def display(self):
        """Create a comparison display."""
        names = ["signal_verification", "suspicious_improvement", "shuffled_target"]
        statuses = ["PASS", "WARN", "HALT"]
        return GateComparisonDisplay(names, statuses)

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

    def test_plot_horizontal_orientation(self, display):
        """Horizontal orientation works."""
        display.plot(orientation="horizontal")

        assert display.ax_ is not None
        plt.close(display.figure_)

    def test_plot_vertical_orientation(self, display):
        """Vertical orientation works."""
        display.plot(orientation="vertical")

        assert display.ax_ is not None
        plt.close(display.figure_)

    def test_plot_with_title(self, display):
        """Custom title is applied."""
        display.plot(title="Validation Gates")

        # Tufte-style uses left-aligned titles
        assert display.ax_.get_title(loc="left") == "Validation Gates"
        plt.close(display.figure_)


class TestPlotGateFunctions:
    """Tests for function API."""

    def test_plot_gate_result(self):
        """plot_gate_result function works."""
        gate = MockGateResult("test_gate", MockGateStatus.PASS, "OK")

        ax = plot_gate_result(gate)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_plot_gate_comparison_from_list(self):
        """plot_gate_comparison works with list of gates."""
        gates = [
            MockGateResult("gate1", MockGateStatus.PASS, "OK"),
            MockGateResult("gate2", MockGateStatus.HALT, "Stop"),
        ]

        ax = plot_gate_comparison(gates)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_plot_gate_comparison_from_report(self):
        """plot_gate_comparison works with GateReport."""
        gates = [
            MockGateResult("gate1", MockGateStatus.PASS, "OK"),
            MockGateResult("gate2", MockGateStatus.PASS, "OK"),
        ]
        report = MockGateReport(results=gates)

        ax = plot_gate_comparison(report)

        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)
