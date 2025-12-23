"""
Tests for regime-stratified validation reports.

Tests the run_gates_stratified() function and StratifiedValidationReport.

Knowledge Tier: [T2] - Regime-conditional evaluation from myga-forecasting-v4
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import (
    GateResult,
    GateStatus,
    StratifiedValidationReport,
    ValidationReport,
    gate_suspicious_improvement,
    run_gates_stratified,
)


class TestRunGatesStratifiedBasic:
    """Basic functionality tests for run_gates_stratified."""

    def test_no_stratification_returns_overall_only(self) -> None:
        """With regimes=None, should return overall only."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        assert isinstance(report, StratifiedValidationReport)
        assert len(report.by_regime) == 0
        assert len(report.regime_counts) == 0
        assert len(report.masked_regimes) == 0

    def test_auto_regime_classification(self) -> None:
        """With regimes='auto', should auto-classify volatility regimes."""
        rng = np.random.default_rng(42)
        # Create data with clear volatility regimes
        n = 100
        actuals = np.concatenate([
            rng.standard_normal(50) * 0.1,  # Low volatility
            rng.standard_normal(50) * 2.0,  # High volatility
        ])
        predictions = actuals + rng.standard_normal(n) * 0.1

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        report = run_gates_stratified(
            overall_gates, actuals, predictions, regimes="auto"
        )

        # Should have regime stratification
        assert len(report.by_regime) > 0 or len(report.masked_regimes) > 0

    def test_explicit_regime_labels(self) -> None:
        """With explicit regime array, should use those labels."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        predictions = np.array([1.1, 2.1, 3.1, 4.1, 5.1] * 10)
        regimes = np.array(["A", "A", "B", "B", "A"] * 10)

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        report = run_gates_stratified(
            overall_gates, actuals, predictions, regimes=regimes
        )

        assert "A" in report.by_regime or "A" in report.masked_regimes
        assert "B" in report.by_regime or "B" in report.masked_regimes


class TestRegimeCounts:
    """Tests for regime counting and masking."""

    def test_regime_counts_correct(self) -> None:
        """Regime counts should match actual data."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        predictions = actuals * 1.1
        regimes = np.array(["A"] * 30 + ["B"] * 20)

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        report = run_gates_stratified(
            overall_gates, actuals, predictions, regimes=regimes
        )

        # A has 30, B has 20
        if "A" in report.regime_counts:
            assert report.regime_counts["A"] == 30
        if "B" in report.regime_counts:
            assert report.regime_counts["B"] == 20

    def test_low_n_regimes_masked(self) -> None:
        """Regimes with fewer than min_n samples should be masked."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        predictions = actuals * 1.1
        # A has 45 samples, B has only 5
        regimes = np.array(["A"] * 45 + ["B"] * 5)

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        report = run_gates_stratified(
            overall_gates,
            actuals,
            predictions,
            regimes=regimes,
            min_n_per_regime=10,
        )

        # B should be masked (only 5 samples)
        assert "B" in report.masked_regimes
        assert "B" not in report.by_regime

    def test_custom_min_n(self) -> None:
        """Custom min_n_per_regime should be respected."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        predictions = actuals * 1.1
        regimes = np.array(["A"] * 45 + ["B"] * 5)

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        # With min_n=3, B should NOT be masked
        report = run_gates_stratified(
            overall_gates,
            actuals,
            predictions,
            regimes=regimes,
            min_n_per_regime=3,
        )

        # B should NOT be masked (5 >= 3)
        assert "B" not in report.masked_regimes


class TestStatusProperty:
    """Tests for the status property."""

    def test_pass_status(self) -> None:
        """Status should be PASS if all gates pass."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        assert report.status == "PASS"

    def test_halt_from_overall(self) -> None:
        """Status should be HALT if overall has HALT."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.HALT,
                message="Critical failure",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        assert report.status == "HALT"

    def test_warn_from_overall(self) -> None:
        """Status should be WARN if overall has WARN but no HALT."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.WARN,
                message="Warning",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        assert report.status == "WARN"


class TestPerRegimeGates:
    """Tests for per-regime gate execution."""

    def test_suspicious_improvement_per_regime(self) -> None:
        """gate_suspicious_improvement should run per regime."""
        # Create data where one regime has suspicious improvement
        n = 100
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        # Normal predictions for regime A, suspiciously good for B
        predictions_a = actuals[:60] + np.random.default_rng(42).standard_normal(60) * 0.5
        predictions_b = actuals[60:] * 1.001  # Almost perfect
        predictions = np.concatenate([predictions_a, predictions_b])
        regimes = np.array(["A"] * 60 + ["B"] * 40)

        # Create overall gates - need to compute actual MAEs
        persistence_mae = np.mean(np.abs(np.diff(actuals)))
        model_mae = np.mean(np.abs(actuals - predictions))

        overall_gates = [
            gate_suspicious_improvement(model_mae, persistence_mae)
        ]

        report = run_gates_stratified(
            overall_gates,
            actuals,
            predictions,
            regimes=regimes,
        )

        # Should have per-regime results
        assert len(report.by_regime) > 0


class TestSummaryMethod:
    """Tests for the summary() method."""

    def test_summary_includes_overall(self) -> None:
        """Summary should include overall status."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)
        summary = report.summary()

        assert "Overall" in summary or "overall" in summary.lower()

    def test_summary_includes_regime_info(self) -> None:
        """Summary should include regime information when stratified."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        predictions = actuals * 1.1
        regimes = np.array(["A"] * 30 + ["B"] * 20)

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        report = run_gates_stratified(
            overall_gates, actuals, predictions, regimes=regimes
        )
        summary = report.summary()

        # Should mention regimes or stratification
        assert "regime" in summary.lower() or "A" in summary or "B" in summary


class TestEdgeCases:
    """Edge case tests for stratified reports."""

    def test_all_samples_in_one_regime(self) -> None:
        """Should handle all samples in one regime."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
        predictions = actuals * 1.1
        regimes = np.array(["A"] * 50)

        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]

        report = run_gates_stratified(
            overall_gates, actuals, predictions, regimes=regimes
        )

        # Should have only one regime
        assert len(report.by_regime) + len(report.masked_regimes) == 1

    def test_empty_overall_gates(self) -> None:
        """Should handle empty overall gates list."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified([], actuals, predictions, regimes=None)

        assert len(report.overall.gates) == 0
        assert report.status == "PASS"

    def test_mixed_status_overall(self) -> None:
        """Should handle mixed statuses in overall gates."""
        overall_gates = [
            GateResult(
                name="pass_gate",
                status=GateStatus.PASS,
                message="Passed",
                details={},
            ),
            GateResult(
                name="warn_gate",
                status=GateStatus.WARN,
                message="Warning",
                details={},
            ),
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        # WARN should propagate
        assert report.status == "WARN"


class TestDataclassProperties:
    """Tests for StratifiedValidationReport dataclass."""

    def test_all_fields_present(self) -> None:
        """Report should have all expected fields."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        assert hasattr(report, "overall")
        assert hasattr(report, "by_regime")
        assert hasattr(report, "regime_counts")
        assert hasattr(report, "masked_regimes")
        assert hasattr(report, "status")
        assert hasattr(report, "summary")

    def test_overall_is_validation_report(self) -> None:
        """Overall should be a ValidationReport instance."""
        overall_gates = [
            GateResult(
                name="test_gate",
                status=GateStatus.PASS,
                message="Test passed",
                details={},
            )
        ]
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = actuals * 1.1

        report = run_gates_stratified(overall_gates, actuals, predictions, regimes=None)

        assert isinstance(report.overall, ValidationReport)
