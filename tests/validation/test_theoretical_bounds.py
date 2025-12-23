"""
Tests for gate_theoretical_bounds.

Tests the AR(1) theoretical minimum detection gate.

Knowledge Tier: [T1] - Based on standard AR(1) theory
References:
- E[|N(0,σ²)|] = σ * sqrt(2/π) ≈ 0.7979σ
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import (
    GateStatus,
    gate_theoretical_bounds,
)


class TestTheoreticalBoundsBasic:
    """Basic functionality tests for theoretical bounds gate."""

    def test_plausible_mae_passes(self) -> None:
        """
        Model MAE above theoretical minimum should PASS.

        Generate AR(1) data and verify a reasonable MAE passes.
        """
        rng = np.random.default_rng(42)
        n = 200
        phi = 0.7
        sigma = 1.0

        # Generate AR(1) series
        y = np.zeros(n)
        y[0] = rng.standard_normal() * sigma / np.sqrt(1 - phi**2)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + sigma * rng.standard_normal()

        # Theoretical MAE for innovations
        theoretical_mae = sigma * np.sqrt(2 / np.pi)

        # Model MAE above theoretical - should PASS
        model_mae = theoretical_mae * 1.5  # 50% above theoretical

        result = gate_theoretical_bounds(model_mae, y)

        assert result.status == GateStatus.PASS, (
            f"MAE above theoretical should pass, got {result.status}"
        )
        assert "phi_estimate" in result.details
        assert "theoretical_mae" in result.details

    def test_impossible_mae_halts(self) -> None:
        """
        Model MAE below theoretical minimum should HALT.

        This indicates potential leakage or evaluation error.
        """
        rng = np.random.default_rng(42)
        n = 200
        phi = 0.7
        sigma = 1.0

        # Generate AR(1) series
        y = np.zeros(n)
        y[0] = rng.standard_normal() * sigma / np.sqrt(1 - phi**2)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + sigma * rng.standard_normal()

        # "Impossible" MAE - way below theoretical
        model_mae = 0.01  # Impossibly good

        result = gate_theoretical_bounds(model_mae, y)

        assert result.status == GateStatus.HALT, (
            f"Impossible MAE should HALT, got {result.status}"
        )
        assert "beats theoretical minimum" in result.message
        assert result.recommendation is not None
        assert "leakage" in result.recommendation.lower()

    def test_insufficient_data_skips(self) -> None:
        """Gate should SKIP with insufficient data (n < 30)."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(20)  # Too few samples

        result = gate_theoretical_bounds(model_mae=0.5, y_train=y)

        assert result.status == GateStatus.SKIP
        assert "insufficient" in result.message.lower()
        assert result.details["n_samples"] == 20

    def test_tolerance_parameter(self) -> None:
        """Custom tolerance should affect threshold."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)

        # Get result to find theoretical MAE
        result_default = gate_theoretical_bounds(model_mae=0.5, y_train=y)
        theoretical = result_default.details["theoretical_mae"]

        # MAE at 95% of theoretical
        model_mae = theoretical * 0.95

        # With 10% tolerance (default), should PASS (0.95 > 0.90)
        result_10 = gate_theoretical_bounds(model_mae, y, tolerance=0.10)
        assert result_10.status == GateStatus.PASS

        # With 3% tolerance, should HALT (0.95 < 0.97)
        result_3 = gate_theoretical_bounds(model_mae, y, tolerance=0.03)
        assert result_3.status == GateStatus.HALT


class TestTheoreticalBoundsTheory:
    """Tests verifying correct AR(1) theory implementation."""

    def test_phi_estimation_known_value(self) -> None:
        """
        ACF(1) estimation should be close to true phi.

        Generate AR(1) with known phi and verify estimation.
        """
        rng = np.random.default_rng(42)
        n = 1000  # Large sample for accurate estimation
        true_phi = 0.8
        sigma = 1.0

        # Generate AR(1)
        y = np.zeros(n)
        y[0] = rng.standard_normal() * sigma / np.sqrt(1 - true_phi**2)
        for t in range(1, n):
            y[t] = true_phi * y[t - 1] + sigma * rng.standard_normal()

        result = gate_theoretical_bounds(model_mae=1.0, y_train=y)

        estimated_phi = result.details["phi_estimate"]

        # Allow 10% relative error
        assert abs(estimated_phi - true_phi) / true_phi < 0.10, (
            f"Estimated phi ({estimated_phi:.3f}) should be close to "
            f"true phi ({true_phi})"
        )

    def test_innovation_sigma_estimation(self) -> None:
        """
        Innovation sigma should be estimated from residuals, not series.

        For AR(1): y_t = phi * y_{t-1} + epsilon_t
        sigma_innovation = std(epsilon_t), NOT std(y_t)
        """
        rng = np.random.default_rng(42)
        n = 1000
        true_phi = 0.9  # High persistence
        true_sigma = 0.5

        # Generate AR(1)
        y = np.zeros(n)
        y[0] = rng.standard_normal() * true_sigma / np.sqrt(1 - true_phi**2)
        for t in range(1, n):
            y[t] = true_phi * y[t - 1] + true_sigma * rng.standard_normal()

        result = gate_theoretical_bounds(model_mae=1.0, y_train=y)

        estimated_sigma = result.details["sigma_innovation"]

        # Series variance >> innovation variance for high phi
        series_std = np.std(y)
        assert estimated_sigma < series_std, (
            "Innovation sigma should be smaller than series std for high phi"
        )

        # Should be close to true innovation sigma
        assert abs(estimated_sigma - true_sigma) / true_sigma < 0.15, (
            f"Estimated sigma ({estimated_sigma:.3f}) should be close to "
            f"true sigma ({true_sigma})"
        )

    def test_theoretical_mae_formula(self) -> None:
        """
        theoretical_mae = sigma_innovation * sqrt(2/pi).

        Verify the formula is correctly applied.
        """
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)

        result = gate_theoretical_bounds(model_mae=0.5, y_train=y)

        sigma = result.details["sigma_innovation"]
        theoretical = result.details["theoretical_mae"]

        expected = sigma * np.sqrt(2 / np.pi)

        assert abs(theoretical - expected) < 1e-10, (
            f"Theoretical MAE ({theoretical}) should equal "
            f"sigma * sqrt(2/pi) ({expected})"
        )


class TestAR1AssumptionWarning:
    """Tests for AR(1) assumption checking via Ljung-Box."""

    def test_ar1_data_no_warning(self) -> None:
        """
        True AR(1) data should not trigger assumption warning.
        """
        rng = np.random.default_rng(42)
        n = 200
        phi = 0.7

        # Generate AR(1)
        y = np.zeros(n)
        y[0] = rng.standard_normal()
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.standard_normal()

        result = gate_theoretical_bounds(model_mae=1.0, y_train=y)

        # AR(1) residuals should be white noise
        assert result.details["ar1_assumption_warning"] is False, (
            "AR(1) data should not trigger assumption warning"
        )

    def test_ar2_data_triggers_warning(self) -> None:
        """
        AR(2) data should trigger AR(1) assumption warning.

        AR(1) model can't capture second-order dynamics.
        """
        rng = np.random.default_rng(42)
        n = 300
        phi1 = 0.6
        phi2 = 0.3

        # Generate AR(2)
        y = np.zeros(n)
        y[0] = rng.standard_normal()
        y[1] = phi1 * y[0] + rng.standard_normal()
        for t in range(2, n):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.standard_normal()

        result = gate_theoretical_bounds(model_mae=1.0, y_train=y)

        # AR(1) residuals should show autocorrelation
        assert result.details["ar1_assumption_warning"] is True, (
            "AR(2) data should trigger AR(1) assumption warning"
        )
        assert "ar1_assumption_message" in result.details

    def test_warning_included_in_pass_message(self) -> None:
        """
        When AR(1) assumption fails but MAE is valid, message should note it.
        """
        rng = np.random.default_rng(42)
        n = 300
        phi1 = 0.6
        phi2 = 0.3

        # Generate AR(2)
        y = np.zeros(n)
        y[0] = rng.standard_normal()
        y[1] = phi1 * y[0] + rng.standard_normal()
        for t in range(2, n):
            y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.standard_normal()

        # Use high MAE so we get PASS
        result = gate_theoretical_bounds(model_mae=10.0, y_train=y)

        if result.details["ar1_assumption_warning"]:
            assert "may not hold" in result.message, (
                "Pass message should note AR(1) assumption caveat"
            )


class TestTheoreticalBoundsEdgeCases:
    """Edge case tests for theoretical bounds gate."""

    def test_constant_series(self) -> None:
        """Constant series should handle gracefully."""
        y = np.ones(50)

        result = gate_theoretical_bounds(model_mae=0.5, y_train=y)

        # Constant series has zero innovation - should handle
        assert result.status in (GateStatus.PASS, GateStatus.WARN, GateStatus.SKIP)

    def test_trending_series(self) -> None:
        """Trending series should work but may trigger warning."""
        n = 100
        y = np.arange(n, dtype=float) + np.random.default_rng(42).standard_normal(n) * 0.1

        result = gate_theoretical_bounds(model_mae=1.0, y_train=y)

        # Should complete without error
        assert result.status in (GateStatus.PASS, GateStatus.WARN, GateStatus.HALT)

    def test_minimum_valid_sample_size(self) -> None:
        """Test with exactly 30 samples (minimum)."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(30)

        result = gate_theoretical_bounds(model_mae=1.0, y_train=y)

        assert result.status != GateStatus.SKIP
        assert result.details["n_samples"] == 30

    def test_details_structure(self) -> None:
        """Result details should have expected structure."""
        rng = np.random.default_rng(42)
        y = rng.standard_normal(100)

        result = gate_theoretical_bounds(model_mae=0.5, y_train=y)

        # Check structure
        assert "phi_estimate" in result.details
        assert "sigma_innovation" in result.details
        assert "theoretical_mae" in result.details
        assert "threshold" in result.details
        assert "tolerance" in result.details
        assert "n_samples" in result.details
        assert "ar1_assumption_warning" in result.details

        # Check types
        assert isinstance(result.details["phi_estimate"], float)
        assert isinstance(result.details["sigma_innovation"], float)
        assert isinstance(result.details["ar1_assumption_warning"], bool)
