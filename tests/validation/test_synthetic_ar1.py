"""
Validation tests for synthetic AR(1) gate.

The synthetic AR(1) test validates model performance against theoretical bounds.
For an AR(1) process with known parameters (phi, sigma), the optimal predictor's
MAE has a theoretical minimum: sigma * sqrt(2/pi).

A model beating this bound is exploiting information beyond what's theoretically
possible, indicating leakage.

Knowledge Tier: [T1] - Based on known AR(1) properties
Reference: See docs/knowledge/mathematical_foundations.md
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import (
    GateStatus,
    gate_synthetic_ar1,
)


class MockMeanPredictor:
    """Predicts mean of training y."""

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockMeanPredictor":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)


class MockOptimalAR1Predictor:
    """
    Optimal predictor for AR(1): y_t = phi * y_{t-1}.

    This is the best possible predictor given proper features.
    """

    def __init__(self, phi: float = 0.95) -> None:
        self.phi = phi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockOptimalAR1Predictor":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Assume first column is lag-1
        return self.phi * X[:, 0]


class TestSyntheticAR1Validation:
    """Validation tests for synthetic AR(1) gate correctness."""

    def test_mean_predictor_passes(self) -> None:
        """
        Mean predictor should PASS.

        A mean predictor is worse than optimal, so it should not
        trigger a theoretical bound violation.
        """
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.95,
            sigma=1.0,
            n_samples=200,
            random_state=42,
        )

        assert result.status == GateStatus.PASS, (
            f"Mean predictor should pass, got {result.status}. "
            f"Ratio: {result.metric_value}"
        )

    def test_optimal_predictor_passes(self) -> None:
        """
        Optimal lag-1 predictor should PASS.

        An optimal predictor should be close to (but not beat)
        the theoretical bound.
        """
        model = MockOptimalAR1Predictor(phi=0.95)

        result = gate_synthetic_ar1(
            model=model,
            phi=0.95,
            sigma=1.0,
            n_samples=500,  # Large sample for stable estimate
            n_lags=3,
            random_state=42,
        )

        assert result.status == GateStatus.PASS, (
            f"Optimal predictor should pass, got {result.status}. "
            f"Ratio: {result.metric_value}"
        )

        # Ratio should be near 1.0 (within reasonable tolerance)
        assert result.metric_value is not None
        assert 0.5 < result.metric_value < 2.0, (
            f"Ratio {result.metric_value} too far from 1.0"
        )

    def test_theoretical_mae_calculation(self) -> None:
        """
        Theoretical MAE should be sigma * sqrt(2/pi).

        This is the MAE of the residuals for the optimal predictor.
        """
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.9,
            sigma=2.5,
            random_state=42,
        )

        expected_theoretical = 2.5 * np.sqrt(2 / np.pi)
        actual_theoretical = result.details["theoretical_mae"]

        assert abs(actual_theoretical - expected_theoretical) < 0.01, (
            f"Theoretical MAE: expected {expected_theoretical:.4f}, "
            f"got {actual_theoretical:.4f}"
        )


class TestSyntheticAR1Parameters:
    """Tests for AR(1) parameter handling."""

    def test_phi_stored_in_details(self) -> None:
        """AR(1) coefficient phi should be in result details."""
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.85,
            sigma=1.0,
            random_state=42,
        )

        assert result.details["phi"] == 0.85

    def test_sigma_stored_in_details(self) -> None:
        """Innovation standard deviation sigma should be in details."""
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.9,
            sigma=1.5,
            random_state=42,
        )

        assert result.details["sigma"] == 1.5

    def test_different_phi_values(self) -> None:
        """Gate should work for various phi values."""
        model = MockMeanPredictor()

        for phi in [0.1, 0.5, 0.9, 0.99]:
            result = gate_synthetic_ar1(
                model=model,
                phi=phi,
                sigma=1.0,
                n_samples=100,
                random_state=42,
            )

            assert result.status in (
                GateStatus.PASS, GateStatus.WARN, GateStatus.HALT
            ), f"Invalid status for phi={phi}"


class TestSyntheticAR1Statistics:
    """Tests for statistical properties of synthetic AR(1) validation."""

    def test_model_mae_in_details(self) -> None:
        """Model MAE should be reported in details."""
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.9,
            sigma=1.0,
            random_state=42,
        )

        assert "model_mae" in result.details
        assert result.details["model_mae"] > 0

    def test_ratio_is_model_over_theoretical(self) -> None:
        """
        metric_value should be model_mae / theoretical_mae.

        Ratio < 1 means model is "too good" (suspicious).
        Ratio near 1 means model matches theory.
        Ratio > 1 means model is worse than optimal.
        """
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.9,
            sigma=1.0,
            random_state=42,
        )

        model_mae = result.details["model_mae"]
        theoretical_mae = result.details["theoretical_mae"]
        expected_ratio = model_mae / theoretical_mae

        assert abs(result.metric_value - expected_ratio) < 0.001

    def test_reproducibility(self) -> None:
        """Same random_state should give same results."""
        model = MockMeanPredictor()

        result1 = gate_synthetic_ar1(model, phi=0.9, sigma=1.0, random_state=123)
        result2 = gate_synthetic_ar1(model, phi=0.9, sigma=1.0, random_state=123)

        assert result1.metric_value == result2.metric_value
        assert result1.details["model_mae"] == result2.details["model_mae"]


class TestSyntheticAR1EdgeCases:
    """Edge case tests for synthetic AR(1) gate."""

    def test_high_persistence(self) -> None:
        """Should handle high-persistence (phi close to 1)."""
        model = MockOptimalAR1Predictor(phi=0.99)

        result = gate_synthetic_ar1(
            model=model,
            phi=0.99,
            sigma=1.0,
            n_samples=200,
            random_state=42,
        )

        # Should complete without error
        assert result.status in (GateStatus.PASS, GateStatus.WARN, GateStatus.HALT)

    def test_low_persistence(self) -> None:
        """Should handle low-persistence (phi close to 0)."""
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.1,
            sigma=1.0,
            n_samples=200,
            random_state=42,
        )

        # For phi near 0, mean predictor is nearly optimal
        assert result.status in (GateStatus.PASS, GateStatus.WARN)

    def test_small_sample(self) -> None:
        """Should handle small samples with appropriate warning."""
        model = MockMeanPredictor()

        result = gate_synthetic_ar1(
            model=model,
            phi=0.9,
            sigma=1.0,
            n_samples=50,  # Small
            random_state=42,
        )

        # Should complete (may warn about sample size)
        assert result.status is not None
