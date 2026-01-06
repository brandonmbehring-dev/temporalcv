"""
Anti-pattern tests for lag leakage detection.

Lag leakage occurs when future information leaks into features through:
- Improper lagging (using t instead of t-1)
- Feature engineering without proper temporal alignment
- Look-ahead bias in preprocessing

These tests verify that temporalcv gates correctly detect these patterns.

Bug Category: #1 from lever_of_archimedes/patterns/data_leakage_prevention.md
Gate: gate_signal_verification
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import (
    GateStatus,
    gate_signal_verification,
)


class LeakyModel:
    """
    Model that simulates lag leakage.

    This model learns a strong X->y mapping that only works when
    there's an illegitimate relationship between features and target.
    """

    def __init__(self) -> None:
        self._coeffs: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LeakyModel":
        """Fit using least squares."""
        X = np.asarray(X)
        y = np.asarray(y)
        XtX = X.T @ X + 0.01 * np.eye(X.shape[1])
        Xty = X.T @ y
        self._coeffs = np.linalg.solve(XtX, Xty)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using learned coefficients."""
        if self._coeffs is None:
            return np.zeros(len(X))
        return X @ self._coeffs


class TestLagLeakageDetection:
    """Tests that verify lag leakage is detected by gates."""

    def test_detects_strong_xy_relationship(self) -> None:
        """
        Scenario: Features are constructed with future information.

        When X is a direct function of y (simulating lag leakage),
        the shuffled target test should HALT.
        """
        rng = np.random.default_rng(42)
        n = 100

        # Simulate lag leakage: X contains transformed y
        # This represents improperly constructed features
        y = rng.standard_normal(n)
        noise = rng.standard_normal((n, 3)) * 0.1
        X = np.column_stack([
            y + noise[:, 0],      # Leaked feature 1
            y * 0.5 + noise[:, 1], # Leaked feature 2
            noise[:, 2],           # Noise feature
        ])

        model = LeakyModel()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=5,
            threshold=0.05,
            method="effect_size",  # Use effect_size mode to test improvement ratio
            random_state=42,
        )

        assert result.status == GateStatus.HALT, (
            f"Expected HALT for leaked data, got {result.status}. "
            f"Improvement ratio: {result.metric_value}"
        )

    def test_detects_perfect_prediction(self) -> None:
        """
        Scenario: Model achieves near-perfect prediction (classic leakage sign).

        When model MAE is suspiciously low compared to shuffled,
        should trigger HALT.
        """
        rng = np.random.default_rng(123)
        n = 100

        # Create data where y is almost perfectly predictable from X
        X = rng.standard_normal((n, 5))
        true_coeffs = np.array([2.0, -1.5, 1.0, 0.5, -0.8])
        y = X @ true_coeffs + rng.standard_normal(n) * 0.05  # Very low noise

        model = LeakyModel()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            threshold=0.05,
            method="effect_size",  # Use effect_size mode to test improvement ratio
            random_state=42,
        )

        # This should catch the suspiciously good fit
        assert result.status == GateStatus.HALT
        assert result.metric_value is not None
        assert result.metric_value > 0.05, (
            f"Expected significant improvement, got {result.metric_value}"
        )

    def test_no_false_positive_on_random_data(self) -> None:
        """
        Scenario: Truly random features with no relationship to target.

        Should NOT trigger HALT when there's no actual leakage.
        """
        rng = np.random.default_rng(42)
        n = 100

        # Completely random data - no relationship
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)

        model = LeakyModel()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=5,
            threshold=0.05,
            method="effect_size",  # Use effect_size mode to test threshold
            random_state=42,
        )

        # Should pass or warn, not halt
        assert result.status in (GateStatus.PASS, GateStatus.WARN), (
            f"False positive: got {result.status} on random data"
        )

    def test_threshold_sensitivity(self) -> None:
        """
        Verify that threshold parameter controls detection sensitivity.
        """
        rng = np.random.default_rng(42)
        n = 100

        # Moderate relationship - should be threshold-dependent
        X = rng.standard_normal((n, 5))
        y = X[:, 0] * 0.5 + rng.standard_normal(n) * 0.5

        model = LeakyModel()

        # Strict threshold should catch
        strict = gate_signal_verification(
            model, X, y, n_shuffles=3, threshold=0.01,
            method="effect_size", random_state=42
        )

        # Lenient threshold might not
        lenient = gate_signal_verification(
            model, X, y, n_shuffles=3, threshold=0.50,
            method="effect_size", random_state=42
        )

        # Strict should be at least as severe as lenient
        severity = {GateStatus.HALT: 2, GateStatus.WARN: 1, GateStatus.PASS: 0}
        assert severity.get(strict.status, 0) >= severity.get(lenient.status, 0)


class TestLagLeakageMetrics:
    """Tests for metrics reported by lag leakage detection."""

    def test_reports_mae_values(self) -> None:
        """Gate should report both real and shuffled MAE values."""
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = LeakyModel()

        result = gate_signal_verification(
            model, X, y, n_shuffles=3, method="effect_size", random_state=42
        )

        assert "mae_real" in result.details
        assert "mae_shuffled_avg" in result.details
        assert "mae_shuffled_all" in result.details
        assert "n_shuffles" in result.details

        # MAE values should be positive
        assert result.details["mae_real"] > 0
        assert result.details["mae_shuffled_avg"] > 0

    def test_reproducible_with_seed(self) -> None:
        """Results should be reproducible with same random_state."""
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = LeakyModel()

        result1 = gate_signal_verification(
            model, X, y, n_shuffles=3, method="effect_size", random_state=99
        )
        result2 = gate_signal_verification(
            model, X, y, n_shuffles=3, method="effect_size", random_state=99
        )

        assert result1.metric_value == result2.metric_value
        assert result1.status == result2.status
