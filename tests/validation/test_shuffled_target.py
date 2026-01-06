"""
Validation tests for shuffled target gate.

The shuffled target test is the definitive external validation method for
detecting data leakage. It works by comparing model performance on:
1. Real (ordered) data
2. Shuffled (randomized) target

If the model performs significantly better on real data, it's exploiting
temporal patterns - which may be legitimate or may indicate leakage.

Knowledge Tier: [T1] - Based on standard permutation testing methodology
Reference: See SPECIFICATION.md for threshold definitions
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import (
    GateStatus,
    gate_signal_verification,
)


class MockMeanPredictor:
    """Predicts mean of training y - no temporal exploitation."""

    def __init__(self) -> None:
        self._mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockMeanPredictor":
        self._mean = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mean)


class MockLag1Predictor:
    """Uses lag-1 for prediction - legitimate temporal exploitation."""

    def __init__(self, coefficient: float = 0.95) -> None:
        self.coefficient = coefficient

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MockLag1Predictor":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Assume first column is lag-1
        if X.ndim > 1:
            return self.coefficient * X[:, 0]
        return self.coefficient * X


class TestShuffledTargetValidation:
    """Validation tests for shuffled target gate correctness."""

    def test_mean_predictor_passes(self) -> None:
        """
        Mean predictor should PASS.

        A model that just predicts the mean doesn't use temporal
        information, so it shouldn't look better on ordered vs shuffled.

        Note: With out-of-sample CV evaluation, there's more variance in small
        samples. Using threshold=0.10 to account for this while still detecting
        genuine leakage.
        """
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 5))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=5,
            threshold=0.10,  # Higher threshold for out-of-sample variance
            method="effect_size",  # Use effect size mode for this test
            random_state=42,
        )

        assert result.status in (GateStatus.PASS, GateStatus.WARN), (
            f"Mean predictor should pass, got {result.status}"
        )

    def test_lag1_predictor_on_ar1_triggers_correctly(self) -> None:
        """
        Legitimate lag-1 predictor on AR(1) data WILL trigger the gate.

        This is expected behavior - the shuffled target test detects
        ANY temporal exploitation, legitimate or not. The gate correctly
        identifies that the model is using temporal information.

        In practice, this means:
        - HALT on shuffled target ≠ proof of leakage
        - It's a signal to investigate further
        - Legitimate models with strong temporal signal will trigger
        """
        rng = np.random.default_rng(42)
        n = 200
        phi = 0.9

        # Generate AR(1) process
        y = np.zeros(n)
        y[0] = rng.standard_normal()
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.standard_normal()

        # Features include lag-1 (legitimate)
        X = np.column_stack([
            np.roll(y, 1),  # Lag 1
            np.roll(y, 2),  # Lag 2
            rng.standard_normal(n),  # Noise
        ])
        X[0, :2] = 0  # Clean up edge

        model = MockLag1Predictor(coefficient=phi)

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=5,
            threshold=0.05,
            method="effect_size",  # Use effect size mode to check improvement_ratio
            random_state=42,
        )

        # A model that exploits temporal structure WILL beat shuffled
        # significantly - this is expected and correct behavior
        assert result.status == GateStatus.HALT, (
            "Temporal model should trigger gate (it's exploiting structure)"
        )
        # The key insight: large improvement ratio confirms temporal exploitation
        assert result.metric_value > 0.3, (
            "Strong temporal model should show large improvement over shuffled"
        )

    def test_threshold_default_value(self) -> None:
        """Default threshold should be 0.05 per SPECIFICATION.md (effect_size mode)."""
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            method="effect_size",  # Test effect_size mode threshold
            random_state=42,
        )

        # Verify threshold is applied
        assert result.threshold == 0.05

    def test_alpha_default_value(self) -> None:
        """Default alpha should be 0.05 per SPECIFICATION.md (permutation mode)."""
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=20,  # Enough for p < 0.05
            method="permutation",  # Test permutation mode alpha
            random_state=42,
        )

        # Verify alpha is used as threshold in permutation mode
        assert result.threshold == 0.05

    def test_n_shuffles_parameter(self) -> None:
        """n_shuffles should control number of permutations."""
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=7,
            method="effect_size",  # Use effect_size mode to test n_shuffles
            random_state=42,
        )

        assert result.details["n_shuffles"] == 7
        assert len(result.details["mae_shuffled_all"]) == 7


class TestShuffledTargetStatistics:
    """Tests for statistical properties of shuffled target test."""

    def test_improvement_ratio_calculation(self) -> None:
        """
        Improvement ratio should be:
        1 - (mae_real / mae_shuffled) = (mae_shuffled - mae_real) / mae_shuffled
        """
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            method="effect_size",  # Use effect_size mode to check improvement_ratio
            random_state=42,
        )

        mae_real = result.details["mae_real"]
        mae_shuffled = result.details["mae_shuffled_avg"]

        expected_ratio = 1 - (mae_real / mae_shuffled)
        assert abs(result.metric_value - expected_ratio) < 0.001
        # Also verify improvement_ratio is stored in details
        assert abs(result.details["improvement_ratio"] - expected_ratio) < 0.001

    def test_variance_across_shuffles(self) -> None:
        """Multiple shuffles should show variance in MAE."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=10,
            method="effect_size",  # Use effect_size mode for this test
            random_state=42,
        )

        maes = result.details["mae_shuffled_all"]
        assert len(set(maes)) > 1, "All shuffles produced same MAE"


class TestShuffledTargetEdgeCases:
    """Edge case tests for shuffled target gate."""

    def test_small_dataset(self) -> None:
        """Should handle small datasets gracefully."""
        rng = np.random.default_rng(42)
        n = 20  # Small
        X = rng.standard_normal((n, 2))
        y = rng.standard_normal(n)

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            random_state=42,
        )

        # Should complete without error
        assert result.status in (GateStatus.PASS, GateStatus.WARN, GateStatus.HALT)

    def test_constant_target(self) -> None:
        """Should handle constant target appropriately."""
        rng = np.random.default_rng(42)
        n = 50
        X = rng.standard_normal((n, 3))
        y = np.ones(n)  # Constant

        model = MockMeanPredictor()

        result = gate_signal_verification(
            model=model,
            X=X,
            y=y,
            n_shuffles=3,
            random_state=42,
        )

        # Mean predictor on constant y = perfect prediction
        # Should pass (or skip) since shuffling constant has no effect
        assert result.status in (GateStatus.PASS, GateStatus.SKIP, GateStatus.WARN)
