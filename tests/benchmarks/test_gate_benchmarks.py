"""Benchmark tests for validation gates.

Measures performance of gate evaluation across various configurations.
"""

import numpy as np
import pytest

from temporalcv.gates import (
    gate_signal_verification,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
    gate_temporal_boundary,
    gate_residual_diagnostics,
    run_gates,
)


class DummyModel:
    """Simple model for benchmarking gates."""

    def __init__(self, noise_std: float = 0.5):
        self.noise_std = noise_std
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyModel":
        """Simple linear fit."""
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with learned coefficients."""
        if self.coef_ is None:
            raise ValueError("Model not fitted")
        return X @ self.coef_


class TestShuffledTargetGateBenchmarks:
    """Benchmarks for shuffled target gate."""

    @pytest.fixture
    def small_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Small dataset: n=200."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = X @ np.array([1.0, 0.5, -0.3, 0.2, 0.1]) + rng.standard_normal(200) * 0.5
        return X, y

    @pytest.fixture
    def medium_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: n=500."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 10))
        y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(500) * 0.5
        return X, y

    def test_shuffled_target_10_shuffles(
        self, benchmark, small_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shuffled target gate with 10 shuffles."""
        X, y = small_data
        model = DummyModel()

        def run_gate() -> object:
            return gate_signal_verification(
                model=model, X=X, y=y, n_shuffles=10, random_state=42
            )

        result = benchmark(run_gate)
        assert result is not None

    def test_shuffled_target_50_shuffles(
        self, benchmark, small_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shuffled target gate with 50 shuffles."""
        X, y = small_data
        model = DummyModel()

        def run_gate() -> object:
            return gate_signal_verification(
                model=model, X=X, y=y, n_shuffles=50, random_state=42
            )

        result = benchmark(run_gate)
        assert result is not None

    def test_shuffled_target_100_shuffles(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Shuffled target gate with 100 shuffles on larger data."""
        X, y = medium_data
        model = DummyModel()

        def run_gate() -> object:
            return gate_signal_verification(
                model=model, X=X, y=y, n_shuffles=100, random_state=42
            )

        result = benchmark(run_gate)
        assert result is not None


class TestSyntheticAR1GateBenchmarks:
    """Benchmarks for synthetic AR(1) gate."""

    def test_synthetic_ar1_short_series(self, benchmark) -> None:
        """Synthetic AR(1) gate with n=200."""
        model = DummyModel()

        def run_gate() -> object:
            return gate_synthetic_ar1(
                model=model, n_samples=200, phi=0.6, sigma=1.0, random_state=42
            )

        result = benchmark(run_gate)
        assert result is not None

    def test_synthetic_ar1_medium_series(self, benchmark) -> None:
        """Synthetic AR(1) gate with n=500."""
        model = DummyModel()

        def run_gate() -> object:
            return gate_synthetic_ar1(
                model=model, n_samples=500, phi=0.6, sigma=1.0, random_state=42
            )

        result = benchmark(run_gate)
        assert result is not None


class TestSuspiciousImprovementGateBenchmarks:
    """Benchmarks for suspicious improvement gate."""

    @pytest.fixture
    def predictions_actuals(self) -> tuple[np.ndarray, np.ndarray]:
        """Predictions and actuals for testing."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(500)
        predictions = actuals + rng.standard_normal(500) * 0.3
        return predictions, actuals

    def test_suspicious_improvement(
        self, benchmark, predictions_actuals: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Suspicious improvement gate."""
        predictions, actuals = predictions_actuals
        # Compute MAE for model and baseline
        model_mae = float(np.mean(np.abs(predictions - actuals)))
        persistence_baseline = np.roll(actuals, 1)
        baseline_mae = float(np.mean(np.abs(persistence_baseline - actuals)))

        def run_gate() -> object:
            return gate_suspicious_improvement(
                model_metric=model_mae,
                baseline_metric=baseline_mae,
            )

        result = benchmark(run_gate)
        assert result is not None


class TestTemporalBoundaryGateBenchmarks:
    """Benchmarks for temporal boundary gate."""

    def test_temporal_boundary(self, benchmark) -> None:
        """Temporal boundary gate with index-based validation."""

        def run_gate() -> object:
            # Validate proper gap enforcement: train_end=99, test_start=100, horizon=1
            return gate_temporal_boundary(
                train_end_idx=99, test_start_idx=100, horizon=1, extra_gap=0
            )

        result = benchmark(run_gate)
        assert result is not None


class TestResidualDiagnosticsGateBenchmarks:
    """Benchmarks for residual diagnostics gate."""

    @pytest.fixture
    def residuals(self) -> np.ndarray:
        """Generate residuals for testing."""
        rng = np.random.default_rng(42)
        return rng.standard_normal(500)

    @pytest.fixture
    def autocorrelated_residuals(self) -> np.ndarray:
        """Generate autocorrelated residuals."""
        rng = np.random.default_rng(42)
        n = 500
        residuals = np.zeros(n)
        residuals[0] = rng.standard_normal()
        for i in range(1, n):
            residuals[i] = 0.6 * residuals[i - 1] + rng.standard_normal()
        return residuals

    def test_residual_diagnostics_white_noise(
        self, benchmark, residuals: np.ndarray
    ) -> None:
        """Residual diagnostics gate with white noise residuals."""

        def run_gate() -> object:
            return gate_residual_diagnostics(residuals=residuals, max_lag=10)

        result = benchmark(run_gate)
        assert result is not None

    def test_residual_diagnostics_autocorrelated(
        self, benchmark, autocorrelated_residuals: np.ndarray
    ) -> None:
        """Residual diagnostics gate with autocorrelated residuals."""

        def run_gate() -> object:
            return gate_residual_diagnostics(
                residuals=autocorrelated_residuals, max_lag=10
            )

        result = benchmark(run_gate)
        assert result is not None


class TestRunGatesBenchmarks:
    """Benchmarks for run_gates aggregation."""

    @pytest.fixture
    def simple_gates(self) -> list:
        """Pre-computed gate results."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = X @ np.array([1.0, 0.5, -0.3, 0.2, 0.1]) + rng.standard_normal(200) * 0.5
        model = DummyModel()

        # Compute MAE metrics for suspicious improvement gate
        predictions = y + rng.standard_normal(200) * 0.1
        model_mae = float(np.mean(np.abs(predictions - y)))
        persistence_baseline = np.roll(y, 1)
        baseline_mae = float(np.mean(np.abs(persistence_baseline - y)))

        # Pre-compute gates
        gates = [
            gate_signal_verification(model=model, X=X, y=y, n_shuffles=10, random_state=42),
            gate_suspicious_improvement(
                model_metric=model_mae,
                baseline_metric=baseline_mae,
            ),
            gate_temporal_boundary(
                train_end_idx=99, test_start_idx=100, horizon=1, extra_gap=0
            ),
        ]
        return gates

    def test_run_gates_aggregation(self, benchmark, simple_gates: list) -> None:
        """Aggregating multiple gate results."""

        def run_aggregation() -> object:
            return run_gates(simple_gates)

        result = benchmark(run_aggregation)
        assert result is not None
