"""Benchmark tests for metric computation.

Measures performance of metric functions across various data sizes.
"""

import numpy as np
import pytest

from temporalcv.metrics import (
    compute_mae,
    compute_mse,
    compute_rmse,
    compute_mape,
    compute_smape,
    compute_mase,
    compute_mrae,
    compute_theils_u,
    compute_pinball_loss,
    compute_crps,
    compute_interval_score,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_linex_loss,
    compute_huber_loss,
)
from temporalcv.persistence import (
    compute_move_threshold,
    compute_move_conditional_metrics,
)


class TestCoreMetricBenchmarks:
    """Benchmarks for core error metrics."""

    @pytest.fixture
    def small_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Small arrays: n=500."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(500)
        predictions = actuals + rng.standard_normal(500) * 0.3
        return predictions, actuals

    @pytest.fixture
    def medium_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium arrays: n=5000."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(5000)
        predictions = actuals + rng.standard_normal(5000) * 0.3
        return predictions, actuals

    @pytest.fixture
    def large_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Large arrays: n=50000."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(50000)
        predictions = actuals + rng.standard_normal(50000) * 0.3
        return predictions, actuals

    def test_mae_small(
        self, benchmark, small_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """MAE with n=500."""
        predictions, actuals = small_arrays
        result = benchmark(lambda: compute_mae(predictions, actuals))
        assert result >= 0

    def test_mae_large(
        self, benchmark, large_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """MAE with n=50000."""
        predictions, actuals = large_arrays
        result = benchmark(lambda: compute_mae(predictions, actuals))
        assert result >= 0

    def test_mse_large(
        self, benchmark, large_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """MSE with n=50000."""
        predictions, actuals = large_arrays
        result = benchmark(lambda: compute_mse(predictions, actuals))
        assert result >= 0

    def test_rmse_large(
        self, benchmark, large_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """RMSE with n=50000."""
        predictions, actuals = large_arrays
        result = benchmark(lambda: compute_rmse(predictions, actuals))
        assert result >= 0

    def test_mape_medium(
        self, benchmark, medium_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """MAPE with n=5000."""
        predictions, actuals = medium_arrays
        # Add offset to avoid division by zero
        actuals_safe = actuals + 10
        predictions_safe = predictions + 10
        result = benchmark(lambda: compute_mape(predictions_safe, actuals_safe))
        assert result >= 0

    def test_smape_medium(
        self, benchmark, medium_arrays: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """SMAPE with n=5000."""
        predictions, actuals = medium_arrays
        # Add offset to avoid division by zero
        actuals_safe = actuals + 10
        predictions_safe = predictions + 10
        result = benchmark(lambda: compute_smape(predictions_safe, actuals_safe))
        assert result >= 0


class TestScaledMetricBenchmarks:
    """Benchmarks for scaled/relative metrics."""

    @pytest.fixture
    def time_series_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Time series with naive baseline."""
        rng = np.random.default_rng(42)
        n = 2000
        actuals = np.cumsum(rng.standard_normal(n))  # Random walk
        predictions = actuals + rng.standard_normal(n) * 0.3
        naive = np.roll(actuals, 1)
        naive[0] = actuals[0]
        return predictions, actuals, naive

    def test_mase(
        self, benchmark, time_series_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """MASE with n=2000."""
        predictions, actuals, _ = time_series_data
        # Precompute naive MAE (random walk: use lag-1 naive)
        naive_mae = float(np.mean(np.abs(np.diff(actuals))))
        result = benchmark(
            lambda: compute_mase(predictions, actuals, naive_mae)
        )
        assert result >= 0

    def test_mrae(
        self, benchmark, time_series_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """MRAE with n=2000."""
        predictions, actuals, naive = time_series_data
        result = benchmark(lambda: compute_mrae(predictions, actuals, naive))
        assert result >= 0

    def test_theils_u(
        self, benchmark, time_series_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Theil's U with n=2000."""
        predictions, actuals, naive = time_series_data
        result = benchmark(lambda: compute_theils_u(predictions, actuals, naive))
        assert result >= 0


class TestQuantileMetricBenchmarks:
    """Benchmarks for quantile/probabilistic metrics."""

    @pytest.fixture
    def quantile_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Quantile predictions and actuals."""
        rng = np.random.default_rng(42)
        n = 2000
        actuals = rng.standard_normal(n)
        # Quantile predictions for tau=0.5 (median)
        predictions = actuals + rng.standard_normal(n) * 0.2
        return predictions, actuals

    @pytest.fixture
    def interval_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interval predictions (lower, upper) and actuals."""
        rng = np.random.default_rng(42)
        n = 2000
        actuals = rng.standard_normal(n)
        lower = actuals - 1.96  # ~95% interval
        upper = actuals + 1.96
        return lower, upper, actuals

    @pytest.fixture
    def ensemble_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble predictions for CRPS."""
        rng = np.random.default_rng(42)
        n = 500
        n_members = 100
        actuals = rng.standard_normal(n)
        # Ensemble predictions
        ensemble = actuals[:, np.newaxis] + rng.standard_normal((n, n_members)) * 0.5
        return ensemble, actuals

    def test_pinball_loss(
        self, benchmark, quantile_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Pinball loss with n=2000."""
        predictions, actuals = quantile_data
        result = benchmark(
            lambda: compute_pinball_loss(actuals, predictions, tau=0.5)
        )
        assert result >= 0

    def test_interval_score(
        self, benchmark, interval_data: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Interval score with n=2000."""
        lower, upper, actuals = interval_data
        result = benchmark(
            lambda: compute_interval_score(actuals, lower, upper, alpha=0.05)
        )
        assert result >= 0

    def test_crps_ensemble(
        self, benchmark, ensemble_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CRPS with n=500, 100 ensemble members."""
        ensemble, actuals = ensemble_data
        result = benchmark(lambda: compute_crps(actuals, ensemble))
        assert result >= 0


class TestFinancialMetricBenchmarks:
    """Benchmarks for financial/trading metrics."""

    @pytest.fixture
    def returns_data(self) -> np.ndarray:
        """Daily returns series."""
        rng = np.random.default_rng(42)
        # 5 years of daily returns
        return rng.standard_normal(252 * 5) * 0.01

    def test_sharpe_ratio(self, benchmark, returns_data: np.ndarray) -> None:
        """Sharpe ratio with 5 years of daily returns."""
        result = benchmark(
            lambda: compute_sharpe_ratio(returns_data, annualization=252.0)
        )
        assert isinstance(result, float)

    def test_max_drawdown(self, benchmark, returns_data: np.ndarray) -> None:
        """Max drawdown with 5 years of daily returns."""
        # Convert returns to prices
        prices = 100 * np.cumprod(1 + returns_data)
        result = benchmark(lambda: compute_max_drawdown(prices))
        assert 0 <= result <= 1


class TestAsymmetricLossBenchmarks:
    """Benchmarks for asymmetric loss functions."""

    @pytest.fixture
    def prediction_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Predictions and actuals for loss computation."""
        rng = np.random.default_rng(42)
        n = 5000
        actuals = rng.standard_normal(n)
        predictions = actuals + rng.standard_normal(n) * 0.3
        return predictions, actuals

    def test_linex_loss(
        self, benchmark, prediction_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """LinEx loss with n=5000."""
        predictions, actuals = prediction_data
        result = benchmark(
            lambda: compute_linex_loss(predictions, actuals, a=0.5)
        )
        assert result >= 0

    def test_huber_loss(
        self, benchmark, prediction_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Huber loss with n=5000."""
        predictions, actuals = prediction_data
        result = benchmark(
            lambda: compute_huber_loss(predictions, actuals, delta=1.0)
        )
        assert result >= 0


class TestMoveConditionalBenchmarks:
    """Benchmarks for move-conditional metrics (high-persistence)."""

    @pytest.fixture
    def high_persistence_data(self) -> tuple[np.ndarray, np.ndarray]:
        """High-persistence series (random walk with drift)."""
        rng = np.random.default_rng(42)
        n = 1000
        # Random walk
        actuals = np.cumsum(rng.standard_normal(n) * 0.1) + np.arange(n) * 0.001
        predictions = actuals + rng.standard_normal(n) * 0.05
        return predictions, actuals

    def test_compute_move_threshold(
        self, benchmark, high_persistence_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Move threshold computation with n=1000."""
        _, actuals = high_persistence_data
        result = benchmark(lambda: compute_move_threshold(actuals))
        assert result >= 0

    def test_move_conditional_metrics(
        self, benchmark, high_persistence_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Move-conditional metrics with n=1000."""
        predictions, actuals = high_persistence_data
        threshold = compute_move_threshold(actuals)
        result = benchmark(
            lambda: compute_move_conditional_metrics(predictions, actuals, threshold)
        )
        assert result is not None
