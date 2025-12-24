"""
Tests for Residual Bootstrap with STL Decomposition.

Test categories:
1. Basic functionality
2. STL decomposition
3. Bootstrap sampling
4. Edge cases
5. Factory function
"""

import numpy as np
import pytest

from temporalcv.bagging import (
    ResidualBootstrap,
    TimeSeriesBagger,
    create_residual_bagger,
)


def generate_seasonal_series(
    n: int,
    trend_slope: float = 0.01,
    seasonal_amplitude: float = 1.0,
    seasonal_period: int = 12,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate series with trend + seasonality + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    trend = trend_slope * t
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    noise = rng.normal(0, noise_std, n)

    y = trend + seasonal + noise

    # Simple features (just lagged values for testing)
    X = np.column_stack([t, np.roll(y, 1)])
    X[0, 1] = 0  # Fix first value

    return X, y


class TestResidualBootstrapBasic:
    """Basic functionality tests."""

    def test_initialization(self) -> None:
        """Should initialize with default parameters."""
        strategy = ResidualBootstrap()

        assert strategy.seasonal_period is None
        assert strategy.block_length is None
        assert strategy.robust is True
        assert strategy.stl_kwargs == {}
        assert strategy.decomposition_ is None

    def test_initialization_custom(self) -> None:
        """Should initialize with custom parameters."""
        strategy = ResidualBootstrap(
            seasonal_period=52,
            block_length=8,
            robust=False,
            stl_kwargs={"seasonal_deg": 0},
        )

        assert strategy.seasonal_period == 52
        assert strategy.block_length == 8
        assert strategy.robust is False
        assert strategy.stl_kwargs == {"seasonal_deg": 0}

    def test_invalid_seasonal_period(self) -> None:
        """Should raise for invalid seasonal period."""
        with pytest.raises(ValueError, match="seasonal_period must be >= 2"):
            ResidualBootstrap(seasonal_period=1)

    def test_invalid_block_length(self) -> None:
        """Should raise for invalid block length."""
        with pytest.raises(ValueError, match="block_length must be >= 1"):
            ResidualBootstrap(block_length=0)

    def test_repr(self) -> None:
        """Should have informative repr."""
        strategy = ResidualBootstrap(seasonal_period=12, block_length=4)
        repr_str = repr(strategy)

        assert "ResidualBootstrap" in repr_str
        assert "seasonal_period=12" in repr_str
        assert "block_length=4" in repr_str


class TestResidualBootstrapSampling:
    """Tests for bootstrap sample generation."""

    def test_generates_correct_number_samples(self) -> None:
        """Should generate requested number of samples."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=42)
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=5, rng=rng)

        assert len(samples) == 5

    def test_samples_have_correct_shape(self) -> None:
        """Samples should have same shape as input."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=43)
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(43)

        samples = strategy.generate_samples(X, y, n_samples=3, rng=rng)

        for X_boot, y_boot in samples:
            assert X_boot.shape == X.shape
            assert y_boot.shape == y.shape

    def test_x_unchanged(self) -> None:
        """X should be unchanged in residual bootstrap."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=44)
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(44)

        samples = strategy.generate_samples(X, y, n_samples=3, rng=rng)

        for X_boot, _ in samples:
            np.testing.assert_array_equal(X_boot, X)

    def test_y_different(self) -> None:
        """y should be different (bootstrapped) across samples."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=45)
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(45)

        samples = strategy.generate_samples(X, y, n_samples=5, rng=rng)

        # y values should differ between samples
        y_boots = [y_boot for _, y_boot in samples]
        for i in range(len(y_boots) - 1):
            assert not np.allclose(y_boots[i], y_boots[i + 1])

    def test_decomposition_stored(self) -> None:
        """Decomposition should be stored after generate_samples."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=46)
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(46)

        strategy.generate_samples(X, y, n_samples=1, rng=rng)

        assert strategy.decomposition_ is not None
        assert hasattr(strategy.decomposition_, "trend")
        assert hasattr(strategy.decomposition_, "seasonal")
        assert hasattr(strategy.decomposition_, "resid")

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed should produce same samples."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=47)
        strategy = ResidualBootstrap(seasonal_period=12)

        rng1 = np.random.default_rng(100)
        samples1 = strategy.generate_samples(X, y, n_samples=3, rng=rng1)

        rng2 = np.random.default_rng(100)
        samples2 = strategy.generate_samples(X, y, n_samples=3, rng=rng2)

        for (X1, y1), (X2, y2) in zip(samples1, samples2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_samples(self) -> None:
        """Different seeds should produce different samples."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=48)
        strategy = ResidualBootstrap(seasonal_period=12)

        rng1 = np.random.default_rng(100)
        samples1 = strategy.generate_samples(X, y, n_samples=1, rng=rng1)

        rng2 = np.random.default_rng(200)
        samples2 = strategy.generate_samples(X, y, n_samples=1, rng=rng2)

        assert not np.allclose(samples1[0][1], samples2[0][1])


class TestResidualBootstrapSTL:
    """Tests for STL decomposition behavior."""

    def test_preserves_trend(self) -> None:
        """Bootstrapped series should preserve trend structure."""
        X, y = generate_seasonal_series(
            150, trend_slope=0.05, seasonal_period=12, noise_std=0.1, seed=50
        )
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(50)

        samples = strategy.generate_samples(X, y, n_samples=10, rng=rng)

        # All bootstrapped series should have similar trend
        trends = []
        for _, y_boot in samples:
            # Simple trend estimate: slope of linear fit
            t = np.arange(len(y_boot))
            slope = np.polyfit(t, y_boot, 1)[0]
            trends.append(slope)

        # Trends should be similar (within some tolerance)
        assert np.std(trends) < 0.01  # Low variance in estimated trends

    def test_preserves_seasonality(self) -> None:
        """Bootstrapped series should preserve seasonal structure."""
        X, y = generate_seasonal_series(
            120, seasonal_amplitude=2.0, seasonal_period=12, noise_std=0.1, seed=51
        )
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(51)

        samples = strategy.generate_samples(X, y, n_samples=5, rng=rng)

        # Check that seasonality is preserved (via autocorrelation at lag 12)
        for _, y_boot in samples:
            # Simple check: correlation at seasonal lag should be high
            corr = np.corrcoef(y_boot[12:], y_boot[:-12])[0, 1]
            assert corr > 0.5  # Should maintain seasonal correlation

    def test_auto_detects_period(self) -> None:
        """Should auto-detect reasonable seasonal period."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=52)
        strategy = ResidualBootstrap()  # No seasonal_period specified
        rng = np.random.default_rng(52)

        # Should not raise
        samples = strategy.generate_samples(X, y, n_samples=1, rng=rng)
        assert len(samples) == 1

    def test_warns_on_large_period(self) -> None:
        """Should warn when seasonal period is too large."""
        X, y = generate_seasonal_series(50, seasonal_period=12, seed=53)
        strategy = ResidualBootstrap(seasonal_period=30)  # Too large
        rng = np.random.default_rng(53)

        with pytest.warns(UserWarning, match="seasonal_period"):
            strategy.generate_samples(X, y, n_samples=1, rng=rng)


class TestResidualBootstrapBlockLength:
    """Tests for block length behavior."""

    def test_auto_computes_block_length(self) -> None:
        """Should auto-compute block length as n^(1/3)."""
        X, y = generate_seasonal_series(125, seasonal_period=12, seed=60)
        strategy = ResidualBootstrap(seasonal_period=12)  # No block_length
        rng = np.random.default_rng(60)

        samples = strategy.generate_samples(X, y, n_samples=1, rng=rng)

        # Should complete without error - block length auto-computed
        assert len(samples) == 1

    def test_respects_custom_block_length(self) -> None:
        """Should use custom block length when specified."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=61)
        strategy = ResidualBootstrap(seasonal_period=12, block_length=10)
        rng = np.random.default_rng(61)

        samples = strategy.generate_samples(X, y, n_samples=1, rng=rng)
        assert len(samples) == 1


class TestResidualBootstrapEdgeCases:
    """Edge case tests."""

    def test_short_series_raises(self) -> None:
        """Should raise for series too short for STL."""
        X = np.ones((3, 2))
        y = np.array([1, 2, 3])
        strategy = ResidualBootstrap()
        rng = np.random.default_rng(70)

        with pytest.raises(ValueError, match="too short"):
            strategy.generate_samples(X, y, n_samples=1, rng=rng)

    def test_minimum_viable_series(self) -> None:
        """Should work with minimum viable series length."""
        # Need enough data for STL decomposition
        n = 24  # 2 cycles of period 12
        X, y = generate_seasonal_series(n, seasonal_period=12, seed=71)
        strategy = ResidualBootstrap(seasonal_period=12)
        rng = np.random.default_rng(71)

        samples = strategy.generate_samples(X, y, n_samples=1, rng=rng)
        assert len(samples) == 1

    def test_list_input(self) -> None:
        """Should accept list input."""
        X_list = [[1, 2], [2, 3], [3, 4]] * 20
        y_list = list(range(60))
        strategy = ResidualBootstrap(seasonal_period=10)
        rng = np.random.default_rng(72)

        samples = strategy.generate_samples(X_list, y_list, n_samples=1, rng=rng)
        assert len(samples) == 1

    def test_robust_handles_outliers(self) -> None:
        """Robust STL should handle outliers."""
        X, y = generate_seasonal_series(100, seasonal_period=12, seed=73)
        # Add outliers
        y[50] += 100
        y[75] -= 100

        strategy = ResidualBootstrap(seasonal_period=12, robust=True)
        rng = np.random.default_rng(73)

        # Should complete without error
        samples = strategy.generate_samples(X, y, n_samples=1, rng=rng)
        assert len(samples) == 1


class TestCreateResidualBagger:
    """Tests for create_residual_bagger factory function."""

    def test_creates_bagger(self) -> None:
        """Should create TimeSeriesBagger with ResidualBootstrap."""

        class DummyModel:
            def fit(self, X, y):
                self.coef_ = np.mean(y)
                return self

            def predict(self, X):
                return np.full(len(X), self.coef_)

        bagger = create_residual_bagger(
            DummyModel(),
            seasonal_period=12,
            n_estimators=5,
            random_state=42,
        )

        assert isinstance(bagger, TimeSeriesBagger)

    def test_fit_predict(self) -> None:
        """Should fit and predict successfully."""

        class SimpleModel:
            def fit(self, X, y):
                self.mean_ = np.mean(y)
                return self

            def predict(self, X):
                return np.full(len(X), self.mean_)

        X, y = generate_seasonal_series(100, seasonal_period=12, seed=80)
        X_test, _ = generate_seasonal_series(20, seasonal_period=12, seed=81)

        bagger = create_residual_bagger(
            SimpleModel(),
            seasonal_period=12,
            n_estimators=5,
            random_state=42,
        )

        bagger.fit(X, y)
        predictions = bagger.predict(X_test)

        assert predictions.shape == (len(X_test),)


class TestIntegrationWithBagger:
    """Integration tests with TimeSeriesBagger."""

    def test_full_workflow(self) -> None:
        """Test complete fit/predict workflow."""

        class LinearModel:
            def fit(self, X, y):
                # Simple least squares
                self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
                return self

            def predict(self, X):
                return X @ self.coef_

        X, y = generate_seasonal_series(100, seasonal_period=12, seed=90)

        strategy = ResidualBootstrap(seasonal_period=12, block_length=5)
        bagger = TimeSeriesBagger(
            base_model=LinearModel(),
            strategy=strategy,
            n_estimators=10,
            random_state=42,
        )

        bagger.fit(X, y)

        # Test prediction
        X_test = X[:10]
        predictions = bagger.predict(X_test)

        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))

    def test_uncertainty_estimation(self) -> None:
        """Test uncertainty estimation with residual bootstrap."""

        class LinearModel:
            def fit(self, X, y):
                self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
                return self

            def predict(self, X):
                return X @ self.coef_

        X, y = generate_seasonal_series(100, seasonal_period=12, seed=91)

        bagger = create_residual_bagger(
            LinearModel(),
            seasonal_period=12,
            n_estimators=20,
            random_state=42,
        )

        bagger.fit(X, y)

        # Get uncertainty estimates
        X_test = X[:10]
        mean, std = bagger.predict_with_uncertainty(X_test)

        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(std >= 0)  # Standard deviation should be non-negative
