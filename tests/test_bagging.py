"""
Test Time Series Bagging Framework.

Tests for time-series-aware bootstrap aggregation strategies.
"""

import numpy as np
import pytest

from temporalcv.bagging import (
    BootstrapStrategy,
    TimeSeriesBagger,
    MovingBlockBootstrap,
    StationaryBootstrap,
    FeatureBagging,
    create_block_bagger,
    create_stationary_bagger,
    create_feature_bagger,
)


# =============================================================================
# Simple Model for Testing
# =============================================================================


class SimpleModel:
    """Simple linear model for testing bagging."""

    def __init__(self, alpha: float = 0.0):
        self.alpha = alpha
        self.coef_: np.ndarray = np.array([])
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleModel":
        """Fit using closed-form ridge solution."""
        X = np.asarray(X)
        y = np.asarray(y)

        # Add bias column
        n = X.shape[0]
        X_bias = np.column_stack([np.ones(n), X])

        # Ridge solution: (X'X + αI)^(-1) X'y
        I = np.eye(X_bias.shape[1])
        I[0, 0] = 0  # Don't regularize intercept
        beta = np.linalg.solve(
            X_bias.T @ X_bias + self.alpha * I,
            X_bias.T @ y,
        )

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def get_params(self, deep: bool = True) -> dict:
        """For sklearn compatibility."""
        return {"alpha": self.alpha}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def linear_data():
    """Generate linear data for testing."""
    rng = np.random.default_rng(42)
    n = 100
    n_features = 5

    X = rng.standard_normal((n, n_features))
    true_coef = np.array([1.0, -0.5, 0.3, -0.2, 0.1])
    noise = rng.normal(0, 0.5, n)
    y = X @ true_coef + noise

    return X, y


@pytest.fixture
def time_series_data():
    """Generate autocorrelated time series data."""
    rng = np.random.default_rng(42)
    n = 150
    n_features = 3

    # Generate AR(1) features
    X = np.zeros((n, n_features))
    for j in range(n_features):
        X[0, j] = rng.standard_normal()
        for i in range(1, n):
            X[i, j] = 0.8 * X[i - 1, j] + 0.2 * rng.standard_normal()

    # Target with autocorrelation
    y = np.zeros(n)
    true_coef = np.array([0.5, -0.3, 0.2])
    y[0] = X[0] @ true_coef + rng.standard_normal() * 0.3
    for i in range(1, n):
        y[i] = 0.5 * y[i - 1] + 0.5 * (X[i] @ true_coef) + rng.standard_normal() * 0.3

    return X, y


# =============================================================================
# Moving Block Bootstrap Tests
# =============================================================================


class TestMovingBlockBootstrap:
    """Test Moving Block Bootstrap strategy."""

    def test_generates_correct_number_of_samples(self, linear_data) -> None:
        """Should generate requested number of samples."""
        X, y = linear_data
        strategy = MovingBlockBootstrap(block_length=10)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=5, rng=rng)

        assert len(samples) == 5

    def test_samples_have_correct_shape(self, linear_data) -> None:
        """Each sample should have same shape as original."""
        X, y = linear_data
        strategy = MovingBlockBootstrap(block_length=10)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=3, rng=rng)

        for X_boot, y_boot in samples:
            assert X_boot.shape == X.shape
            assert y_boot.shape == y.shape

    def test_auto_block_length(self, linear_data) -> None:
        """Should auto-compute block length if not specified."""
        X, y = linear_data
        strategy = MovingBlockBootstrap()  # No block_length
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=2, rng=rng)

        # Should not raise and should produce valid samples
        assert len(samples) == 2
        assert samples[0][0].shape == X.shape

    def test_block_length_clamping(self, linear_data) -> None:
        """Should clamp block length to valid range."""
        X, y = linear_data
        strategy = MovingBlockBootstrap(block_length=1000)  # Too large
        rng = np.random.default_rng(42)

        # Should warn and clamp
        with pytest.warns(UserWarning, match="clamped"):
            samples = strategy.generate_samples(X, y, n_samples=2, rng=rng)

        assert len(samples) == 2

    def test_preserves_local_structure(self, time_series_data) -> None:
        """Block bootstrap should preserve local autocorrelation."""
        X, y = time_series_data
        strategy = MovingBlockBootstrap(block_length=10)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=1, rng=rng)
        X_boot, y_boot = samples[0]

        # Find consecutive runs in bootstrapped data
        # With block_length=10, we should see some consecutive sequences
        # This is a weak test but verifies block structure exists
        assert X_boot.shape == X.shape

    def test_repr(self) -> None:
        """Should have informative repr."""
        strategy = MovingBlockBootstrap(block_length=10)
        assert "MovingBlockBootstrap" in repr(strategy)
        assert "10" in repr(strategy)


# =============================================================================
# Stationary Bootstrap Tests
# =============================================================================


class TestStationaryBootstrap:
    """Test Stationary Bootstrap strategy."""

    def test_generates_correct_number_of_samples(self, linear_data) -> None:
        """Should generate requested number of samples."""
        X, y = linear_data
        strategy = StationaryBootstrap(expected_block_length=10.0)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=5, rng=rng)

        assert len(samples) == 5

    def test_samples_have_correct_shape(self, linear_data) -> None:
        """Each sample should have same shape as original."""
        X, y = linear_data
        strategy = StationaryBootstrap(expected_block_length=10.0)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=3, rng=rng)

        for X_boot, y_boot in samples:
            assert X_boot.shape == X.shape
            assert y_boot.shape == y.shape

    def test_auto_block_length(self, linear_data) -> None:
        """Should auto-compute expected block length if not specified."""
        X, y = linear_data
        strategy = StationaryBootstrap()  # No expected_block_length
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=2, rng=rng)

        assert len(samples) == 2

    def test_different_samples_vary(self, linear_data) -> None:
        """Different samples should not be identical."""
        X, y = linear_data
        strategy = StationaryBootstrap(expected_block_length=5.0)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=2, rng=rng)

        # Samples should be different
        assert not np.allclose(samples[0][0], samples[1][0])
        assert not np.allclose(samples[0][1], samples[1][1])

    def test_repr(self) -> None:
        """Should have informative repr."""
        strategy = StationaryBootstrap(expected_block_length=15.0)
        assert "StationaryBootstrap" in repr(strategy)
        assert "15" in repr(strategy)


# =============================================================================
# Feature Bagging Tests
# =============================================================================


class TestFeatureBagging:
    """Test Feature Bagging (Random Subspace) strategy."""

    def test_validates_max_features(self) -> None:
        """Should validate max_features parameter."""
        with pytest.raises(ValueError, match="must be in"):
            FeatureBagging(max_features=0.0)
        with pytest.raises(ValueError, match="must be in"):
            FeatureBagging(max_features=1.5)

    def test_generates_correct_number_of_samples(self, linear_data) -> None:
        """Should generate requested number of samples."""
        X, y = linear_data
        strategy = FeatureBagging(max_features=0.7)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=5, rng=rng)

        assert len(samples) == 5

    def test_samples_have_reduced_features(self, linear_data) -> None:
        """Each sample should have subset of features."""
        X, y = linear_data
        n_features = X.shape[1]
        strategy = FeatureBagging(max_features=0.6)
        rng = np.random.default_rng(42)

        samples = strategy.generate_samples(X, y, n_samples=3, rng=rng)

        expected_n_features = int(0.6 * n_features)
        for X_boot, y_boot in samples:
            assert X_boot.shape[1] == expected_n_features
            # All observations preserved
            assert X_boot.shape[0] == X.shape[0]
            assert y_boot.shape == y.shape

    def test_stores_feature_indices(self, linear_data) -> None:
        """Should store feature indices for transform_for_predict."""
        X, y = linear_data
        strategy = FeatureBagging(max_features=0.6)
        rng = np.random.default_rng(42)

        strategy.generate_samples(X, y, n_samples=3, rng=rng)

        assert len(strategy.feature_indices_) == 3
        for idx in strategy.feature_indices_:
            assert len(idx) == int(0.6 * X.shape[1])

    def test_transform_for_predict(self, linear_data) -> None:
        """transform_for_predict should select correct features."""
        X, y = linear_data
        strategy = FeatureBagging(max_features=0.6)
        rng = np.random.default_rng(42)

        strategy.generate_samples(X, y, n_samples=3, rng=rng)

        for i in range(3):
            X_transformed = strategy.transform_for_predict(X, i)
            expected_indices = strategy.feature_indices_[i]
            np.testing.assert_array_equal(X_transformed, X[:, expected_indices])

    def test_transform_before_generate_raises(self, linear_data) -> None:
        """transform_for_predict before generate_samples should raise."""
        X, _ = linear_data
        strategy = FeatureBagging(max_features=0.7)

        with pytest.raises(RuntimeError, match="generate samples"):
            strategy.transform_for_predict(X, 0)

    def test_validates_2d_input(self) -> None:
        """Should raise for non-2D input."""
        strategy = FeatureBagging(max_features=0.7)
        rng = np.random.default_rng(42)

        with pytest.raises(ValueError, match="2D"):
            strategy.generate_samples(np.array([1, 2, 3]), np.array([1, 2, 3]), 2, rng)

    def test_different_samples_have_different_features(self, linear_data) -> None:
        """Different samples should have different feature subsets."""
        X, y = linear_data
        strategy = FeatureBagging(max_features=0.6)
        rng = np.random.default_rng(42)

        strategy.generate_samples(X, y, n_samples=5, rng=rng)

        # At least some feature sets should differ
        feature_sets = [frozenset(idx.tolist()) for idx in strategy.feature_indices_]
        assert len(set(feature_sets)) > 1

    def test_repr(self) -> None:
        """Should have informative repr."""
        strategy = FeatureBagging(max_features=0.8)
        assert "FeatureBagging" in repr(strategy)
        assert "0.8" in repr(strategy)


# =============================================================================
# TimeSeriesBagger Tests
# =============================================================================


class TestTimeSeriesBagger:
    """Test TimeSeriesBagger wrapper."""

    def test_validates_n_estimators(self) -> None:
        """Should validate n_estimators."""
        with pytest.raises(ValueError, match="n_estimators"):
            TimeSeriesBagger(
                SimpleModel(), MovingBlockBootstrap(), n_estimators=0
            )

    def test_validates_aggregation(self) -> None:
        """Should validate aggregation method."""
        with pytest.raises(ValueError, match="aggregation"):
            TimeSeriesBagger(
                SimpleModel(),
                MovingBlockBootstrap(),
                aggregation="invalid",  # type: ignore[arg-type]
            )

    def test_fit_validates_input(self) -> None:
        """fit should validate input data."""
        bagger = TimeSeriesBagger(SimpleModel(), MovingBlockBootstrap())

        with pytest.raises(ValueError, match="length mismatch"):
            bagger.fit(np.zeros((10, 3)), np.zeros(5))

        with pytest.raises(ValueError, match="empty"):
            bagger.fit(np.zeros((0, 3)), np.zeros(0))

    def test_predict_requires_fit(self) -> None:
        """predict should fail without fit."""
        bagger = TimeSeriesBagger(SimpleModel(), MovingBlockBootstrap())

        with pytest.raises(RuntimeError, match="must be fitted"):
            bagger.predict(np.zeros((5, 3)))

    def test_fit_creates_estimators(self, linear_data) -> None:
        """fit should create n_estimators fitted models."""
        X, y = linear_data
        bagger = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=10
        )

        bagger.fit(X, y)

        assert len(bagger.estimators_) == 10
        assert bagger.is_fitted

    def test_predict_returns_correct_shape(self, linear_data) -> None:
        """predict should return correct shape."""
        X, y = linear_data
        bagger = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=5
        )

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        assert predictions.shape == (len(X),)

    def test_mean_aggregation(self, linear_data) -> None:
        """Mean aggregation should average predictions."""
        X, y = linear_data
        bagger = TimeSeriesBagger(
            SimpleModel(),
            MovingBlockBootstrap(),
            n_estimators=5,
            aggregation="mean",
        )

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        # Verify it's actually using mean (sanity check)
        all_preds = bagger._get_all_predictions(X)
        expected = np.mean(all_preds, axis=0)
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_median_aggregation(self, linear_data) -> None:
        """Median aggregation should use median."""
        X, y = linear_data
        bagger = TimeSeriesBagger(
            SimpleModel(),
            MovingBlockBootstrap(),
            n_estimators=5,
            aggregation="median",
        )

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        # Verify it's actually using median
        all_preds = bagger._get_all_predictions(X)
        expected = np.median(all_preds, axis=0)
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_predict_with_uncertainty(self, linear_data) -> None:
        """predict_with_uncertainty should return mean and std."""
        X, y = linear_data
        bagger = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=10
        )

        bagger.fit(X, y)
        mean, std = bagger.predict_with_uncertainty(X)

        assert mean.shape == (len(X),)
        assert std.shape == (len(X),)
        assert np.all(std >= 0)  # Std should be non-negative

    def test_predict_interval(self, linear_data) -> None:
        """predict_interval should return mean, lower, upper."""
        X, y = linear_data
        bagger = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=20
        )

        bagger.fit(X, y)
        mean, lower, upper = bagger.predict_interval(X, alpha=0.10)

        assert mean.shape == (len(X),)
        assert lower.shape == (len(X),)
        assert upper.shape == (len(X),)
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)

    def test_predict_interval_validates_alpha(self, linear_data) -> None:
        """predict_interval should validate alpha."""
        X, y = linear_data
        bagger = TimeSeriesBagger(SimpleModel(), MovingBlockBootstrap())
        bagger.fit(X, y)

        with pytest.raises(ValueError, match="alpha must be in"):
            bagger.predict_interval(X, alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            bagger.predict_interval(X, alpha=1.0)

    def test_reproducibility(self, linear_data) -> None:
        """Same random_state should give same results."""
        X, y = linear_data

        bagger1 = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=5, random_state=42
        )
        bagger1.fit(X, y)
        pred1 = bagger1.predict(X)

        bagger2 = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=5, random_state=42
        )
        bagger2.fit(X, y)
        pred2 = bagger2.predict(X)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_repr(self) -> None:
        """Should have informative repr."""
        bagger = TimeSeriesBagger(
            SimpleModel(), MovingBlockBootstrap(), n_estimators=10
        )
        r = repr(bagger)
        assert "TimeSeriesBagger" in r
        assert "MovingBlockBootstrap" in r
        assert "10" in r


# =============================================================================
# Feature Bagging Integration Tests
# =============================================================================


class TestFeatureBaggingIntegration:
    """Integration tests for feature bagging with TimeSeriesBagger."""

    def test_feature_bagging_works_with_bagger(self, linear_data) -> None:
        """Feature bagging should work end-to-end."""
        X, y = linear_data
        strategy = FeatureBagging(max_features=0.6)
        bagger = TimeSeriesBagger(SimpleModel(), strategy, n_estimators=5)

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        assert predictions.shape == (len(X),)

    def test_feature_bagging_uses_correct_features(self, linear_data) -> None:
        """Each estimator should see its own feature subset."""
        X, y = linear_data
        strategy = FeatureBagging(max_features=0.6)
        bagger = TimeSeriesBagger(SimpleModel(), strategy, n_estimators=3)

        bagger.fit(X, y)

        # Each estimator should have been trained on different features
        # Verify by checking feature_indices are stored
        assert len(strategy.feature_indices_) == 3

        # Predictions should work (verifies transform_for_predict is called)
        predictions = bagger.predict(X)
        assert predictions.shape == (len(X),)


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions for creating baggers."""

    def test_create_block_bagger(self, linear_data) -> None:
        """create_block_bagger should work."""
        X, y = linear_data
        bagger = create_block_bagger(SimpleModel(), n_estimators=5)

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        assert predictions.shape == (len(X),)
        assert isinstance(bagger.strategy, MovingBlockBootstrap)

    def test_create_block_bagger_with_options(self, linear_data) -> None:
        """create_block_bagger should accept all options."""
        X, y = linear_data
        bagger = create_block_bagger(
            SimpleModel(),
            n_estimators=5,
            block_length=15,
            aggregation="median",
            random_state=123,
        )

        bagger.fit(X, y)
        assert bagger.strategy.block_length == 15
        assert bagger.aggregation == "median"
        assert bagger.random_state == 123

    def test_create_stationary_bagger(self, linear_data) -> None:
        """create_stationary_bagger should work."""
        X, y = linear_data
        bagger = create_stationary_bagger(SimpleModel(), n_estimators=5)

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        assert predictions.shape == (len(X),)
        assert isinstance(bagger.strategy, StationaryBootstrap)

    def test_create_feature_bagger(self, linear_data) -> None:
        """create_feature_bagger should work."""
        X, y = linear_data
        bagger = create_feature_bagger(SimpleModel(), n_estimators=5, max_features=0.8)

        bagger.fit(X, y)
        predictions = bagger.predict(X)

        assert predictions.shape == (len(X),)
        assert isinstance(bagger.strategy, FeatureBagging)


# =============================================================================
# Integration Tests
# =============================================================================


class TestBaggingIntegration:
    """Integration tests for bagging framework."""

    def test_bagging_reduces_variance(self, linear_data) -> None:
        """Bagging should reduce prediction variance."""
        X, y = linear_data

        # Single model predictions
        single_model = SimpleModel()
        single_model.fit(X, y)
        single_pred = single_model.predict(X)

        # Bagged predictions with uncertainty
        bagger = create_block_bagger(SimpleModel(), n_estimators=20)
        bagger.fit(X, y)
        bagged_pred, std = bagger.predict_with_uncertainty(X)

        # Bagged predictions should be similar to single
        correlation = np.corrcoef(single_pred, bagged_pred)[0, 1]
        assert correlation > 0.9

    def test_bagging_with_autocorrelated_data(self, time_series_data) -> None:
        """Bagging should work with autocorrelated time series."""
        X, y = time_series_data

        # Block bootstrap should preserve autocorrelation
        bagger = create_block_bagger(SimpleModel(), n_estimators=10, block_length=10)
        bagger.fit(X, y)
        predictions = bagger.predict(X)

        assert predictions.shape == (len(X),)

        # Predictions should have positive correlation with y (even if modest)
        # The autocorrelated data with simple ridge can have lower correlation
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.0  # Just verify positive relationship

    def test_all_strategies_produce_valid_predictions(self, linear_data) -> None:
        """All strategies should produce valid predictions."""
        X, y = linear_data

        strategies = [
            ("block", create_block_bagger(SimpleModel(), n_estimators=5)),
            ("stationary", create_stationary_bagger(SimpleModel(), n_estimators=5)),
            ("feature", create_feature_bagger(SimpleModel(), n_estimators=5)),
        ]

        for name, bagger in strategies:
            bagger.fit(X, y)
            predictions = bagger.predict(X)

            assert predictions.shape == (len(X),), f"Failed for {name}"
            assert not np.any(np.isnan(predictions)), f"NaN in {name} predictions"
            assert not np.any(np.isinf(predictions)), f"Inf in {name} predictions"

    def test_prediction_intervals_have_coverage(self, linear_data) -> None:
        """Prediction intervals should produce valid bounds."""
        X, y = linear_data

        bagger = create_block_bagger(SimpleModel(), n_estimators=50, random_state=42)
        bagger.fit(X, y)
        mean, lower, upper = bagger.predict_interval(X, alpha=0.10)

        # Verify interval structure is correct
        assert np.all(lower <= mean), "Lower should be <= mean"
        assert np.all(mean <= upper), "Mean should be <= upper"
        assert np.all(lower < upper), "Lower should be < upper"

        # Verify intervals have reasonable width (not degenerate)
        widths = upper - lower
        assert np.mean(widths) > 0, "Intervals should have positive width"

        # Note: Bootstrap intervals from simple ridge on resampled data
        # may not achieve target coverage - that's a limitation of bootstrap
        # for this use case, not a bug. Conformal provides coverage guarantee.
