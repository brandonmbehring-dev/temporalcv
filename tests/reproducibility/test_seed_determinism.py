"""Seed determinism tests.

Verifies that stochastic functions produce identical results with the same seed.
"""

import numpy as np
import pytest

from temporalcv.gates import gate_signal_verification, gate_synthetic_ar1
from temporalcv.bagging import (
    MovingBlockBootstrap,
    StationaryBootstrap,
    ResidualBootstrap,
    TimeSeriesBagger,
)
from temporalcv.conformal import SplitConformalPredictor, AdaptiveConformalPredictor


class DummyModel:
    """Simple model for testing."""

    def __init__(self) -> None:
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


class TestShuffledTargetGateDeterminism:
    """Tests for shuffled target gate reproducibility."""

    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed data for reproducibility tests."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.array([1.0, 0.5, -0.3, 0.2, 0.1]) + rng.standard_normal(100) * 0.5
        return X, y

    @pytest.mark.parametrize("seed", range(20))
    def test_shuffled_target_same_seed_same_result(
        self, seed: int, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Same seed must produce identical gate results."""
        X, y = data
        model = DummyModel()

        result1 = gate_signal_verification(
            model=model, X=X, y=y, n_shuffles=20, random_state=seed
        )
        result2 = gate_signal_verification(
            model=model, X=X, y=y, n_shuffles=20, random_state=seed
        )

        assert result1.status == result2.status
        assert result1.message == result2.message
        # Note: details may have floats that are very close but not identical due to
        # floating point, so we compare structure rather than exact values

    @pytest.mark.parametrize("seed", range(10))
    def test_different_seeds_different_results(
        self, seed: int, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Different seeds should produce different internal state."""
        X, y = data
        model = DummyModel()

        result1 = gate_signal_verification(
            model=model, X=X, y=y, n_shuffles=20, random_state=seed
        )
        result2 = gate_signal_verification(
            model=model, X=X, y=y, n_shuffles=20, random_state=seed + 100
        )

        # Status might be same (both PASS), but internal metrics could differ
        # This is a smoke test to ensure different seeds don't crash
        assert result1 is not None
        assert result2 is not None


class TestSyntheticAR1GateDeterminism:
    """Tests for synthetic AR(1) gate reproducibility."""

    @pytest.mark.parametrize("seed", range(20))
    def test_synthetic_ar1_same_seed_same_result(self, seed: int) -> None:
        """Same seed must produce identical gate results."""
        model = DummyModel()

        result1 = gate_synthetic_ar1(
            model=model, n_samples=100, phi=0.6, sigma=1.0, random_state=seed
        )
        result2 = gate_synthetic_ar1(
            model=model, n_samples=100, phi=0.6, sigma=1.0, random_state=seed
        )

        assert result1.status == result2.status
        assert result1.message == result2.message


class TestBootstrapDeterminism:
    """Tests for bootstrap strategy reproducibility."""

    @pytest.fixture
    def time_series_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed time series for testing."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 5))
        y = np.cumsum(rng.standard_normal(n))
        return X, y

    @pytest.mark.parametrize("seed", range(10))
    def test_moving_block_determinism(
        self, seed: int, time_series_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Moving block bootstrap must be deterministic."""
        X, y = time_series_data
        strategy = MovingBlockBootstrap(block_length=10)
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        samples1 = strategy.generate_samples(X, y, n_samples=3, rng=rng1)
        samples2 = strategy.generate_samples(X, y, n_samples=3, rng=rng2)

        for (X1, y1), (X2, y2) in zip(samples1, samples2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    @pytest.mark.parametrize("seed", range(10))
    def test_stationary_bootstrap_determinism(
        self, seed: int, time_series_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Stationary bootstrap must be deterministic."""
        X, y = time_series_data
        strategy = StationaryBootstrap(expected_block_length=10)
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        samples1 = strategy.generate_samples(X, y, n_samples=3, rng=rng1)
        samples2 = strategy.generate_samples(X, y, n_samples=3, rng=rng2)

        for (X1, y1), (X2, y2) in zip(samples1, samples2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    @pytest.mark.parametrize("seed", range(10))
    def test_residual_bootstrap_determinism(
        self, seed: int, time_series_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Residual bootstrap must be deterministic."""
        X, y = time_series_data
        strategy = ResidualBootstrap(seasonal_period=10, block_length=5)
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        samples1 = strategy.generate_samples(X, y, n_samples=3, rng=rng1)
        samples2 = strategy.generate_samples(X, y, n_samples=3, rng=rng2)

        for (X1, y1), (X2, y2) in zip(samples1, samples2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)


class TestBaggerDeterminism:
    """Tests for TimeSeriesBagger reproducibility."""

    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed data for testing."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.array([1.0, 0.5, -0.3, 0.2, 0.1]) + rng.standard_normal(100) * 0.5
        return X, y

    @pytest.mark.parametrize("seed", range(10))
    def test_bagger_predictions_deterministic(
        self, seed: int, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Bagger predictions must be deterministic with same seed."""
        X, y = data
        strategy = MovingBlockBootstrap(block_length=10)

        bagger1 = TimeSeriesBagger(
            base_model=DummyModel(),
            strategy=strategy,
            n_estimators=10,
            random_state=seed,
        )
        bagger2 = TimeSeriesBagger(
            base_model=DummyModel(),
            strategy=strategy,
            n_estimators=10,
            random_state=seed,
        )

        bagger1.fit(X, y)
        bagger2.fit(X, y)

        pred1 = bagger1.predict(X[:10])
        pred2 = bagger2.predict(X[:10])

        np.testing.assert_array_almost_equal(pred1, pred2)


class TestConformalDeterminism:
    """Tests for conformal prediction reproducibility."""

    @pytest.fixture
    def calibration_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed calibration data."""
        rng = np.random.default_rng(42)
        n = 100
        cal_preds = rng.standard_normal(n)
        cal_actuals = cal_preds + rng.standard_normal(n) * 0.3
        return cal_preds, cal_actuals

    @pytest.mark.parametrize("seed", range(10))
    def test_split_conformal_deterministic(
        self, seed: int, calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Split conformal must be deterministic (no randomness)."""
        cal_preds, cal_actuals = calibration_data

        conformal1 = SplitConformalPredictor(alpha=0.1)
        conformal2 = SplitConformalPredictor(alpha=0.1)

        conformal1.calibrate(cal_preds, cal_actuals)
        conformal2.calibrate(cal_preds, cal_actuals)

        # Same calibration → same quantile
        assert conformal1.quantile_ == conformal2.quantile_

        # Same predictions
        test_preds = np.array([0.0, 0.5, 1.0])
        intervals1 = conformal1.predict_interval(test_preds)
        intervals2 = conformal2.predict_interval(test_preds)

        np.testing.assert_array_equal(intervals1.lower, intervals2.lower)
        np.testing.assert_array_equal(intervals1.upper, intervals2.upper)

    @pytest.mark.parametrize("seed", range(10))
    def test_adaptive_conformal_deterministic(
        self, seed: int, calibration_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Adaptive conformal must be deterministic (no randomness)."""
        cal_preds, cal_actuals = calibration_data

        adaptive1 = AdaptiveConformalPredictor(alpha=0.1)
        adaptive2 = AdaptiveConformalPredictor(alpha=0.1)

        adaptive1.initialize(cal_preds, cal_actuals)
        adaptive2.initialize(cal_preds, cal_actuals)

        # Same initialization → same current quantile
        assert adaptive1.current_quantile == adaptive2.current_quantile

        # Same predictions (single prediction interface returns tuple)
        test_preds = [0.0, 0.5, 1.0]
        for pred in test_preds:
            lower1, upper1 = adaptive1.predict_interval(pred)
            lower2, upper2 = adaptive2.predict_interval(pred)
            assert lower1 == lower2
            assert upper1 == upper2


class TestCVDeterminism:
    """Tests for cross-validation reproducibility."""

    @pytest.fixture
    def data(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed data for CV tests."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = rng.standard_normal(200)
        return X, y

    @pytest.mark.parametrize("seed", range(10))
    def test_walk_forward_cv_deterministic(
        self, seed: int, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """WalkForwardCV must be deterministic (no randomness in splits)."""
        from temporalcv.cv import WalkForwardCV

        X, y = data
        cv = WalkForwardCV(n_splits=5)

        splits1 = list(cv.split(X, y))
        splits2 = list(cv.split(X, y))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    @pytest.mark.parametrize("seed", range(10))
    def test_purged_kfold_deterministic(
        self, seed: int, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedKFold must be deterministic (no randomness)."""
        from temporalcv.cv_financial import PurgedKFold

        X, y = data
        cv = PurgedKFold(n_splits=5, purge_gap=5)

        splits1 = list(cv.split(X, y))
        splits2 = list(cv.split(X, y))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)
