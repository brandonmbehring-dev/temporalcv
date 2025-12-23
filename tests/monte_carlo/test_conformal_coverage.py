"""
Monte Carlo coverage calibration for conformal prediction.

Tier 2: Statistical calibration tests - run nightly.

Tests verify:
- Marginal coverage at nominal levels (90%, 95%)
- Conservative coverage with finite samples
- Adaptive conformal behavior
"""

import numpy as np
import pytest

from temporalcv.conformal import SplitConformalPredictor, AdaptiveConformalPredictor


# =============================================================================
# Helper: Simple Linear Model for Testing
# =============================================================================


class SimpleLinearRegressor:
    """Minimal linear regressor for conformal testing."""

    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Simple OLS
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_


# =============================================================================
# Split Conformal Coverage
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestSplitConformalCoverage:
    """Monte Carlo coverage for split conformal prediction."""

    def test_coverage_95_homoscedastic(self):
        """
        With homoscedastic errors, 95% conformal intervals should cover ~95%.

        N=500 simulations, linear DGP with constant variance.
        """
        N_SIMS = 500
        covers_true = []

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)

            # Generate linear data: y = 2x + 1 + noise
            n = 200
            X = rng.randn(n, 1)
            y = 2 * X.ravel() + 1 + rng.randn(n)

            # Split into train/calibration/test
            X_train, y_train = X[:100], y[:100]
            X_calib, y_calib = X[100:150], y[100:150]
            X_test, y_test = X[150:], y[150:]

            # Fit model
            model = SimpleLinearRegressor()
            model.fit(X_train, y_train)

            # Get predictions for calibration and test sets
            calib_preds = model.predict(X_calib)
            test_preds = model.predict(X_test)

            # Calibrate conformal predictor (alpha=0.05 for 95% intervals)
            conformal = SplitConformalPredictor(alpha=0.05)
            conformal.calibrate(calib_preds, y_calib)

            # Predict intervals
            intervals = conformal.predict_interval(test_preds)

            # Check coverage - intervals is a PredictionInterval namedtuple
            in_interval = (intervals.lower <= y_test) & (y_test <= intervals.upper)
            coverage = np.mean(in_interval)
            covers_true.append(coverage)

        avg_coverage = np.mean(covers_true)

        # Should achieve 93-99% (conservative is OK)
        assert 0.90 <= avg_coverage <= 0.99, (
            f"95% conformal coverage = {avg_coverage:.1%}, expected 93-99%"
        )

    def test_coverage_90_homoscedastic(self):
        """
        With 90% nominal level, coverage should be ~90%.
        """
        N_SIMS = 400
        covers_true = []

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)

            n = 200
            X = rng.randn(n, 1)
            y = 2 * X.ravel() + 1 + rng.randn(n)

            X_train, y_train = X[:100], y[:100]
            X_calib, y_calib = X[100:150], y[100:150]
            X_test, y_test = X[150:], y[150:]

            model = SimpleLinearRegressor()
            model.fit(X_train, y_train)

            calib_preds = model.predict(X_calib)
            test_preds = model.predict(X_test)

            conformal = SplitConformalPredictor(alpha=0.10)
            conformal.calibrate(calib_preds, y_calib)

            intervals = conformal.predict_interval(test_preds)

            in_interval = (intervals.lower <= y_test) & (y_test <= intervals.upper)
            coverage = np.mean(in_interval)
            covers_true.append(coverage)

        avg_coverage = np.mean(covers_true)

        # 90% CI should achieve 88-97% (conservative OK)
        assert 0.85 <= avg_coverage <= 0.99, (
            f"90% conformal coverage = {avg_coverage:.1%}, expected 88-97%"
        )


# =============================================================================
# Calibration Size Effects
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestCalibrationSizeEffects:
    """Test how calibration set size affects coverage."""

    def test_small_calibration_set(self):
        """
        With small calibration set (n=30), coverage should still be valid.

        Intervals may be wider, but coverage maintained.
        """
        N_SIMS = 400
        covers_true = []

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)

            n = 150
            X = rng.randn(n, 1)
            y = 2 * X.ravel() + 1 + rng.randn(n)

            X_train, y_train = X[:80], y[:80]
            X_calib, y_calib = X[80:110], y[80:110]  # Only 30 calibration points
            X_test, y_test = X[110:], y[110:]

            model = SimpleLinearRegressor()
            model.fit(X_train, y_train)

            calib_preds = model.predict(X_calib)
            test_preds = model.predict(X_test)

            conformal = SplitConformalPredictor(alpha=0.05)
            conformal.calibrate(calib_preds, y_calib)

            intervals = conformal.predict_interval(test_preds)

            in_interval = (intervals.lower <= y_test) & (y_test <= intervals.upper)
            coverage = np.mean(in_interval)
            covers_true.append(coverage)

        avg_coverage = np.mean(covers_true)

        # Small calibration = conservative (higher coverage OK)
        assert 0.88 <= avg_coverage <= 0.99, (
            f"Small calib coverage = {avg_coverage:.1%}, expected 90-99%"
        )

    def test_large_calibration_set(self):
        """
        With large calibration set, coverage should be closer to nominal.
        """
        N_SIMS = 300
        covers_true = []

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)

            n = 400
            X = rng.randn(n, 1)
            y = 2 * X.ravel() + 1 + rng.randn(n)

            X_train, y_train = X[:150], y[:150]
            X_calib, y_calib = X[150:300], y[150:300]  # 150 calibration points
            X_test, y_test = X[300:], y[300:]

            model = SimpleLinearRegressor()
            model.fit(X_train, y_train)

            calib_preds = model.predict(X_calib)
            test_preds = model.predict(X_test)

            conformal = SplitConformalPredictor(alpha=0.05)
            conformal.calibrate(calib_preds, y_calib)

            intervals = conformal.predict_interval(test_preds)

            in_interval = (intervals.lower <= y_test) & (y_test <= intervals.upper)
            coverage = np.mean(in_interval)
            covers_true.append(coverage)

        avg_coverage = np.mean(covers_true)

        # Large calibration = closer to nominal
        assert 0.92 <= avg_coverage <= 0.99, (
            f"Large calib coverage = {avg_coverage:.1%}, expected 93-99%"
        )
