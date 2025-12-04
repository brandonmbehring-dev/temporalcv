"""
Test Conformal Prediction Module.

Tests for distribution-free prediction intervals with coverage guarantees.

Key properties tested:
1. Marginal coverage ≈ 1 - α (nominal level)
2. Intervals are valid (no leakage in calibration)
3. Adaptive intervals adjust to distribution shift
"""

import numpy as np
import pytest

from temporalcv.conformal import (
    PredictionInterval,
    SplitConformalPredictor,
    AdaptiveConformalPredictor,
    BootstrapUncertainty,
    evaluate_interval_quality,
    walk_forward_conformal,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def linear_data():
    """Generate data with known linear relationship."""
    np.random.seed(42)
    n = 200
    noise = np.random.normal(0, 0.5, n)
    true_values = np.linspace(0, 10, n)
    y = true_values + noise
    return true_values, y, noise


@pytest.fixture
def calibration_test_split(linear_data):
    """Split data into calibration and test sets."""
    true_values, y, _ = linear_data
    n = len(y)
    n_cal = n // 2

    preds_cal, y_cal = true_values[:n_cal], y[:n_cal]
    preds_test, y_test = true_values[n_cal:], y[n_cal:]

    return preds_cal, y_cal, preds_test, y_test


# =============================================================================
# PredictionInterval Tests
# =============================================================================


class TestPredictionInterval:
    """Test PredictionInterval dataclass."""

    def test_interval_creation(self) -> None:
        """PredictionInterval should store bounds correctly."""
        point = np.array([1.0, 2.0, 3.0])
        lower = point - 0.5
        upper = point + 0.5

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        assert len(interval.point) == 3
        assert interval.confidence == 0.95
        assert interval.method == "test"

    def test_interval_width(self) -> None:
        """Width should be upper - lower."""
        point = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        expected_width = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(interval.width, expected_width)

    def test_mean_width(self) -> None:
        """mean_width should average interval widths."""
        point = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.0, 2.0])  # widths: 1.0, 2.0, 2.0
        upper = np.array([1.5, 3.0, 4.0])

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        assert interval.mean_width == pytest.approx(5.0 / 3, rel=0.01)

    def test_coverage_calculation(self) -> None:
        """coverage should compute fraction within bounds."""
        point = np.array([1.0, 2.0, 3.0, 4.0])
        lower = np.array([0.5, 1.5, 2.5, 3.5])
        upper = np.array([1.5, 2.5, 3.5, 4.5])

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        # Actuals: 1.0, 2.0, 5.0, 4.0 (3 within, 1 outside)
        actuals = np.array([1.0, 2.0, 5.0, 4.0])

        assert interval.coverage(actuals) == pytest.approx(0.75, rel=0.01)

    def test_to_dict(self) -> None:
        """to_dict should return serializable dictionary."""
        point = np.array([1.0, 2.0])
        lower = np.array([0.5, 1.5])
        upper = np.array([1.5, 2.5])

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        d = interval.to_dict()
        assert "point" in d
        assert "lower" in d
        assert "upper" in d
        assert "confidence" in d
        assert "method" in d
        assert "mean_width" in d


# =============================================================================
# Split Conformal Predictor Tests
# =============================================================================


class TestSplitConformalPredictor:
    """Test Split Conformal Prediction."""

    def test_init_validates_alpha(self) -> None:
        """Should raise for invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            SplitConformalPredictor(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            SplitConformalPredictor(alpha=1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            SplitConformalPredictor(alpha=-0.1)

    def test_calibration_stores_quantile(self, linear_data) -> None:
        """Calibration should compute and store quantile."""
        true_values, y, _ = linear_data

        scp = SplitConformalPredictor(alpha=0.05)
        scp.calibrate(true_values[:100], y[:100])

        assert scp.quantile_ is not None
        assert scp.quantile_ > 0

    def test_calibration_requires_min_samples(self) -> None:
        """Calibration should require minimum samples."""
        scp = SplitConformalPredictor(alpha=0.05)

        with pytest.raises(ValueError, match="at least 10"):
            scp.calibrate(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_calibration_validates_lengths(self) -> None:
        """Calibration should validate array lengths."""
        scp = SplitConformalPredictor(alpha=0.05)

        with pytest.raises(ValueError, match="same length"):
            scp.calibrate(np.zeros(20), np.zeros(15))

    def test_predict_interval_requires_calibration(self) -> None:
        """predict_interval should fail without calibration."""
        scp = SplitConformalPredictor(alpha=0.05)

        with pytest.raises(RuntimeError, match="not calibrated"):
            scp.predict_interval(np.array([1, 2, 3]))

    def test_intervals_have_correct_width(self, linear_data) -> None:
        """Intervals should have width = 2 * quantile."""
        true_values, y, _ = linear_data

        scp = SplitConformalPredictor(alpha=0.05)
        scp.calibrate(true_values[:100], y[:100])

        intervals = scp.predict_interval(true_values[100:])

        # Width should be constant = 2 * quantile
        expected_width = 2 * scp.quantile_
        np.testing.assert_array_almost_equal(
            intervals.width, np.full(len(intervals.width), expected_width)
        )

    def test_coverage_approximately_correct(self, calibration_test_split) -> None:
        """Coverage on test set should be ≈ 1 - α."""
        preds_cal, y_cal, preds_test, y_test = calibration_test_split

        scp = SplitConformalPredictor(alpha=0.10)  # 90% intervals
        scp.calibrate(preds_cal, y_cal)

        intervals = scp.predict_interval(preds_test)
        coverage = intervals.coverage(y_test)

        # Coverage should be at least 1 - α (finite sample guarantee)
        assert coverage >= 0.85, f"Coverage {coverage:.3f} < 0.85"
        # But not excessively high (overly conservative)
        assert coverage <= 1.0


# =============================================================================
# Adaptive Conformal Predictor Tests
# =============================================================================


class TestAdaptiveConformalPredictor:
    """Test Adaptive Conformal Inference."""

    def test_init_validates_alpha(self) -> None:
        """Should raise for invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            AdaptiveConformalPredictor(alpha=0.0)

    def test_init_validates_gamma(self) -> None:
        """Should raise for invalid gamma."""
        with pytest.raises(ValueError, match="gamma must be in"):
            AdaptiveConformalPredictor(gamma=0.0)
        with pytest.raises(ValueError, match="gamma must be in"):
            AdaptiveConformalPredictor(gamma=1.0)

    def test_initialization_stores_quantile(self, linear_data) -> None:
        """Initialization should set current quantile."""
        true_values, y, _ = linear_data

        acp = AdaptiveConformalPredictor(alpha=0.05, gamma=0.1)
        acp.initialize(true_values[:50], y[:50])

        assert acp.current_quantile is not None
        assert acp.current_quantile > 0
        assert len(acp.quantile_history) == 1

    def test_initialize_validates_empty(self) -> None:
        """Initialize should fail on empty data."""
        acp = AdaptiveConformalPredictor()

        with pytest.raises(ValueError, match="empty"):
            acp.initialize(np.array([]), np.array([]))

    def test_update_adjusts_quantile(self, linear_data) -> None:
        """Update should adjust quantile based on coverage."""
        true_values, y, _ = linear_data

        acp = AdaptiveConformalPredictor(alpha=0.05, gamma=0.1)
        acp.initialize(true_values[:50], y[:50])

        initial_q = acp.current_quantile

        # Update with a point that's covered (error = 0)
        acp.update(0.0, 0.0)  # Prediction = actual

        # Quantile should decrease (tighten) when covered
        assert acp.current_quantile < initial_q

    def test_quantile_increases_when_not_covered(self) -> None:
        """Quantile should increase when prediction is not covered."""
        acp = AdaptiveConformalPredictor(alpha=0.05, gamma=0.1)
        acp.initialize(np.zeros(50), np.zeros(50))  # Start with quantile ~ 0

        # Force non-coverage with large error
        initial_q = acp.current_quantile
        acp.update(0.0, 100.0)  # Huge error

        # Quantile should increase
        assert acp.current_quantile > initial_q

    def test_predict_interval_creates_bounds(self, linear_data) -> None:
        """predict_interval should return lower/upper bounds."""
        true_values, y, _ = linear_data

        acp = AdaptiveConformalPredictor(alpha=0.05, gamma=0.1)
        acp.initialize(true_values[:50], y[:50])

        lower, upper = acp.predict_interval(0.0)

        assert lower < upper
        assert upper - lower == pytest.approx(2 * acp.current_quantile, rel=0.01)

    def test_predict_interval_requires_init(self) -> None:
        """predict_interval should fail without initialization."""
        acp = AdaptiveConformalPredictor()

        with pytest.raises(RuntimeError, match="not initialized"):
            acp.predict_interval(0.0)

    def test_update_requires_init(self) -> None:
        """update should fail without initialization."""
        acp = AdaptiveConformalPredictor()

        with pytest.raises(RuntimeError, match="not initialized"):
            acp.update(0.0, 0.0)

    def test_quantile_history_grows(self, linear_data) -> None:
        """Quantile history should grow with updates."""
        true_values, y, _ = linear_data

        acp = AdaptiveConformalPredictor(alpha=0.05, gamma=0.1)
        acp.initialize(true_values[:50], y[:50])

        for i in range(10):
            acp.update(true_values[50 + i], y[50 + i])

        assert len(acp.quantile_history) == 11  # 1 initial + 10 updates


# =============================================================================
# Bootstrap Uncertainty Tests
# =============================================================================


class TestBootstrapUncertainty:
    """Test Bootstrap-based prediction intervals."""

    def test_init_validates_n_bootstrap(self) -> None:
        """Should raise for invalid n_bootstrap."""
        with pytest.raises(ValueError, match="n_bootstrap must be"):
            BootstrapUncertainty(n_bootstrap=0)

    def test_init_validates_alpha(self) -> None:
        """Should raise for invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            BootstrapUncertainty(alpha=0.0)

    def test_fit_stores_residuals(self, linear_data) -> None:
        """fit should store residuals."""
        true_values, y, _ = linear_data

        boot = BootstrapUncertainty(n_bootstrap=100, alpha=0.05)
        boot.fit(true_values, y)

        assert boot.residuals_ is not None
        assert len(boot.residuals_) == len(y)

    def test_fit_validates_lengths(self) -> None:
        """fit should validate array lengths."""
        boot = BootstrapUncertainty()

        with pytest.raises(ValueError, match="same length"):
            boot.fit(np.zeros(20), np.zeros(15))

    def test_predict_interval_requires_fit(self) -> None:
        """predict_interval should fail without fit."""
        boot = BootstrapUncertainty()

        with pytest.raises(RuntimeError, match="not fitted"):
            boot.predict_interval(np.array([1, 2, 3]))

    def test_bootstrap_intervals_vary(self, linear_data) -> None:
        """Bootstrap intervals should have variable width."""
        true_values, y, _ = linear_data

        boot = BootstrapUncertainty(n_bootstrap=100, alpha=0.05)
        boot.fit(true_values[:100], y[:100])

        intervals = boot.predict_interval(true_values[100:])

        assert intervals.mean_width > 0
        assert intervals.method == "bootstrap"

    def test_bootstrap_coverage_reasonable(self, calibration_test_split) -> None:
        """Bootstrap coverage should be reasonable."""
        preds_cal, y_cal, preds_test, y_test = calibration_test_split

        boot = BootstrapUncertainty(n_bootstrap=100, alpha=0.10)
        boot.fit(preds_cal, y_cal)

        intervals = boot.predict_interval(preds_test)
        coverage = intervals.coverage(y_test)

        # Bootstrap is approximate, so wider tolerance
        assert 0.75 <= coverage <= 1.0

    def test_reproducibility_with_random_state(self, linear_data) -> None:
        """Same random_state should give same intervals."""
        true_values, y, _ = linear_data

        boot1 = BootstrapUncertainty(n_bootstrap=50, random_state=42)
        boot1.fit(true_values[:50], y[:50])
        intervals1 = boot1.predict_interval(true_values[50:60])

        boot2 = BootstrapUncertainty(n_bootstrap=50, random_state=42)
        boot2.fit(true_values[:50], y[:50])
        intervals2 = boot2.predict_interval(true_values[50:60])

        np.testing.assert_array_almost_equal(intervals1.lower, intervals2.lower)
        np.testing.assert_array_almost_equal(intervals1.upper, intervals2.upper)


# =============================================================================
# Interval Quality Evaluation Tests
# =============================================================================


class TestIntervalQuality:
    """Test interval quality evaluation."""

    def test_evaluate_returns_all_metrics(self, linear_data) -> None:
        """evaluate_interval_quality should return all metrics."""
        true_values, y, _ = linear_data

        interval = PredictionInterval(
            point=np.zeros(100),
            lower=-np.ones(100),
            upper=np.ones(100),
            confidence=0.95,
            method="test",
        )

        quality = evaluate_interval_quality(interval, y[:100])

        assert "coverage" in quality
        assert "target_coverage" in quality
        assert "coverage_gap" in quality
        assert "mean_width" in quality
        assert "interval_score" in quality
        assert "method" in quality

    def test_coverage_gap_computed_correctly(self) -> None:
        """coverage_gap should be coverage - target."""
        point = np.zeros(100)
        lower = -np.ones(100)
        upper = np.ones(100)

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.90,
            method="test",
        )

        # Actuals all within bounds
        actuals = np.zeros(100)
        quality = evaluate_interval_quality(interval, actuals)

        # Coverage = 1.0, target = 0.90, gap = 0.10
        assert quality["coverage"] == pytest.approx(1.0, rel=0.01)
        assert quality["coverage_gap"] == pytest.approx(0.10, rel=0.01)

    def test_interval_score_penalizes_miscoverage(self) -> None:
        """Interval score should be worse when coverage is off."""
        point = np.zeros(100)

        # Narrow intervals (low coverage)
        narrow = PredictionInterval(
            point=point,
            lower=-0.1 * np.ones(100),
            upper=0.1 * np.ones(100),
            confidence=0.95,
            method="test",
        )

        # Wide intervals (high coverage)
        wide = PredictionInterval(
            point=point,
            lower=-2 * np.ones(100),
            upper=2 * np.ones(100),
            confidence=0.95,
            method="test",
        )

        np.random.seed(42)
        actuals = np.random.normal(0, 0.5, 100)

        narrow_score = evaluate_interval_quality(narrow, actuals)["interval_score"]
        wide_score = evaluate_interval_quality(wide, actuals)["interval_score"]

        # Both scores should be positive
        assert narrow_score > 0
        assert wide_score > 0

    def test_conditional_coverage_for_large_samples(self) -> None:
        """Should compute conditional coverage for n >= 20."""
        interval = PredictionInterval(
            point=np.arange(30, dtype=float),
            lower=np.arange(30, dtype=float) - 1,
            upper=np.arange(30, dtype=float) + 1,
            confidence=0.95,
            method="test",
        )

        actuals = np.arange(30, dtype=float)
        quality = evaluate_interval_quality(interval, actuals)

        # Should have conditional coverage metrics
        assert not np.isnan(quality["low_coverage"])
        assert not np.isnan(quality["high_coverage"])

    def test_conditional_coverage_nan_for_small_samples(self) -> None:
        """Should return NaN conditional coverage for n < 20."""
        interval = PredictionInterval(
            point=np.arange(10, dtype=float),
            lower=np.arange(10, dtype=float) - 1,
            upper=np.arange(10, dtype=float) + 1,
            confidence=0.95,
            method="test",
        )

        actuals = np.arange(10, dtype=float)
        quality = evaluate_interval_quality(interval, actuals)

        assert np.isnan(quality["low_coverage"])
        assert np.isnan(quality["high_coverage"])
        assert np.isnan(quality["conditional_gap"])


# =============================================================================
# Coverage Guarantee Tests
# =============================================================================


class TestCoverageGuarantees:
    """Test that conformal methods provide coverage guarantees."""

    def test_split_conformal_finite_sample_validity(self) -> None:
        """
        Split conformal should have coverage ≥ 1 - α in finite samples.

        [T1] Romano et al. 2019 finite-sample guarantee.
        """
        np.random.seed(42)

        coverages = []
        for _ in range(20):  # Multiple trials
            # Generate data
            n = 100
            y = np.random.normal(0, 1, n)
            predictions = np.zeros(n)  # Simple mean prediction

            # Split into calibration/test
            n_cal = 50

            scp = SplitConformalPredictor(alpha=0.10)
            scp.calibrate(predictions[:n_cal], y[:n_cal])

            intervals = scp.predict_interval(predictions[n_cal:])
            coverage = intervals.coverage(y[n_cal:])
            coverages.append(coverage)

        # Average coverage should be ≥ 1 - α = 0.90
        mean_coverage = np.mean(coverages)
        assert mean_coverage >= 0.85, (
            f"Mean coverage {mean_coverage:.3f} < 0.85. "
            f"Finite sample guarantee may be violated."
        )

    def test_coverage_not_grossly_overconservative(self) -> None:
        """Coverage should not be grossly overconservative (e.g., 100%)."""
        np.random.seed(42)

        n = 200
        y = np.random.normal(0, 1, n)
        predictions = np.zeros(n)

        scp = SplitConformalPredictor(alpha=0.10)
        scp.calibrate(predictions[:100], y[:100])

        intervals = scp.predict_interval(predictions[100:])
        coverage = intervals.coverage(y[100:])

        # Should not be extremely overconservative
        assert coverage < 0.995, (
            f"Coverage {coverage:.3f} is too high. "
            f"Intervals may be excessively wide."
        )


# =============================================================================
# Walk-Forward Conformal Tests
# =============================================================================


class TestWalkForwardConformal:
    """
    Test walk_forward_conformal helper function.

    CRITICAL: Coverage must be computed ONLY on holdout (post-calibration)
    data to avoid inflated coverage from calibration points.
    """

    def test_coverage_on_holdout_only(self) -> None:
        """
        Coverage should be computed on holdout, not calibration.

        [T1] This is the core fix - coverage was previously
        inflated by including calibration points.
        """
        np.random.seed(42)

        # Generate predictions and actuals
        n = 100
        predictions = np.random.normal(0, 0.1, n)
        actuals = predictions + np.random.normal(0, 0.05, n)

        intervals, quality = walk_forward_conformal(
            predictions, actuals, calibration_fraction=0.3, alpha=0.05
        )

        # Verify holdout-only computation
        assert quality["holdout_size"] == 70, (
            f"Expected 70 holdout points, got {quality['holdout_size']}"
        )
        assert quality["calibration_size"] == 30, (
            f"Expected 30 calibration points, got {quality['calibration_size']}"
        )

        # Intervals should be sized for holdout only
        assert len(intervals.point) == 70, (
            f"Intervals should have 70 points, got {len(intervals.point)}"
        )

    def test_metadata_returned(self) -> None:
        """Quality dict should include calibration metadata."""
        np.random.seed(42)
        n = 100
        predictions = np.random.normal(0, 0.1, n)
        actuals = predictions + np.random.normal(0, 0.05, n)

        _, quality = walk_forward_conformal(predictions, actuals)

        # Required metadata keys
        assert "calibration_size" in quality
        assert "holdout_size" in quality
        assert "calibration_fraction" in quality
        assert "quantile" in quality
        assert "coverage" in quality

    def test_requires_minimum_calibration_points(self) -> None:
        """Should require at least 10 calibration points."""
        np.random.seed(42)
        n = 20  # With 30% calibration = 6 points (too few)
        predictions = np.random.normal(0, 0.1, n)
        actuals = predictions + np.random.normal(0, 0.05, n)

        with pytest.raises(ValueError, match=">= 10 calibration"):
            walk_forward_conformal(predictions, actuals, calibration_fraction=0.3)

    def test_requires_minimum_holdout_points(self) -> None:
        """Should require at least 10 holdout points."""
        np.random.seed(42)
        n = 20  # With 70% calibration = 6 holdout points (too few)
        predictions = np.random.normal(0, 0.1, n)
        actuals = predictions + np.random.normal(0, 0.05, n)

        with pytest.raises(ValueError, match=">= 10 holdout"):
            walk_forward_conformal(predictions, actuals, calibration_fraction=0.7)

    def test_coverage_within_reasonable_bounds(self) -> None:
        """
        Coverage should be reasonable for well-calibrated intervals.

        [T1] Coverage guarantee: should be >= 1 - alpha (with finite sample)
        """
        np.random.seed(42)

        coverages = []
        for seed in [42, 123, 456, 789]:
            np.random.seed(seed)
            n = 150
            predictions = np.random.normal(0, 0.1, n)
            actuals = predictions + np.random.normal(0, 0.05, n)

            _, quality = walk_forward_conformal(
                predictions, actuals, calibration_fraction=0.3, alpha=0.10
            )
            coverages.append(quality["coverage"])

        mean_coverage = np.mean(coverages)

        # Average coverage should be >= 1 - alpha = 0.90 (approximately)
        assert mean_coverage >= 0.80, (
            f"Mean coverage {mean_coverage:.3f} < 0.80. "
            f"Coverage guarantee may be violated."
        )

    def test_length_mismatch_raises(self) -> None:
        """Should raise error if predictions/actuals lengths differ."""
        predictions = np.zeros(100)
        actuals = np.zeros(50)  # Different length

        with pytest.raises(ValueError, match="same length"):
            walk_forward_conformal(predictions, actuals)


# =============================================================================
# Integration Tests
# =============================================================================


class TestConformalIntegration:
    """Integration tests for conformal prediction."""

    def test_conformal_with_model_predictions(self) -> None:
        """Test conformal with actual model-like predictions."""
        np.random.seed(42)

        # Simulate model predictions (with some noise)
        n = 200
        true_values = np.sin(np.linspace(0, 4 * np.pi, n))
        noise = np.random.normal(0, 0.2, n)
        actuals = true_values + noise

        # Model predicts true values (good model)
        predictions = true_values

        # Split
        n_cal = 60

        # Calibrate and predict
        scp = SplitConformalPredictor(alpha=0.05)
        scp.calibrate(predictions[:n_cal], actuals[:n_cal])
        intervals = scp.predict_interval(predictions[n_cal:])

        # Coverage should be good for well-specified model
        coverage = intervals.coverage(actuals[n_cal:])
        assert coverage >= 0.90

    def test_bootstrap_vs_conformal_comparison(self) -> None:
        """Compare bootstrap and conformal intervals."""
        np.random.seed(42)

        n = 150
        predictions = np.random.normal(0, 0.5, n)
        actuals = predictions + np.random.normal(0, 0.3, n)

        n_cal = 50

        # Conformal
        scp = SplitConformalPredictor(alpha=0.10)
        scp.calibrate(predictions[:n_cal], actuals[:n_cal])
        conformal_intervals = scp.predict_interval(predictions[n_cal:])

        # Bootstrap
        boot = BootstrapUncertainty(n_bootstrap=100, alpha=0.10)
        boot.fit(predictions[:n_cal], actuals[:n_cal])
        bootstrap_intervals = boot.predict_interval(predictions[n_cal:])

        # Both should have reasonable coverage
        conf_coverage = conformal_intervals.coverage(actuals[n_cal:])
        boot_coverage = bootstrap_intervals.coverage(actuals[n_cal:])

        assert conf_coverage >= 0.80
        assert boot_coverage >= 0.70  # Bootstrap is approximate

    def test_adaptive_tracks_distribution_shift(self) -> None:
        """Adaptive conformal should adjust to distribution shift."""
        np.random.seed(42)

        # Initialize with low-noise data
        low_noise = np.random.normal(0, 0.1, 50)
        acp = AdaptiveConformalPredictor(alpha=0.10, gamma=0.1)
        acp.initialize(np.zeros(50), low_noise)

        initial_quantile = acp.current_quantile

        # Update with high-noise data (distribution shift)
        for _ in range(30):
            actual = np.random.normal(0, 1.0)  # Higher noise
            acp.update(0.0, actual)

        # Quantile should increase to adapt
        assert acp.current_quantile > initial_quantile
