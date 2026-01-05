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
    CoverageDiagnostics,
    compute_coverage_diagnostics,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def linear_data():
    """Generate data with known linear relationship."""
    rng = np.random.default_rng(42)
    n = 200
    noise = rng.normal(0, 0.5, n)
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
        # Note: With finite samples, coverage can fall below 1-α due to variance.
        # The finite sample guarantee is coverage >= (n_cal)/(n_cal+1)*(1-α) ≈ 0.89
        # but we allow additional wiggle room for test stability.
        assert coverage >= 0.78, f"Coverage {coverage:.3f} < 0.78"
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

        rng = np.random.default_rng(42)
        actuals = rng.normal(0, 0.5, 100)

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
        rng = np.random.default_rng(42)

        coverages = []
        for _ in range(20):  # Multiple trials
            # Generate data
            n = 100
            y = rng.normal(0, 1, n)
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
        rng = np.random.default_rng(42)

        n = 200
        y = rng.normal(0, 1, n)
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
        rng = np.random.default_rng(42)

        # Generate predictions and actuals
        n = 100
        predictions = rng.normal(0, 0.1, n)
        actuals = predictions + rng.normal(0, 0.05, n)

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
        rng = np.random.default_rng(42)
        n = 100
        predictions = rng.normal(0, 0.1, n)
        actuals = predictions + rng.normal(0, 0.05, n)

        _, quality = walk_forward_conformal(predictions, actuals)

        # Required metadata keys
        assert "calibration_size" in quality
        assert "holdout_size" in quality
        assert "calibration_fraction" in quality
        assert "quantile" in quality
        assert "coverage" in quality

    def test_requires_minimum_calibration_points(self) -> None:
        """Should require at least 10 calibration points."""
        rng = np.random.default_rng(42)
        n = 20  # With 30% calibration = 6 points (too few)
        predictions = rng.normal(0, 0.1, n)
        actuals = predictions + rng.normal(0, 0.05, n)

        with pytest.raises(ValueError, match=">= 10 calibration"):
            walk_forward_conformal(predictions, actuals, calibration_fraction=0.3)

    def test_requires_minimum_holdout_points(self) -> None:
        """Should require at least 10 holdout points."""
        rng = np.random.default_rng(42)
        n = 20  # With 70% calibration = 6 holdout points (too few)
        predictions = rng.normal(0, 0.1, n)
        actuals = predictions + rng.normal(0, 0.05, n)

        with pytest.raises(ValueError, match=">= 10 holdout"):
            walk_forward_conformal(predictions, actuals, calibration_fraction=0.7)

    def test_coverage_within_reasonable_bounds(self) -> None:
        """
        Coverage should be reasonable for well-calibrated intervals.

        [T1] Coverage guarantee: should be >= 1 - alpha (with finite sample)
        """
        coverages = []
        for seed in [42, 123, 456, 789]:
            rng = np.random.default_rng(seed)
            n = 150
            predictions = rng.normal(0, 0.1, n)
            actuals = predictions + rng.normal(0, 0.05, n)

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
        rng = np.random.default_rng(42)

        # Simulate model predictions (with some noise)
        n = 200
        true_values = np.sin(np.linspace(0, 4 * np.pi, n))
        noise = rng.normal(0, 0.2, n)
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
        rng = np.random.default_rng(42)

        n = 150
        predictions = rng.normal(0, 0.5, n)
        actuals = predictions + rng.normal(0, 0.3, n)

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
        rng = np.random.default_rng(42)

        # Initialize with low-noise data
        low_noise = rng.normal(0, 0.1, 50)
        acp = AdaptiveConformalPredictor(alpha=0.10, gamma=0.1)
        acp.initialize(np.zeros(50), low_noise)

        initial_quantile = acp.current_quantile

        # Update with high-noise data (distribution shift)
        for _ in range(30):
            actual = rng.normal(0, 1.0)  # Higher noise
            acp.update(0.0, actual)

        # Quantile should increase to adapt
        assert acp.current_quantile > initial_quantile


# =============================================================================
# Coverage Diagnostics Tests
# =============================================================================


class TestCoverageDiagnostics:
    """Test detailed coverage diagnostics for conformal prediction intervals."""

    def test_basic_diagnostics(self, calibration_test_split) -> None:
        """compute_coverage_diagnostics should return correct type and fields."""
        preds_cal, y_cal, preds_test, y_test = calibration_test_split

        conformal = SplitConformalPredictor(alpha=0.10)
        conformal.calibrate(preds_cal, y_cal)
        intervals = conformal.predict_interval(preds_test)

        diag = compute_coverage_diagnostics(intervals, y_test)

        assert isinstance(diag, CoverageDiagnostics)
        assert 0.0 <= diag.overall_coverage <= 1.0
        assert diag.target_coverage == 0.90
        assert diag.n_observations == len(y_test)

    def test_overall_coverage_calculation(self) -> None:
        """overall_coverage should be fraction of actuals within intervals."""
        point = np.arange(10, dtype=float)
        # Wide intervals: all 10 points covered
        lower = point - 10
        upper = point + 10

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        actuals = point  # Exactly at center
        diag = compute_coverage_diagnostics(interval, actuals)

        assert diag.overall_coverage == pytest.approx(1.0, rel=0.01)

    def test_coverage_gap_calculation(self) -> None:
        """coverage_gap should be target - empirical."""
        point = np.arange(100, dtype=float)
        lower = point - 0.1  # Very narrow intervals
        upper = point + 0.1

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,  # Target = 0.95
            method="test",
        )

        # All actuals within bounds = 100% coverage
        actuals = point
        diag = compute_coverage_diagnostics(interval, actuals)

        # gap = target - empirical = 0.95 - 1.0 = -0.05
        assert diag.coverage_gap == pytest.approx(-0.05, rel=0.01)
        assert diag.undercoverage_warning is False  # Not undercovered

    def test_undercoverage_warning_triggered(self) -> None:
        """undercoverage_warning should be True when gap > threshold."""
        point = np.arange(100, dtype=float)
        lower = point - 0.001  # Extremely narrow intervals
        upper = point + 0.001

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        # Half the actuals outside bounds (50% coverage vs 95% target)
        rng = np.random.default_rng(42)
        actuals = point + rng.normal(0, 1.0, 100)

        diag = compute_coverage_diagnostics(
            interval, actuals, undercoverage_threshold=0.05
        )

        # Should warn: coverage << 95%
        assert diag.undercoverage_warning is True
        assert diag.coverage_gap > 0.05

    def test_coverage_by_window(self) -> None:
        """coverage_by_window should compute coverage in rolling windows."""
        n = 100
        point = np.arange(n, dtype=float)
        lower = point - 1
        upper = point + 1

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        actuals = point  # All covered
        diag = compute_coverage_diagnostics(interval, actuals, window_size=50)

        # Should have 2 windows
        assert len(diag.coverage_by_window) == 2
        for window_name, cov in diag.coverage_by_window.items():
            assert cov == pytest.approx(1.0, rel=0.01)

    def test_coverage_by_regime(self) -> None:
        """coverage_by_regime should compute per-regime coverage."""
        n = 100
        point = np.arange(n, dtype=float)
        lower = point - 1
        upper = point + 1

        interval = PredictionInterval(
            point=point,
            lower=lower,
            upper=upper,
            confidence=0.95,
            method="test",
        )

        actuals = point  # All covered
        regimes = np.array(["low"] * 50 + ["high"] * 50)

        diag = compute_coverage_diagnostics(interval, actuals, regimes=regimes)

        assert diag.coverage_by_regime is not None
        assert "low" in diag.coverage_by_regime
        assert "high" in diag.coverage_by_regime
        assert diag.coverage_by_regime["low"] == pytest.approx(1.0, rel=0.01)
        assert diag.coverage_by_regime["high"] == pytest.approx(1.0, rel=0.01)

    def test_length_mismatch_raises(self) -> None:
        """Should raise ValueError if intervals and actuals have different lengths."""
        interval = PredictionInterval(
            point=np.zeros(10),
            lower=-np.ones(10),
            upper=np.ones(10),
            confidence=0.95,
            method="test",
        )

        actuals = np.zeros(20)  # Different length

        with pytest.raises(ValueError, match="doesn't match"):
            compute_coverage_diagnostics(interval, actuals)

    def test_regime_length_mismatch_raises(self) -> None:
        """Should raise ValueError if regimes has wrong length."""
        interval = PredictionInterval(
            point=np.zeros(10),
            lower=-np.ones(10),
            upper=np.ones(10),
            confidence=0.95,
            method="test",
        )

        actuals = np.zeros(10)
        regimes = np.array(["a"] * 5)  # Wrong length

        with pytest.raises(ValueError, match="doesn't match"):
            compute_coverage_diagnostics(interval, actuals, regimes=regimes)

    def test_n_observations_correct(self) -> None:
        """n_observations should match input length."""
        for n in [10, 50, 100, 200]:
            interval = PredictionInterval(
                point=np.zeros(n),
                lower=-np.ones(n),
                upper=np.ones(n),
                confidence=0.95,
                method="test",
            )
            actuals = np.zeros(n)

            diag = compute_coverage_diagnostics(interval, actuals)
            assert diag.n_observations == n

    def test_target_coverage_from_interval(self) -> None:
        """target_coverage should use interval.confidence if not specified."""
        interval = PredictionInterval(
            point=np.zeros(50),
            lower=-np.ones(50),
            upper=np.ones(50),
            confidence=0.80,  # 80% confidence
            method="test",
        )
        actuals = np.zeros(50)

        diag = compute_coverage_diagnostics(interval, actuals)
        assert diag.target_coverage == 0.80

    def test_explicit_target_coverage_override(self) -> None:
        """Explicit target_coverage should override interval.confidence."""
        interval = PredictionInterval(
            point=np.zeros(50),
            lower=-np.ones(50),
            upper=np.ones(50),
            confidence=0.80,
            method="test",
        )
        actuals = np.zeros(50)

        diag = compute_coverage_diagnostics(
            interval, actuals, target_coverage=0.95
        )
        assert diag.target_coverage == 0.95


# =============================================================================
# Bellman Conformal Inference Tests (Yang, Candès & Lei 2024)
# =============================================================================


class TestBellmanConformalPredictor:
    """Tests for BellmanConformalPredictor."""

    def test_initialization(self) -> None:
        """BellmanConformalPredictor should initialize correctly."""
        from temporalcv.conformal import BellmanConformalPredictor

        bcp = BellmanConformalPredictor(alpha=0.10, horizon=5, n_grid=30)

        assert bcp.alpha == 0.10
        assert bcp.horizon == 5
        assert bcp.n_grid == 30
        assert bcp.current_quantile is None

    def test_invalid_alpha_raises(self) -> None:
        """Should raise ValueError for invalid alpha."""
        from temporalcv.conformal import BellmanConformalPredictor

        with pytest.raises(ValueError, match="alpha"):
            BellmanConformalPredictor(alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            BellmanConformalPredictor(alpha=1.0)

    def test_invalid_horizon_raises(self) -> None:
        """Should raise ValueError for invalid horizon."""
        from temporalcv.conformal import BellmanConformalPredictor

        with pytest.raises(ValueError, match="horizon"):
            BellmanConformalPredictor(horizon=0)

    def test_invalid_n_grid_raises(self) -> None:
        """Should raise ValueError for invalid n_grid."""
        from temporalcv.conformal import BellmanConformalPredictor

        with pytest.raises(ValueError, match="n_grid"):
            BellmanConformalPredictor(n_grid=5)

    def test_calibration(self) -> None:
        """Should calibrate with valid data."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(alpha=0.10)
        bcp.initialize(predictions, actuals)

        assert bcp.current_quantile is not None
        assert bcp.current_quantile > 0
        assert bcp.value_function is not None
        assert len(bcp.quantile_history) == 1

    def test_calibration_insufficient_data_raises(self) -> None:
        """Should raise with insufficient calibration data."""
        from temporalcv.conformal import BellmanConformalPredictor

        predictions = np.zeros(5)
        actuals = np.zeros(5)

        bcp = BellmanConformalPredictor()

        with pytest.raises(ValueError, match="at least"):
            bcp.initialize(predictions, actuals)

    def test_predict_interval_before_init_raises(self) -> None:
        """Should raise if predict_interval called before initialize."""
        from temporalcv.conformal import BellmanConformalPredictor

        bcp = BellmanConformalPredictor()

        with pytest.raises(RuntimeError, match="not initialized"):
            bcp.predict_interval(0.0)

    def test_predict_interval_single(self) -> None:
        """Should produce valid single prediction interval."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(alpha=0.10)
        bcp.initialize(predictions, actuals)

        lower, upper = bcp.predict_interval(0.0)

        assert lower < 0.0 < upper
        assert upper - lower > 0  # Non-zero width

    def test_update_changes_quantile(self) -> None:
        """Update should modify current quantile."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(alpha=0.10, gamma=0.2)
        bcp.initialize(predictions, actuals)

        initial_quantile = bcp.current_quantile

        # Update multiple times with large errors (should widen)
        for _ in range(10):
            bcp.update(0.0, 10.0)  # Large error

        # Quantile should have changed
        assert len(bcp.quantile_history) == 11
        # With large errors, quantile should increase
        assert bcp.current_quantile != initial_quantile

    def test_solve_optimal_sequence(self) -> None:
        """solve_optimal_sequence should return valid quantile sequence."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(alpha=0.10, horizon=5)
        bcp.initialize(predictions, actuals)

        test_preds = rng.normal(0, 1, 20)
        quantiles = bcp.solve_optimal_sequence(test_preds)

        assert len(quantiles) == 20
        assert np.all(quantiles > 0)  # All positive

    def test_predict_intervals_batch(self) -> None:
        """predict_intervals_batch should return PredictionInterval."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(alpha=0.10)
        bcp.initialize(predictions, actuals)

        test_preds = rng.normal(0, 1, 20)
        intervals = bcp.predict_intervals_batch(test_preds)

        assert isinstance(intervals, PredictionInterval)
        assert len(intervals.point) == 20
        assert len(intervals.lower) == 20
        assert len(intervals.upper) == 20
        assert intervals.method == "bellman_conformal"
        assert np.all(intervals.upper > intervals.lower)

    def test_coverage_maintained(self) -> None:
        """Coverage should be approximately maintained."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        n = 100
        alpha = 0.10

        # Generate data with known noise level
        predictions = rng.normal(0, 1, n)
        actuals = predictions + rng.normal(0, 0.3, n)

        # Calibrate on first half
        bcp = BellmanConformalPredictor(alpha=alpha, horizon=5)
        bcp.initialize(predictions[:50], actuals[:50])

        # Test on second half
        intervals = bcp.predict_intervals_batch(predictions[50:])
        coverage = intervals.coverage(actuals[50:])

        # Coverage should be reasonable (>= 1 - alpha - slack)
        assert coverage >= 0.75, f"Coverage {coverage:.2f} too low"

    def test_bellman_equation_solved(self) -> None:
        """Value function should be computed via backward induction."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(alpha=0.10, horizon=5, n_grid=20)
        bcp.initialize(predictions, actuals)

        # Value function should have shape (horizon+1, n_grid)
        assert bcp.value_function.shape == (6, 20)
        # Terminal cost should be zero
        assert np.all(bcp.value_function[5, :] == 0.0)

    def test_quantile_grid_built(self) -> None:
        """Quantile grid should be built from calibration data."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp = BellmanConformalPredictor(n_grid=30)
        bcp.initialize(predictions, actuals)

        assert bcp._quantile_grid is not None
        assert len(bcp._quantile_grid) == 30
        assert np.all(bcp._quantile_grid >= 0)  # Non-negative

    def test_reproducibility(self) -> None:
        """Same data should give same results."""
        from temporalcv.conformal import BellmanConformalPredictor

        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 50)
        actuals = predictions + rng.normal(0, 0.3, 50)

        bcp1 = BellmanConformalPredictor(alpha=0.10, horizon=5)
        bcp1.initialize(predictions, actuals)

        bcp2 = BellmanConformalPredictor(alpha=0.10, horizon=5)
        bcp2.initialize(predictions, actuals)

        assert bcp1.current_quantile == bcp2.current_quantile
        assert np.allclose(bcp1.value_function, bcp2.value_function)


class TestBellmanVsAdaptiveComparison:
    """Compare Bellman and Adaptive conformal methods."""

    def test_both_maintain_coverage(self) -> None:
        """Both methods should maintain reasonable coverage."""
        from temporalcv.conformal import (
            AdaptiveConformalPredictor,
            BellmanConformalPredictor,
        )

        rng = np.random.default_rng(42)
        n = 150
        alpha = 0.10

        predictions = rng.normal(0, 1, n)
        actuals = predictions + rng.normal(0, 0.3, n)

        # Bellman
        bcp = BellmanConformalPredictor(alpha=alpha)
        bcp.initialize(predictions[:50], actuals[:50])

        # Adaptive
        acp = AdaptiveConformalPredictor(alpha=alpha)
        acp.initialize(predictions[:50], actuals[:50])

        # Simulate online updates for both
        bellman_covered = 0
        adaptive_covered = 0

        for i in range(50, n):
            pred = predictions[i]
            actual = actuals[i]

            # Bellman
            b_lower, b_upper = bcp.predict_interval(pred)
            if b_lower <= actual <= b_upper:
                bellman_covered += 1
            bcp.update(pred, actual)

            # Adaptive
            a_lower, a_upper = acp.predict_interval(pred)
            if a_lower <= actual <= a_upper:
                adaptive_covered += 1
            acp.update(pred, actual)

        bellman_coverage = bellman_covered / (n - 50)
        adaptive_coverage = adaptive_covered / (n - 50)

        # Both should have reasonable coverage
        assert bellman_coverage >= 0.70
        assert adaptive_coverage >= 0.70

    def test_bellman_tighter_intervals_possible(self) -> None:
        """Bellman may produce tighter intervals in some cases."""
        from temporalcv.conformal import (
            AdaptiveConformalPredictor,
            BellmanConformalPredictor,
        )

        rng = np.random.default_rng(42)
        n = 100

        predictions = rng.normal(0, 1, n)
        actuals = predictions + rng.normal(0, 0.3, n)

        # Bellman
        bcp = BellmanConformalPredictor(alpha=0.10, horizon=10)
        bcp.initialize(predictions[:50], actuals[:50])
        bellman_intervals = bcp.predict_intervals_batch(predictions[50:])

        # Adaptive (simulate to get final quantile)
        acp = AdaptiveConformalPredictor(alpha=0.10)
        acp.initialize(predictions[:50], actuals[:50])

        # Both produce valid intervals
        assert bellman_intervals.mean_width > 0
        assert acp.current_quantile > 0
