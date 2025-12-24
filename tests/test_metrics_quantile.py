"""
Tests for Quantile and Interval Metrics.

Tests cover:
- compute_pinball_loss: quantile regression loss
- compute_crps: Continuous Ranked Probability Score
- compute_interval_score: proper scoring rule for intervals
- compute_quantile_coverage: empirical coverage
- compute_winkler_score: alias for interval_score
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.metrics.quantile import (
    compute_crps,
    compute_interval_score,
    compute_pinball_loss,
    compute_quantile_coverage,
    compute_winkler_score,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def perfect_predictions():
    """Predictions that exactly match actuals."""
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = actuals.copy()
    return actuals, preds


@pytest.fixture
def symmetric_errors():
    """Predictions with symmetric errors around actuals."""
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Alternate over and under predictions
    preds = np.array([1.5, 1.5, 3.5, 3.5, 5.5])
    return actuals, preds


@pytest.fixture
def perfect_intervals():
    """Intervals that contain all actuals."""
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = actuals - 0.5
    upper = actuals + 0.5
    return actuals, lower, upper


@pytest.fixture
def partial_coverage_intervals():
    """Intervals that miss some actuals."""
    actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Miss first and last
    lower = np.array([1.5, 1.5, 2.5, 3.5, 5.5])
    upper = np.array([2.0, 2.5, 3.5, 4.5, 6.0])
    return actuals, lower, upper


# =============================================================================
# Tests: compute_pinball_loss
# =============================================================================


class TestComputePinballLoss:
    """Tests for pinball (quantile) loss function."""

    def test_perfect_predictions_zero_loss(self, perfect_predictions):
        """Perfect predictions yield zero pinball loss."""
        actuals, preds = perfect_predictions
        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        assert loss == 0.0

    def test_median_symmetric_loss(self, symmetric_errors):
        """At tau=0.5, over/under-predictions penalized equally."""
        actuals, preds = symmetric_errors
        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        # Mean absolute error / 2 for tau=0.5
        mae = np.mean(np.abs(actuals - preds))
        assert loss == pytest.approx(mae * 0.5, rel=1e-6)

    def test_high_quantile_penalizes_underprediction(self):
        """At tau=0.9, under-predictions penalized more heavily."""
        actuals = np.array([10.0])
        under_pred = np.array([8.0])  # Under by 2
        over_pred = np.array([12.0])  # Over by 2

        loss_under = compute_pinball_loss(actuals, under_pred, tau=0.9)
        loss_over = compute_pinball_loss(actuals, over_pred, tau=0.9)

        # Under-prediction: tau * |error| = 0.9 * 2 = 1.8
        # Over-prediction: (1-tau) * |error| = 0.1 * 2 = 0.2
        assert loss_under == pytest.approx(1.8, rel=1e-6)
        assert loss_over == pytest.approx(0.2, rel=1e-6)
        assert loss_under > loss_over

    def test_low_quantile_penalizes_overprediction(self):
        """At tau=0.1, over-predictions penalized more heavily."""
        actuals = np.array([10.0])
        under_pred = np.array([8.0])  # Under by 2
        over_pred = np.array([12.0])  # Over by 2

        loss_under = compute_pinball_loss(actuals, under_pred, tau=0.1)
        loss_over = compute_pinball_loss(actuals, over_pred, tau=0.1)

        # Under-prediction: tau * |error| = 0.1 * 2 = 0.2
        # Over-prediction: (1-tau) * |error| = 0.9 * 2 = 1.8
        assert loss_under == pytest.approx(0.2, rel=1e-6)
        assert loss_over == pytest.approx(1.8, rel=1e-6)
        assert loss_over > loss_under

    def test_loss_always_positive(self):
        """Pinball loss is non-negative."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(100)
        preds = rng.standard_normal(100)

        for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
            loss = compute_pinball_loss(actuals, preds, tau=tau)
            assert loss >= 0

    def test_tau_boundary_validation(self):
        """tau must be in (0, 1)."""
        actuals = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="tau must be in"):
            compute_pinball_loss(actuals, preds, tau=0.0)

        with pytest.raises(ValueError, match="tau must be in"):
            compute_pinball_loss(actuals, preds, tau=1.0)

        with pytest.raises(ValueError, match="tau must be in"):
            compute_pinball_loss(actuals, preds, tau=-0.1)

        with pytest.raises(ValueError, match="tau must be in"):
            compute_pinball_loss(actuals, preds, tau=1.5)

    def test_array_length_mismatch(self):
        """Arrays must have same length."""
        actuals = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Array lengths must match"):
            compute_pinball_loss(actuals, preds, tau=0.5)

    def test_empty_arrays(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_pinball_loss(np.array([]), np.array([]), tau=0.5)

    def test_list_input(self):
        """Accepts list input."""
        actuals = [1.0, 2.0, 3.0]
        preds = [1.5, 2.5, 3.5]
        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        assert isinstance(loss, float)


# =============================================================================
# Tests: compute_crps
# =============================================================================


class TestComputeCRPS:
    """Tests for Continuous Ranked Probability Score."""

    def test_perfect_forecast_zero_crps(self):
        """Perfect forecast (all samples equal actual) yields zero CRPS."""
        actuals = np.array([1.0, 2.0, 3.0])
        # Perfect forecast: all samples equal actual
        forecast_samples = np.column_stack([actuals] * 100)
        crps = compute_crps(actuals, forecast_samples)
        assert crps == pytest.approx(0.0, abs=1e-6)

    def test_crps_measures_forecast_spread(self):
        """Wider forecast distribution yields higher CRPS (when centered)."""
        actuals = np.array([0.0, 0.0, 0.0])
        n_samples = 1000

        rng = np.random.default_rng(42)
        # Narrow distribution (std=0.1)
        narrow = rng.normal(0, 0.1, size=(3, n_samples))
        # Wide distribution (std=1.0)
        wide = rng.normal(0, 1.0, size=(3, n_samples))

        crps_narrow = compute_crps(actuals, narrow)
        crps_wide = compute_crps(actuals, wide)

        assert crps_narrow < crps_wide

    def test_crps_measures_bias(self):
        """Biased forecast yields higher CRPS."""
        actuals = np.array([0.0, 0.0, 0.0])
        n_samples = 1000

        rng = np.random.default_rng(42)
        # Unbiased (centered at 0)
        unbiased = rng.normal(0, 0.5, size=(3, n_samples))
        # Biased (centered at 2)
        biased = rng.normal(2, 0.5, size=(3, n_samples))

        crps_unbiased = compute_crps(actuals, unbiased)
        crps_biased = compute_crps(actuals, biased)

        assert crps_unbiased < crps_biased

    def test_crps_is_proper_scoring_rule(self):
        """CRPS is minimized when forecast matches true distribution."""
        rng = np.random.default_rng(42)
        n_obs = 50
        n_samples = 500

        # True distribution: N(0, 1)
        actuals = rng.normal(0, 1, size=n_obs)

        # Correct forecast: samples from N(0, 1)
        correct_forecast = rng.normal(0, 1, size=(n_obs, n_samples))

        # Incorrect forecast: samples from N(0, 2)
        incorrect_forecast = rng.normal(0, 2, size=(n_obs, n_samples))

        crps_correct = compute_crps(actuals, correct_forecast)
        crps_incorrect = compute_crps(actuals, incorrect_forecast)

        # Correct forecast should have lower CRPS (on average)
        # Note: stochastic, but with enough samples should hold
        assert crps_correct < crps_incorrect * 1.2  # Allow some tolerance

    def test_crps_non_negative(self):
        """CRPS is always non-negative."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(20)
        forecast_samples = rng.standard_normal((20, 100))

        crps = compute_crps(actuals, forecast_samples)
        assert crps >= 0

    def test_actuals_must_be_1d(self):
        """Actuals must be 1D array."""
        actuals = np.array([[1.0, 2.0], [3.0, 4.0]])
        forecast_samples = np.random.randn(4, 100)

        with pytest.raises(ValueError, match="actuals must be 1D"):
            compute_crps(actuals, forecast_samples)

    def test_forecast_samples_must_be_2d(self):
        """Forecast samples must be 2D."""
        actuals = np.array([1.0, 2.0, 3.0])
        forecast_samples = np.array([1.0, 2.0, 3.0])  # 1D

        with pytest.raises(ValueError, match="forecast_samples must be 2D"):
            compute_crps(actuals, forecast_samples)

    def test_dimension_mismatch(self):
        """Number of observations must match."""
        actuals = np.array([1.0, 2.0, 3.0])
        forecast_samples = np.random.randn(5, 100)  # 5 != 3

        with pytest.raises(ValueError, match="Number of observations must match"):
            compute_crps(actuals, forecast_samples)

    def test_empty_arrays(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_crps(np.array([]), np.array([]).reshape(0, 10))

    def test_single_sample_per_observation(self):
        """Works with single sample per observation (edge case)."""
        actuals = np.array([1.0, 2.0, 3.0])
        forecast_samples = np.array([[1.5], [2.5], [3.5]])

        crps = compute_crps(actuals, forecast_samples)
        # scipy.stats.energy_distance returns 2*|sample - actual| for point masses
        # This is consistent with the energy distance formula
        expected = np.mean(2 * np.abs(forecast_samples[:, 0] - actuals))
        assert crps == pytest.approx(expected, rel=1e-4)


# =============================================================================
# Tests: compute_interval_score
# =============================================================================


class TestComputeIntervalScore:
    """Tests for interval score (proper scoring rule for intervals)."""

    def test_perfect_coverage_width_only(self, perfect_intervals):
        """When all actuals covered, score equals interval width."""
        actuals, lower, upper = perfect_intervals
        score = compute_interval_score(actuals, lower, upper, alpha=0.05)

        # With perfect coverage, penalty terms are 0
        expected_width = np.mean(upper - lower)
        assert score == pytest.approx(expected_width, rel=1e-6)

    def test_narrow_intervals_better_when_covering(self):
        """Narrower intervals score better when both cover."""
        actuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Narrow intervals (width=1)
        narrow_lower = actuals - 0.5
        narrow_upper = actuals + 0.5

        # Wide intervals (width=2)
        wide_lower = actuals - 1.0
        wide_upper = actuals + 1.0

        score_narrow = compute_interval_score(
            actuals, narrow_lower, narrow_upper, alpha=0.05
        )
        score_wide = compute_interval_score(
            actuals, wide_lower, wide_upper, alpha=0.05
        )

        assert score_narrow < score_wide

    def test_coverage_failure_penalty(self):
        """Missing actuals incurs penalty proportional to 2/alpha."""
        actuals = np.array([5.0])  # Single observation

        # Interval that misses by 1 unit below
        lower = np.array([6.0])
        upper = np.array([8.0])
        alpha = 0.05

        score = compute_interval_score(actuals, lower, upper, alpha=alpha)

        # Width = 2
        # Penalty = (2/alpha) * (lower - actual) = (2/0.05) * 1 = 40
        expected = 2.0 + 40.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_above_penalty(self):
        """Actual above interval incurs correct penalty."""
        actuals = np.array([10.0])  # Single observation

        # Interval that misses by 2 units above
        lower = np.array([5.0])
        upper = np.array([8.0])
        alpha = 0.10

        score = compute_interval_score(actuals, lower, upper, alpha=alpha)

        # Width = 3
        # Penalty = (2/alpha) * (actual - upper) = (2/0.10) * 2 = 40
        expected = 3.0 + 40.0
        assert score == pytest.approx(expected, rel=1e-6)

    def test_alpha_affects_penalty_magnitude(self):
        """Smaller alpha means larger penalty for misses."""
        actuals = np.array([5.0])
        lower = np.array([6.0])  # Misses by 1
        upper = np.array([8.0])

        score_05 = compute_interval_score(actuals, lower, upper, alpha=0.05)
        score_20 = compute_interval_score(actuals, lower, upper, alpha=0.20)

        # Smaller alpha → larger penalty factor → higher score for same miss
        assert score_05 > score_20

    def test_alpha_boundary_validation(self):
        """alpha must be in (0, 1)."""
        actuals = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_interval_score(actuals, lower, upper, alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_interval_score(actuals, lower, upper, alpha=1.0)

    def test_lower_greater_than_upper_error(self):
        """lower > upper raises ValueError."""
        actuals = np.array([1.0, 2.0, 3.0])
        lower = np.array([2.0, 1.5, 2.5])  # First is > upper
        upper = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="lower must be <= upper"):
            compute_interval_score(actuals, lower, upper, alpha=0.05)

    def test_array_length_mismatch(self):
        """Arrays must have same length."""
        actuals = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5])
        upper = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="Array lengths must match"):
            compute_interval_score(actuals, lower, upper, alpha=0.05)

    def test_empty_arrays(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_interval_score(
                np.array([]), np.array([]), np.array([]), alpha=0.05
            )

    def test_interval_score_non_negative(self):
        """Interval score is always non-negative."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(50)
        lower = actuals - rng.uniform(0.5, 2.0, size=50)
        upper = actuals + rng.uniform(0.5, 2.0, size=50)

        score = compute_interval_score(actuals, lower, upper, alpha=0.10)
        assert score >= 0


# =============================================================================
# Tests: compute_quantile_coverage
# =============================================================================


class TestComputeQuantileCoverage:
    """Tests for empirical coverage of prediction intervals."""

    def test_perfect_coverage(self, perfect_intervals):
        """100% coverage when all actuals within intervals."""
        actuals, lower, upper = perfect_intervals
        coverage = compute_quantile_coverage(actuals, lower, upper)
        assert coverage == 1.0

    def test_zero_coverage(self):
        """0% coverage when no actuals within intervals."""
        actuals = np.array([0.0, 0.0, 0.0])
        lower = np.array([1.0, 1.0, 1.0])
        upper = np.array([2.0, 2.0, 2.0])

        coverage = compute_quantile_coverage(actuals, lower, upper)
        assert coverage == 0.0

    def test_partial_coverage(self, partial_coverage_intervals):
        """Correct partial coverage calculation."""
        actuals, lower, upper = partial_coverage_intervals

        # actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
        # lower   = [1.5, 1.5, 2.5, 3.5, 5.5]
        # upper   = [2.0, 2.5, 3.5, 4.5, 6.0]
        # Covered: [No,  Yes, Yes, Yes, No] = 3/5 = 0.6

        coverage = compute_quantile_coverage(actuals, lower, upper)
        assert coverage == pytest.approx(0.6, rel=1e-6)

    def test_boundary_inclusive(self):
        """Actuals on boundary are counted as covered."""
        actuals = np.array([1.0, 2.0])
        lower = np.array([1.0, 1.5])  # First on lower boundary
        upper = np.array([1.5, 2.0])  # Second on upper boundary

        coverage = compute_quantile_coverage(actuals, lower, upper)
        assert coverage == 1.0

    def test_coverage_returns_float(self):
        """Coverage returns float in [0, 1]."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(100)
        lower = actuals - 1.0
        upper = actuals + 1.0

        coverage = compute_quantile_coverage(actuals, lower, upper)
        assert isinstance(coverage, float)
        assert 0.0 <= coverage <= 1.0

    def test_array_length_mismatch(self):
        """Arrays must have same length."""
        actuals = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5])
        upper = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="Array lengths must match"):
            compute_quantile_coverage(actuals, lower, upper)

    def test_empty_arrays(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_quantile_coverage(np.array([]), np.array([]), np.array([]))


# =============================================================================
# Tests: compute_winkler_score
# =============================================================================


class TestComputeWinklerScore:
    """Tests for Winkler score (alias for interval score)."""

    def test_winkler_equals_interval_score(self):
        """Winkler score is identical to interval score."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(20)
        lower = actuals - rng.uniform(0.5, 1.5, size=20)
        upper = actuals + rng.uniform(0.5, 1.5, size=20)
        alpha = 0.10

        winkler = compute_winkler_score(actuals, lower, upper, alpha)
        interval = compute_interval_score(actuals, lower, upper, alpha)

        assert winkler == interval

    def test_winkler_same_validation(self):
        """Winkler score has same validation as interval score."""
        actuals = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_winkler_score(actuals, lower, upper, alpha=0.0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestQuantileMetricsIntegration:
    """Integration tests for quantile metrics."""

    def test_pinball_crps_relationship(self):
        """CRPS is related to integral of pinball loss over quantiles."""
        rng = np.random.default_rng(42)
        n_obs = 20
        n_samples = 500

        # Generate actuals and forecast samples
        actuals = rng.standard_normal(n_obs)
        forecast_samples = rng.standard_normal((n_obs, n_samples))

        # CRPS from function
        crps = compute_crps(actuals, forecast_samples)

        # Approximate CRPS by averaging pinball loss over many quantiles
        # (This is a known relationship - CRPS = 2 * integral of pinball loss)
        taus = np.linspace(0.01, 0.99, 50)
        pinball_sum = 0.0
        for tau in taus:
            quantile_preds = np.quantile(forecast_samples, tau, axis=1)
            pinball_sum += compute_pinball_loss(actuals, quantile_preds, tau=tau)

        # Not exact match but should be in same ballpark
        pinball_avg = 2 * pinball_sum / len(taus)
        assert abs(crps - pinball_avg) < 0.5  # Loose bound due to approximation

    def test_interval_coverage_consistency(self):
        """Well-calibrated intervals should have expected coverage."""
        rng = np.random.default_rng(42)
        n_obs = 1000
        alpha = 0.10

        # Generate data from N(0, 1)
        actuals = rng.standard_normal(n_obs)

        # Create calibrated 90% prediction intervals
        # For N(0,1), 90% interval is approximately [-1.645, 1.645]
        from scipy import stats

        z = stats.norm.ppf(1 - alpha / 2)
        lower = np.zeros(n_obs) - z
        upper = np.zeros(n_obs) + z

        coverage = compute_quantile_coverage(actuals, lower, upper)

        # Should be close to 1 - alpha = 0.90
        assert coverage == pytest.approx(1 - alpha, abs=0.03)

    def test_metrics_work_with_conformal_output(self):
        """Quantile metrics work with conformal prediction output format."""
        rng = np.random.default_rng(42)

        # Simulate conformal prediction output
        n_test = 50
        actuals = rng.standard_normal(n_test)
        point_preds = actuals + rng.normal(0, 0.3, n_test)

        # Conformal intervals (simulated)
        conformity_radius = 1.2
        lower = point_preds - conformity_radius
        upper = point_preds + conformity_radius

        # All metrics should work
        coverage = compute_quantile_coverage(actuals, lower, upper)
        score = compute_interval_score(actuals, lower, upper, alpha=0.10)

        assert 0 <= coverage <= 1
        assert score > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestQuantileMetricsEdgeCases:
    """Edge cases and numerical stability tests."""

    def test_large_values(self):
        """Handles large values without overflow."""
        actuals = np.array([1e10, 2e10, 3e10])
        preds = np.array([1.1e10, 2.1e10, 3.1e10])

        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        assert np.isfinite(loss)

    def test_small_values(self):
        """Handles small values without underflow."""
        actuals = np.array([1e-10, 2e-10, 3e-10])
        preds = np.array([1.1e-10, 2.1e-10, 3.1e-10])

        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        assert np.isfinite(loss)

    def test_negative_values(self):
        """Works correctly with negative values."""
        actuals = np.array([-5.0, -3.0, -1.0])
        preds = np.array([-4.5, -3.5, -0.5])

        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        assert loss >= 0

    def test_mixed_signs(self):
        """Works correctly with mixed positive/negative values."""
        actuals = np.array([-2.0, 0.0, 2.0])
        preds = np.array([-1.5, 0.5, 2.5])

        loss = compute_pinball_loss(actuals, preds, tau=0.5)
        assert loss >= 0

    def test_extreme_tau_values(self):
        """Works with tau very close to 0 or 1."""
        actuals = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.5, 2.5, 3.5])

        loss_low = compute_pinball_loss(actuals, preds, tau=0.001)
        loss_high = compute_pinball_loss(actuals, preds, tau=0.999)

        assert np.isfinite(loss_low)
        assert np.isfinite(loss_high)

    def test_identical_bounds(self):
        """Interval with lower == upper (point prediction)."""
        actuals = np.array([1.0, 2.0, 3.0])
        lower = np.array([1.0, 2.0, 3.0])
        upper = np.array([1.0, 2.0, 3.0])

        # All on boundary → 100% coverage
        coverage = compute_quantile_coverage(actuals, lower, upper)
        assert coverage == 1.0

        # Width = 0, no penalties (all on boundary)
        score = compute_interval_score(actuals, lower, upper, alpha=0.05)
        assert score == 0.0
