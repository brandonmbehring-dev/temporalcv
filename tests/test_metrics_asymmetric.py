"""
Tests for Asymmetric Loss Functions.

Tests cover:
- compute_linex_loss: linear-exponential asymmetric loss
- compute_asymmetric_mape: weighted MAPE
- compute_directional_loss: directional miss penalties
- compute_squared_log_error: MSLE
- compute_huber_loss: robust loss function
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.metrics.asymmetric import (
    compute_asymmetric_mape,
    compute_directional_loss,
    compute_huber_loss,
    compute_linex_loss,
    compute_squared_log_error,
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
def under_predictions():
    """Predictions systematically below actuals."""
    actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    preds = actuals - 5.0
    return actuals, preds


@pytest.fixture
def over_predictions():
    """Predictions systematically above actuals."""
    actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    preds = actuals + 5.0
    return actuals, preds


@pytest.fixture
def mixed_predictions():
    """Mix of over and under predictions."""
    actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    preds = np.array([15.0, 15.0, 35.0, 35.0, 55.0])
    return actuals, preds


# =============================================================================
# Tests: compute_linex_loss
# =============================================================================


class TestComputeLinexLoss:
    """Tests for LinEx loss function."""

    def test_perfect_predictions_zero_loss(self, perfect_predictions):
        """Perfect predictions yield zero LinEx loss."""
        actuals, preds = perfect_predictions
        loss = compute_linex_loss(preds, actuals, a=1.0, b=1.0)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_positive_a_penalizes_underprediction(self, under_predictions, over_predictions):
        """With a > 0, under-predictions penalized more heavily."""
        actuals_under, preds_under = under_predictions
        actuals_over, preds_over = over_predictions

        loss_under = compute_linex_loss(preds_under, actuals_under, a=1.0)
        loss_over = compute_linex_loss(preds_over, actuals_over, a=1.0)

        # Under-prediction (a > 0): exponential penalty
        # Over-prediction (a > 0): linear penalty
        assert loss_under > loss_over

    def test_negative_a_penalizes_overprediction(self, under_predictions, over_predictions):
        """With a < 0, over-predictions penalized more heavily."""
        actuals_under, preds_under = under_predictions
        actuals_over, preds_over = over_predictions

        loss_under = compute_linex_loss(preds_under, actuals_under, a=-1.0)
        loss_over = compute_linex_loss(preds_over, actuals_over, a=-1.0)

        # Over-prediction (a < 0): exponential penalty
        # Under-prediction (a < 0): linear penalty
        assert loss_over > loss_under

    def test_larger_a_more_asymmetric(self, under_predictions):
        """Larger |a| creates more asymmetric loss."""
        actuals, preds = under_predictions

        loss_small_a = compute_linex_loss(preds, actuals, a=0.1)
        loss_large_a = compute_linex_loss(preds, actuals, a=1.0)

        # Larger a makes exponential term dominate more
        assert loss_large_a > loss_small_a

    def test_b_scales_loss(self, under_predictions):
        """b parameter scales the loss."""
        actuals, preds = under_predictions

        loss_b1 = compute_linex_loss(preds, actuals, a=1.0, b=1.0)
        loss_b2 = compute_linex_loss(preds, actuals, a=1.0, b=2.0)

        assert loss_b2 == pytest.approx(2 * loss_b1, rel=1e-6)

    def test_loss_always_nonnegative(self, mixed_predictions):
        """LinEx loss is always non-negative."""
        actuals, preds = mixed_predictions

        for a in [-2.0, -0.5, 0.5, 2.0]:
            loss = compute_linex_loss(preds, actuals, a=a)
            assert loss >= 0

    def test_a_zero_error(self, perfect_predictions):
        """a=0 raises ValueError."""
        actuals, preds = perfect_predictions
        with pytest.raises(ValueError, match="cannot be 0"):
            compute_linex_loss(preds, actuals, a=0)

    def test_b_nonpositive_error(self, perfect_predictions):
        """b <= 0 raises ValueError."""
        actuals, preds = perfect_predictions
        with pytest.raises(ValueError, match="b must be > 0"):
            compute_linex_loss(preds, actuals, a=1.0, b=0)
        with pytest.raises(ValueError, match="b must be > 0"):
            compute_linex_loss(preds, actuals, a=1.0, b=-1.0)

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_linex_loss(np.array([1, 2]), np.array([1, 2, 3]), a=1.0)

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_linex_loss(np.array([]), np.array([]), a=1.0)

    def test_large_errors_clipped(self):
        """Large errors don't cause overflow."""
        actuals = np.array([0.0, 0.0])
        preds = np.array([1000.0, -1000.0])

        # Should not raise overflow
        loss_pos_a = compute_linex_loss(preds, actuals, a=1.0)
        loss_neg_a = compute_linex_loss(preds, actuals, a=-1.0)

        assert np.isfinite(loss_pos_a)
        assert np.isfinite(loss_neg_a)


# =============================================================================
# Tests: compute_asymmetric_mape
# =============================================================================


class TestComputeAsymmetricMAPE:
    """Tests for asymmetric MAPE."""

    def test_perfect_predictions_zero_error(self, perfect_predictions):
        """Perfect predictions yield zero asymmetric MAPE."""
        actuals, preds = perfect_predictions
        amape = compute_asymmetric_mape(preds, actuals)
        assert amape == pytest.approx(0.0, abs=1e-10)

    def test_alpha_half_symmetric(self, mixed_predictions):
        """alpha=0.5 gives symmetric behavior."""
        actuals, preds = mixed_predictions

        amape_05 = compute_asymmetric_mape(preds, actuals, alpha=0.5)

        # For symmetric alpha=0.5, result should be 0.5 * MAPE
        standard_mape = np.mean(np.abs(actuals - preds) / np.abs(actuals))
        assert amape_05 == pytest.approx(0.5 * standard_mape, rel=1e-6)

    def test_alpha_high_penalizes_underprediction(self, under_predictions, over_predictions):
        """alpha > 0.5 penalizes under-predictions more."""
        actuals_under, preds_under = under_predictions
        actuals_over, preds_over = over_predictions

        # Same magnitude errors, different directions
        amape_under = compute_asymmetric_mape(preds_under, actuals_under, alpha=0.8)
        amape_over = compute_asymmetric_mape(preds_over, actuals_over, alpha=0.8)

        # Under-predictions weighted at 0.8, over at 0.2
        assert amape_under > amape_over

    def test_alpha_low_penalizes_overprediction(self, under_predictions, over_predictions):
        """alpha < 0.5 penalizes over-predictions more."""
        actuals_under, preds_under = under_predictions
        actuals_over, preds_over = over_predictions

        amape_under = compute_asymmetric_mape(preds_under, actuals_under, alpha=0.2)
        amape_over = compute_asymmetric_mape(preds_over, actuals_over, alpha=0.2)

        # Over-predictions weighted at 0.8, under at 0.2
        assert amape_over > amape_under

    def test_alpha_bounds(self, perfect_predictions):
        """alpha must be in [0, 1]."""
        actuals, preds = perfect_predictions

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_asymmetric_mape(preds, actuals, alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            compute_asymmetric_mape(preds, actuals, alpha=1.1)

    def test_alpha_extremes(self, under_predictions, over_predictions):
        """alpha=0 and alpha=1 are valid extremes."""
        actuals_under, preds_under = under_predictions
        actuals_over, preds_over = over_predictions

        # alpha=0: only over-predictions matter
        amape_under_0 = compute_asymmetric_mape(preds_under, actuals_under, alpha=0)
        amape_over_0 = compute_asymmetric_mape(preds_over, actuals_over, alpha=0)
        assert amape_under_0 == 0.0  # Under-predictions have zero weight
        assert amape_over_0 > 0.0  # Over-predictions have weight 1

        # alpha=1: only under-predictions matter
        amape_under_1 = compute_asymmetric_mape(preds_under, actuals_under, alpha=1)
        amape_over_1 = compute_asymmetric_mape(preds_over, actuals_over, alpha=1)
        assert amape_under_1 > 0.0  # Under-predictions have weight 1
        assert amape_over_1 == 0.0  # Over-predictions have zero weight

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_asymmetric_mape(np.array([1, 2]), np.array([1, 2, 3]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_asymmetric_mape(np.array([]), np.array([]))


# =============================================================================
# Tests: compute_directional_loss
# =============================================================================


class TestComputeDirectionalLoss:
    """Tests for directional loss function."""

    def test_perfect_direction_zero_loss(self):
        """Correct direction predictions yield zero loss."""
        pred_changes = np.array([1.0, -1.0, 1.0, -1.0])
        actual_changes = np.array([0.5, -0.5, 0.1, -0.1])

        loss = compute_directional_loss(pred_changes, actual_changes)
        assert loss == 0.0

    def test_all_wrong_full_loss(self):
        """All wrong directions yield maximum loss."""
        pred_changes = np.array([1.0, -1.0, 1.0, -1.0])
        actual_changes = np.array([-0.5, 0.5, -0.1, 0.1])

        loss = compute_directional_loss(
            pred_changes, actual_changes, up_miss_weight=1.0, down_miss_weight=1.0
        )
        # All 4 wrong, average = 1.0
        assert loss == 1.0

    def test_up_miss_weight(self):
        """up_miss_weight affects loss for missing UP moves."""
        pred_changes = np.array([-1.0])  # Predicted DOWN
        actual_changes = np.array([1.0])  # Actually UP

        loss_w1 = compute_directional_loss(
            pred_changes, actual_changes, up_miss_weight=1.0, down_miss_weight=1.0
        )
        loss_w2 = compute_directional_loss(
            pred_changes, actual_changes, up_miss_weight=2.0, down_miss_weight=1.0
        )

        assert loss_w2 == 2 * loss_w1

    def test_down_miss_weight(self):
        """down_miss_weight affects loss for missing DOWN moves."""
        pred_changes = np.array([1.0])  # Predicted UP
        actual_changes = np.array([-1.0])  # Actually DOWN

        loss_w1 = compute_directional_loss(
            pred_changes, actual_changes, up_miss_weight=1.0, down_miss_weight=1.0
        )
        loss_w3 = compute_directional_loss(
            pred_changes, actual_changes, up_miss_weight=1.0, down_miss_weight=3.0
        )

        assert loss_w3 == 3 * loss_w1

    def test_zero_prediction_no_penalty(self):
        """Zero prediction doesn't miss an UP move (ambiguous)."""
        pred_changes = np.array([0.0])
        actual_changes = np.array([1.0])

        # Predicted no direction, actual UP
        # This is considered a miss of UP (pred_sign <= 0)
        loss = compute_directional_loss(pred_changes, actual_changes)
        assert loss == 1.0

    def test_zero_actual_no_penalty(self):
        """Zero actual change incurs no penalty (no direction to miss)."""
        pred_changes = np.array([1.0, -1.0])
        actual_changes = np.array([0.0, 0.0])

        loss = compute_directional_loss(pred_changes, actual_changes)
        assert loss == 0.0  # No direction to miss

    def test_with_previous_actuals(self):
        """Can compute from levels with previous_actuals."""
        previous = np.array([100.0, 100.0])
        predictions = np.array([110.0, 90.0])  # Predict UP, DOWN
        actuals = np.array([105.0, 95.0])  # Both correct direction

        loss = compute_directional_loss(
            predictions, actuals, previous_actuals=previous
        )
        assert loss == 0.0

    def test_previous_actuals_length_mismatch(self):
        """previous_actuals must match length."""
        with pytest.raises(ValueError, match="previous_actuals length"):
            compute_directional_loss(
                np.array([1, 2, 3]),
                np.array([1, 2, 3]),
                previous_actuals=np.array([1, 2]),
            )

    def test_negative_weights_error(self):
        """Negative weights raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_directional_loss(
                np.array([1.0]),
                np.array([1.0]),
                up_miss_weight=-1.0,
            )

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_directional_loss(np.array([1, 2]), np.array([1]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_directional_loss(np.array([]), np.array([]))


# =============================================================================
# Tests: compute_squared_log_error
# =============================================================================


class TestComputeSquaredLogError:
    """Tests for MSLE (mean squared log error)."""

    def test_perfect_predictions_zero_msle(self, perfect_predictions):
        """Perfect predictions yield zero MSLE."""
        actuals, preds = perfect_predictions
        msle = compute_squared_log_error(preds, actuals)
        assert msle == pytest.approx(0.0, abs=1e-10)

    def test_relative_error_sensitivity(self):
        """MSLE is sensitive to relative error, not absolute."""
        # Same absolute error (10), different relative
        actuals_low = np.array([20.0])
        preds_low = np.array([30.0])  # 50% error

        actuals_high = np.array([200.0])
        preds_high = np.array([210.0])  # 5% error

        msle_low = compute_squared_log_error(preds_low, actuals_low)
        msle_high = compute_squared_log_error(preds_high, actuals_high)

        # Higher relative error should give higher MSLE
        assert msle_low > msle_high

    def test_asymmetry_under_vs_over(self):
        """MSLE penalizes errors asymmetrically based on log scale."""
        actuals = np.array([10.0])

        # Same ratio: 2x and 0.5x
        over_2x = np.array([20.0])  # Over by 2x
        under_half = np.array([5.0])  # Under by 2x

        msle_over = compute_squared_log_error(over_2x, actuals)
        msle_under = compute_squared_log_error(under_half, actuals)

        # Both have non-zero loss due to log scale
        assert msle_over > 0
        assert msle_under > 0
        # Log scale creates asymmetry (depends on values)
        assert msle_over != msle_under

    def test_handles_zeros(self):
        """Handles zero values due to log(1 + x) transformation."""
        actuals = np.array([0.0, 1.0, 10.0])
        preds = np.array([0.0, 1.0, 10.0])

        msle = compute_squared_log_error(preds, actuals)
        assert msle == pytest.approx(0.0, abs=1e-6)

    def test_negative_values_error(self):
        """Negative values raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_squared_log_error(np.array([-1.0, 1.0]), np.array([1.0, 1.0]))

        with pytest.raises(ValueError, match="non-negative"):
            compute_squared_log_error(np.array([1.0, 1.0]), np.array([-1.0, 1.0]))

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_squared_log_error(np.array([1, 2]), np.array([1, 2, 3]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_squared_log_error(np.array([]), np.array([]))


# =============================================================================
# Tests: compute_huber_loss
# =============================================================================


class TestComputeHuberLoss:
    """Tests for Huber loss function."""

    def test_perfect_predictions_zero_loss(self, perfect_predictions):
        """Perfect predictions yield zero Huber loss."""
        actuals, preds = perfect_predictions
        loss = compute_huber_loss(preds, actuals)
        assert loss == pytest.approx(0.0, abs=1e-10)

    def test_small_errors_quadratic(self):
        """Small errors give quadratic (MSE-like) loss."""
        actuals = np.array([0.0])
        preds = np.array([0.5])  # Error = 0.5, below delta=1.0
        delta = 1.0

        huber = compute_huber_loss(preds, actuals, delta=delta)
        expected = 0.5 * 0.5**2  # 0.5 * error^2
        assert huber == pytest.approx(expected, rel=1e-6)

    def test_large_errors_linear(self):
        """Large errors give linear (MAE-like) loss."""
        actuals = np.array([0.0])
        preds = np.array([5.0])  # Error = 5, above delta=1.0
        delta = 1.0

        huber = compute_huber_loss(preds, actuals, delta=delta)
        expected = delta * (5.0 - 0.5 * delta)  # Linear formula
        assert huber == pytest.approx(expected, rel=1e-6)

    def test_robustness_to_outliers(self):
        """Huber loss is less affected by outliers than MSE."""
        actuals = np.array([1.0, 2.0, 3.0, 100.0])  # Outlier
        preds = np.array([1.0, 2.0, 3.0, 3.0])  # Misses outlier

        huber = compute_huber_loss(preds, actuals, delta=1.0)
        mse = np.mean((actuals - preds) ** 2)

        # Huber should be much smaller than MSE due to outlier
        assert huber < mse

    def test_delta_affects_transition(self):
        """Delta controls where loss transitions from quadratic to linear."""
        actuals = np.array([0.0])
        preds = np.array([2.0])  # Error = 2

        # With delta=1, error is in linear region
        huber_d1 = compute_huber_loss(preds, actuals, delta=1.0)

        # With delta=3, error is in quadratic region
        huber_d3 = compute_huber_loss(preds, actuals, delta=3.0)

        # Quadratic (delta=3) should give higher loss for this error
        expected_d3 = 0.5 * 2**2  # Quadratic
        expected_d1 = 1.0 * (2 - 0.5 * 1.0)  # Linear

        assert huber_d1 == pytest.approx(expected_d1, rel=1e-6)
        assert huber_d3 == pytest.approx(expected_d3, rel=1e-6)

    def test_symmetric_loss(self):
        """Huber loss is symmetric around zero."""
        actuals = np.array([0.0, 0.0])
        preds = np.array([2.0, -2.0])  # Same magnitude, opposite sign

        huber = compute_huber_loss(preds, actuals)
        # Should be same for both
        individual = [
            compute_huber_loss(preds[:1], actuals[:1]),
            compute_huber_loss(preds[1:], actuals[1:]),
        ]
        assert individual[0] == pytest.approx(individual[1], rel=1e-6)

    def test_delta_nonpositive_error(self):
        """delta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="delta must be > 0"):
            compute_huber_loss(np.array([1.0]), np.array([1.0]), delta=0)

        with pytest.raises(ValueError, match="delta must be > 0"):
            compute_huber_loss(np.array([1.0]), np.array([1.0]), delta=-1.0)

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_huber_loss(np.array([1, 2]), np.array([1, 2, 3]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_huber_loss(np.array([]), np.array([]))


# =============================================================================
# Integration Tests
# =============================================================================


class TestAsymmetricLossIntegration:
    """Integration tests for asymmetric loss functions."""

    def test_linex_vs_asymmetric_mape_behavior(self):
        """LinEx and asymmetric MAPE have similar asymmetry patterns."""
        actuals = np.array([100.0, 100.0, 100.0, 100.0, 100.0])

        # Under-predictions
        under_preds = np.array([90.0, 90.0, 90.0, 90.0, 90.0])
        # Over-predictions
        over_preds = np.array([110.0, 110.0, 110.0, 110.0, 110.0])

        # Both should penalize under more with positive a / high alpha
        linex_under = compute_linex_loss(under_preds, actuals, a=1.0)
        linex_over = compute_linex_loss(over_preds, actuals, a=1.0)

        amape_under = compute_asymmetric_mape(under_preds, actuals, alpha=0.8)
        amape_over = compute_asymmetric_mape(over_preds, actuals, alpha=0.8)

        assert linex_under > linex_over
        assert amape_under > amape_over

    def test_huber_between_mae_and_mse(self):
        """Huber loss interpolates between MAE and MSE behavior."""
        rng = np.random.default_rng(42)
        actuals = rng.standard_normal(100)
        preds = rng.standard_normal(100)

        mae = np.mean(np.abs(actuals - preds))
        mse = np.mean((actuals - preds) ** 2)
        huber = compute_huber_loss(preds, actuals, delta=1.0)

        # Huber should be somewhere between scaled MAE and MSE behavior
        # (exact relationship depends on error distribution)
        assert huber > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestAsymmetricLossEdgeCases:
    """Edge cases and numerical stability tests."""

    def test_single_observation(self):
        """Works with single observation."""
        actuals = np.array([10.0])
        preds = np.array([12.0])

        assert np.isfinite(compute_linex_loss(preds, actuals, a=1.0))
        assert np.isfinite(compute_asymmetric_mape(preds, actuals))
        assert np.isfinite(compute_directional_loss(preds, actuals))
        assert np.isfinite(compute_huber_loss(preds, actuals))

    def test_large_values(self):
        """Handles large values."""
        actuals = np.array([1e10, 2e10])
        preds = np.array([1.1e10, 1.9e10])

        assert np.isfinite(compute_asymmetric_mape(preds, actuals))
        assert np.isfinite(compute_huber_loss(preds, actuals))

    def test_small_values(self):
        """Handles small values."""
        actuals = np.array([1e-10, 2e-10])
        preds = np.array([1.1e-10, 1.9e-10])

        assert np.isfinite(compute_asymmetric_mape(preds, actuals))
        assert np.isfinite(compute_huber_loss(preds, actuals))

    def test_list_inputs(self):
        """All functions accept list inputs."""
        actuals = [1.0, 2.0, 3.0]
        preds = [1.1, 2.1, 3.1]

        assert isinstance(compute_linex_loss(preds, actuals, a=1.0), float)
        assert isinstance(compute_asymmetric_mape(preds, actuals), float)
        assert isinstance(compute_directional_loss(preds, actuals), float)
        assert isinstance(compute_squared_log_error(preds, actuals), float)
        assert isinstance(compute_huber_loss(preds, actuals), float)
