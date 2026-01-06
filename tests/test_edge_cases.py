"""
Edge Case Tests for temporalcv.

Tests boundary conditions, degenerate inputs, and stress cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from temporalcv import WalkForwardCV
from temporalcv.bagging import create_block_bagger
from temporalcv.conformal import AdaptiveConformalPredictor
from temporalcv.gates import (
    gate_signal_verification,
    gate_suspicious_improvement,
    gate_temporal_boundary,
    GateStatus,
)
from temporalcv.persistence import (
    compute_move_conditional_metrics,
    compute_move_threshold,
    compute_direction_accuracy,
)
from temporalcv.statistical_tests import dm_test, pt_test


# =============================================================================
# NaN Input Validation
# =============================================================================


class TestNaNInputValidation:
    """Test that NaN inputs are properly rejected."""

    def test_persistence_rejects_nan_predictions(self) -> None:
        """MC metrics should reject NaN predictions."""
        preds = np.array([1.0, np.nan, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="NaN"):
            compute_move_conditional_metrics(preds, actuals, threshold=0.5)

    def test_persistence_rejects_nan_actuals(self) -> None:
        """MC metrics should reject NaN actuals."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, np.nan, 3.0])

        with pytest.raises(ValueError, match="NaN"):
            compute_move_conditional_metrics(preds, actuals, threshold=0.5)

    def test_dm_test_handles_nan(self) -> None:
        """DM test should handle NaN gracefully or reject."""
        errors1 = np.array([1.0, np.nan, 3.0])
        errors2 = np.array([1.0, 2.0, 3.0])

        # Should either reject or handle
        try:
            dm_test(errors1, errors2, h=1)
        except (ValueError, RuntimeError):
            pass  # Expected for NaN input

    def test_pt_test_handles_nan(self) -> None:
        """PT test should handle NaN gracefully or reject."""
        pred_changes = np.array([1.0, np.nan, -1.0])
        actual_changes = np.array([1.0, 1.0, -1.0])

        # Should either reject or handle
        try:
            result = pt_test(pred_changes, actual_changes)
            # If it succeeds, result should be valid
            assert hasattr(result, "test_statistic")
        except (ValueError, RuntimeError):
            pass  # Expected for NaN input


# =============================================================================
# Empty and Minimum Size Arrays
# =============================================================================


class TestMinimumSizeArrays:
    """Test minimum valid input sizes."""

    def test_dm_test_minimum_size(self) -> None:
        """DM test requires minimum samples (n >= 30 recommended)."""
        # Very short arrays (DM test should warn or work with degraded quality)
        errors1 = np.array([1.0, 2.0, 3.0])
        errors2 = np.array([1.1, 2.1, 3.1])

        # Should not crash; may warn about small sample
        try:
            result = dm_test(errors1, errors2, h=1)
            assert hasattr(result, "statistic")
        except ValueError as e:
            assert "samples" in str(e).lower() or "length" in str(e).lower()

    def test_pt_test_minimum_size(self) -> None:
        """PT test requires minimum samples."""
        # PT test needs changes, so minimum 2 values for 1 change
        pred_changes = np.array([1.0])
        actual_changes = np.array([1.0])

        # Should not crash
        try:
            result = pt_test(pred_changes, actual_changes)
            assert hasattr(result, "test_statistic")
        except ValueError:
            pass  # OK if it rejects very small input

    def test_mc_metrics_empty_arrays(self) -> None:
        """MC metrics should handle empty arrays gracefully."""
        preds = np.array([])
        actuals = np.array([])

        result = compute_move_conditional_metrics(preds, actuals, threshold=0.5)

        assert result.n_total == 0
        assert np.isnan(result.skill_score)

    def test_direction_accuracy_empty(self) -> None:
        """Direction accuracy should handle empty arrays."""
        preds = np.array([])
        actuals = np.array([])

        acc = compute_direction_accuracy(preds, actuals)
        assert acc == 0.0


# =============================================================================
# Constant and Degenerate Data
# =============================================================================


class TestDegenerateData:
    """Test behavior with constant or degenerate data."""

    def test_constant_predictions(self) -> None:
        """Constant predictions should work but may have low skill."""
        np.random.seed(42)
        preds = np.full(50, 0.5)
        actuals = np.random.randn(50) * 0.1

        threshold = compute_move_threshold(actuals)
        result = compute_move_conditional_metrics(preds, actuals, threshold=threshold)

        # Should compute without error
        assert result.n_total == 50

    def test_constant_actuals(self) -> None:
        """Constant actuals (no moves) should result in NaN skill."""
        np.random.seed(42)
        preds = np.random.randn(50) * 0.1
        actuals = np.zeros(50)  # All flat

        threshold = compute_move_threshold(actuals)
        # Threshold from zeros will be 0, so all are FLAT

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = compute_move_conditional_metrics(preds, actuals, threshold=threshold)

        # No moves means NaN skill score
        assert result.n_moves == 0 or np.isnan(result.skill_score)

    def test_pt_test_with_sufficient_samples(self) -> None:
        """PT test with all same-direction changes (needs >= 30 samples)."""
        np.random.seed(42)
        # PT test requires >= 30 samples
        pred_changes = np.ones(35) + np.random.randn(35) * 0.01  # All positive
        actual_changes = np.ones(35) + np.random.randn(35) * 0.01  # All positive

        result = pt_test(pred_changes, actual_changes)

        # Very high agreement on direction
        assert result.accuracy > 0.9

    def test_opposite_predictions(self) -> None:
        """Predictions exactly opposite to actuals."""
        preds = np.array([1.0, -1.0, 1.0, -1.0])
        actuals = np.array([-1.0, 1.0, -1.0, 1.0])

        acc = compute_direction_accuracy(preds, actuals)

        # All wrong direction
        assert acc == 0.0


# =============================================================================
# Boundary Conditions for CV
# =============================================================================


class TestCVBoundaryConditions:
    """Test CV edge cases."""

    def test_cv_minimum_data(self) -> None:
        """CV should work with minimum viable data size."""
        # Sliding window of 20, test size 5, 3 splits = need at least 35 points
        y = np.random.randn(40)

        cv = WalkForwardCV(
            n_splits=2,
            window_type="sliding",
            window_size=20,
            test_size=5,
        )

        splits = list(cv.split(y))
        assert len(splits) == 2

        for train_idx, test_idx in splits:
            assert len(train_idx) == 20
            assert len(test_idx) == 5

    def test_cv_gap_at_data_end(self) -> None:
        """CV gap should be respected at data boundaries."""
        y = np.random.randn(50)

        cv = WalkForwardCV(
            n_splits=2,
            window_type="expanding",
            extra_gap=5,
            test_size=5,
        )

        for train_idx, test_idx in cv.split(y):
            gap_actual = test_idx[0] - train_idx[-1] - 1
            assert gap_actual >= cv.extra_gap

    def test_cv_single_split(self) -> None:
        """CV with single split should work."""
        y = np.random.randn(100)

        cv = WalkForwardCV(
            n_splits=1,
            window_type="expanding",
            test_size=10,
        )

        splits = list(cv.split(y))
        assert len(splits) == 1


# =============================================================================
# Gate Edge Cases
# =============================================================================


class TestGateEdgeCases:
    """Test gate boundary conditions."""

    def test_suspicious_improvement_exact_threshold(self) -> None:
        """Improvement exactly at threshold."""
        result = gate_suspicious_improvement(
            model_metric=0.80,
            baseline_metric=1.0,
            threshold=0.20,  # Exactly 20% improvement
        )
        # At boundary - should be PASS (not exceeding threshold)
        assert result.status in [GateStatus.PASS, GateStatus.WARN]

    def test_suspicious_improvement_zero_baseline(self) -> None:
        """Zero baseline should be handled with SKIP."""
        result = gate_suspicious_improvement(
            model_metric=0.5,
            baseline_metric=0.0,  # Zero baseline
            threshold=0.20,
        )
        # Zero baseline cannot compute improvement ratio - SKIP is appropriate
        assert result.status == GateStatus.SKIP

    def test_temporal_boundary_sufficient_gap(self) -> None:
        """actual_gap >= required_gap (horizon + gap) should pass."""
        # actual_gap = test_start_idx - train_end_idx - 1 = 103 - 99 - 1 = 3
        # required_gap = horizon + gap = 1 + 2 = 3
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=103,  # actual_gap = 3
            horizon=1,
            extra_gap=2,  # required_gap = 1 + 2 = 3, so 3 >= 3 passes
        )
        assert result.status == GateStatus.PASS

    def test_temporal_boundary_insufficient_gap(self) -> None:
        """Gap < horizon should fail."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=100,  # gap of 0
            horizon=2,  # Need gap >= 2
            extra_gap=0,
        )
        assert result.status == GateStatus.HALT

    def test_temporal_boundary_negative_gap(self) -> None:
        """Negative gap (overlap) should fail."""
        result = gate_temporal_boundary(
            train_end_idx=105,  # Train ends AFTER test starts
            test_start_idx=100,
            horizon=1,
            extra_gap=0,
        )
        assert result.status == GateStatus.HALT


# =============================================================================
# Conformal Edge Cases
# =============================================================================


class TestConformalEdgeCases:
    """Test conformal prediction edge cases."""

    def test_adaptive_conformal_small_calibration(self) -> None:
        """AdaptiveConformalPredictor with small calibration set."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actuals = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        acp = AdaptiveConformalPredictor(alpha=0.1, gamma=0.1)
        acp.initialize(predictions, actuals)

        # Should produce valid intervals
        lower, upper = acp.predict_interval(3.0)
        assert lower < upper

    def test_adaptive_conformal_extreme_alpha(self) -> None:
        """AdaptiveConformalPredictor with extreme alpha values."""
        np.random.seed(42)
        predictions = np.random.randn(50)
        actuals = predictions + np.random.randn(50) * 0.1

        # Very low alpha = very wide intervals (99% coverage)
        acp_wide = AdaptiveConformalPredictor(alpha=0.01, gamma=0.1)
        acp_wide.initialize(predictions, actuals)

        # Very high alpha = narrow intervals (60% coverage)
        acp_narrow = AdaptiveConformalPredictor(alpha=0.4, gamma=0.1)
        acp_narrow.initialize(predictions, actuals)

        lower_wide, upper_wide = acp_wide.predict_interval(0.0)
        lower_narrow, upper_narrow = acp_narrow.predict_interval(0.0)

        # Wide intervals should be wider
        width_wide = upper_wide - lower_wide
        width_narrow = upper_narrow - lower_narrow
        assert width_wide > width_narrow


# =============================================================================
# Bagging Edge Cases
# =============================================================================


class TestBaggingEdgeCases:
    """Test bagging edge cases."""

    def test_bagger_single_estimator(self) -> None:
        """Bagger with single estimator (degenerate case)."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        bagger = create_block_bagger(
            base_model=Ridge(alpha=1.0),
            n_estimators=1,  # Single estimator
            block_length=5,
            random_state=42,
        )
        bagger.fit(X[:40], y[:40])

        mean_pred, lower, upper = bagger.predict_interval(X[40:])

        assert len(mean_pred) == 10
        # With single estimator, interval width should be 0
        assert np.all(lower <= mean_pred)

    def test_bagger_very_short_blocks(self) -> None:
        """Bagger with block length = 1."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        bagger = create_block_bagger(
            base_model=Ridge(alpha=1.0),
            n_estimators=5,
            block_length=1,  # Very short blocks
            random_state=42,
        )
        bagger.fit(X[:40], y[:40])

        mean_pred, lower, upper = bagger.predict_interval(X[40:])

        assert len(mean_pred) == 10


# =============================================================================
# Statistical Test Edge Cases
# =============================================================================


class TestStatisticalTestEdgeCases:
    """Test statistical test edge cases."""

    def test_dm_test_identical_errors(self) -> None:
        """DM test with identical error arrays."""
        np.random.seed(42)
        errors = np.random.randn(50)

        result = dm_test(errors, errors.copy(), h=1)

        # Identical errors = no difference, statistic should be ~0 or NaN
        # (zero variance in difference can cause NaN)
        if not np.isnan(result.statistic):
            assert result.statistic == pytest.approx(0.0, abs=0.1)

    def test_dm_test_one_clearly_better(self) -> None:
        """DM test where one model is clearly better."""
        np.random.seed(42)
        good_errors = np.abs(np.random.randn(50)) * 0.1  # Small positive errors
        bad_errors = np.abs(np.random.randn(50)) * 2.0  # Large positive errors

        result = dm_test(good_errors, bad_errors, h=1)

        # Good model should be significantly better
        assert result.statistic < 0  # Negative = first is better
        assert result.pvalue < 0.05  # Significant (note: 'pvalue' not 'p_value')

    def test_pt_test_random_predictions(self) -> None:
        """PT test with random predictions (no skill)."""
        np.random.seed(42)
        actual_changes = np.random.randn(100)
        pred_changes = np.random.randn(100)  # Random (no relationship)

        result = pt_test(pred_changes, actual_changes)

        # Random predictions = ~50% accuracy
        assert 0.3 <= result.accuracy <= 0.7
        # Likely not significant (note: 'pvalue' not 'p_value')
        assert result.pvalue > 0.01


# =============================================================================
# Type Coercion Edge Cases
# =============================================================================


class TestTypeCoercion:
    """Test that functions handle various input types."""

    def test_list_inputs_to_persistence(self) -> None:
        """Persistence functions should accept lists."""
        preds = [1.0, 2.0, 3.0]
        actuals = [1.0, 2.0, 3.0]

        result = compute_move_conditional_metrics(preds, actuals, threshold=0.5)
        assert result.n_total == 3

    def test_integer_inputs(self) -> None:
        """Functions should handle integer arrays."""
        preds = np.array([1, 2, 3])
        actuals = np.array([1, 2, 3])

        result = compute_move_conditional_metrics(preds, actuals, threshold=1)
        assert result.n_total == 3

    def test_mixed_precision(self) -> None:
        """Functions should handle mixed precision."""
        preds = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        actuals = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        result = compute_move_conditional_metrics(preds, actuals, threshold=0.5)
        assert result.n_total == 3
