"""
Test High-Persistence Series Metrics Module.

Tests for move-conditional metrics, MC-SS skill score, and
direction accuracy with thresholded signs.
"""

import numpy as np
import pytest

from temporalcv.persistence import (
    MoveConditionalResult,
    MoveDirection,
    classify_moves,
    compute_direction_accuracy,
    compute_move_conditional_metrics,
    compute_move_only_mae,
    compute_move_threshold,
    compute_persistence_mae,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_actuals() -> np.ndarray:
    """Sample actual changes with known distribution."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.05, 100)


@pytest.fixture
def perfect_predictions(sample_actuals: np.ndarray) -> np.ndarray:
    """Perfect predictions (equal to actuals)."""
    return sample_actuals.copy()


@pytest.fixture
def persistence_predictions(sample_actuals: np.ndarray) -> np.ndarray:
    """Persistence predictions (all zeros)."""
    return np.zeros_like(sample_actuals)


# =============================================================================
# Move Threshold Tests
# =============================================================================


class TestComputeMoveThreshold:
    """Test move threshold computation."""

    def test_computes_percentile(self) -> None:
        """Should compute percentile of absolute values."""
        actuals = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])

        threshold = compute_move_threshold(actuals, percentile=50.0)

        # 50th percentile of |values| = 0.05
        assert threshold == pytest.approx(0.05, rel=0.01)

    def test_default_percentile_is_70(self) -> None:
        """Default should be 70th percentile."""
        rng = np.random.default_rng(42)
        actuals = rng.normal(0, 1, 1000)

        threshold_default = compute_move_threshold(actuals)
        threshold_70 = compute_move_threshold(actuals, percentile=70.0)

        assert threshold_default == pytest.approx(threshold_70)

    def test_empty_array_raises(self) -> None:
        """Should raise for empty array."""
        with pytest.raises(ValueError, match="empty"):
            compute_move_threshold(np.array([]))

    def test_invalid_percentile_raises(self) -> None:
        """Should raise for invalid percentile."""
        actuals = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="percentile"):
            compute_move_threshold(actuals, percentile=0.0)

        with pytest.raises(ValueError, match="percentile"):
            compute_move_threshold(actuals, percentile=101.0)

    def test_uses_absolute_values(self) -> None:
        """Should use absolute values for percentile."""
        # All negative values
        actuals = np.array([-0.1, -0.2, -0.3])

        threshold = compute_move_threshold(actuals, percentile=50.0)

        assert threshold > 0  # Should be positive (absolute)


# =============================================================================
# Move Classification Tests
# =============================================================================


class TestClassifyMoves:
    """Test move classification."""

    def test_classifies_correctly(self) -> None:
        """Should classify UP/DOWN/FLAT correctly."""
        values = np.array([0.1, -0.1, 0.02, -0.02, 0.0])
        threshold = 0.05

        moves = classify_moves(values, threshold)

        assert moves[0] == MoveDirection.UP  # 0.1 > 0.05
        assert moves[1] == MoveDirection.DOWN  # -0.1 < -0.05
        assert moves[2] == MoveDirection.FLAT  # |0.02| <= 0.05
        assert moves[3] == MoveDirection.FLAT  # |-0.02| <= 0.05
        assert moves[4] == MoveDirection.FLAT  # |0.0| <= 0.05

    def test_boundary_is_flat(self) -> None:
        """Values exactly at threshold should be FLAT."""
        threshold = 0.05
        values = np.array([0.05, -0.05])

        moves = classify_moves(values, threshold)

        assert moves[0] == MoveDirection.FLAT
        assert moves[1] == MoveDirection.FLAT

    def test_negative_threshold_raises(self) -> None:
        """Should raise for negative threshold."""
        with pytest.raises(ValueError, match="non-negative"):
            classify_moves(np.array([0.1]), threshold=-0.05)

    def test_returns_array_of_enum(self) -> None:
        """Should return array of MoveDirection enums."""
        moves = classify_moves(np.array([0.1, -0.1, 0.0]), threshold=0.05)

        assert all(isinstance(m, MoveDirection) for m in moves)


# =============================================================================
# Move Conditional Metrics Tests
# =============================================================================


class TestMoveConditionalMetrics:
    """Test move-conditional metrics computation."""

    def test_computes_conditional_maes(self) -> None:
        """Should compute MAE for each move direction."""
        # Construct controlled data
        actuals = np.array([0.1, 0.2, -0.1, -0.2, 0.01, 0.02])  # UP, UP, DOWN, DOWN, FLAT, FLAT
        predictions = np.array([0.08, 0.18, -0.12, -0.22, 0.0, 0.01])
        threshold = 0.05

        mc = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)

        # UP: errors are 0.02, 0.02 -> MAE = 0.02
        assert mc.mae_up == pytest.approx(0.02, abs=0.001)
        # DOWN: errors are 0.02, 0.02 -> MAE = 0.02
        assert mc.mae_down == pytest.approx(0.02, abs=0.001)
        # Counts
        assert mc.n_up == 2
        assert mc.n_down == 2
        assert mc.n_flat == 2

    def test_mc_ss_formula(self) -> None:
        """MC-SS = 1 - (model_mae_moves / persistence_mae_moves)."""
        # Known values for testing formula
        actuals = np.array([0.1, -0.1, 0.0])  # UP, DOWN, FLAT
        predictions = np.array([0.05, -0.05, 0.0])  # Half the actual
        threshold = 0.05

        mc = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)

        # Model MAE on moves: (|0.1-0.05| + |-0.1-(-0.05)|) / 2 = 0.05
        model_mae_moves = 0.05
        # Persistence MAE on moves: (|0.1| + |-0.1|) / 2 = 0.1
        persistence_mae_moves = 0.1
        # MC-SS = 1 - (0.05 / 0.1) = 0.5
        expected_skill = 1 - (model_mae_moves / persistence_mae_moves)

        assert mc.skill_score == pytest.approx(expected_skill, rel=0.01)

    def test_persistence_gets_zero_skill(
        self, sample_actuals: np.ndarray, persistence_predictions: np.ndarray
    ) -> None:
        """Persistence baseline should get MC-SS = 0."""
        threshold = compute_move_threshold(sample_actuals)
        mc = compute_move_conditional_metrics(
            persistence_predictions, sample_actuals, threshold=threshold
        )

        # Persistence MAE = mean(|actuals|), model MAE = same
        # So skill = 1 - 1 = 0
        assert mc.skill_score == pytest.approx(0.0, abs=0.01)

    def test_perfect_model_skill(
        self, sample_actuals: np.ndarray, perfect_predictions: np.ndarray
    ) -> None:
        """Perfect model should get MC-SS = 1."""
        threshold = compute_move_threshold(sample_actuals)
        mc = compute_move_conditional_metrics(
            perfect_predictions, sample_actuals, threshold=threshold
        )

        # Perfect predictions have MAE = 0, so skill = 1 - 0 = 1
        assert mc.skill_score == pytest.approx(1.0, abs=0.01)

    def test_computes_threshold_if_not_provided(self, sample_actuals: np.ndarray) -> None:
        """Should compute threshold from actuals if not provided."""
        predictions = np.zeros_like(sample_actuals)

        mc = compute_move_conditional_metrics(predictions, sample_actuals)

        # Should have computed a threshold
        assert mc.move_threshold > 0

    def test_is_reliable_property(self) -> None:
        """is_reliable should require >= 10 per move direction."""
        # Reliable case
        actuals = np.concatenate([np.ones(15) * 0.1, np.ones(15) * -0.1])
        predictions = np.zeros(30)
        mc = compute_move_conditional_metrics(predictions, actuals, threshold=0.05)
        assert mc.is_reliable is True

        # Unreliable case (not enough DOWN)
        actuals_few_down = np.concatenate([np.ones(15) * 0.1, np.ones(5) * -0.1])
        predictions_few = np.zeros(20)
        mc_few = compute_move_conditional_metrics(
            predictions_few, actuals_few_down, threshold=0.05
        )
        assert mc_few.is_reliable is False

    def test_empty_arrays(self) -> None:
        """Should handle empty arrays."""
        mc = compute_move_conditional_metrics(np.array([]), np.array([]))

        assert np.isnan(mc.skill_score)
        assert mc.n_total == 0

    def test_length_mismatch_raises(self) -> None:
        """Should raise if arrays have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            compute_move_conditional_metrics(np.array([1, 2]), np.array([1]))

    def test_to_dict(self, sample_actuals: np.ndarray) -> None:
        """Should convert to dictionary."""
        predictions = np.zeros_like(sample_actuals)
        threshold = compute_move_threshold(sample_actuals)

        mc = compute_move_conditional_metrics(predictions, sample_actuals, threshold=threshold)
        d = mc.to_dict()

        assert "mae_up" in d
        assert "mae_down" in d
        assert "skill_score" in d
        assert "is_reliable" in d
        assert "move_fraction" in d


# =============================================================================
# Direction Accuracy Tests
# =============================================================================


class TestDirectionAccuracy:
    """Test direction accuracy computation."""

    def test_two_class_sign_based(self) -> None:
        """Without threshold, should use sign comparison."""
        predictions = np.array([0.1, -0.1, 0.1, -0.1])
        actuals = np.array([0.2, -0.2, -0.2, 0.2])  # First 2 match, last 2 don't

        acc = compute_direction_accuracy(predictions, actuals)

        assert acc == pytest.approx(0.5)  # 2/4 correct

    def test_three_class_with_threshold(self) -> None:
        """With threshold, should use UP/DOWN/FLAT comparison."""
        predictions = np.array([0.0, 0.0, 0.1, -0.1])  # FLAT, FLAT, UP, DOWN
        actuals = np.array([0.01, 0.1, 0.1, -0.1])  # FLAT, UP, UP, DOWN
        threshold = 0.05

        acc = compute_direction_accuracy(predictions, actuals, move_threshold=threshold)

        # FLAT==FLAT (correct), FLAT!=UP (wrong), UP==UP (correct), DOWN==DOWN (correct)
        assert acc == pytest.approx(0.75)  # 3/4 correct

    def test_persistence_with_threshold(self) -> None:
        """Persistence (predicts 0) should get credit for FLAT actuals."""
        # Persistence predictions (all zeros = FLAT)
        predictions = np.zeros(100)
        # Mix of actuals: some FLAT, some moves
        actuals = np.concatenate(
            [
                np.random.uniform(-0.02, 0.02, 70),  # FLAT
                np.random.uniform(0.1, 0.2, 15),  # UP
                np.random.uniform(-0.2, -0.1, 15),  # DOWN
            ]
        )
        threshold = 0.05

        acc = compute_direction_accuracy(predictions, actuals, move_threshold=threshold)

        # Persistence should get ~70% (correct on FLAT, wrong on UP/DOWN)
        assert acc == pytest.approx(0.70, abs=0.05)

    def test_persistence_without_threshold(self) -> None:
        """Persistence without threshold gets 0% (zero excluded)."""
        predictions = np.zeros(100)
        actuals = np.random.randn(100)  # Non-zero actuals

        acc = compute_direction_accuracy(predictions, actuals)

        # Sign(0) = 0, sign of non-zero != 0
        # But we exclude zeros from actuals, so all predictions of 0
        # should compare against non-zero actuals as sign mismatch
        # Actually np.sign(0) = 0, so 0 != sign(non-zero)
        assert acc == 0.0

    def test_excludes_zero_actuals_without_threshold(self) -> None:
        """Without threshold, zero actuals should be excluded."""
        predictions = np.array([0.1, 0.1, 0.1])
        actuals = np.array([0.1, 0.0, -0.1])  # Middle is zero

        acc = compute_direction_accuracy(predictions, actuals)

        # Only compare index 0 (match) and 2 (no match)
        assert acc == pytest.approx(0.5)  # 1/2

    def test_empty_arrays(self) -> None:
        """Should return 0 for empty arrays."""
        acc = compute_direction_accuracy(np.array([]), np.array([]))
        assert acc == 0.0

    def test_length_mismatch_raises(self) -> None:
        """Should raise if arrays differ in length."""
        with pytest.raises(ValueError, match="same length"):
            compute_direction_accuracy(np.array([1, 2]), np.array([1]))


# =============================================================================
# Move-Only MAE Tests
# =============================================================================


class TestMoveOnlyMAE:
    """Test move-only MAE computation."""

    def test_excludes_flat(self) -> None:
        """Should exclude FLAT from MAE calculation."""
        actuals = np.array([0.1, -0.1, 0.02])  # UP, DOWN, FLAT
        predictions = np.array([0.08, -0.12, 0.01])
        threshold = 0.05

        mae, n = compute_move_only_mae(predictions, actuals, threshold)

        # Only UP and DOWN: errors are 0.02, 0.02
        assert mae == pytest.approx(0.02, abs=0.001)
        assert n == 2

    def test_all_flat_returns_nan(self) -> None:
        """Should return NaN if all points are FLAT."""
        actuals = np.array([0.01, -0.01, 0.0])
        predictions = np.zeros(3)
        threshold = 0.05

        mae, n = compute_move_only_mae(predictions, actuals, threshold)

        assert np.isnan(mae)
        assert n == 0

    def test_negative_threshold_raises(self) -> None:
        """Should raise for negative threshold."""
        with pytest.raises(ValueError, match="non-negative"):
            compute_move_only_mae(np.array([1]), np.array([1]), threshold=-0.05)


# =============================================================================
# Persistence MAE Tests
# =============================================================================


class TestPersistenceMAE:
    """Test persistence MAE computation."""

    def test_equals_mean_abs_actual(self) -> None:
        """Persistence MAE should equal mean(|actual|)."""
        actuals = np.array([0.1, -0.1, 0.2, -0.2])

        mae = compute_persistence_mae(actuals)

        expected = np.mean(np.abs(actuals))
        assert mae == pytest.approx(expected)

    def test_with_threshold_uses_moves_only(self) -> None:
        """With threshold, should compute on moves only."""
        actuals = np.array([0.1, -0.1, 0.01])  # UP, DOWN, FLAT
        threshold = 0.05

        mae = compute_persistence_mae(actuals, threshold=threshold)

        # Only UP and DOWN: mean(|0.1| + |-0.1|) / 2 = 0.1
        assert mae == pytest.approx(0.1)

    def test_empty_returns_nan(self) -> None:
        """Should return NaN for empty array."""
        mae = compute_persistence_mae(np.array([]))
        assert np.isnan(mae)

    def test_all_flat_with_threshold_returns_nan(self) -> None:
        """All FLAT with threshold should return NaN."""
        actuals = np.array([0.01, -0.01])
        threshold = 0.05

        mae = compute_persistence_mae(actuals, threshold=threshold)

        assert np.isnan(mae)


# =============================================================================
# MoveConditionalResult Tests
# =============================================================================


class TestMoveConditionalResult:
    """Test MoveConditionalResult dataclass."""

    def test_properties(self) -> None:
        """Test computed properties."""
        mc = MoveConditionalResult(
            mae_up=0.05,
            mae_down=0.06,
            mae_flat=0.01,
            n_up=15,
            n_down=12,
            n_flat=73,
            skill_score=0.25,
            move_threshold=0.03,
        )

        assert mc.n_total == 100
        assert mc.n_moves == 27
        assert mc.move_fraction == pytest.approx(0.27)
        assert mc.is_reliable is True  # Both >= 10


# =============================================================================
# Integration Tests
# =============================================================================


class TestPersistenceIntegration:
    """Integration tests for persistence metrics."""

    def test_full_evaluation_workflow(self) -> None:
        """Test full evaluation workflow."""
        rng = np.random.default_rng(42)

        # Simulate training and test data
        train_actuals = rng.normal(0, 0.05, 200)
        test_actuals = rng.normal(0, 0.05, 50)

        # Compute threshold from TRAINING data (critical!)
        threshold = compute_move_threshold(train_actuals, percentile=70.0)

        # Model predictions (slightly better than persistence)
        test_predictions = test_actuals * 0.3  # Captures some signal

        # Compute metrics
        mc = compute_move_conditional_metrics(
            test_predictions, test_actuals, threshold=threshold
        )

        # Verify reasonable results
        assert -1 < mc.skill_score < 1  # Bounded
        assert mc.n_total == 50
        assert mc.move_threshold == threshold

    def test_mc_ss_bounds(self) -> None:
        """MC-SS should be in reasonable bounds."""
        rng = np.random.default_rng(42)
        actuals = rng.normal(0, 0.1, 100)
        threshold = compute_move_threshold(actuals)

        # Various prediction scenarios
        scenarios = [
            ("perfect", actuals.copy()),
            ("persistence", np.zeros(100)),
            ("random", rng.normal(0, 0.1, 100)),
            ("inverse", -actuals),
        ]

        for name, predictions in scenarios:
            mc = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)
            # Skill should be bounded (can be negative for bad models)
            assert mc.skill_score <= 1.0, f"{name} exceeded upper bound"
            # Very bad models can go negative but not infinitely
            assert mc.skill_score > -10, f"{name} too negative"
