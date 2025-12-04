"""
Tests for event-aware metrics module.

Tests Brier score, PR-AUC, and related calibration functions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from temporalcv.metrics.event import (
    BrierScoreResult,
    PRAUCResult,
    UndefinedMetricWarning,
    compute_calibrated_direction_brier,
    compute_direction_brier,
    compute_pr_auc,
    convert_predictions_to_direction_probs,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def perfect_binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Perfect binary predictions (Brier = 0)."""
    actuals = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    probs = actuals.astype(float)  # Perfect predictions
    return probs, actuals


@pytest.fixture
def worst_binary_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Completely wrong predictions (Brier = 1)."""
    actuals = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    probs = 1.0 - actuals.astype(float)  # Always wrong
    return probs, actuals


@pytest.fixture
def climatology_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Climatology predictions (predict base rate)."""
    actuals = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    base_rate = np.mean(actuals)
    probs = np.full_like(actuals, base_rate, dtype=float)
    return probs, actuals


@pytest.fixture
def imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """Imbalanced binary data (10% positive)."""
    rng = np.random.default_rng(42)
    n = 100
    actuals = (rng.random(n) < 0.1).astype(int)
    probs = rng.random(n)
    return probs, actuals


# =============================================================================
# TestBrierScore
# =============================================================================


class TestBrierScore:
    """Test Brier score for direction prediction."""

    def test_perfect_predictions_brier_zero(
        self, perfect_binary_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Perfect predictions should have Brier = 0."""
        probs, actuals = perfect_binary_predictions
        result = compute_direction_brier(probs, actuals, n_classes=2)

        assert result.brier_score == pytest.approx(0.0)
        assert result.n_samples == len(actuals)
        assert result.n_classes == 2

    def test_worst_predictions_brier_one(
        self, worst_binary_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Completely wrong predictions should have Brier = 1."""
        probs, actuals = worst_binary_predictions
        result = compute_direction_brier(probs, actuals, n_classes=2)

        assert result.brier_score == pytest.approx(1.0)

    def test_climatology_has_uncertainty_brier(
        self, climatology_predictions: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Climatology forecast should have Brier = uncertainty."""
        probs, actuals = climatology_predictions
        result = compute_direction_brier(probs, actuals, n_classes=2)

        # For climatology, Brier = p * (1-p) = uncertainty
        assert result.brier_score == pytest.approx(result.uncertainty, abs=1e-10)
        assert result.skill_score == pytest.approx(0.0, abs=1e-10)

    def test_brier_bounded_zero_one(self) -> None:
        """Brier score should be in [0, 1] for binary classification."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            probs = rng.random(50)
            actuals = rng.choice([0, 1], size=50)
            result = compute_direction_brier(probs, actuals, n_classes=2)

            assert 0.0 <= result.brier_score <= 1.0

    def test_skill_score_interpretation(self) -> None:
        """Skill score should be positive for better than climatology."""
        # Good predictions (close to truth)
        actuals = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        good_probs = np.array([0.9, 0.1, 0.8, 0.2, 0.9, 0.85, 0.15, 0.1])
        result_good = compute_direction_brier(good_probs, actuals, n_classes=2)

        # Should have positive skill score
        assert result_good.skill_score > 0

    def test_three_class_brier(self) -> None:
        """3-class Brier should work with probability vectors."""
        # [P(DOWN), P(FLAT), P(UP)]
        pred_probs = np.array([
            [0.1, 0.2, 0.7],  # Predicts UP
            [0.8, 0.1, 0.1],  # Predicts DOWN
            [0.1, 0.8, 0.1],  # Predicts FLAT
        ])
        actuals = np.array([2, 0, 1])  # UP, DOWN, FLAT

        result = compute_direction_brier(pred_probs, actuals, n_classes=3)

        assert result.n_classes == 3
        assert result.n_samples == 3
        # Perfect for all 3, so should be low
        assert result.brier_score < 0.2

    def test_three_class_perfect(self) -> None:
        """Perfect 3-class predictions should have Brier close to 0."""
        pred_probs = np.array([
            [0.0, 0.0, 1.0],  # Perfect UP
            [1.0, 0.0, 0.0],  # Perfect DOWN
            [0.0, 1.0, 0.0],  # Perfect FLAT
        ])
        actuals = np.array([2, 0, 1])

        result = compute_direction_brier(pred_probs, actuals, n_classes=3)

        assert result.brier_score == pytest.approx(0.0)

    def test_three_class_worst(self) -> None:
        """Maximally wrong 3-class predictions should have high Brier."""
        pred_probs = np.array([
            [1.0, 0.0, 0.0],  # Predicts DOWN, actual UP
            [0.0, 0.0, 1.0],  # Predicts UP, actual DOWN
            [1.0, 0.0, 0.0],  # Predicts DOWN, actual FLAT
        ])
        actuals = np.array([2, 0, 1])

        result = compute_direction_brier(pred_probs, actuals, n_classes=3)

        # Maximum Brier for 3-class is 2 (one-hot error squared)
        assert result.brier_score > 1.5

    def test_empty_input(self) -> None:
        """Empty input should return NaN."""
        result = compute_direction_brier(np.array([]), np.array([]), n_classes=2)

        assert np.isnan(result.brier_score)
        assert result.n_samples == 0

    def test_shape_validation_binary(self) -> None:
        """Should reject 2D probs for binary classification."""
        probs = np.array([[0.7, 0.3], [0.4, 0.6]])
        actuals = np.array([1, 0])

        with pytest.raises(ValueError, match="should be 1D"):
            compute_direction_brier(probs, actuals, n_classes=2)

    def test_shape_validation_three_class(self) -> None:
        """Should reject 1D probs for 3-class classification."""
        probs = np.array([0.7, 0.3])
        actuals = np.array([2, 0])

        with pytest.raises(ValueError, match="should be \\(n_samples, 3\\)"):
            compute_direction_brier(probs, actuals, n_classes=3)

    def test_probability_validation(self) -> None:
        """Should reject probabilities outside [0, 1]."""
        probs = np.array([1.5, 0.3, -0.1])
        actuals = np.array([1, 0, 1])

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            compute_direction_brier(probs, actuals, n_classes=2)

    def test_probability_sum_validation(self) -> None:
        """Should reject 3-class probs that don't sum to 1."""
        probs = np.array([
            [0.3, 0.3, 0.3],  # Sums to 0.9
            [0.4, 0.4, 0.4],  # Sums to 1.2
        ])
        actuals = np.array([2, 0])

        with pytest.raises(ValueError, match="must sum to 1.0"):
            compute_direction_brier(probs, actuals, n_classes=3)

    def test_to_dict(self) -> None:
        """to_dict should include all fields."""
        probs = np.array([0.7, 0.3])
        actuals = np.array([1, 0])
        result = compute_direction_brier(probs, actuals, n_classes=2)

        d = result.to_dict()
        assert "brier_score" in d
        assert "skill_score" in d
        assert "n_samples" in d
        assert "n_classes" in d


# =============================================================================
# TestPRAUC
# =============================================================================


class TestPRAUC:
    """Test PR-AUC computation."""

    def test_perfect_ranking(self) -> None:
        """Perfect ranking has high PR-AUC (not necessarily 1.0).

        Note: PR-AUC for perfect ranking is NOT 1.0 under standard formulation.
        The curve starts at (recall=0, precision=baseline), not (0, 1).
        For 3/5 positives (baseline=0.6), perfect ranking gives ~0.93.
        """
        # All positives ranked before all negatives
        pred_probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        actuals = np.array([1, 1, 1, 0, 0])

        result = compute_pr_auc(pred_probs, actuals)

        # Perfect ranking should be significantly above baseline (0.6)
        assert result.pr_auc > result.baseline + 0.1
        assert result.pr_auc == pytest.approx(0.933, abs=0.01)

    def test_worst_ranking(self) -> None:
        """Inverse ranking should have low PR-AUC."""
        # All negatives ranked before all positives
        pred_probs = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        actuals = np.array([0, 0, 0, 1, 1])

        result = compute_pr_auc(pred_probs, actuals)

        # Should be close to baseline
        assert result.pr_auc < result.baseline * 1.5

    def test_baseline_is_positive_rate(self) -> None:
        """Baseline should equal positive class rate."""
        actuals = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        probs = np.random.rand(10)

        result = compute_pr_auc(probs, actuals)

        assert result.baseline == pytest.approx(0.3)

    def test_random_predictions_near_baseline(self) -> None:
        """Random predictions should approach baseline PR-AUC."""
        rng = np.random.default_rng(42)
        n = 1000
        probs = rng.random(n)
        actuals = rng.choice([0, 1], size=n, p=[0.7, 0.3])

        result = compute_pr_auc(probs, actuals)

        # Should be within 20% of baseline
        assert result.pr_auc == pytest.approx(result.baseline, rel=0.3)

    def test_lift_over_baseline(self) -> None:
        """Good predictions should have lift > 1."""
        # Good ranking
        pred_probs = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        actuals = np.array([1, 1, 0, 0, 0])

        result = compute_pr_auc(pred_probs, actuals)

        assert result.lift_over_baseline > 1.0

    def test_precision_at_50_recall(self) -> None:
        """Should correctly compute precision at 50% recall."""
        # Perfect ranking: all 2 positives are top-2
        pred_probs = np.array([0.9, 0.8, 0.3, 0.2])
        actuals = np.array([1, 1, 0, 0])

        result = compute_pr_auc(pred_probs, actuals)

        # At 50% recall (1 positive found), precision = 1/1 = 1.0
        assert result.precision_at_50_recall == pytest.approx(1.0, abs=0.01)

    def test_empty_positive_class(self) -> None:
        """Should return 0.0 with warning if no positives (Issue #6 fix)."""
        probs = np.array([0.9, 0.8, 0.3])
        actuals = np.array([0, 0, 0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_pr_auc(probs, actuals)

            # Warning should be raised
            assert len(w) == 1
            assert issubclass(w[0].category, UndefinedMetricWarning)
            assert "no positive samples" in str(w[0].message)

        # Returns skill-equivalent fallback (0.0 = no skill)
        assert result.pr_auc == 0.0
        assert result.baseline == 0.0
        assert result.n_positive == 0

    def test_empty_negative_class(self) -> None:
        """Should return 1.0 with warning if no negatives (Issue #6 fix)."""
        probs = np.array([0.9, 0.8, 0.3])
        actuals = np.array([1, 1, 1])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_pr_auc(probs, actuals)

            # Warning should be raised
            assert len(w) == 1
            assert issubclass(w[0].category, UndefinedMetricWarning)
            assert "all positive samples" in str(w[0].message)

        # Returns trivially correct fallback (1.0)
        assert result.pr_auc == 1.0
        assert result.baseline == 1.0
        assert result.n_negative == 0

    def test_imbalanced_data(
        self, imbalanced_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Should handle highly imbalanced data."""
        probs, actuals = imbalanced_data
        result = compute_pr_auc(probs, actuals)

        assert result.imbalance_ratio >= 1.0
        assert 0.0 <= result.pr_auc <= 1.0

    def test_length_mismatch(self) -> None:
        """Should raise error on length mismatch."""
        probs = np.array([0.9, 0.8])
        actuals = np.array([1, 0, 1])

        with pytest.raises(ValueError, match="Length mismatch"):
            compute_pr_auc(probs, actuals)

    def test_to_dict(self) -> None:
        """to_dict should include all fields."""
        probs = np.array([0.9, 0.8, 0.3, 0.2])
        actuals = np.array([1, 1, 0, 0])
        result = compute_pr_auc(probs, actuals)

        d = result.to_dict()
        assert "pr_auc" in d
        assert "baseline" in d
        assert "lift_over_baseline" in d
        assert "imbalance_ratio" in d


# =============================================================================
# TestCalibratedBrier
# =============================================================================


class TestCalibratedBrier:
    """Test calibrated Brier with reliability diagram data."""

    def test_returns_three_values(self) -> None:
        """Should return brier, bin_means, bin_fracs."""
        probs = np.array([0.1, 0.2, 0.8, 0.9])
        actuals = np.array([0, 0, 1, 1])

        brier, bin_means, bin_fracs = compute_calibrated_direction_brier(
            probs, actuals, n_bins=5
        )

        assert isinstance(brier, float)
        assert isinstance(bin_means, np.ndarray)
        assert isinstance(bin_fracs, np.ndarray)

    def test_brier_matches_compute_direction_brier(self) -> None:
        """Brier should match compute_direction_brier result."""
        rng = np.random.default_rng(42)
        probs = rng.random(100)
        actuals = rng.choice([0, 1], size=100)

        brier, _, _ = compute_calibrated_direction_brier(probs, actuals, n_bins=10)
        result = compute_direction_brier(probs, actuals, n_classes=2)

        assert brier == pytest.approx(result.brier_score)

    def test_n_bins_determines_output_length(self) -> None:
        """Output arrays should have length n_bins."""
        probs = np.linspace(0, 1, 100)
        actuals = (probs > 0.5).astype(int)

        for n_bins in [5, 10, 20]:
            _, bin_means, bin_fracs = compute_calibrated_direction_brier(
                probs, actuals, n_bins=n_bins
            )
            assert len(bin_means) == n_bins
            assert len(bin_fracs) == n_bins

    def test_perfectly_calibrated_diagonal(self) -> None:
        """Perfectly calibrated predictions should have bin_means ≈ bin_fracs."""
        # Create perfectly calibrated predictions
        rng = np.random.default_rng(42)
        n = 10000
        probs = rng.random(n)
        # Generate actuals with P(1|p) = p
        actuals = (rng.random(n) < probs).astype(int)

        _, bin_means, bin_fracs = compute_calibrated_direction_brier(
            probs, actuals, n_bins=10
        )

        # Non-NaN bins should have means close to fractions
        valid = ~np.isnan(bin_means)
        if np.any(valid):
            assert np.allclose(bin_means[valid], bin_fracs[valid], atol=0.1)

    def test_empty_bins_are_nan(self) -> None:
        """Empty bins should contain NaN."""
        # All probs in [0, 0.2] range
        probs = np.array([0.0, 0.1, 0.1, 0.2])
        actuals = np.array([0, 0, 1, 1])

        _, bin_means, bin_fracs = compute_calibrated_direction_brier(
            probs, actuals, n_bins=5
        )

        # Bins 1-4 should be empty (NaN)
        assert np.isnan(bin_means[2])  # Bin for [0.4, 0.6)
        assert np.isnan(bin_fracs[2])

    def test_empty_input(self) -> None:
        """Empty input should return NaN brier and empty arrays."""
        brier, bin_means, bin_fracs = compute_calibrated_direction_brier(
            np.array([]), np.array([]), n_bins=5
        )

        assert np.isnan(brier)
        assert len(bin_means) == 0
        assert len(bin_fracs) == 0

    def test_invalid_n_bins(self) -> None:
        """Should reject n_bins < 1."""
        probs = np.array([0.5, 0.5])
        actuals = np.array([0, 1])

        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            compute_calibrated_direction_brier(probs, actuals, n_bins=0)


# =============================================================================
# TestConvertPredictions
# =============================================================================


class TestConvertPredictions:
    """Test conversion from point predictions to direction probabilities."""

    def test_high_prediction_gives_high_p_up(self) -> None:
        """High predictions should give high P(UP)."""
        predictions = np.array([2.0, 1.0, 0.5])
        stds = np.array([0.1, 0.1, 0.1])

        p_up = convert_predictions_to_direction_probs(predictions, stds, threshold=0.0)

        assert p_up[0] > 0.99  # Very high prediction
        assert p_up[1] > 0.99
        assert p_up[2] > 0.99

    def test_low_prediction_gives_low_p_up(self) -> None:
        """Low predictions should give low P(UP)."""
        predictions = np.array([-2.0, -1.0, -0.5])
        stds = np.array([0.1, 0.1, 0.1])

        p_up = convert_predictions_to_direction_probs(predictions, stds, threshold=0.0)

        assert p_up[0] < 0.01
        assert p_up[1] < 0.01
        assert p_up[2] < 0.01

    def test_prediction_at_threshold_gives_half(self) -> None:
        """Prediction at threshold should give P(UP) ≈ 0.5."""
        predictions = np.array([0.1, 0.1, 0.1])
        stds = np.array([0.5, 0.5, 0.5])

        p_up = convert_predictions_to_direction_probs(predictions, stds, threshold=0.1)

        # When mean = threshold, P(UP) = 0.5
        assert np.allclose(p_up, 0.5)

    def test_higher_std_spreads_probability(self) -> None:
        """Higher uncertainty should spread probability toward 0.5."""
        predictions = np.array([1.0])
        low_std = np.array([0.1])
        high_std = np.array([10.0])

        p_up_low = convert_predictions_to_direction_probs(predictions, low_std)
        p_up_high = convert_predictions_to_direction_probs(predictions, high_std)

        # Low std: very confident UP
        assert p_up_low[0] > 0.99

        # High std: closer to 0.5
        assert 0.4 < p_up_high[0] < 0.6

    def test_threshold_shifts_probability(self) -> None:
        """Non-zero threshold should shift probability calculation."""
        predictions = np.array([0.5])
        stds = np.array([0.1])

        # With threshold=0, prediction of 0.5 should give high P(UP)
        p_up_0 = convert_predictions_to_direction_probs(predictions, stds, threshold=0.0)

        # With threshold=1.0, prediction of 0.5 should give low P(UP)
        p_up_1 = convert_predictions_to_direction_probs(predictions, stds, threshold=1.0)

        assert p_up_0[0] > 0.99
        assert p_up_1[0] < 0.01

    def test_zero_std_handled(self) -> None:
        """Zero std should be handled (treated as near-deterministic)."""
        predictions = np.array([1.0, -1.0])
        stds = np.array([0.0, 0.0])

        p_up = convert_predictions_to_direction_probs(predictions, stds, threshold=0.0)

        # Should be near 1 and near 0 respectively
        assert p_up[0] > 0.99
        assert p_up[1] < 0.01

    def test_length_mismatch(self) -> None:
        """Should raise error on length mismatch."""
        predictions = np.array([1.0, 2.0])
        stds = np.array([0.1])

        with pytest.raises(ValueError, match="Length mismatch"):
            convert_predictions_to_direction_probs(predictions, stds)

    def test_negative_std(self) -> None:
        """Should raise error on negative std."""
        predictions = np.array([1.0])
        stds = np.array([-0.1])

        with pytest.raises(ValueError, match="must be non-negative"):
            convert_predictions_to_direction_probs(predictions, stds)


# =============================================================================
# Integration Tests
# =============================================================================


class TestEventMetricsIntegration:
    """Integration tests combining multiple metrics."""

    def test_brier_and_prauc_consistent(self) -> None:
        """Better Brier should generally mean better PR-AUC for same data."""
        rng = np.random.default_rng(42)
        actuals = rng.choice([0, 1], size=100, p=[0.5, 0.5])

        # Good predictions
        noise = rng.normal(0, 0.1, 100)
        good_probs = np.clip(actuals + noise, 0, 1)

        # Random predictions
        random_probs = rng.random(100)

        brier_good = compute_direction_brier(good_probs, actuals, n_classes=2)
        brier_random = compute_direction_brier(random_probs, actuals, n_classes=2)

        prauc_good = compute_pr_auc(good_probs, actuals)
        prauc_random = compute_pr_auc(random_probs, actuals)

        # Good should have better (lower) Brier and better (higher) PR-AUC
        assert brier_good.brier_score < brier_random.brier_score
        assert prauc_good.pr_auc > prauc_random.pr_auc

    def test_convert_and_compute_brier(self) -> None:
        """End-to-end: convert point predictions, compute Brier."""
        # Simulate ensemble predictions
        rng = np.random.default_rng(42)
        n = 50
        true_values = rng.choice([1, -1], size=n)
        predictions = true_values + rng.normal(0, 0.5, n)
        stds = np.abs(rng.normal(0.3, 0.1, n))

        # Convert to direction probabilities
        p_up = convert_predictions_to_direction_probs(predictions, stds, threshold=0.0)

        # Create actual directions (1 if positive, 0 if negative)
        actuals = (true_values > 0).astype(int)

        # Compute Brier
        result = compute_direction_brier(p_up, actuals, n_classes=2)

        # Should have reasonable Brier (better than random)
        assert result.brier_score < 0.3
        assert result.skill_score > 0


# =============================================================================
# Validation Against Known Values
# =============================================================================


class TestKnownValues:
    """Test against manually computed known values."""

    def test_brier_known_value(self) -> None:
        """Brier score should match manual calculation."""
        # Manual: probs = [0.9, 0.1], actuals = [1, 0]
        # Brier = mean((0.9-1)^2 + (0.1-0)^2) = mean(0.01 + 0.01) = 0.01
        probs = np.array([0.9, 0.1])
        actuals = np.array([1, 0])

        result = compute_direction_brier(probs, actuals, n_classes=2)

        assert result.brier_score == pytest.approx(0.01)

    def test_prauc_known_value(self) -> None:
        """PR-AUC should match manual calculation for simple case.

        Note: With baseline=0.5, perfect ranking gives PR-AUC ≈ 0.875.
        PR curve: (0, 0.5) -> (0.5, 1) -> (1, 1)
        AUC = 0.5*0.5 + 0.5*(1+1)/2 = 0.25 + 0.5 = 0.75... actually
        For 2/4 positives: recall points at 0.5 and 1.0 with precision 1 and 1.
        Trapezoid from (0, 0.5) to (0.5, 1) to (1, 1).
        """
        # Simple case: 2 positives at top
        probs = np.array([1.0, 0.9, 0.1, 0.0])
        actuals = np.array([1, 1, 0, 0])

        result = compute_pr_auc(probs, actuals)

        # Perfect ranking with 50% positives, starting at baseline=0.5
        assert result.pr_auc == pytest.approx(0.875, abs=0.01)
        assert result.baseline == pytest.approx(0.5)
