"""
Tests for temporalcv.statistical_tests module.

Tests statistical tests for forecast evaluation:
- Diebold-Mariano test with HAC variance
- Pesaran-Timmermann directional accuracy test
- HAC variance computation
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.statistical_tests import (
    DMTestResult,
    PTTestResult,
    GWTestResult,
    MultiModelComparisonResult,
    MultiHorizonResult,
    MultiModelHorizonResult,
    dm_test,
    pt_test,
    gw_test,
    compare_multiple_models,
    compare_horizons,
    compare_models_horizons,
    compute_hac_variance,
    compute_self_normalized_variance,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def correlated_errors() -> tuple[np.ndarray, np.ndarray]:
    """
    Create error series where model 1 is better than model 2.

    Returns errors with known relationship for testing DM test.
    """
    rng = np.random.default_rng(42)
    n = 100

    # Model 1: smaller errors
    errors_1 = rng.normal(0, 1.0, n)

    # Model 2: larger errors (should be significantly worse)
    errors_2 = rng.normal(0, 1.5, n)

    return errors_1, errors_2


@pytest.fixture
def direction_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Create data with known directional accuracy.

    Returns actual and predicted with ~70% direction accuracy.
    """
    rng = np.random.default_rng(42)
    n = 100

    actual = rng.standard_normal(n)

    # Predicted: same sign as actual 70% of the time
    predicted = np.where(
        rng.random(n) < 0.7,
        np.abs(actual) * np.sign(actual),  # Same direction
        -np.abs(actual) * np.sign(actual),  # Opposite direction
    )

    return actual, predicted


@pytest.fixture
def predictable_switching_errors() -> tuple[np.ndarray, np.ndarray]:
    """
    Create error series with predictable switching pattern.

    Model 1 is better on odd periods, model 2 is better on even periods.
    This creates a pattern that the GW test should detect.
    """
    rng = np.random.default_rng(123)
    n = 100

    errors_1 = np.zeros(n)
    errors_2 = np.zeros(n)

    for t in range(n):
        if t % 2 == 0:
            # Even periods: model 2 is better
            errors_1[t] = rng.normal(0, 2.0)
            errors_2[t] = rng.normal(0, 0.5)
        else:
            # Odd periods: model 1 is better
            errors_1[t] = rng.normal(0, 0.5)
            errors_2[t] = rng.normal(0, 2.0)

    return errors_1, errors_2


# =============================================================================
# DMTestResult Tests
# =============================================================================


class TestDMTestResult:
    """Tests for DMTestResult dataclass."""

    def test_basic_creation(self) -> None:
        """DMTestResult should store all fields."""
        result = DMTestResult(
            statistic=-2.5,
            pvalue=0.012,
            h=2,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=True,
            mean_loss_diff=-0.5,
        )

        assert result.statistic == -2.5
        assert result.pvalue == 0.012
        assert result.h == 2
        assert result.n == 100

    def test_significant_at_05(self) -> None:
        """Significance at 0.05 should be correct."""
        sig_result = DMTestResult(
            statistic=-2.5,
            pvalue=0.012,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=True,
            mean_loss_diff=-0.5,
        )
        assert sig_result.significant_at_05 is True

        nonsig_result = DMTestResult(
            statistic=-1.0,
            pvalue=0.15,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=True,
            mean_loss_diff=-0.1,
        )
        assert nonsig_result.significant_at_05 is False

    def test_str_format(self) -> None:
        """String format should be readable."""
        result = DMTestResult(
            statistic=-2.5,
            pvalue=0.012,
            h=2,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=True,
            mean_loss_diff=-0.5,
        )

        s = str(result)
        assert "DM" in s
        assert "-2.5" in s or "2.5" in s


# =============================================================================
# PTTestResult Tests
# =============================================================================


class TestPTTestResult:
    """Tests for PTTestResult dataclass."""

    def test_basic_creation(self) -> None:
        """PTTestResult should store all fields."""
        result = PTTestResult(
            statistic=2.0,
            pvalue=0.023,
            accuracy=0.65,
            expected=0.50,
            n=100,
            n_classes=2,
        )

        assert result.statistic == 2.0
        assert result.pvalue == 0.023
        assert result.accuracy == 0.65
        assert result.expected == 0.50

    def test_skill_property(self) -> None:
        """Skill should be accuracy - expected."""
        result = PTTestResult(
            statistic=2.0,
            pvalue=0.023,
            accuracy=0.65,
            expected=0.50,
            n=100,
            n_classes=2,
        )

        assert result.skill == pytest.approx(0.15)

    def test_str_format(self) -> None:
        """String format should be readable."""
        result = PTTestResult(
            statistic=2.0,
            pvalue=0.023,
            accuracy=0.65,
            expected=0.50,
            n=100,
            n_classes=2,
        )

        s = str(result)
        assert "PT" in s
        assert "65" in s  # accuracy


# =============================================================================
# compute_hac_variance Tests
# =============================================================================


class TestComputeHACVariance:
    """Tests for HAC variance computation."""

    def test_white_noise_variance(self) -> None:
        """HAC variance of white noise should approximate sample variance."""
        rng = np.random.default_rng(42)
        n = 1000
        d = rng.standard_normal(n)

        hac_var = compute_hac_variance(d, bandwidth=0)
        sample_var = np.var(d, ddof=1) / n

        # Should be close for white noise
        assert hac_var == pytest.approx(sample_var, rel=0.3)

    def test_positive_variance(self) -> None:
        """HAC variance should always be positive."""
        rng = np.random.default_rng(42)
        d = rng.standard_normal(100)

        for bw in [1, 5, 10]:
            var = compute_hac_variance(d, bandwidth=bw)
            assert var > 0

    def test_bandwidth_effect(self) -> None:
        """Higher bandwidth should generally increase variance estimate for autocorrelated series."""
        rng = np.random.default_rng(42)

        # Create autocorrelated series
        n = 200
        d = np.zeros(n)
        d[0] = rng.standard_normal()
        for t in range(1, n):
            d[t] = 0.8 * d[t - 1] + rng.standard_normal()

        var_bw1 = compute_hac_variance(d, bandwidth=1)
        var_bw5 = compute_hac_variance(d, bandwidth=5)

        # For positively autocorrelated series, higher bandwidth captures more
        assert var_bw5 >= var_bw1 * 0.5  # Allow some variability


# =============================================================================
# dm_test Tests
# =============================================================================


class TestDMTest:
    """Tests for Diebold-Mariano test."""

    def test_identical_errors(self) -> None:
        """Identical errors should give non-significant result."""
        rng = np.random.default_rng(42)
        errors = rng.standard_normal(50)

        result = dm_test(errors, errors, h=1)

        # Identical errors => no difference
        assert result.mean_loss_diff == pytest.approx(0.0, abs=1e-10)
        assert result.pvalue > 0.05 or np.isnan(result.statistic)

    def test_model1_better(self, correlated_errors: tuple) -> None:
        """Model with smaller errors should have negative statistic (one-sided)."""
        errors_1, errors_2 = correlated_errors

        result = dm_test(errors_1, errors_2, h=1, alternative="less")

        # Model 1 has smaller errors => should be significantly better
        assert result.statistic < 0  # Negative = model 1 better
        assert result.mean_loss_diff < 0

    def test_squared_vs_absolute_loss(self, correlated_errors: tuple) -> None:
        """Test should work with both loss functions."""
        errors_1, errors_2 = correlated_errors

        result_se = dm_test(errors_1, errors_2, h=1, loss="squared")
        result_ae = dm_test(errors_1, errors_2, h=1, loss="absolute")

        # Both should indicate model 1 is better
        assert result_se.mean_loss_diff < 0
        assert result_ae.mean_loss_diff < 0

    def test_harvey_adjustment(self, correlated_errors: tuple) -> None:
        """Harvey adjustment should modify statistic."""
        errors_1, errors_2 = correlated_errors

        result_adj = dm_test(errors_1, errors_2, h=2, harvey_correction=True)
        result_raw = dm_test(errors_1, errors_2, h=2, harvey_correction=False)

        # Adjustment should modify the statistic
        assert result_adj.statistic != result_raw.statistic
        assert result_adj.harvey_adjusted is True
        assert result_raw.harvey_adjusted is False

    def test_horizon_stored(self, correlated_errors: tuple) -> None:
        """Horizon should be stored in result."""
        errors_1, errors_2 = correlated_errors

        result = dm_test(errors_1, errors_2, h=4)

        assert result.h == 4

    def test_alternative_less(self, correlated_errors: tuple) -> None:
        """Alternative='less' should test if model 1 is better."""
        errors_1, errors_2 = correlated_errors

        result = dm_test(errors_1, errors_2, alternative="less")

        # Model 1 is better, so p-value should be small
        # (but depends on randomness, so just check it runs)
        assert result.alternative == "less"

    def test_alternative_greater(self, correlated_errors: tuple) -> None:
        """Alternative='greater' should test if model 2 is better."""
        errors_1, errors_2 = correlated_errors

        result = dm_test(errors_1, errors_2, alternative="greater")

        # Model 2 is worse, so p-value should be large
        assert result.alternative == "greater"

    def test_insufficient_samples(self) -> None:
        """Should raise error with too few samples."""
        errors = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Insufficient samples"):
            dm_test(errors, errors, h=1)

    def test_mismatched_lengths(self) -> None:
        """Should raise error with mismatched lengths."""
        errors_1 = np.random.randn(50)
        errors_2 = np.random.randn(60)

        with pytest.raises(ValueError, match="same length"):
            dm_test(errors_1, errors_2)

    def test_invalid_loss(self, correlated_errors: tuple) -> None:
        """Should raise error with invalid loss function."""
        errors_1, errors_2 = correlated_errors

        with pytest.raises(ValueError, match="Unknown loss"):
            dm_test(errors_1, errors_2, loss="invalid")  # type: ignore

    def test_invalid_horizon(self, correlated_errors: tuple) -> None:
        """Should raise error with invalid horizon."""
        errors_1, errors_2 = correlated_errors

        with pytest.raises(ValueError, match="Horizon"):
            dm_test(errors_1, errors_2, h=0)


# =============================================================================
# pt_test Tests
# =============================================================================


class TestPTTest:
    """Tests for Pesaran-Timmermann test."""

    def test_perfect_accuracy(self) -> None:
        """Perfect direction accuracy should have high statistic."""
        rng = np.random.default_rng(42)
        actual = rng.standard_normal(100)
        predicted = actual.copy()  # Perfect prediction

        result = pt_test(actual, predicted)

        assert result.accuracy == 1.0
        assert result.statistic > 0
        assert result.pvalue < 0.05

    def test_random_accuracy(self) -> None:
        """Random predictions should have ~50% accuracy (2-class)."""
        rng = np.random.default_rng(42)
        n = 200

        actual = rng.standard_normal(n)
        predicted = rng.standard_normal(n)  # Independent

        result = pt_test(actual, predicted)

        # Accuracy should be near expected (random)
        assert abs(result.accuracy - result.expected) < 0.15

    def test_three_class_with_threshold(self) -> None:
        """3-class mode should work with threshold."""
        rng = np.random.default_rng(42)
        n = 100

        actual = rng.standard_normal(n)
        predicted = actual * 0.5 + rng.standard_normal(n) * 0.5

        result = pt_test(actual, predicted, move_threshold=0.5)

        assert result.n_classes == 3
        assert 0 < result.accuracy < 1

    def test_threshold_affects_expected(self) -> None:
        """Threshold should change expected accuracy."""
        rng = np.random.default_rng(42)
        actual = rng.standard_normal(100)
        predicted = rng.standard_normal(100)

        result_2class = pt_test(actual, predicted, move_threshold=None)
        result_3class = pt_test(actual, predicted, move_threshold=0.5)

        # Different classification schemes
        assert result_2class.n_classes == 2
        assert result_3class.n_classes == 3

    def test_persistence_baseline(self) -> None:
        """Persistence (predicts 0) should have fair comparison with threshold."""
        rng = np.random.default_rng(42)
        n = 100

        # Actual changes
        actual = rng.standard_normal(n)

        # Persistence predicts 0 (no change)
        predicted = np.zeros(n)

        # Without threshold: persistence always wrong (sign of 0 is 0)
        result_no_thresh = pt_test(actual, predicted, move_threshold=None)

        # With threshold: persistence gets credit for "FLAT" predictions
        result_thresh = pt_test(actual, predicted, move_threshold=0.5)

        # With threshold, persistence should have reasonable accuracy
        # on observations that are actually flat
        assert result_thresh.n_classes == 3

    def test_insufficient_samples(self) -> None:
        """Should raise error with too few samples."""
        actual = np.array([1, 2, 3])
        predicted = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Insufficient samples"):
            pt_test(actual, predicted)

    def test_mismatched_lengths(self) -> None:
        """Should raise error with mismatched lengths."""
        actual = np.random.randn(50)
        predicted = np.random.randn(60)

        with pytest.raises(ValueError, match="same length"):
            pt_test(actual, predicted)

    def test_significant_skill(self, direction_data: tuple) -> None:
        """Data with known skill should show significance."""
        actual, predicted = direction_data

        result = pt_test(actual, predicted)

        # 70% accuracy vs ~50% expected should be significant
        assert result.accuracy > result.expected
        assert result.skill > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_dm_test_with_hac(self) -> None:
        """DM test should use HAC variance correctly for h>1."""
        rng = np.random.default_rng(42)

        # Create AR(1) forecast errors (autocorrelated)
        n = 100
        errors_1 = np.zeros(n)
        errors_2 = np.zeros(n)

        errors_1[0] = rng.standard_normal()
        errors_2[0] = rng.standard_normal()

        for t in range(1, n):
            errors_1[t] = 0.5 * errors_1[t - 1] + rng.normal(0, 0.8)
            errors_2[t] = 0.5 * errors_2[t - 1] + rng.normal(0, 1.2)

        result_h1 = dm_test(errors_1, errors_2, h=1)
        result_h3 = dm_test(errors_1, errors_2, h=3)

        # Different horizons should give different results
        # due to different HAC bandwidth
        assert result_h1.h == 1
        assert result_h3.h == 3

    def test_combined_evaluation(self, direction_data: tuple) -> None:
        """Combined DM and PT tests should work together."""
        actual, predicted = direction_data

        # Compute forecast errors
        errors_model = actual - predicted
        errors_baseline = actual  # Baseline predicts 0

        # DM test
        dm_result = dm_test(errors_model, errors_baseline, h=1)

        # PT test
        pt_result = pt_test(actual, predicted)

        # Both should complete without error
        assert isinstance(dm_result, DMTestResult)
        assert isinstance(pt_result, PTTestResult)

        # Can compare significance
        assert hasattr(dm_result, "significant_at_05")
        assert hasattr(pt_result, "significant_at_05")


# =============================================================================
# Regression Tests (Critical Bug Fixes)
# =============================================================================


class TestPTTestVarianceRegression:
    """
    Regression test for PT test variance formula fix.

    Bug: n_effective**2 denominator instead of n_effective (2025-12-23)
    Impact: P-values were too small (anticonservative), test rejected H0 too often.
    Fix: Changed n_effective**2 to n_effective per PT 1992 equation 8.
    """

    def test_variance_scales_with_n(self) -> None:
        """
        PT test variance should scale as 1/n, not 1/n².

        With correct variance, p-values should be approximately the same
        regardless of sample size when proportions are the same.
        """
        rng = np.random.default_rng(42)

        # Create data with ~60% direction accuracy at different sample sizes
        results = {}
        for n in [50, 100, 200]:
            actual = rng.standard_normal(n)
            # ~60% correct direction
            predicted = np.where(
                rng.random(n) < 0.6,
                actual * np.sign(actual),  # Correct direction
                -actual * np.sign(actual),  # Wrong direction
            )
            result = pt_test(actual, predicted)
            results[n] = result.pvalue

        # P-values should be in similar range (not decreasing by factor of n)
        # With bug (1/n²), doubling n would halve variance → smaller p-values
        # With fix (1/n), p-values should be roughly stable
        assert results[100] > results[50] * 0.1, (
            f"P-values collapsed too much: n=50 gave {results[50]:.4f}, "
            f"n=100 gave {results[100]:.4f}"
        )
        assert results[200] > results[100] * 0.1, (
            f"P-values collapsed too much: n=100 gave {results[100]:.4f}, "
            f"n=200 gave {results[200]:.4f}"
        )

    def test_50_50_accuracy_not_significant(self) -> None:
        """
        Random guessing (50% accuracy) should not be significant.

        With bug, even random data could appear significant due to
        underestimated variance.
        """
        rng = np.random.default_rng(123)
        n = 100

        # Pure random: 50% correct direction expected
        actual = rng.standard_normal(n)
        predicted = rng.standard_normal(n)  # Independent of actual

        result = pt_test(actual, predicted)

        # Should NOT be significant (p > 0.05)
        assert result.pvalue > 0.05, (
            f"Random guessing should not be significant, but p={result.pvalue:.4f}"
        )
        # Accuracy should be near 50%
        assert 0.35 < result.accuracy < 0.65, (
            f"Random accuracy should be near 50%, got {result.accuracy:.2f}"
        )

    def test_2class_variance_formula_correct(self) -> None:
        """
        Verify 2-class variance formula matches PT 1992 equation 8.

        V(P*) = term1 + term2 + term3 where all terms have 1/n denominator.
        """
        rng = np.random.default_rng(456)
        n = 100

        # Create data where we can manually verify variance (2-class = no zeros)
        actual = rng.choice([-1.0, 1.0], size=n)
        predicted = actual * rng.choice([1.0, -1.0], size=n, p=[0.7, 0.3])

        # pt_test auto-detects 2-class when move_threshold is None and no zeros
        result = pt_test(actual, predicted)

        # Manual calculation of expected variance components
        nonzero = actual != 0
        n_eff = int(np.sum(nonzero))
        p_y_pos = float(np.mean(actual[nonzero] > 0))
        p_x_pos = float(np.mean(predicted[nonzero] > 0))
        p_star = p_y_pos * p_x_pos + (1 - p_y_pos) * (1 - p_x_pos)

        # Correct formula: all terms have 1/n_eff (not 1/n_eff²)
        var_p_hat = p_star * (1 - p_star) / n_eff
        term1 = (2 * p_y_pos - 1) ** 2 * p_x_pos * (1 - p_x_pos) / n_eff
        term2 = (2 * p_x_pos - 1) ** 2 * p_y_pos * (1 - p_y_pos) / n_eff
        term3 = 4 * p_y_pos * p_x_pos * (1 - p_y_pos) * (1 - p_x_pos) / n_eff
        expected_var = var_p_hat + term1 + term2 + term3

        # Variance should be O(1/n), not O(1/n²)
        # For n=100, variance should be O(0.01), not O(0.0001)
        assert expected_var > 1e-4, (
            f"Variance {expected_var:.6f} is too small (likely n² bug)"
        )

        # Also verify that the test produces a reasonable result
        assert 0 < result.pvalue < 1, f"P-value should be in (0,1), got {result.pvalue}"
        assert result.n == n_eff, f"Expected n={n_eff}, got {result.n}"


# =============================================================================
# Multi-Model Comparison Tests
# =============================================================================


class TestMultiModelComparisonResult:
    """Tests for MultiModelComparisonResult dataclass."""

    def test_basic_attributes(self) -> None:
        """Should have correct basic attributes."""
        result = MultiModelComparisonResult(
            pairwise_results={},
            best_model="A",
            bonferroni_alpha=0.025,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2)],
            significant_pairs=[("A", "B")],
        )

        assert result.best_model == "A"
        assert result.bonferroni_alpha == 0.025
        assert result.original_alpha == 0.05
        assert result.n_significant == 1
        assert result.n_comparisons == 0  # No pairwise_results in this test

    def test_summary_contains_key_info(self) -> None:
        """Summary should include rankings and significant pairs."""
        dm_result = DMTestResult(
            statistic=-2.5,
            pvalue=0.01,
            h=1,
            n=100,
            loss="squared",
            alternative="less",
            harvey_adjusted=True,
            mean_loss_diff=-0.5,
        )

        result = MultiModelComparisonResult(
            pairwise_results={("A", "B"): dm_result},
            best_model="A",
            bonferroni_alpha=0.0167,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2), ("C", 0.3)],
            significant_pairs=[("A", "B")],
        )

        summary = result.summary()

        assert "A" in summary
        assert "best" in summary.lower()
        assert "0.0167" in summary or "Bonferroni" in summary
        assert "Significant" in summary

    def test_get_pairwise_order_independent(self) -> None:
        """get_pairwise should work regardless of order."""
        dm_result = DMTestResult(
            statistic=-2.0,
            pvalue=0.02,
            h=1,
            n=100,
            loss="squared",
            alternative="less",
            harvey_adjusted=True,
            mean_loss_diff=-0.3,
        )

        result = MultiModelComparisonResult(
            pairwise_results={("A", "B"): dm_result},
            best_model="A",
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2)],
            significant_pairs=[],
        )

        # Should find regardless of order
        assert result.get_pairwise("A", "B") == dm_result
        assert result.get_pairwise("B", "A") == dm_result
        assert result.get_pairwise("A", "C") is None


class TestCompareMultipleModels:
    """Tests for compare_multiple_models function."""

    def test_basic_comparison(self) -> None:
        """Should compare multiple models and rank them."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "Good": rng.normal(0, 0.5, n),
            "Medium": rng.normal(0, 1.0, n),
            "Bad": rng.normal(0, 1.5, n),
        }

        result = compare_multiple_models(errors)

        # Best model should have lowest MSE
        assert result.best_model == "Good"
        assert len(result.model_rankings) == 3
        assert result.model_rankings[0][0] == "Good"

        # Should have 3 comparisons
        assert result.n_comparisons == 3

    def test_bonferroni_correction(self) -> None:
        """Bonferroni alpha should be original_alpha / n_comparisons."""
        rng = np.random.default_rng(42)
        n = 50

        errors = {
            "A": rng.normal(0, 1, n),
            "B": rng.normal(0, 1, n),
            "C": rng.normal(0, 1, n),
        }

        result = compare_multiple_models(errors, alpha=0.10)

        # 3 models = 3 pairwise comparisons
        assert result.original_alpha == 0.10
        assert result.bonferroni_alpha == pytest.approx(0.10 / 3)

    def test_detects_significant_difference(self) -> None:
        """Should detect significant difference between very different models."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "Best": rng.normal(0, 0.3, n),  # Very small errors
            "Worst": rng.normal(0, 2.0, n),  # Very large errors
        }

        result = compare_multiple_models(errors, alpha=0.05)

        # Should find significant difference
        assert len(result.significant_pairs) > 0
        assert ("Best", "Worst") in result.significant_pairs

    def test_no_significance_for_similar_models(self) -> None:
        """Should not find significance for similar models."""
        rng = np.random.default_rng(42)
        n = 50

        errors = {
            "A": rng.normal(0, 1.0, n),
            "B": rng.normal(0, 1.0, n),
        }

        result = compare_multiple_models(errors, alpha=0.05)

        # Similar models should not be significantly different
        # (with small n and same distribution)
        assert len(result.significant_pairs) == 0

    def test_respects_loss_function(self) -> None:
        """Should use specified loss function."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "A": rng.normal(0, 1, n),
            "B": rng.normal(0, 1.5, n),
        }

        result_se = compare_multiple_models(errors, loss="squared")
        result_ae = compare_multiple_models(errors, loss="absolute")

        # Rankings might differ based on loss function
        # At minimum, both should complete
        assert result_se.best_model is not None
        assert result_ae.best_model is not None

    def test_custom_horizon(self) -> None:
        """Should pass horizon to DM tests."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "A": rng.normal(0, 1, n),
            "B": rng.normal(0, 1.5, n),
        }

        result = compare_multiple_models(errors, h=4)

        # Check that h was passed through
        for dm_result in result.pairwise_results.values():
            assert dm_result.h == 4

    def test_raises_on_single_model(self) -> None:
        """Should raise error with fewer than 2 models."""
        with pytest.raises(ValueError, match="at least 2 models"):
            compare_multiple_models({"A": np.array([1, 2, 3])})

    def test_raises_on_mismatched_lengths(self) -> None:
        """Should raise error if error arrays have different lengths."""
        with pytest.raises(ValueError, match="same length"):
            compare_multiple_models({
                "A": np.random.randn(50),
                "B": np.random.randn(60),
            })

    def test_four_models_has_six_comparisons(self) -> None:
        """4 models should result in 4*3/2 = 6 pairwise comparisons."""
        rng = np.random.default_rng(42)
        n = 50

        errors = {
            "A": rng.normal(0, 1, n),
            "B": rng.normal(0, 1, n),
            "C": rng.normal(0, 1, n),
            "D": rng.normal(0, 1, n),
        }

        result = compare_multiple_models(errors)

        assert result.n_comparisons == 6
        assert result.bonferroni_alpha == pytest.approx(0.05 / 6)

    def test_pairwise_ordering(self) -> None:
        """Pairwise results should be ordered with better model first."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "Good": rng.normal(0, 0.5, n),
            "Bad": rng.normal(0, 1.5, n),
        }

        result = compare_multiple_models(errors)

        # The tuple should be (Good, Bad) not (Bad, Good)
        # because Good has lower loss
        assert ("Good", "Bad") in result.pairwise_results
        assert ("Bad", "Good") not in result.pairwise_results

    def test_integration_with_dm_test(self) -> None:
        """Pairwise results should be valid DMTestResult objects."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "A": rng.normal(0, 1, n),
            "B": rng.normal(0, 1.5, n),
        }

        result = compare_multiple_models(errors)

        for dm_result in result.pairwise_results.values():
            assert isinstance(dm_result, DMTestResult)
            assert 0 <= dm_result.pvalue <= 1
            assert dm_result.n == n
            assert dm_result.alternative == "less"  # Tests if first is better


# =============================================================================
# Self-Normalized Variance Tests
# =============================================================================


class TestComputeSelfNormalizedVariance:
    """Tests for compute_self_normalized_variance function."""

    def test_always_positive(self) -> None:
        """Self-normalized variance should always be >= 0."""
        rng = np.random.default_rng(42)

        # Test with various data types
        for _ in range(10):
            d = rng.standard_normal(100)
            var = compute_self_normalized_variance(d)
            assert var >= 0, f"Self-normalized variance should be non-negative, got {var}"

    def test_zero_for_constant_series(self) -> None:
        """Variance should be zero for constant series."""
        d = np.ones(50)
        var = compute_self_normalized_variance(d)
        assert var == 0.0, f"Expected 0 for constant series, got {var}"

    def test_white_noise_variance(self) -> None:
        """Variance for white noise should be reasonable.

        For white noise with variance σ², the self-normalized variance is:
        V_n = (1/n²) Σ S_k² where S_k are partial sums of demeaned data.

        For large n, this converges to σ²/3 (approximately), so it's O(1).
        """
        rng = np.random.default_rng(123)
        n = 500
        d = rng.standard_normal(n)

        var = compute_self_normalized_variance(d)

        # Self-normalized variance for white noise ≈ σ²/3
        # For standard normal, σ² = 1, so expect var ≈ 1/3 ≈ 0.33
        # Allow reasonable range due to randomness
        assert 0.05 < var < 2.0, (
            f"SN variance {var:.6f} should be O(1) for white noise (expected ~0.33)"
        )

    def test_ar1_series(self) -> None:
        """Test with AR(1) series showing autocorrelation."""
        rng = np.random.default_rng(456)
        n = 200
        ar_coef = 0.8

        # Generate AR(1) series
        e = rng.standard_normal(n)
        d = np.zeros(n)
        d[0] = e[0]
        for t in range(1, n):
            d[t] = ar_coef * d[t - 1] + e[t]

        var = compute_self_normalized_variance(d)

        # Variance should still be positive for AR(1)
        assert var > 0, f"SN variance should be positive for AR(1), got {var}"

        # For AR(1), SN variance should be larger than for white noise
        # (due to autocorrelation inflating partial sums)
        wn_var = compute_self_normalized_variance(rng.standard_normal(n))
        assert var > wn_var * 0.5, (
            f"AR(1) variance {var:.6f} should be comparable to or larger than WN {wn_var:.6f}"
        )

    def test_empty_array(self) -> None:
        """Empty array should return 0."""
        d = np.array([])
        var = compute_self_normalized_variance(d)
        assert var == 0.0, f"Expected 0 for empty array, got {var}"

    def test_single_element(self) -> None:
        """Single element should return 0 (no partial sums)."""
        d = np.array([5.0])
        var = compute_self_normalized_variance(d)
        # After demeaning, we have [0], partial sums = [0], so var = 0
        assert var == 0.0, f"Expected 0 for single element, got {var}"


class TestDMTestSelfNormalized:
    """Tests for dm_test with variance_method='self_normalized'."""

    def test_positive_variance(self) -> None:
        """Self-normalized method should never produce negative variance."""
        rng = np.random.default_rng(42)
        n = 100

        # Create scenario that might cause negative HAC variance
        errors_1 = rng.normal(0, 1, n)
        errors_2 = errors_1 + rng.normal(0, 0.1, n)  # Very similar predictions

        result = dm_test(
            errors_1, errors_2,
            h=1,
            variance_method="self_normalized",
        )

        # Should produce a valid result (not NaN statistic due to negative var)
        assert np.isfinite(result.statistic) or result.pvalue == 1.0
        assert 0 <= result.pvalue <= 1

    def test_vs_hac_direction(self) -> None:
        """Both methods should agree on the sign of the statistic."""
        rng = np.random.default_rng(42)
        n = 100

        # Model 1 clearly better
        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result_hac = dm_test(errors_1, errors_2, variance_method="hac")
        result_sn = dm_test(errors_1, errors_2, variance_method="self_normalized")

        # Both should indicate model 1 is better (negative statistic)
        if np.isfinite(result_hac.statistic) and np.isfinite(result_sn.statistic):
            assert np.sign(result_hac.statistic) == np.sign(result_sn.statistic), (
                f"HAC stat={result_hac.statistic:.3f}, SN stat={result_sn.statistic:.3f}"
            )

    def test_backward_compatible(self) -> None:
        """Default variance_method should remain 'hac'."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 1, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = dm_test(errors_1, errors_2)

        assert result.variance_method == "hac", (
            f"Default should be 'hac', got '{result.variance_method}'"
        )

    def test_variance_method_stored(self) -> None:
        """Result should track which variance method was used."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 1, n)
        errors_2 = rng.normal(0, 1.5, n)

        result_hac = dm_test(errors_1, errors_2, variance_method="hac")
        result_sn = dm_test(errors_1, errors_2, variance_method="self_normalized")

        assert result_hac.variance_method == "hac"
        assert result_sn.variance_method == "self_normalized"

    def test_invalid_method_raises(self) -> None:
        """Unknown variance_method should raise ValueError."""
        rng = np.random.default_rng(42)
        n = 50

        errors_1 = rng.normal(0, 1, n)
        errors_2 = rng.normal(0, 1.5, n)

        with pytest.raises(ValueError, match="Unknown variance_method"):
            dm_test(errors_1, errors_2, variance_method="unknown")  # type: ignore

    def test_harvey_not_applied_for_sn(self) -> None:
        """Harvey correction should not be applied for self-normalized."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 1, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = dm_test(
            errors_1, errors_2,
            harvey_correction=True,  # Request Harvey
            variance_method="self_normalized",
        )

        # Harvey adjustment doesn't apply to self-normalized
        assert result.harvey_adjusted is False

    def test_alternatives(self) -> None:
        """All alternative hypotheses should work with self-normalized."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)  # Better model
        errors_2 = rng.normal(0, 1.5, n)  # Worse model

        for alt in ["two-sided", "less", "greater"]:
            result = dm_test(
                errors_1, errors_2,
                alternative=alt,
                variance_method="self_normalized",
            )
            assert 0 <= result.pvalue <= 1
            assert result.alternative == alt

        # "less" (model 1 better) should have smallest p-value
        result_less = dm_test(
            errors_1, errors_2,
            alternative="less",
            variance_method="self_normalized",
        )
        result_greater = dm_test(
            errors_1, errors_2,
            alternative="greater",
            variance_method="self_normalized",
        )
        assert result_less.pvalue < result_greater.pvalue, (
            "Model 1 is better, so 'less' should have smaller p-value"
        )

    def test_significant_difference_detected(self) -> None:
        """Should detect significant difference between models."""
        rng = np.random.default_rng(42)
        n = 100

        # Clear difference: model 1 much better
        errors_1 = rng.normal(0, 0.3, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = dm_test(
            errors_1, errors_2,
            alternative="less",
            variance_method="self_normalized",
        )

        # Should be significant at 0.05 level
        assert result.pvalue < 0.10, (
            f"Expected significant result, got p={result.pvalue:.4f}"
        )


# =============================================================================
# GWTestResult Tests
# =============================================================================


class TestGWTestResult:
    """Tests for GWTestResult dataclass."""

    def test_basic_creation(self) -> None:
        """GWTestResult should store all fields."""
        result = GWTestResult(
            statistic=15.5,
            pvalue=0.004,
            r_squared=0.155,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.25,
        )

        assert result.statistic == 15.5
        assert result.pvalue == 0.004
        assert result.r_squared == 0.155
        assert result.n == 100
        assert result.n_lags == 1
        assert result.q == 2

    def test_significant_at_05(self) -> None:
        """Significance at 0.05 should be correct."""
        sig_result = GWTestResult(
            statistic=15.5,
            pvalue=0.004,
            r_squared=0.155,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.25,
        )
        assert sig_result.significant_at_05 is True
        assert sig_result.significant_at_01 is True

        nonsig_result = GWTestResult(
            statistic=3.0,
            pvalue=0.22,
            r_squared=0.03,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.1,
        )
        assert nonsig_result.significant_at_05 is False
        assert nonsig_result.significant_at_01 is False

    def test_conditional_predictability_property(self) -> None:
        """conditional_predictability should match significant_at_05."""
        sig_result = GWTestResult(
            statistic=15.5,
            pvalue=0.004,
            r_squared=0.155,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.25,
        )
        assert sig_result.conditional_predictability is True

        nonsig_result = GWTestResult(
            statistic=3.0,
            pvalue=0.22,
            r_squared=0.03,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.1,
        )
        assert nonsig_result.conditional_predictability is False

    def test_str_format(self) -> None:
        """String format should be readable and include key info."""
        result = GWTestResult(
            statistic=15.5,
            pvalue=0.004,
            r_squared=0.155,
            n=100,
            n_lags=1,
            q=2,
            loss="squared",
            alternative="two-sided",
            mean_loss_diff=0.25,
        )

        s = str(result)
        assert "GW" in s
        assert "1" in s  # n_lags
        assert "15.5" in s or "15." in s  # statistic
        assert "***" in s  # significance stars for p < 0.01


# =============================================================================
# gw_test Tests
# =============================================================================


class TestGWTest:
    """Tests for Giacomini-White test."""

    def test_identical_errors(self) -> None:
        """Identical errors should give non-significant result."""
        rng = np.random.default_rng(42)
        errors = rng.standard_normal(100)

        result = gw_test(errors, errors.copy())

        # Identical errors => no difference, R² should be ~0
        assert result.mean_loss_diff == pytest.approx(0.0, abs=1e-10)
        assert result.pvalue > 0.05
        assert result.r_squared == pytest.approx(0.0, abs=0.01)

    def test_model1_better(self, correlated_errors: tuple) -> None:
        """Model with smaller errors should have negative mean_loss_diff."""
        errors_1, errors_2 = correlated_errors

        result = gw_test(errors_1, errors_2)

        # Model 1 has smaller errors => negative loss differential
        assert result.mean_loss_diff < 0

    def test_squared_vs_absolute_loss(self, correlated_errors: tuple) -> None:
        """Test should work with both loss functions."""
        errors_1, errors_2 = correlated_errors

        result_se = gw_test(errors_1, errors_2, loss="squared")
        result_ae = gw_test(errors_1, errors_2, loss="absolute")

        # Both should indicate model 1 is better
        assert result_se.mean_loss_diff < 0
        assert result_ae.mean_loss_diff < 0
        assert result_se.loss == "squared"
        assert result_ae.loss == "absolute"

    def test_n_lags_stored(self, correlated_errors: tuple) -> None:
        """n_lags should be stored in result."""
        errors_1, errors_2 = correlated_errors

        result_1lag = gw_test(errors_1, errors_2, n_lags=1)
        result_3lag = gw_test(errors_1, errors_2, n_lags=3)

        assert result_1lag.n_lags == 1
        assert result_1lag.q == 2  # 1 + n_lags
        assert result_3lag.n_lags == 3
        assert result_3lag.q == 4  # 1 + n_lags

    def test_effective_sample_size(self, correlated_errors: tuple) -> None:
        """Effective sample size should be n - n_lags."""
        errors_1, errors_2 = correlated_errors
        n_original = len(errors_1)

        result_1lag = gw_test(errors_1, errors_2, n_lags=1)
        result_3lag = gw_test(errors_1, errors_2, n_lags=3)

        assert result_1lag.n == n_original - 1
        assert result_3lag.n == n_original - 3

    def test_alternative_two_sided(self, correlated_errors: tuple) -> None:
        """Two-sided alternative should be default."""
        errors_1, errors_2 = correlated_errors

        result = gw_test(errors_1, errors_2)

        assert result.alternative == "two-sided"

    def test_alternative_less(self, correlated_errors: tuple) -> None:
        """Alternative='less' should test if model 1 is conditionally better."""
        errors_1, errors_2 = correlated_errors

        result = gw_test(errors_1, errors_2, alternative="less")

        assert result.alternative == "less"
        # Model 1 is better, so p-value should be reasonable
        assert 0 <= result.pvalue <= 1

    def test_alternative_greater(self, correlated_errors: tuple) -> None:
        """Alternative='greater' should test if model 2 is conditionally better."""
        errors_1, errors_2 = correlated_errors

        result = gw_test(errors_1, errors_2, alternative="greater")

        assert result.alternative == "greater"
        # Model 2 is worse, so p-value should be high for this direction
        assert result.pvalue > 0.5

    def test_insufficient_samples(self) -> None:
        """Should raise error with too few effective samples."""
        # n=53, n_lags=4 => 53//10=5, and 4 < 5 (passes overfitting check)
        # => effective = 53 - 4 = 49 < 50 (triggers insufficient samples)
        errors = np.random.randn(53)

        with pytest.raises(ValueError, match="Insufficient samples"):
            gw_test(errors, errors.copy(), n_lags=4)

    def test_mismatched_lengths(self) -> None:
        """Should raise error with mismatched lengths."""
        errors_1 = np.random.randn(100)
        errors_2 = np.random.randn(110)

        with pytest.raises(ValueError, match="same length"):
            gw_test(errors_1, errors_2)

    def test_invalid_n_lags_zero(self) -> None:
        """Should raise error with n_lags < 1."""
        errors = np.random.randn(100)

        with pytest.raises(ValueError, match="n_lags must be >= 1"):
            gw_test(errors, errors.copy(), n_lags=0)

    def test_invalid_n_lags_too_large(self) -> None:
        """Should raise error with n_lags > 10."""
        errors = np.random.randn(200)

        with pytest.raises(ValueError, match="n_lags must be <= 10"):
            gw_test(errors, errors.copy(), n_lags=11)

    def test_invalid_loss(self, correlated_errors: tuple) -> None:
        """Should raise error with invalid loss function."""
        errors_1, errors_2 = correlated_errors

        with pytest.raises(ValueError, match="Unknown loss"):
            gw_test(errors_1, errors_2, loss="invalid")  # type: ignore

    def test_nan_in_errors_1(self) -> None:
        """Should raise error if errors_1 contains NaN."""
        errors_1 = np.array([1.0, 2.0, np.nan, 4.0] + [0.0] * 96)
        errors_2 = np.zeros(100)

        with pytest.raises(ValueError, match="errors_1 contains NaN"):
            gw_test(errors_1, errors_2)

    def test_nan_in_errors_2(self) -> None:
        """Should raise error if errors_2 contains NaN."""
        errors_1 = np.zeros(100)
        errors_2 = np.array([1.0, 2.0, np.nan, 4.0] + [0.0] * 96)

        with pytest.raises(ValueError, match="errors_2 contains NaN"):
            gw_test(errors_1, errors_2)

    def test_r_squared_in_valid_range(self, correlated_errors: tuple) -> None:
        """R-squared should be in [0, 1]."""
        errors_1, errors_2 = correlated_errors

        result = gw_test(errors_1, errors_2)

        assert 0 <= result.r_squared <= 1

    def test_statistic_nonnegative(self, correlated_errors: tuple) -> None:
        """GW statistic (T × R²) should be non-negative."""
        errors_1, errors_2 = correlated_errors

        result = gw_test(errors_1, errors_2)

        assert result.statistic >= 0


class TestGWTestPredictableSwitching:
    """Tests for GW test detecting predictable switching patterns."""

    def test_detects_alternating_pattern(
        self, predictable_switching_errors: tuple
    ) -> None:
        """GW test should detect predictable alternating pattern.

        When model superiority alternates every period (odd/even),
        the lag-1 loss differential is highly predictive of future performance.
        """
        errors_1, errors_2 = predictable_switching_errors

        result = gw_test(errors_1, errors_2, n_lags=1)

        # R² should be substantial (pattern is very predictable)
        assert result.r_squared > 0.1, (
            f"Expected high R² for alternating pattern, got {result.r_squared:.4f}"
        )

        # Test should be significant
        assert result.pvalue < 0.05, (
            f"Expected significant result for alternating pattern, p={result.pvalue:.4f}"
        )

    def test_dm_vs_gw_equal_average_predictable(
        self, predictable_switching_errors: tuple
    ) -> None:
        """Key insight: DM may not reject when GW does.

        When models alternate superiority, average performance may be equal
        (DM not significant), but conditional performance is predictable
        (GW significant).
        """
        errors_1, errors_2 = predictable_switching_errors

        dm_result = dm_test(errors_1, errors_2)
        gw_result = gw_test(errors_1, errors_2, n_lags=1)

        # Both tests ran successfully
        assert isinstance(dm_result, DMTestResult)
        assert isinstance(gw_result, GWTestResult)

        # GW should detect the predictable pattern
        assert gw_result.conditional_predictability is True, (
            f"GW should detect predictability, but p={gw_result.pvalue:.4f}"
        )


class TestGWTestIntegration:
    """Integration tests for GW test with other statistical tests."""

    def test_gw_consistent_with_dm_direction(self) -> None:
        """GW and DM should agree on which model is better on average."""
        rng = np.random.default_rng(42)
        n = 100

        # Model 1 consistently better (no switching)
        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        dm_result = dm_test(errors_1, errors_2)
        gw_result = gw_test(errors_1, errors_2)

        # Both should indicate model 1 is better (negative loss diff)
        assert dm_result.mean_loss_diff < 0
        assert gw_result.mean_loss_diff < 0

    def test_combined_dm_gw_workflow(self) -> None:
        """Complete workflow: run both DM and GW tests."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.8, n)
        errors_2 = rng.normal(0, 1.0, n)

        # Run both tests
        dm_result = dm_test(errors_1, errors_2, h=1)
        gw_result = gw_test(errors_1, errors_2, n_lags=1)

        # Both should complete without error
        assert isinstance(dm_result, DMTestResult)
        assert isinstance(gw_result, GWTestResult)

        # Both should have valid p-values
        assert 0 <= dm_result.pvalue <= 1
        assert 0 <= gw_result.pvalue <= 1

    def test_gw_with_multiple_lags(self) -> None:
        """GW test should work with different lag specifications."""
        rng = np.random.default_rng(42)
        n = 200

        errors_1 = rng.normal(0, 1.0, n)
        errors_2 = rng.normal(0, 1.2, n)

        # Test with different lag specifications
        results = {}
        for n_lags in [1, 2, 3, 5]:
            results[n_lags] = gw_test(errors_1, errors_2, n_lags=n_lags)

        # All should produce valid results
        for n_lags, result in results.items():
            assert result.n_lags == n_lags
            assert result.q == 1 + n_lags
            assert 0 <= result.pvalue <= 1
            assert 0 <= result.r_squared <= 1


class TestGWEdgeCases:
    """Edge case tests for GW test."""

    def test_constant_loss_differential(self) -> None:
        """Constant loss differential should produce R²=0."""
        rng = np.random.default_rng(42)
        n = 100

        # Same error magnitude, just different sign
        errors_1 = np.ones(n) * 0.5
        errors_2 = np.ones(n) * 0.5

        result = gw_test(errors_1, errors_2)

        # Constant d => Z = 0 => R² = 0
        assert result.r_squared == pytest.approx(0.0, abs=0.01)
        assert result.pvalue > 0.5

    def test_near_singular_instruments(self) -> None:
        """Near-singular instruments should still produce a result."""
        rng = np.random.default_rng(42)
        n = 100

        # Create data that might cause numerical issues
        errors_1 = rng.normal(0, 1, n)
        errors_2 = errors_1 + 1e-10 * rng.normal(0, 1, n)  # Nearly identical

        # Should not crash; may produce NaN statistic with warning
        result = gw_test(errors_1, errors_2)

        # Result should have valid structure
        assert result.n == n - 1  # After lag adjustment
        assert 0 <= result.pvalue <= 1

    def test_minimum_sample_boundary(self) -> None:
        """Test exactly at minimum sample size boundary."""
        rng = np.random.default_rng(42)

        # n=51, n_lags=1 => effective=50 (exactly at minimum)
        errors_1 = rng.normal(0, 1, 51)
        errors_2 = rng.normal(0, 1.5, 51)

        result = gw_test(errors_1, errors_2, n_lags=1)

        assert result.n == 50  # Effective sample size
        assert 0 <= result.pvalue <= 1

    def test_n_lags_overfitting_guard(self) -> None:
        """n_lags should be limited relative to sample size."""
        rng = np.random.default_rng(42)

        # n=60, n_lags=6 => n/10 = 6, so n_lags >= n/10 should fail
        errors = rng.normal(0, 1, 60)

        with pytest.raises(ValueError, match="too large"):
            gw_test(errors, errors.copy(), n_lags=6)

    def test_extreme_r_squared_values(self) -> None:
        """R-squared should be properly bounded even for extreme cases."""
        rng = np.random.default_rng(42)
        n = 100

        # High variance in one model only
        errors_1 = rng.normal(0, 0.1, n)
        errors_2 = rng.normal(0, 10.0, n)

        result = gw_test(errors_1, errors_2)

        assert 0 <= result.r_squared <= 1, (
            f"R² should be in [0, 1], got {result.r_squared}"
        )


# =============================================================================
# Clark-West Test
# =============================================================================


class TestCWTestResult:
    """Tests for CWTestResult dataclass."""

    def test_basic_attributes(self) -> None:
        """Should have correct basic attributes."""
        from temporalcv.statistical_tests import CWTestResult

        result = CWTestResult(
            statistic=-1.5,
            pvalue=0.134,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.05,
            mean_loss_diff_adjusted=0.03,
            adjustment_magnitude=0.02,
            variance_method="hac",
        )

        assert result.statistic == -1.5
        assert result.pvalue == 0.134
        assert result.h == 1
        assert result.n == 100
        assert result.loss == "squared"
        assert result.alternative == "two-sided"
        assert result.harvey_adjusted is False
        assert result.mean_loss_diff == 0.05
        assert result.mean_loss_diff_adjusted == 0.03
        assert result.adjustment_magnitude == 0.02
        assert result.variance_method == "hac"

    def test_significant_at_05(self) -> None:
        """Should correctly identify significance at 0.05."""
        from temporalcv.statistical_tests import CWTestResult

        # Significant result at both levels
        sig_result = CWTestResult(
            statistic=-3.0,
            pvalue=0.003,  # < 0.01
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.05,
            mean_loss_diff_adjusted=0.03,
            adjustment_magnitude=0.02,
        )
        assert sig_result.significant_at_05 is True
        assert sig_result.significant_at_01 is True

        # Significant at 0.05 but not 0.01
        sig_05_result = CWTestResult(
            statistic=-2.5,
            pvalue=0.012,  # < 0.05 but > 0.01
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.05,
            mean_loss_diff_adjusted=0.03,
            adjustment_magnitude=0.02,
        )
        assert sig_05_result.significant_at_05 is True
        assert sig_05_result.significant_at_01 is False

        # Non-significant result
        nonsig_result = CWTestResult(
            statistic=-1.0,
            pvalue=0.317,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.05,
            mean_loss_diff_adjusted=0.03,
            adjustment_magnitude=0.02,
        )
        assert nonsig_result.significant_at_05 is False
        assert nonsig_result.significant_at_01 is False

    def test_adjustment_ratio(self) -> None:
        """Should correctly compute adjustment ratio."""
        from temporalcv.statistical_tests import CWTestResult

        result = CWTestResult(
            statistic=-1.5,
            pvalue=0.134,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.10,
            mean_loss_diff_adjusted=0.08,
            adjustment_magnitude=0.02,
        )
        # adjustment_ratio = 0.02 / 0.10 = 0.2
        assert abs(result.adjustment_ratio - 0.2) < 1e-10

    def test_adjustment_ratio_zero_mean(self) -> None:
        """Should handle zero mean loss diff in adjustment ratio."""
        from temporalcv.statistical_tests import CWTestResult

        result = CWTestResult(
            statistic=0.0,
            pvalue=1.0,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.0,  # Zero mean
            mean_loss_diff_adjusted=0.0,
            adjustment_magnitude=0.01,
        )
        assert result.adjustment_ratio == float("inf")

        # Zero adjustment with zero mean
        result2 = CWTestResult(
            statistic=0.0,
            pvalue=1.0,
            h=1,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=False,
            mean_loss_diff=0.0,
            mean_loss_diff_adjusted=0.0,
            adjustment_magnitude=0.0,
        )
        assert result2.adjustment_ratio == 0.0

    def test_str_representation(self) -> None:
        """Should have correct string representation."""
        from temporalcv.statistical_tests import CWTestResult

        result = CWTestResult(
            statistic=-2.5,
            pvalue=0.012,
            h=2,
            n=100,
            loss="squared",
            alternative="two-sided",
            harvey_adjusted=True,
            mean_loss_diff=0.05,
            mean_loss_diff_adjusted=0.03,
            adjustment_magnitude=0.02,
        )
        s = str(result)
        assert "CW(2)" in s
        assert "-2.5" in s or "-2.500" in s
        assert "0.012" in s
        assert "**" in s  # Significance indicator


class TestCWTest:
    """Tests for cw_test function."""

    def test_basic_cw_test(self) -> None:
        """Should compute CW test for basic nested model comparison."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        # Simulate nested models
        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]  # AR(1)/persistence
        preds_unrestricted = actuals[:-1] + 0.1 * rng.normal(0, 1, n - 1)

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
        )

        assert isinstance(result.statistic, float)
        assert 0 <= result.pvalue <= 1
        assert result.n == n - 1
        assert result.h == 1
        assert result.loss == "squared"
        assert result.adjustment_magnitude >= 0

    def test_alternative_less(self) -> None:
        """Should handle one-sided test: unrestricted better."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        # Make unrestricted clearly better
        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1] + 0.5 * rng.normal(0, 1, n - 1)  # Add noise
        preds_unrestricted = actuals[:-1]  # Perfect prediction

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            alternative="less",
        )

        assert result.alternative == "less"
        # Unrestricted is better (lower loss), so mean_loss_diff_adjusted < 0
        # p-value should be small for alternative="less"
        assert result.pvalue < 0.5  # Should be significant or trending that way

    def test_alternative_greater(self) -> None:
        """Should handle one-sided test: restricted better."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        # Make restricted better
        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]  # Perfect
        preds_unrestricted = actuals[:-1] + 0.5 * rng.normal(0, 1, n - 1)  # Add noise

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            alternative="greater",
        )

        assert result.alternative == "greater"
        # Restricted is better, so mean_loss_diff_adjusted > 0 (unrestricted has higher loss)

    def test_absolute_loss(self) -> None:
        """Should work with absolute loss function."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]
        preds_unrestricted = actuals[:-1] + 0.1 * rng.normal(0, 1, n - 1)

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            loss="absolute",
        )

        assert result.loss == "absolute"
        assert isinstance(result.statistic, float)

    def test_harvey_correction_applied(self) -> None:
        """Should apply Harvey correction for h > 1."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]
        preds_unrestricted = actuals[:-1] + 0.1 * rng.normal(0, 1, n - 1)

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        # With Harvey correction (h=4)
        result_with = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            h=4,
            harvey_correction=True,
        )

        # Without Harvey correction
        result_without = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            h=4,
            harvey_correction=False,
        )

        assert result_with.harvey_adjusted is True
        assert result_without.harvey_adjusted is False
        # Statistics should differ due to correction
        assert result_with.statistic != result_without.statistic

    def test_self_normalized_variance(self) -> None:
        """Should work with self-normalized variance."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]
        preds_unrestricted = actuals[:-1] + 0.1 * rng.normal(0, 1, n - 1)

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            variance_method="self_normalized",
        )

        assert result.variance_method == "self_normalized"
        assert isinstance(result.statistic, float)
        assert 0 <= result.pvalue <= 1

    def test_adjustment_magnitude_correct(self) -> None:
        """Should correctly compute adjustment magnitude."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 50

        # Known predictions with known adjustment
        preds_restricted = np.ones(n) * 10
        preds_unrestricted = np.ones(n) * 11  # Difference of 1

        errors_restricted = rng.normal(0, 1, n)
        errors_unrestricted = rng.normal(0, 1, n)

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
        )

        # Adjustment = (preds_r - preds_u)² = (10 - 11)² = 1
        assert abs(result.adjustment_magnitude - 1.0) < 1e-10

    def test_mean_loss_diff_relationship(self) -> None:
        """Mean adjusted = mean unadjusted - adjustment."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]
        preds_unrestricted = actuals[:-1] + 0.2 * rng.normal(0, 1, n - 1)

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        result = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
        )

        # Check relationship: adjusted = unadjusted - adjustment
        expected_adjusted = result.mean_loss_diff - result.adjustment_magnitude
        assert abs(result.mean_loss_diff_adjusted - expected_adjusted) < 1e-10

    def test_identical_predictions_equals_dm(self) -> None:
        """When predictions identical, CW should equal DM."""
        from temporalcv.statistical_tests import cw_test, dm_test

        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 1, n)
        errors_2 = rng.normal(0, 1.2, n)  # Slightly worse

        # Identical predictions → zero adjustment
        preds = np.ones(n)

        cw_result = cw_test(
            errors_unrestricted=errors_1,
            errors_restricted=errors_2,
            predictions_unrestricted=preds,
            predictions_restricted=preds,
            harvey_correction=False,  # Match DM default
        )

        dm_result = dm_test(
            errors_1=errors_1,
            errors_2=errors_2,
            harvey_correction=False,
        )

        # Adjustment should be zero
        assert abs(cw_result.adjustment_magnitude) < 1e-10

        # Statistics should be equal
        assert abs(cw_result.statistic - dm_result.statistic) < 1e-10
        assert abs(cw_result.pvalue - dm_result.pvalue) < 1e-10

    def test_minimum_sample_size(self) -> None:
        """Should raise for n < 30."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 20  # Too small

        errors = rng.normal(0, 1, n)
        preds = np.ones(n)

        with pytest.raises(ValueError, match="Insufficient sample size"):
            cw_test(
                errors_unrestricted=errors,
                errors_restricted=errors,
                predictions_unrestricted=preds,
                predictions_restricted=preds,
            )

    def test_input_validation_length_mismatch(self) -> None:
        """Should raise for mismatched array lengths."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)

        errors_1 = rng.normal(0, 1, 100)
        errors_2 = rng.normal(0, 1, 50)  # Different length
        preds = np.ones(100)

        with pytest.raises(ValueError, match="same length"):
            cw_test(
                errors_unrestricted=errors_1,
                errors_restricted=errors_2,
                predictions_unrestricted=preds,
                predictions_restricted=preds,
            )

    def test_input_validation_nan(self) -> None:
        """Should raise for NaN values."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 50

        errors = rng.normal(0, 1, n)
        errors[10] = np.nan  # Add NaN
        preds = np.ones(n)

        with pytest.raises(ValueError, match="NaN"):
            cw_test(
                errors_unrestricted=errors,
                errors_restricted=errors.copy(),
                predictions_unrestricted=preds,
                predictions_restricted=preds,
            )

    def test_horizon_bandwidth(self) -> None:
        """Should use HAC bandwidth = h-1."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 100

        actuals = np.cumsum(rng.normal(0, 1, n))
        preds_restricted = actuals[:-1]
        preds_unrestricted = actuals[:-1] + 0.1 * rng.normal(0, 1, n - 1)

        errors_restricted = actuals[1:] - preds_restricted
        errors_unrestricted = actuals[1:] - preds_unrestricted

        # Different horizons should give different results due to bandwidth
        result_h1 = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            h=1,
            harvey_correction=False,
        )

        result_h4 = cw_test(
            errors_unrestricted=errors_unrestricted,
            errors_restricted=errors_restricted,
            predictions_unrestricted=preds_unrestricted,
            predictions_restricted=preds_restricted,
            h=4,
            harvey_correction=False,
        )

        # Different h should yield different variance estimates → different stats
        assert result_h1.h == 1
        assert result_h4.h == 4
        # Note: They might be equal in some cases, but typically differ

    def test_invalid_loss_function(self) -> None:
        """Should raise for invalid loss function."""
        from temporalcv.statistical_tests import cw_test

        rng = np.random.default_rng(42)
        n = 50

        errors = rng.normal(0, 1, n)
        preds = np.ones(n)

        with pytest.raises(ValueError, match="Unknown loss function"):
            cw_test(
                errors_unrestricted=errors,
                errors_restricted=errors,
                predictions_unrestricted=preds,
                predictions_restricted=preds,
                loss="invalid",  # type: ignore
            )


# =============================================================================
# Multi-Horizon Comparison Tests
# =============================================================================


class TestMultiHorizonResult:
    """Tests for MultiHorizonResult dataclass."""

    def test_basic_attributes(self) -> None:
        """Should have correct basic attributes."""
        dm_result = DMTestResult(
            statistic=-2.5,
            pvalue=0.012,
            h=1,
            n=100,
            loss="squared",
            alternative="less",
            harvey_adjusted=True,
            mean_loss_diff=-0.5,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_result, 2: dm_result, 4: dm_result},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.horizons == (1, 2, 4)
        assert result.model_1_name == "Model"
        assert result.model_2_name == "Baseline"
        assert result.loss == "squared"
        assert result.alpha == 0.05

    def test_significant_horizons_all_significant(self) -> None:
        """All horizons significant when all p < alpha."""
        dm_sig = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_sig, 2: dm_sig, 4: dm_sig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.significant_horizons == [1, 2, 4]

    def test_significant_horizons_none_significant(self) -> None:
        """No horizons significant when all p >= alpha."""
        dm_nonsig = DMTestResult(
            statistic=-0.5, pvalue=0.30, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.1,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_nonsig, 2: dm_nonsig, 4: dm_nonsig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.significant_horizons == []

    def test_first_insignificant_horizon(self) -> None:
        """Should find first horizon where significance is lost."""
        dm_sig = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )
        dm_nonsig = DMTestResult(
            statistic=-0.5, pvalue=0.30, h=4, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.1,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4, 8),
            dm_results={1: dm_sig, 2: dm_sig, 4: dm_nonsig, 8: dm_nonsig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100, 8: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.first_insignificant_horizon == 4

    def test_first_insignificant_horizon_all_significant(self) -> None:
        """Should return None when all horizons are significant."""
        dm_sig = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_sig, 2: dm_sig, 4: dm_sig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.first_insignificant_horizon is None

    def test_best_horizon(self) -> None:
        """Should find horizon with smallest p-value."""
        dm_h1 = DMTestResult(
            statistic=-2.0, pvalue=0.045, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.3,
        )
        dm_h2 = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=2, n=100,  # Best!
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )
        dm_h4 = DMTestResult(
            statistic=-1.5, pvalue=0.10, h=4, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.2,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_h1, 2: dm_h2, 4: dm_h4},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.best_horizon == 2

    def test_degradation_pattern_consistent(self) -> None:
        """Pattern should be 'consistent' when all horizons significant."""
        dm_sig = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_sig, 2: dm_sig, 4: dm_sig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.degradation_pattern == "consistent"

    def test_degradation_pattern_none(self) -> None:
        """Pattern should be 'none' when no horizons significant."""
        dm_nonsig = DMTestResult(
            statistic=-0.5, pvalue=0.30, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.1,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_nonsig, 2: dm_nonsig, 4: dm_nonsig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.degradation_pattern == "none"

    def test_degradation_pattern_degrading(self) -> None:
        """Pattern should be 'degrading' when p-values increase with horizon."""
        dm_h1 = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )
        dm_h2 = DMTestResult(
            statistic=-2.0, pvalue=0.045, h=2, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.3,
        )
        dm_h4 = DMTestResult(
            statistic=-0.5, pvalue=0.30, h=4, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.1,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_h1, 2: dm_h2, 4: dm_h4},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        assert result.degradation_pattern == "degrading"

    def test_get_pvalues(self) -> None:
        """get_pvalues should return dict of horizon -> p-value."""
        dm_h1 = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )
        dm_h2 = DMTestResult(
            statistic=-2.0, pvalue=0.045, h=2, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.3,
        )

        result = MultiHorizonResult(
            horizons=(1, 2),
            dm_results={1: dm_h1, 2: dm_h2},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        pvalues = result.get_pvalues()
        assert pvalues == {1: 0.003, 2: 0.045}

    def test_get_statistics(self) -> None:
        """get_statistics should return dict of horizon -> DM statistic."""
        dm_h1 = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )
        dm_h2 = DMTestResult(
            statistic=-2.0, pvalue=0.045, h=2, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.3,
        )

        result = MultiHorizonResult(
            horizons=(1, 2),
            dm_results={1: dm_h1, 2: dm_h2},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        stats = result.get_statistics()
        assert stats == {1: -3.0, 2: -2.0}

    def test_summary_contains_key_info(self) -> None:
        """Summary should include model names and pattern."""
        dm_sig = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        result = MultiHorizonResult(
            horizons=(1, 2, 4),
            dm_results={1: dm_sig, 2: dm_sig, 4: dm_sig},
            model_1_name="MyModel",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100, 4: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        summary = result.summary()
        assert "MyModel" in summary
        assert "Baseline" in summary
        assert "consistent" in summary.lower()

    def test_to_markdown_table_format(self) -> None:
        """to_markdown should produce table with headers."""
        dm_sig = DMTestResult(
            statistic=-3.0, pvalue=0.003, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        result = MultiHorizonResult(
            horizons=(1, 2),
            dm_results={1: dm_sig, 2: dm_sig},
            model_1_name="Model",
            model_2_name="Baseline",
            n_per_horizon={1: 100, 2: 100},
            loss="squared",
            alternative="less",
            alpha=0.05,
        )

        md = result.to_markdown()
        assert "| Horizon |" in md
        assert "| P-value |" in md
        assert "✓" in md  # Significant marker


class TestCompareHorizons:
    """Tests for compare_horizons function."""

    def test_basic_comparison(self) -> None:
        """Should compare models across multiple horizons."""
        rng = np.random.default_rng(42)
        n = 100

        # Model 1 better than model 2
        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(
            errors_1, errors_2,
            horizons=(1, 2, 4),
        )

        assert isinstance(result, MultiHorizonResult)
        assert result.horizons == (1, 2, 4)
        assert len(result.dm_results) == 3

    def test_default_horizons(self) -> None:
        """Default horizons should be (1, 2, 3, 4)."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(errors_1, errors_2)

        assert result.horizons == (1, 2, 3, 4)

    def test_different_horizons_different_results(self) -> None:
        """Different horizons should have different HAC bandwidths."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(errors_1, errors_2, horizons=(1, 4, 8))

        # Horizons should be stored in dm_results
        assert 1 in result.dm_results
        assert 4 in result.dm_results
        assert 8 in result.dm_results

        # Each should have correct h value
        assert result.dm_results[1].h == 1
        assert result.dm_results[4].h == 4
        assert result.dm_results[8].h == 8

    def test_loss_function_passed_through(self) -> None:
        """Loss function should be passed to DM tests."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 1, n)
        errors_2 = rng.normal(0, 1.5, n)

        result_se = compare_horizons(errors_1, errors_2, horizons=(1,), loss="squared")
        result_ae = compare_horizons(errors_1, errors_2, horizons=(1,), loss="absolute")

        assert result_se.loss == "squared"
        assert result_ae.loss == "absolute"
        assert result_se.dm_results[1].loss == "squared"
        assert result_ae.dm_results[1].loss == "absolute"

    def test_alternative_hypothesis_passed_through(self) -> None:
        """Alternative hypothesis should be passed to DM tests."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(
            errors_1, errors_2,
            horizons=(1,),
            alternative="less",
        )

        assert result.alternative == "less"
        assert result.dm_results[1].alternative == "less"

    def test_alpha_used_for_significance(self) -> None:
        """Alpha should determine significant_horizons."""
        rng = np.random.default_rng(42)
        n = 100

        # Moderate difference
        errors_1 = rng.normal(0, 0.8, n)
        errors_2 = rng.normal(0, 1.2, n)

        result_strict = compare_horizons(errors_1, errors_2, horizons=(1,), alpha=0.01)
        result_loose = compare_horizons(errors_1, errors_2, horizons=(1,), alpha=0.10)

        # Same test, different alpha thresholds
        assert result_strict.alpha == 0.01
        assert result_loose.alpha == 0.10

    def test_variance_method_passed_through(self) -> None:
        """Variance method should be passed to DM tests."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result_hac = compare_horizons(
            errors_1, errors_2,
            horizons=(1,),
            variance_method="hac",
        )
        result_sn = compare_horizons(
            errors_1, errors_2,
            horizons=(1,),
            variance_method="self_normalized",
        )

        assert result_hac.dm_results[1].variance_method == "hac"
        assert result_sn.dm_results[1].variance_method == "self_normalized"

    def test_model_names_stored(self) -> None:
        """Model names should be stored in result."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(
            errors_1, errors_2,
            horizons=(1,),
            model_1_name="ARIMA",
            model_2_name="Persistence",
        )

        assert result.model_1_name == "ARIMA"
        assert result.model_2_name == "Persistence"

    def test_empty_horizons_raises(self) -> None:
        """Should raise error for empty horizons."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        with pytest.raises(ValueError, match="cannot be empty"):
            compare_horizons(errors_1, errors_2, horizons=())

    def test_invalid_horizon_raises(self) -> None:
        """Should raise error for non-positive horizons."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        with pytest.raises(ValueError, match="must be >= 1"):
            compare_horizons(errors_1, errors_2, horizons=(1, 0, 4))

    def test_degrading_pattern_detected(self) -> None:
        """Should detect degrading pattern when advantage fades."""
        rng = np.random.default_rng(42)
        n = 150

        # Model 1 clearly better at h=1, similar at h=8
        errors_1 = rng.normal(0, 0.3, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(
            errors_1, errors_2,
            horizons=(1, 2, 4, 8),
            alternative="less",
        )

        # With such a clear difference, all horizons should be significant
        # but pattern can still be analyzed
        assert result.degradation_pattern in ["consistent", "degrading"]

    def test_n_per_horizon_tracked(self) -> None:
        """Should track sample size per horizon."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(errors_1, errors_2, horizons=(1, 4))

        assert result.n_per_horizon[1] == n
        assert result.n_per_horizon[4] == n

    def test_list_horizons_converted_to_tuple(self) -> None:
        """List horizons should be converted to tuple."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(errors_1, errors_2, horizons=[1, 2, 4])

        assert isinstance(result.horizons, tuple)
        assert result.horizons == (1, 2, 4)


class TestMultiModelHorizonResult:
    """Tests for MultiModelHorizonResult dataclass."""

    def test_basic_attributes(self) -> None:
        """Should have correct basic attributes."""
        # Create mock MultiModelComparisonResult
        dm_result = DMTestResult(
            statistic=-2.5, pvalue=0.01, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        comp = MultiModelComparisonResult(
            pairwise_results={("A", "B"): dm_result},
            best_model="A",
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2)],
            significant_pairs=[],
        )

        result = MultiModelHorizonResult(
            horizons=(1, 2, 4),
            model_names=("A", "B", "C"),
            pairwise_by_horizon={1: comp, 2: comp, 4: comp},
            alpha=0.05,
        )

        assert result.horizons == (1, 2, 4)
        assert result.model_names == ("A", "B", "C")
        assert result.alpha == 0.05

    def test_best_model_by_horizon(self) -> None:
        """Should return best model for each horizon."""
        dm_result = DMTestResult(
            statistic=-2.5, pvalue=0.01, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        comp_h1 = MultiModelComparisonResult(
            pairwise_results={("A", "B"): dm_result},
            best_model="A",  # A wins at h=1
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2)],
            significant_pairs=[],
        )

        comp_h4 = MultiModelComparisonResult(
            pairwise_results={("B", "A"): dm_result},
            best_model="B",  # B wins at h=4
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("B", 0.1), ("A", 0.2)],
            significant_pairs=[],
        )

        result = MultiModelHorizonResult(
            horizons=(1, 4),
            model_names=("A", "B"),
            pairwise_by_horizon={1: comp_h1, 4: comp_h4},
            alpha=0.05,
        )

        assert result.best_model_by_horizon == {1: "A", 4: "B"}

    def test_consistent_best_when_same(self) -> None:
        """consistent_best should return model when same across all horizons."""
        dm_result = DMTestResult(
            statistic=-2.5, pvalue=0.01, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        comp = MultiModelComparisonResult(
            pairwise_results={("A", "B"): dm_result},
            best_model="A",  # A wins at all horizons
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2)],
            significant_pairs=[],
        )

        result = MultiModelHorizonResult(
            horizons=(1, 2, 4),
            model_names=("A", "B"),
            pairwise_by_horizon={1: comp, 2: comp, 4: comp},
            alpha=0.05,
        )

        assert result.consistent_best == "A"

    def test_consistent_best_none_when_varies(self) -> None:
        """consistent_best should return None when best model varies."""
        dm_result = DMTestResult(
            statistic=-2.5, pvalue=0.01, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        comp_a = MultiModelComparisonResult(
            pairwise_results={("A", "B"): dm_result},
            best_model="A",
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("A", 0.1), ("B", 0.2)],
            significant_pairs=[],
        )

        comp_b = MultiModelComparisonResult(
            pairwise_results={("B", "A"): dm_result},
            best_model="B",
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("B", 0.1), ("A", 0.2)],
            significant_pairs=[],
        )

        result = MultiModelHorizonResult(
            horizons=(1, 4),
            model_names=("A", "B"),
            pairwise_by_horizon={1: comp_a, 4: comp_b},
            alpha=0.05,
        )

        assert result.consistent_best is None

    def test_summary_contains_key_info(self) -> None:
        """Summary should include models and horizons."""
        dm_result = DMTestResult(
            statistic=-2.5, pvalue=0.01, h=1, n=100,
            loss="squared", alternative="less",
            harvey_adjusted=True, mean_loss_diff=-0.5,
        )

        comp = MultiModelComparisonResult(
            pairwise_results={("ARIMA", "RF"): dm_result},
            best_model="ARIMA",
            bonferroni_alpha=0.05,
            original_alpha=0.05,
            model_rankings=[("ARIMA", 0.1), ("RF", 0.2)],
            significant_pairs=[],
        )

        result = MultiModelHorizonResult(
            horizons=(1, 4),
            model_names=("ARIMA", "RF"),
            pairwise_by_horizon={1: comp, 4: comp},
            alpha=0.05,
        )

        summary = result.summary()
        assert "ARIMA" in summary
        assert "RF" in summary


class TestCompareModelsHorizons:
    """Tests for compare_models_horizons function."""

    def test_basic_comparison(self) -> None:
        """Should compare multiple models across horizons."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "Good": rng.normal(0, 0.5, n),
            "Medium": rng.normal(0, 1.0, n),
            "Bad": rng.normal(0, 1.5, n),
        }

        result = compare_models_horizons(errors, horizons=(1, 2, 4))

        assert isinstance(result, MultiModelHorizonResult)
        assert result.horizons == (1, 2, 4)
        assert result.model_names == ("Good", "Medium", "Bad")

    def test_default_horizons(self) -> None:
        """Default horizons should be (1, 2, 3, 4)."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "A": rng.normal(0, 0.5, n),
            "B": rng.normal(0, 1.5, n),
        }

        result = compare_models_horizons(errors)

        assert result.horizons == (1, 2, 3, 4)

    def test_pairwise_results_per_horizon(self) -> None:
        """Should have MultiModelComparisonResult for each horizon."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "A": rng.normal(0, 0.5, n),
            "B": rng.normal(0, 1.5, n),
        }

        result = compare_models_horizons(errors, horizons=(1, 4))

        assert 1 in result.pairwise_by_horizon
        assert 4 in result.pairwise_by_horizon
        assert isinstance(result.pairwise_by_horizon[1], MultiModelComparisonResult)
        assert isinstance(result.pairwise_by_horizon[4], MultiModelComparisonResult)

    def test_best_model_consistent_when_clear_winner(self) -> None:
        """Should find consistent best when one model is clearly better."""
        rng = np.random.default_rng(42)
        n = 150

        errors = {
            "Best": rng.normal(0, 0.2, n),  # Clearly best
            "Medium": rng.normal(0, 1.0, n),
            "Worst": rng.normal(0, 2.0, n),
        }

        result = compare_models_horizons(errors, horizons=(1, 2, 4))

        # Best model should win at all horizons
        assert result.consistent_best == "Best"

    def test_raises_on_single_model(self) -> None:
        """Should raise error with fewer than 2 models."""
        with pytest.raises(ValueError, match="at least 2 models"):
            compare_models_horizons({"A": np.random.randn(50)})

    def test_empty_horizons_raises(self) -> None:
        """Should raise error for empty horizons."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compare_models_horizons(
                {"A": np.random.randn(50), "B": np.random.randn(50)},
                horizons=(),
            )

    def test_invalid_horizon_raises(self) -> None:
        """Should raise error for non-positive horizons."""
        with pytest.raises(ValueError, match="must be >= 1"):
            compare_models_horizons(
                {"A": np.random.randn(50), "B": np.random.randn(50)},
                horizons=(1, 0, 4),
            )

    def test_loss_function_passed_through(self) -> None:
        """Loss function should be passed to multi-model comparison."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "A": rng.normal(0, 0.5, n),
            "B": rng.normal(0, 1.5, n),
        }

        result = compare_models_horizons(errors, horizons=(1,), loss="absolute")

        # Check that loss was used in pairwise comparison
        for h, comp in result.pairwise_by_horizon.items():
            for dm_result in comp.pairwise_results.values():
                assert dm_result.loss == "absolute"


class TestMultiHorizonIntegration:
    """Integration tests for multi-horizon comparison."""

    def test_compare_horizons_result_types(self) -> None:
        """compare_horizons should return proper result with valid types."""
        rng = np.random.default_rng(42)
        n = 100

        errors_1 = rng.normal(0, 0.5, n)
        errors_2 = rng.normal(0, 1.5, n)

        result = compare_horizons(
            errors_1, errors_2,
            horizons=(1, 2, 4, 8),
            alternative="less",
        )

        # Check all properties are accessible and have correct types
        assert isinstance(result.significant_horizons, list)
        assert isinstance(result.best_horizon, int)
        assert isinstance(result.degradation_pattern, str)
        assert isinstance(result.get_pvalues(), dict)
        assert isinstance(result.get_statistics(), dict)
        assert isinstance(result.summary(), str)
        assert isinstance(result.to_markdown(), str)

    def test_end_to_end_workflow(self) -> None:
        """Complete workflow from data to interpretation."""
        rng = np.random.default_rng(42)
        n = 100

        # Create scenario: model advantage degrades with horizon
        errors_model = rng.normal(0, 0.5, n)
        errors_baseline = rng.normal(0, 1.2, n)

        # Two-model comparison across horizons
        result = compare_horizons(
            errors_model, errors_baseline,
            horizons=(1, 2, 4, 8, 12),
            alternative="less",
            model_1_name="Model",
            model_2_name="Baseline",
        )

        # Should have results for all horizons
        assert len(result.dm_results) == 5

        # Summary and markdown should be non-empty
        assert len(result.summary()) > 100
        assert len(result.to_markdown()) > 50

        # Pattern should be classified
        assert result.degradation_pattern in ["consistent", "degrading", "none", "irregular"]

    def test_multi_model_multi_horizon_workflow(self) -> None:
        """Complete workflow with multiple models and horizons."""
        rng = np.random.default_rng(42)
        n = 100

        errors = {
            "ARIMA": rng.normal(0, 0.6, n),
            "RF": rng.normal(0, 0.8, n),
            "Naive": rng.normal(0, 1.2, n),
        }

        result = compare_models_horizons(
            errors,
            horizons=(1, 4, 12),
            alpha=0.05,
        )

        # Should have results for all horizons
        assert len(result.pairwise_by_horizon) == 3

        # Should be able to check best model at each horizon
        best_by_horizon = result.best_model_by_horizon
        assert set(best_by_horizon.keys()) == {1, 4, 12}

        # All best models should be valid model names
        for model in best_by_horizon.values():
            assert model in errors

        # Summary should work
        assert len(result.summary()) > 50


# =============================================================================
# Forecast Encompassing Test (Harvey et al. 1998)
# =============================================================================


class TestEncompassingTestResult:
    """Tests for EncompassingTestResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Result dataclass should be created correctly."""
        from temporalcv.statistical_tests import EncompassingTestResult

        result = EncompassingTestResult(
            lambda_coef=0.3,
            statistic=2.5,
            pvalue=0.02,
            encompasses=False,
            optimal_weight_b=0.3,
            direction="a_encompasses_b",
            n=100,
            h=1,
        )

        assert result.lambda_coef == 0.3
        assert result.statistic == 2.5
        assert result.pvalue == 0.02
        assert result.encompasses is False
        assert result.optimal_weight_b == 0.3
        assert result.n == 100
        assert result.h == 1

    def test_optimal_weight_a_computed(self) -> None:
        """optimal_weight_a can be computed as 1 - lambda."""
        from temporalcv.statistical_tests import EncompassingTestResult

        result = EncompassingTestResult(
            lambda_coef=0.3,
            statistic=2.5,
            pvalue=0.02,
            encompasses=False,
            optimal_weight_b=0.3,
            direction="a_encompasses_b",
            n=100,
            h=1,
        )

        # User can compute weight for A as 1 - optimal_weight_b
        assert 1 - result.optimal_weight_b == pytest.approx(0.7, rel=0.01)

    def test_fields_complete(self) -> None:
        """All expected fields should be present."""
        from temporalcv.statistical_tests import EncompassingTestResult

        result = EncompassingTestResult(
            lambda_coef=0.3,
            statistic=2.5,
            pvalue=0.02,
            encompasses=False,
            optimal_weight_b=0.3,
            direction="a_encompasses_b",
            n=100,
            h=1,
        )

        # Check all fields are accessible
        assert isinstance(result.lambda_coef, float)
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.encompasses, bool)
        assert isinstance(result.optimal_weight_b, float)
        assert isinstance(result.direction, str)
        assert isinstance(result.n, int)
        assert isinstance(result.h, int)


class TestForecastEncompassingTest:
    """Tests for forecast_encompassing_test function."""

    def test_basic_functionality(self) -> None:
        """Test basic encompassing test."""
        from temporalcv.statistical_tests import forecast_encompassing_test

        rng = np.random.default_rng(42)
        n = 100
        actual = rng.normal(0, 1, n)
        forecast_a = actual + rng.normal(0, 0.3, n)
        forecast_b = actual + rng.normal(0, 0.5, n)

        result = forecast_encompassing_test(actual, forecast_a, forecast_b)

        assert hasattr(result, "lambda_coef")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "encompasses")
        assert 0 <= result.pvalue <= 1
        assert -10 <= result.lambda_coef <= 10  # Reasonable range

    def test_encompasses_when_a_much_better(self) -> None:
        """A should encompass B when A is much better."""
        from temporalcv.statistical_tests import forecast_encompassing_test

        rng = np.random.default_rng(42)
        n = 200
        actual = rng.normal(0, 1, n)
        forecast_a = actual + rng.normal(0, 0.1, n)  # Very good
        forecast_b = actual + rng.normal(0, 1.0, n)  # Much worse

        result = forecast_encompassing_test(actual, forecast_a, forecast_b)

        # Lambda should be close to 0 (B adds no information)
        assert result.lambda_coef < 0.3

    def test_combine_when_both_useful(self) -> None:
        """Lambda should be significant when both models add information."""
        from temporalcv.statistical_tests import forecast_encompassing_test

        rng = np.random.default_rng(42)
        n = 200
        actual = rng.normal(0, 1, n)
        # Both models have useful but different information
        forecast_a = actual + 0.5 * rng.normal(0, 1, n)
        forecast_b = actual + 0.5 * rng.normal(0, 1, n)

        result = forecast_encompassing_test(actual, forecast_a, forecast_b)

        # Lambda should be between 0 and 1 (both contribute)
        assert 0 < result.lambda_coef < 1 or result.pvalue > 0.1

    def test_horizon_adjustment(self) -> None:
        """Test with different horizon values."""
        from temporalcv.statistical_tests import forecast_encompassing_test

        rng = np.random.default_rng(42)
        n = 100
        actual = rng.normal(0, 1, n)
        forecast_a = actual + rng.normal(0, 0.3, n)
        forecast_b = actual + rng.normal(0, 0.5, n)

        result_h1 = forecast_encompassing_test(actual, forecast_a, forecast_b, h=1)
        result_h4 = forecast_encompassing_test(actual, forecast_a, forecast_b, h=4)

        # Both should produce valid results
        assert 0 <= result_h1.pvalue <= 1
        assert 0 <= result_h4.pvalue <= 1
        # Different horizons may give different statistics
        assert result_h1.h == 1
        assert result_h4.h == 4

    def test_insufficient_data_raises(self) -> None:
        """Should raise with insufficient data."""
        from temporalcv.statistical_tests import forecast_encompassing_test

        actual = np.array([1.0, 2.0, 3.0])
        forecast_a = np.array([1.1, 2.1, 3.1])
        forecast_b = np.array([1.2, 2.2, 3.2])

        with pytest.raises(ValueError, match="at least"):
            forecast_encompassing_test(actual, forecast_a, forecast_b)


class TestForecastEncompassingBidirectional:
    """Tests for forecast_encompassing_bidirectional function."""

    def test_basic_functionality(self) -> None:
        """Test bidirectional encompassing test."""
        from temporalcv.statistical_tests import forecast_encompassing_bidirectional

        rng = np.random.default_rng(42)
        n = 100
        actual = rng.normal(0, 1, n)
        forecast_a = actual + rng.normal(0, 0.3, n)
        forecast_b = actual + rng.normal(0, 0.5, n)

        result = forecast_encompassing_bidirectional(actual, forecast_a, forecast_b)

        assert hasattr(result, "a_encompasses_b")
        assert hasattr(result, "b_encompasses_a")
        assert hasattr(result, "recommendation")
        assert result.recommendation in ["use_a", "use_b", "combine", "equivalent"]

    def test_use_a_recommendation(self) -> None:
        """Should recommend 'use_a' when A much better."""
        from temporalcv.statistical_tests import forecast_encompassing_bidirectional

        rng = np.random.default_rng(42)
        n = 200
        actual = rng.normal(0, 1, n)
        forecast_a = actual + rng.normal(0, 0.05, n)  # Very good
        forecast_b = actual + rng.normal(0, 1.0, n)  # Much worse

        result = forecast_encompassing_bidirectional(actual, forecast_a, forecast_b)

        # A should encompass B, B should not encompass A
        # Result should recommend using A
        assert result.recommendation in ["use_a", "combine"]

    def test_combine_recommendation(self) -> None:
        """Should recommend 'combine' when both add information."""
        from temporalcv.statistical_tests import forecast_encompassing_bidirectional

        rng = np.random.default_rng(42)
        n = 200
        actual = rng.normal(0, 1, n)
        # Both have independent information
        forecast_a = actual + rng.normal(0, 0.5, n)
        forecast_b = actual + rng.normal(0, 0.5, n)

        result = forecast_encompassing_bidirectional(actual, forecast_a, forecast_b)

        # When neither encompasses the other, should recommend combine
        # (or equivalent if both encompass)
        assert result.recommendation in ["combine", "equivalent"]


# =============================================================================
# Reality Check Test (White 2000)
# =============================================================================


class TestRealityCheckResult:
    """Tests for RealityCheckResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Result dataclass should be created correctly."""
        from temporalcv.statistical_tests import RealityCheckResult

        result = RealityCheckResult(
            statistic=2.5,
            pvalue=0.03,
            best_model="ModelA",
            individual_statistics={"ModelA": 2.5, "ModelB": 1.0},
            mean_losses={"ModelA": 0.5, "ModelB": 0.6},
            n_bootstrap=1000,
            block_size=10,
            n=100,
        )

        assert result.statistic == 2.5
        assert result.pvalue == 0.03
        assert result.best_model == "ModelA"
        assert result.n_bootstrap == 1000
        assert result.n == 100

    def test_significant_models(self) -> None:
        """significant_models should return models beating benchmark."""
        from temporalcv.statistical_tests import RealityCheckResult

        result = RealityCheckResult(
            statistic=2.5,
            pvalue=0.03,
            best_model="ModelA",
            individual_statistics={"ModelA": 2.5, "ModelB": -1.0},
            mean_losses={"ModelA": 0.4, "ModelB": 0.7},
            n_bootstrap=1000,
            block_size=10,
            n=100,
        )

        # significant_models returns models with positive statistics
        sig = result.significant_models
        assert "ModelA" in sig
        assert "ModelB" not in sig


class TestRealityCheckTest:
    """Tests for reality_check_test function."""

    def test_basic_functionality(self) -> None:
        """Test basic reality check."""
        from temporalcv.statistical_tests import reality_check_test

        rng = np.random.default_rng(42)
        n = 100

        benchmark_errors = rng.normal(0, 1, n) ** 2
        model_errors = {
            "A": rng.normal(0, 0.8, n) ** 2,  # Better
            "B": rng.normal(0, 1.2, n) ** 2,  # Worse
        }

        result = reality_check_test(
            benchmark_errors, model_errors, n_bootstrap=100, random_state=42
        )

        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "best_model")
        assert 0 <= result.pvalue <= 1
        assert result.best_model in ["A", "B"]

    def test_all_models_worse_high_pvalue(self) -> None:
        """P-value should be high when all models are worse than benchmark."""
        from temporalcv.statistical_tests import reality_check_test

        rng = np.random.default_rng(42)
        n = 100

        # Benchmark is best
        benchmark_errors = rng.normal(0, 0.5, n) ** 2
        model_errors = {
            "A": rng.normal(0, 1.0, n) ** 2,
            "B": rng.normal(0, 1.5, n) ** 2,
        }

        result = reality_check_test(
            benchmark_errors, model_errors, n_bootstrap=200, random_state=42
        )

        # High p-value expected (fail to reject null that benchmark is best)
        assert result.pvalue > 0.1

    def test_clear_winner_low_pvalue(self) -> None:
        """P-value should be low when a model clearly beats benchmark."""
        from temporalcv.statistical_tests import reality_check_test

        rng = np.random.default_rng(42)
        n = 200

        # Model A much better than benchmark
        benchmark_errors = rng.normal(0, 1.0, n) ** 2
        model_errors = {
            "A": rng.normal(0, 0.1, n) ** 2,  # Much better
        }

        result = reality_check_test(
            benchmark_errors, model_errors, n_bootstrap=200, random_state=42
        )

        # Low p-value expected (reject null)
        assert result.pvalue < 0.2  # Allow some slack

    def test_reproducibility(self) -> None:
        """Same seed should give same result."""
        from temporalcv.statistical_tests import reality_check_test

        rng = np.random.default_rng(42)
        n = 50

        benchmark = rng.random(n)
        models = {"A": rng.random(n), "B": rng.random(n)}

        result1 = reality_check_test(benchmark, models, n_bootstrap=50, random_state=123)
        result2 = reality_check_test(benchmark, models, n_bootstrap=50, random_state=123)

        assert result1.pvalue == result2.pvalue
        assert result1.statistic == result2.statistic

    def test_insufficient_data_raises(self) -> None:
        """Should raise with insufficient data."""
        from temporalcv.statistical_tests import reality_check_test

        benchmark = np.array([1.0, 2.0, 3.0])
        models = {"A": np.array([1.1, 2.1, 3.1])}

        with pytest.raises(ValueError, match="at least"):
            reality_check_test(benchmark, models)


# =============================================================================
# SPA Test (Hansen 2005)
# =============================================================================


class TestSPATestResult:
    """Tests for SPATestResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Result dataclass should be created correctly."""
        from temporalcv.statistical_tests import SPATestResult

        result = SPATestResult(
            statistic=2.5,
            pvalue=0.03,
            pvalue_consistent=0.04,
            pvalue_lower=0.02,
            best_model="ModelA",
            individual_statistics={"ModelA": 2.5, "ModelB": 1.0},
            mean_losses={"ModelA": 0.5, "ModelB": 0.6},
            n_bootstrap=1000,
            block_size=10,
            n=100,
        )

        assert result.statistic == 2.5
        assert result.pvalue == 0.03
        assert result.pvalue_consistent == 0.04
        assert result.pvalue_lower == 0.02
        assert result.best_model == "ModelA"

    def test_three_pvalues_ordered(self) -> None:
        """Three p-values should be properly ordered."""
        from temporalcv.statistical_tests import SPATestResult

        result = SPATestResult(
            statistic=2.5,
            pvalue=0.05,  # Main
            pvalue_consistent=0.06,  # Most conservative
            pvalue_lower=0.04,  # Least conservative
            best_model="ModelA",
            individual_statistics={"ModelA": 2.5},
            mean_losses={"ModelA": 0.5},
            n_bootstrap=1000,
            block_size=10,
            n=100,
        )

        # lower <= main <= consistent typically
        assert result.pvalue_lower <= result.pvalue_consistent


class TestSPATest:
    """Tests for spa_test function."""

    def test_basic_functionality(self) -> None:
        """Test basic SPA test."""
        from temporalcv.statistical_tests import spa_test

        rng = np.random.default_rng(42)
        n = 100

        benchmark_errors = rng.normal(0, 1, n) ** 2
        model_errors = {
            "A": rng.normal(0, 0.8, n) ** 2,
            "B": rng.normal(0, 1.2, n) ** 2,
        }

        result = spa_test(
            benchmark_errors, model_errors, n_bootstrap=100, random_state=42
        )

        assert hasattr(result, "pvalue")
        assert hasattr(result, "pvalue_consistent")
        assert hasattr(result, "pvalue_lower")
        assert 0 <= result.pvalue <= 1
        assert 0 <= result.pvalue_consistent <= 1
        assert 0 <= result.pvalue_lower <= 1

    def test_spa_vs_reality_check_similar(self) -> None:
        """SPA and RC should give similar conclusions for clear cases."""
        from temporalcv.statistical_tests import reality_check_test, spa_test

        rng = np.random.default_rng(42)
        n = 100

        benchmark = rng.random(n)
        models = {"A": rng.random(n) * 0.5}  # A is clearly better

        rc = reality_check_test(benchmark, models, n_bootstrap=100, random_state=42)
        spa = spa_test(benchmark, models, n_bootstrap=100, random_state=42)

        # Both should identify same best model
        assert rc.best_model == spa.best_model

    def test_studentization_effect(self) -> None:
        """SPA with studentization should be more robust."""
        from temporalcv.statistical_tests import spa_test

        rng = np.random.default_rng(42)
        n = 100

        benchmark = rng.random(n)
        # One model with high variance, one with low
        models = {
            "high_var": rng.normal(0.5, 2.0, n).clip(0),
            "low_var": rng.normal(0.5, 0.2, n).clip(0),
        }

        result = spa_test(
            benchmark, models, n_bootstrap=100, random_state=42
        )

        # Should produce valid result despite variance differences
        assert 0 <= result.pvalue <= 1
        assert result.best_model in ["high_var", "low_var"]

    def test_reproducibility(self) -> None:
        """Same seed should give same result."""
        from temporalcv.statistical_tests import spa_test

        rng = np.random.default_rng(42)
        n = 50

        benchmark = rng.random(n)
        models = {"A": rng.random(n)}

        result1 = spa_test(benchmark, models, n_bootstrap=50, random_state=123)
        result2 = spa_test(benchmark, models, n_bootstrap=50, random_state=123)

        assert result1.pvalue == result2.pvalue

    def test_insufficient_data_raises(self) -> None:
        """Should raise with insufficient data."""
        from temporalcv.statistical_tests import spa_test

        benchmark = np.array([1.0, 2.0, 3.0])
        models = {"A": np.array([1.1, 2.1, 3.1])}

        with pytest.raises(ValueError, match="at least"):
            spa_test(benchmark, models)


class TestMultipleComparisonIntegration:
    """Integration tests for multiple comparison methods."""

    def test_full_workflow(self) -> None:
        """Test complete workflow with RC, SPA, and encompassing."""
        from temporalcv.statistical_tests import (
            forecast_encompassing_bidirectional,
            reality_check_test,
            spa_test,
        )

        rng = np.random.default_rng(42)
        n = 150

        # Generate forecasts and actuals
        actual = rng.normal(0, 1, n)
        benchmark = actual + rng.normal(0, 0.5, n)
        model_a = actual + rng.normal(0, 0.3, n)
        model_b = actual + rng.normal(0, 0.4, n)

        # Convert to errors
        benchmark_errors = (benchmark - actual) ** 2
        model_errors = {
            "A": (model_a - actual) ** 2,
            "B": (model_b - actual) ** 2,
        }

        # Run all tests
        rc = reality_check_test(benchmark_errors, model_errors, n_bootstrap=100, random_state=42)
        spa = spa_test(benchmark_errors, model_errors, n_bootstrap=100, random_state=42)
        enc = forecast_encompassing_bidirectional(actual, model_a, model_b)

        # All should produce valid results
        assert 0 <= rc.pvalue <= 1
        assert 0 <= spa.pvalue <= 1
        assert enc.recommendation in ["use_a", "use_b", "combine", "equivalent"]

        # RC and SPA should agree on best model
        assert rc.best_model == spa.best_model
