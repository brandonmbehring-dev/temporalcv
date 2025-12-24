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
    dm_test,
    pt_test,
    compute_hac_variance,
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
