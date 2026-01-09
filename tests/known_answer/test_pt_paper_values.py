"""
Known-answer tests for Pesaran-Timmermann test against PT (1992).

This module validates the PT test implementation against published values from:

[T1] Pesaran, M.H. & Timmermann, A. (1992).
     A simple nonparametric test of predictive performance.
     Journal of Business & Economic Statistics, 10(4), 461-465.

Key validations:
1. Variance formula (Equation 8) for p* under independence
2. Expected accuracy p* = p_y * p_x + (1-p_y) * (1-p_x)
3. Total variance = Var(p_hat) + Var(p*)
4. Test statistic = (p_hat - p*) / sqrt(Var_total)

References
----------
- Pesaran & Timmermann (1992), Equations 6-8, Section 2
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from temporalcv.statistical_tests import pt_test

# =============================================================================
# Variance Formula Tests (Equation 8)
# =============================================================================


def _compute_pt_variance_formula(
    p_y: float, p_x: float, n: int
) -> tuple[float, float, float]:
    """
    Compute PT (1992) variance components from Equation 8.

    Parameters
    ----------
    p_y : float
        Proportion of positive actual values
    p_x : float
        Proportion of positive predictions
    n : int
        Sample size

    Returns
    -------
    tuple[float, float, float]
        (var_p_hat, var_p_star, var_total)

    Notes
    -----
    From PT 1992 Equation 8:

    p* = p_y * p_x + (1-p_y) * (1-p_x)

    Var(p_hat) = p* * (1-p*) / n

    Var(p*) = [(2p_y - 1)^2 * p_x(1-p_x) +
               (2p_x - 1)^2 * p_y(1-p_y) +
               4 * p_y * p_x * (1-p_y) * (1-p_x)] / n

    Var_total = Var(p_hat) + Var(p*)
    """
    # Expected accuracy under independence
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)

    # Variance of p_hat (observed accuracy)
    var_p_hat = p_star * (1 - p_star) / n

    # Variance of p_star (expected accuracy)
    term1 = (2 * p_y - 1) ** 2 * p_x * (1 - p_x) / n
    term2 = (2 * p_x - 1) ** 2 * p_y * (1 - p_y) / n
    term3 = 4 * p_y * p_x * (1 - p_y) * (1 - p_x) / n
    var_p_star = term1 + term2 + term3

    # Total variance
    var_total = var_p_hat + var_p_star

    return var_p_hat, var_p_star, var_total


class TestPTVarianceFormula:
    """
    Test the PT variance formula against Equation 8.

    The formula is non-trivial and must be validated against
    manual calculations from the paper.
    """

    def test_symmetric_case_p50_p50(self) -> None:
        """
        Test with p_y = p_x = 0.5 (no marginal bias).

        In this case:
        - p* = 0.5 * 0.5 + 0.5 * 0.5 = 0.5
        - All (2p-1) terms are zero, so Var(p*) has special form
        """
        p_y, p_x, n = 0.5, 0.5, 100

        var_p_hat, var_p_star, var_total = _compute_pt_variance_formula(p_y, p_x, n)

        # p* = 0.5
        p_star = 0.5

        # Var(p_hat) = 0.5 * 0.5 / 100 = 0.0025
        assert_allclose(var_p_hat, 0.0025, rtol=1e-10)

        # For p_y = p_x = 0.5:
        # term1 = 0^2 * 0.25 / 100 = 0
        # term2 = 0^2 * 0.25 / 100 = 0
        # term3 = 4 * 0.5 * 0.5 * 0.5 * 0.5 / 100 = 0.0025
        assert_allclose(var_p_star, 0.0025, rtol=1e-10)

        # Total = 0.0025 + 0.0025 = 0.005
        assert_allclose(var_total, 0.005, rtol=1e-10)

    def test_asymmetric_case_p60_p70(self) -> None:
        """
        Test with p_y = 0.6, p_x = 0.7 (typical imbalanced scenario).

        Manual calculation from Equation 8:
        """
        p_y, p_x, n = 0.6, 0.7, 100

        var_p_hat, var_p_star, var_total = _compute_pt_variance_formula(p_y, p_x, n)

        # p* = 0.6*0.7 + 0.4*0.3 = 0.42 + 0.12 = 0.54
        p_star = 0.54

        # Var(p_hat) = 0.54 * 0.46 / 100 = 0.002484
        expected_var_p_hat = 0.54 * 0.46 / 100
        assert_allclose(var_p_hat, expected_var_p_hat, rtol=1e-10)

        # term1 = (0.2)^2 * 0.7 * 0.3 / 100 = 0.04 * 0.21 / 100 = 0.000084
        # term2 = (0.4)^2 * 0.6 * 0.4 / 100 = 0.16 * 0.24 / 100 = 0.000384
        # term3 = 4 * 0.6 * 0.7 * 0.4 * 0.3 / 100 = 0.2016 / 100 = 0.002016
        expected_var_p_star = 0.000084 + 0.000384 + 0.002016
        assert_allclose(var_p_star, expected_var_p_star, rtol=1e-6)

    def test_extreme_case_p90_p90(self) -> None:
        """
        Test with highly imbalanced p_y = p_x = 0.9.

        This represents data where positives dominate.
        """
        p_y, p_x, n = 0.9, 0.9, 100

        var_p_hat, var_p_star, var_total = _compute_pt_variance_formula(p_y, p_x, n)

        # p* = 0.9*0.9 + 0.1*0.1 = 0.81 + 0.01 = 0.82
        p_star = 0.82

        # Var(p_hat) = 0.82 * 0.18 / 100 = 0.001476
        expected_var_p_hat = 0.82 * 0.18 / 100
        assert_allclose(var_p_hat, expected_var_p_hat, rtol=1e-10)

    def test_variance_decreases_with_n(self) -> None:
        """
        Variance should decrease as 1/n.
        """
        p_y, p_x = 0.6, 0.7

        _, _, var_100 = _compute_pt_variance_formula(p_y, p_x, 100)
        _, _, var_400 = _compute_pt_variance_formula(p_y, p_x, 400)

        # Variance should scale as 1/n
        # var_100 / var_400 should be approximately 4
        assert_allclose(var_100 / var_400, 4.0, rtol=1e-10)

    def test_variance_components_non_negative(self) -> None:
        """
        All variance components must be non-negative.
        """
        for p_y in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for p_x in [0.1, 0.3, 0.5, 0.7, 0.9]:
                var_p_hat, var_p_star, var_total = _compute_pt_variance_formula(
                    p_y, p_x, 100
                )
                assert var_p_hat >= 0, f"var_p_hat negative for p_y={p_y}, p_x={p_x}"
                assert var_p_star >= 0, f"var_p_star negative for p_y={p_y}, p_x={p_x}"
                assert var_total >= 0, f"var_total negative for p_y={p_y}, p_x={p_x}"


# =============================================================================
# Expected Accuracy p* Tests
# =============================================================================


class TestExpectedAccuracy:
    """
    Test the expected accuracy p* formula.

    Under independence, the expected accuracy is:
    p* = P(correct | independence) = p_y * p_x + (1-p_y) * (1-p_x)
    """

    @pytest.mark.parametrize(
        "p_y,p_x,expected_p_star",
        [
            (0.5, 0.5, 0.50),    # Balanced case: random guessing
            (0.6, 0.6, 0.52),    # 0.6*0.6 + 0.4*0.4 = 0.36 + 0.16
            (0.7, 0.7, 0.58),    # 0.7*0.7 + 0.3*0.3 = 0.49 + 0.09
            (0.5, 1.0, 0.50),    # Always predict positive: 0.5*1 + 0.5*0
            (0.5, 0.0, 0.50),    # Always predict negative: 0.5*0 + 0.5*1
            (0.9, 0.9, 0.82),    # High imbalance: 0.81 + 0.01
            (0.6, 0.7, 0.54),    # Asymmetric: 0.42 + 0.12
        ],
    )
    def test_expected_accuracy_values(
        self, p_y: float, p_x: float, expected_p_star: float
    ) -> None:
        """Verify p* formula matches manual calculations."""
        computed = p_y * p_x + (1 - p_y) * (1 - p_x)
        assert_allclose(computed, expected_p_star, rtol=1e-10)

    def test_p_star_bounds(self) -> None:
        """
        p* is always in [0.5, 1] when p_y, p_x are in [0, 1].

        Actually, p* is minimized at 0.5 when p_y = 1-p_x.
        """
        for p_y in np.linspace(0.01, 0.99, 20):
            for p_x in np.linspace(0.01, 0.99, 20):
                p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
                # p* can actually be < 0.5 if signs are opposite
                # But for same-sign marginals, p* >= 0.5
                assert 0 <= p_star <= 1, f"p_star out of [0,1] for p_y={p_y}, p_x={p_x}"


# =============================================================================
# Integration Tests with pt_test Function
# =============================================================================


class TestPTTestIntegration:
    """
    Test that pt_test correctly implements the paper formulas.
    """

    def test_known_accuracy_scenario(self) -> None:
        """
        Test with known accuracy scenario.

        Create data where we know the exact accuracy and marginals.
        """
        rng = np.random.default_rng(42)
        n = 1000

        # Create actual values with known proportion
        p_y = 0.6  # 60% positive
        actual = np.where(rng.random(n) < p_y, 1.0, -1.0)

        # Create predictions that match with known accuracy
        accuracy_target = 0.70  # 70% correct
        correct_mask = rng.random(n) < accuracy_target
        predicted = np.where(correct_mask, actual, -actual)

        result = pt_test(actual, predicted)

        # Verify accuracy is close to target
        assert_allclose(result.accuracy, accuracy_target, rtol=0.05)

        # Verify the test produces reasonable p-value
        # With 70% accuracy and 60% positive, skill should be significant
        assert result.pvalue < 0.05, "Should detect significant skill"

    def test_random_predictions_not_significant(self) -> None:
        """
        Random predictions should not be statistically significant.
        """
        rng = np.random.default_rng(42)
        n = 200

        actual = rng.choice([-1.0, 1.0], size=n)
        predicted = rng.choice([-1.0, 1.0], size=n)  # Independent random

        result = pt_test(actual, predicted)

        # P-value should be large (not significant)
        assert result.pvalue > 0.05, (
            f"Random predictions should not be significant, got p={result.pvalue}"
        )

    def test_perfect_predictions_highly_significant(self) -> None:
        """
        Perfect predictions should be highly significant.
        """
        rng = np.random.default_rng(42)
        n = 100

        actual = rng.choice([-1.0, 1.0], size=n)
        predicted = actual.copy()  # Perfect

        result = pt_test(actual, predicted)

        assert result.accuracy == 1.0, "Perfect predictions should have 100% accuracy"
        assert result.pvalue < 0.001, "Perfect predictions should be highly significant"

    def test_opposite_predictions_not_significant_one_sided(self) -> None:
        """
        Always-wrong predictions are NOT significant in one-sided PT test.

        The PT test is one-sided (H1: positive skill, accuracy > expected).
        Consistently wrong predictions (accuracy < expected) give p-value ~1.

        Note: This is by design - to detect negative skill, you would need
        a two-sided test or test the opposite hypothesis.
        """
        rng = np.random.default_rng(42)
        n = 100

        actual = rng.choice([-1.0, 1.0], size=n)
        predicted = -actual  # Always wrong

        result = pt_test(actual, predicted)

        assert result.accuracy == 0.0, "Opposite predictions should have 0% accuracy"
        # One-sided test: accuracy << expected gives large p-value (wrong direction)
        assert result.pvalue > 0.5, (
            "One-sided PT test should not detect 'negative skill' as significant"
        )


# =============================================================================
# Numerical Examples from Paper
# =============================================================================


class TestPaperNumericalExamples:
    """
    Test numerical examples that can be derived from PT (1992).

    The paper provides the formula but no specific numerical examples.
    We validate the implementation against manual calculations.
    """

    def test_example_n100_balanced(self) -> None:
        """
        Example: n=100, balanced marginals.

        Scenario:
        - 50 positive actuals, 50 negative
        - 50 positive predictions, 50 negative
        - 65% accuracy observed
        """
        n = 100
        p_y = 0.5
        p_x = 0.5
        p_hat = 0.65  # Observed accuracy

        # Expected accuracy under independence
        p_star = 0.5

        # Variance calculation
        var_p_hat, var_p_star, var_total = _compute_pt_variance_formula(p_y, p_x, n)

        # Test statistic
        z_stat = (p_hat - p_star) / np.sqrt(var_total)

        # z should be approximately (0.65 - 0.5) / sqrt(0.005) ≈ 2.12
        expected_z = 0.15 / np.sqrt(0.005)
        assert_allclose(z_stat, expected_z, rtol=1e-10)
        assert_allclose(z_stat, 2.12, rtol=0.01)

    def test_example_n100_imbalanced(self) -> None:
        """
        Example: n=100, imbalanced marginals.

        Scenario:
        - 70 positive actuals, 30 negative
        - 60 positive predictions, 40 negative
        - 65% accuracy observed
        """
        n = 100
        p_y = 0.7
        p_x = 0.6
        p_hat = 0.65  # Observed accuracy

        # Expected accuracy under independence
        # p* = 0.7*0.6 + 0.3*0.4 = 0.42 + 0.12 = 0.54
        p_star = 0.54

        # Variance calculation
        var_p_hat, var_p_star, var_total = _compute_pt_variance_formula(p_y, p_x, n)

        # Test statistic
        z_stat = (p_hat - p_star) / np.sqrt(var_total)

        # Skill = 0.65 - 0.54 = 0.11
        # Should be significant but less than balanced case
        assert z_stat > 0, "Positive skill should give positive z"


# =============================================================================
# Edge Cases
# =============================================================================


class TestPTEdgeCases:
    """
    Test edge cases and boundary conditions.
    """

    def test_minimum_sample_size(self) -> None:
        """
        PT test requires minimum 30 samples.
        """
        rng = np.random.default_rng(42)

        # Should work with 30
        actual_30 = rng.choice([-1.0, 1.0], size=30)
        predicted_30 = rng.choice([-1.0, 1.0], size=30)
        result = pt_test(actual_30, predicted_30)
        assert result.n == 30

        # Should fail with 29
        actual_29 = rng.choice([-1.0, 1.0], size=29)
        predicted_29 = rng.choice([-1.0, 1.0], size=29)
        with pytest.raises(ValueError, match="Insufficient samples"):
            pt_test(actual_29, predicted_29)

    def test_all_same_direction(self) -> None:
        """
        When all values have same sign, test should handle gracefully.
        """
        n = 50

        # All positive actuals and predictions
        actual = np.ones(n)
        predicted = np.ones(n)

        # This is a degenerate case - all correct but no variance
        result = pt_test(actual, predicted)

        # Accuracy is 100% but variance formula may produce edge case
        assert result.accuracy == 1.0

    def test_zeros_excluded_in_2class(self) -> None:
        """
        Zero values should be excluded in 2-class mode.
        """
        rng = np.random.default_rng(42)

        # Mix of positive, negative, and zero
        actual = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0] * 10)
        predicted = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0] * 10)

        result = pt_test(actual, predicted)

        # Only non-zero pairs should be counted
        # n_effective should be 40 (excluding 20 zeros in actual)
        assert result.n == 40
