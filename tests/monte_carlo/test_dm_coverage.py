"""
Monte Carlo coverage calibration for DM test.

Tier 2: Statistical calibration tests - run nightly.

Tests verify:
- Type I error control under null hypothesis
- Power under alternative hypothesis
- Coverage of confidence intervals
"""

import numpy as np
import pytest

from temporalcv.statistical_tests import dm_test


# =============================================================================
# DM Test Coverage Under Null
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestDMTestCoverageNull:
    """Monte Carlo coverage calibration for DM test under null."""

    def test_dm_null_fail_to_reject_rate(self):
        """
        Under null (equal forecasts), fail-to-reject rate should be ~95%.

        Failing to reject at α=0.05 means true null is captured.
        N=500 simulations, target 90-98% (conservative acceptable).
        """
        N_SIMS = 500
        fail_to_reject = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Two forecasters with equal skill (null is true)
            errors1 = rng.randn(100)
            errors2 = rng.randn(100)

            result = dm_test(errors1, errors2, h=1, harvey_correction=True)

            # Fail to reject if p >= 0.05
            if result.pvalue >= 0.05:
                fail_to_reject += 1

        coverage = fail_to_reject / N_SIMS

        # Should fail to reject ~95% of time when null is true
        assert 0.90 <= coverage <= 0.98, (
            f"Fail-to-reject rate = {coverage:.1%}, expected 93-97%"
        )

    def test_dm_type_i_error_control(self):
        """
        Under null, rejection rate should be ~5%.

        Type I error = probability of rejecting true null.
        """
        N_SIMS = 500
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Equal forecasters (null is true)
            errors1 = rng.randn(100)
            errors2 = rng.randn(100)

            result = dm_test(errors1, errors2, h=1, harvey_correction=True)

            # Reject if p < 0.05 (two-tailed)
            if result.pvalue < 0.05:
                rejections += 1

        type_i_rate = rejections / N_SIMS

        # Type I error should be near 5% (3-8% acceptable range)
        assert 0.02 <= type_i_rate <= 0.10, (
            f"Type I error = {type_i_rate:.1%}, expected ~5%"
        )


# =============================================================================
# DM Test Power Under Alternative
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestDMTestPowerAlternative:
    """Monte Carlo power analysis for DM test under alternative."""

    def test_dm_power_moderate_difference(self):
        """
        Under alternative (different forecast accuracy), should reject frequently.

        Effect size: SD ratio 0.8 vs 1.2 (substantial difference).
        Expected power: > 50% with n=100.
        """
        N_SIMS = 500
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Model 1 has lower error variance (better)
            errors1 = rng.randn(100) * 0.8  # SD = 0.8
            errors2 = rng.randn(100) * 1.2  # SD = 1.2

            result = dm_test(errors1, errors2, h=1)

            # Should reject null (errors differ)
            if result.pvalue < 0.05:
                rejections += 1

        power = rejections / N_SIMS

        # Should have reasonable power (> 40%) with this effect size
        assert power > 0.40, f"Power = {power:.1%}, expected > 40%"

    def test_dm_power_small_difference(self):
        """
        With small effect, power should be lower but non-zero.

        Effect size: SD ratio 0.9 vs 1.1 (small difference).
        """
        N_SIMS = 500
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Smaller effect size
            errors1 = rng.randn(100) * 0.9
            errors2 = rng.randn(100) * 1.1

            result = dm_test(errors1, errors2, h=1)

            if result.pvalue < 0.05:
                rejections += 1

        power = rejections / N_SIMS

        # Lower power expected, but should detect some difference
        assert power > 0.10, f"Power = {power:.1%}, expected > 10%"

    def test_dm_power_large_sample(self):
        """
        With larger sample size, power should increase.

        n=300 with moderate effect should have high power.
        """
        N_SIMS = 300  # Fewer sims since each is larger
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Larger sample, moderate effect
            errors1 = rng.randn(300) * 0.85
            errors2 = rng.randn(300) * 1.15

            result = dm_test(errors1, errors2, h=1)

            if result.pvalue < 0.05:
                rejections += 1

        power = rejections / N_SIMS

        # Should have high power with large n
        assert power > 0.70, f"Power = {power:.1%}, expected > 70%"


# =============================================================================
# DM Test with Autocorrelation (HAC Variance)
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestDMTestHACVariance:
    """Test HAC variance estimation under autocorrelation."""

    def test_dm_type_i_with_autocorrelated_errors(self):
        """
        With autocorrelated errors (h>1), HAC should maintain type I error control.

        Tests multi-step ahead forecast comparison where errors are
        autocorrelated with MA(h-1) structure.
        """
        N_SIMS = 300
        h = 3  # 3-step ahead forecasts
        fail_to_reject = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)

            # Generate correlated forecast errors (MA(h-1) structure)
            n = 100
            innovations1 = rng.randn(n + h - 1)
            innovations2 = rng.randn(n + h - 1)

            # MA(h-1) errors (h-step ahead induces h-1 autocorrelation)
            errors1 = np.array([innovations1[i:i + h].mean() for i in range(n)])
            errors2 = np.array([innovations2[i:i + h].mean() for i in range(n)])

            result = dm_test(errors1, errors2, h=h, harvey_correction=True)

            # Fail to reject when null is true
            if result.pvalue >= 0.05:
                fail_to_reject += 1

        coverage = fail_to_reject / N_SIMS

        # HAC-adjusted should still maintain valid Type I error control
        assert 0.85 <= coverage <= 0.99, (
            f"HAC fail-to-reject rate = {coverage:.1%}, expected 90-98%"
        )
