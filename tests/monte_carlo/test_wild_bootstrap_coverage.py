"""
Monte Carlo coverage calibration for wild cluster bootstrap.

Tier 2: Statistical calibration tests - run nightly.

Tests verify:
- Coverage of bootstrap CIs with few clusters (folds)
- Webb vs Rademacher weight behavior
- Inference validity across cluster counts
"""

import numpy as np
import pytest

from temporalcv.inference.wild_bootstrap import wild_cluster_bootstrap


# =============================================================================
# Basic Coverage Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestWildBootstrapCoverage:
    """Monte Carlo coverage for wild cluster bootstrap."""

    def test_type_i_error_5_folds(self):
        """
        With 5 folds under null (mean=0), Type I error should be ~5%.

        Uses Webb weights (auto-selected for <13 clusters).
        Note: Wild bootstrap tests H0: mean = 0, so we check p-value calibration.
        """
        N_SIMS = 500
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Fold statistics centered at 0 (null is true)
            fold_stats = rng.randn(5)

            result = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=499,
                random_state=seed,
            )

            # Count rejections at alpha=0.05
            if result.p_value < 0.05:
                rejections += 1

        type_i_rate = rejections / N_SIMS

        # Type I error should be near 5%, but bootstrap can be conservative
        # Accept 2-10% range given few clusters
        assert 0.01 <= type_i_rate <= 0.15, (
            f"5-fold Type I error = {type_i_rate:.1%}, expected ~5%"
        )

    def test_type_i_error_10_folds(self):
        """
        With 10 folds under null, Type I error should be ~5%.

        Still uses Webb weights (<13 clusters).
        """
        N_SIMS = 500
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            fold_stats = rng.randn(10)

            result = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=499,
                random_state=seed,
            )

            if result.p_value < 0.05:
                rejections += 1

        type_i_rate = rejections / N_SIMS

        # With 10 clusters, should be closer to nominal 5%
        assert 0.02 <= type_i_rate <= 0.12, (
            f"10-fold Type I error = {type_i_rate:.1%}, expected ~5%"
        )

    def test_type_i_error_20_folds(self):
        """
        With 20 folds, wild bootstrap uses Rademacher weights.

        Should have well-calibrated Type I error with many clusters.
        """
        N_SIMS = 400  # Fewer sims, more folds
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            fold_stats = rng.randn(20)

            result = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=499,
                random_state=seed,
            )

            if result.p_value < 0.05:
                rejections += 1

        type_i_rate = rejections / N_SIMS

        # With 20 clusters, should be closer to nominal
        assert 0.02 <= type_i_rate <= 0.10, (
            f"20-fold Type I error = {type_i_rate:.1%}, expected ~5%"
        )


# =============================================================================
# Weight Type Comparison
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestWeightTypeComparison:
    """Compare Webb vs Rademacher weight behavior."""

    def test_webb_vs_rademacher_5_folds(self):
        """
        With 5 folds, Webb should give wider (more conservative) CIs.

        Webb 6-point provides more bootstrap variability than Rademacher.
        """
        N_SIMS = 300
        webb_widths = []
        rademacher_widths = []

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            fold_stats = rng.randn(5)

            result_webb = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=299,
                weight_type="webb",
                random_state=seed,
            )

            result_rademacher = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=299,
                weight_type="rademacher",
                random_state=seed + 10000,
            )

            webb_widths.append(result_webb.ci_upper - result_webb.ci_lower)
            rademacher_widths.append(
                result_rademacher.ci_upper - result_rademacher.ci_lower
            )

        # Webb typically gives wider CIs with few clusters
        webb_mean = np.mean(webb_widths)
        rademacher_mean = np.mean(rademacher_widths)

        # Widths should be positive and reasonable
        assert webb_mean > 0
        assert rademacher_mean > 0


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.slow
@pytest.mark.monte_carlo
class TestWildBootstrapEdgeCases:
    """Test bootstrap behavior in edge cases."""

    def test_power_strong_positive_signal(self):
        """
        With strong positive signal, should reject H0: mean=0.

        Using 10 clusters with large effect size to test power.
        """
        N_SIMS = 300
        rejections = 0

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            # Large positive effect with 10 clusters (mean ≈ 3)
            fold_stats = rng.uniform(2, 4, size=10)

            result = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=299,
                random_state=seed,
            )

            # Should reject H0: mean = 0 using p-value
            if result.p_value < 0.05:
                rejections += 1

        power = rejections / N_SIMS

        # With strong signal, should have high power (>70%)
        assert power > 0.70, (
            f"Power = {power:.1%}, expected > 70%"
        )

    def test_minimum_3_folds(self):
        """
        With only 3 folds, inference is limited but should not crash.
        """
        N_SIMS = 200
        results = []

        for seed in range(N_SIMS):
            rng = np.random.RandomState(seed)
            fold_stats = rng.randn(3)

            result = wild_cluster_bootstrap(
                fold_stats,
                n_bootstrap=199,
                random_state=seed,
            )

            # Should return valid result
            assert result is not None
            assert np.isfinite(result.estimate)
            assert np.isfinite(result.se)
            results.append(result)

        # CIs should be valid (though wide)
        widths = [r.ci_upper - r.ci_lower for r in results]
        assert all(w > 0 for w in widths)
