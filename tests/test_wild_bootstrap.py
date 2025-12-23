"""
Tests for wild cluster bootstrap inference.

Tests the wild bootstrap implementation for CV fold-level inference.

Knowledge Tier: [T2] - Wild bootstrap is established, testing CV adaptation
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.inference import (
    WildBootstrapResult,
    wild_cluster_bootstrap,
)


class TestWildBootstrapBasic:
    """Basic functionality tests for wild cluster bootstrap."""

    def test_basic_bootstrap_runs(self) -> None:
        """Basic bootstrap should run without errors."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99, random_state=42)

        assert isinstance(result, WildBootstrapResult)
        assert result.n_clusters == 5
        assert result.n_bootstrap == 99
        assert len(result.bootstrap_distribution) == 99

    def test_estimate_is_mean(self) -> None:
        """Original estimate should be mean of fold statistics."""
        fold_stats = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        expected_mean = np.mean(fold_stats)
        assert abs(result.estimate - expected_mean) < 1e-10

    def test_standard_error_positive(self) -> None:
        """Standard error should be positive."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert result.se > 0

    def test_confidence_interval_reasonable(self) -> None:
        """Confidence interval should have reasonable width and bounds."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=999, random_state=42)

        # CI should have reasonable properties
        assert result.ci_lower < result.ci_upper  # Lower < upper
        assert result.ci_upper - result.ci_lower > 0  # Non-zero width
        # CI should be centered roughly around estimate (within 3 SE)
        ci_center = (result.ci_lower + result.ci_upper) / 2
        assert abs(ci_center - result.estimate) < 3 * result.se

    def test_p_value_in_range(self) -> None:
        """P-value should be in [0, 1]."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert 0 <= result.p_value <= 1


class TestWeightSelection:
    """Tests for automatic weight distribution selection."""

    def test_webb_weights_for_few_folds(self) -> None:
        """Webb weights should be used for < 13 folds."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])  # 5 folds
        result = wild_cluster_bootstrap(fold_stats, weight_type="auto", random_state=42)

        assert result.weight_type == "webb"

    def test_rademacher_weights_for_many_folds(self) -> None:
        """Rademacher weights should be used for >= 13 folds."""
        fold_stats = np.random.default_rng(42).standard_normal(15)
        result = wild_cluster_bootstrap(fold_stats, weight_type="auto")

        assert result.weight_type == "rademacher"

    def test_explicit_rademacher(self) -> None:
        """Explicit rademacher should be used regardless of fold count."""
        fold_stats = np.array([0.1, 0.2, 0.15])  # 3 folds
        result = wild_cluster_bootstrap(fold_stats, weight_type="rademacher")

        assert result.weight_type == "rademacher"

    def test_explicit_webb(self) -> None:
        """Explicit webb should be used regardless of fold count."""
        fold_stats = np.random.default_rng(42).standard_normal(20)
        result = wild_cluster_bootstrap(fold_stats, weight_type="webb")

        assert result.weight_type == "webb"


class TestReproducibility:
    """Tests for reproducibility with random_state."""

    def test_same_seed_same_result(self) -> None:
        """Same random_state should produce identical results."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])

        result1 = wild_cluster_bootstrap(fold_stats, n_bootstrap=99, random_state=42)
        result2 = wild_cluster_bootstrap(fold_stats, n_bootstrap=99, random_state=42)

        assert result1.se == result2.se
        assert result1.p_value == result2.p_value
        np.testing.assert_array_equal(
            result1.bootstrap_distribution, result2.bootstrap_distribution
        )

    def test_different_seed_different_result(self) -> None:
        """Different random_state should produce different results."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])

        result1 = wild_cluster_bootstrap(fold_stats, n_bootstrap=99, random_state=42)
        result2 = wild_cluster_bootstrap(fold_stats, n_bootstrap=99, random_state=123)

        assert not np.allclose(
            result1.bootstrap_distribution, result2.bootstrap_distribution
        )


class TestEdgeCases:
    """Edge case tests for wild bootstrap."""

    def test_minimum_folds(self) -> None:
        """Should work with minimum 2 folds."""
        fold_stats = np.array([0.1, 0.2])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert isinstance(result, WildBootstrapResult)
        assert result.n_clusters == 2

    def test_one_fold_raises(self) -> None:
        """Should raise error with only 1 fold."""
        fold_stats = np.array([0.1])

        with pytest.raises(ValueError, match="at least 2"):
            wild_cluster_bootstrap(fold_stats)

    def test_empty_array_raises(self) -> None:
        """Should raise error with empty array."""
        fold_stats = np.array([])

        with pytest.raises(ValueError, match="at least 2"):
            wild_cluster_bootstrap(fold_stats)

    def test_all_positive_statistics(self) -> None:
        """Should handle all positive statistics."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert result.estimate > 0

    def test_all_negative_statistics(self) -> None:
        """Should handle all negative statistics."""
        fold_stats = np.array([-0.1, -0.2, -0.15, -0.12, -0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert result.estimate < 0

    def test_mixed_sign_statistics(self) -> None:
        """Should handle mixed sign statistics."""
        fold_stats = np.array([0.1, -0.2, 0.15, -0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert isinstance(result, WildBootstrapResult)

    def test_zero_estimate(self) -> None:
        """Should handle statistics that average to near-zero."""
        fold_stats = np.array([0.1, -0.1, 0.05, -0.05])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        # P-value should be high for near-zero estimate
        assert result.p_value > 0.5

    def test_few_folds_warning(self) -> None:
        """Should warn for very few folds (< 6)."""
        fold_stats = np.array([0.1, 0.2, 0.15])  # 3 folds

        with pytest.warns(UserWarning, match="Only 3 folds"):
            wild_cluster_bootstrap(fold_stats)


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_n_bootstrap(self) -> None:
        """Should raise error for n_bootstrap < 1."""
        fold_stats = np.array([0.1, 0.2, 0.15])

        with pytest.raises(ValueError, match="n_bootstrap"):
            wild_cluster_bootstrap(fold_stats, n_bootstrap=0)

    def test_invalid_alpha(self) -> None:
        """Should raise error for alpha outside (0, 1)."""
        fold_stats = np.array([0.1, 0.2, 0.15])

        with pytest.raises(ValueError, match="alpha"):
            wild_cluster_bootstrap(fold_stats, alpha=0)

        with pytest.raises(ValueError, match="alpha"):
            wild_cluster_bootstrap(fold_stats, alpha=1)

    def test_invalid_weight_type(self) -> None:
        """Should raise error for invalid weight_type."""
        fold_stats = np.array([0.1, 0.2, 0.15])

        with pytest.raises(ValueError, match="weight_type"):
            wild_cluster_bootstrap(fold_stats, weight_type="invalid")  # type: ignore


class TestDataclassProperties:
    """Tests for WildBootstrapResult dataclass."""

    def test_frozen_dataclass(self) -> None:
        """WildBootstrapResult should be frozen."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            result.se = 999.0

    def test_all_fields_present(self) -> None:
        """Result should have all expected fields."""
        fold_stats = np.array([0.1, 0.2, 0.15, 0.12, 0.18])
        result = wild_cluster_bootstrap(fold_stats, n_bootstrap=99)

        assert hasattr(result, "estimate")
        assert hasattr(result, "se")
        assert hasattr(result, "ci_lower")
        assert hasattr(result, "ci_upper")
        assert hasattr(result, "p_value")
        assert hasattr(result, "n_bootstrap")
        assert hasattr(result, "n_clusters")
        assert hasattr(result, "weight_type")
        assert hasattr(result, "bootstrap_distribution")
