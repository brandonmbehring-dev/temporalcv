"""
Tests for influence diagnostics module.

Tests both observation-level and block-level influence computation.

Knowledge Tier: [T2] - Influence functions with HAC adjustment
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.diagnostics import (
    InfluenceDiagnostic,
    compute_dm_influence,
)


class TestInfluenceDiagnosticsBasic:
    """Basic functionality tests for influence diagnostics."""

    def test_basic_influence_computation(self) -> None:
        """
        Basic influence computation should work without errors.
        """
        rng = np.random.default_rng(42)
        n = 100
        errors1 = rng.standard_normal(n)
        errors2 = rng.standard_normal(n)

        result = compute_dm_influence(errors1, errors2)

        assert isinstance(result, InfluenceDiagnostic)
        assert len(result.observation_influence) == n
        assert len(result.observation_high_mask) == n
        assert len(result.block_influence) > 0
        assert len(result.block_indices) == len(result.block_influence)

    def test_outlier_detection_observation_level(self) -> None:
        """
        Single large outlier should be detected at observation level.
        """
        rng = np.random.default_rng(42)
        n = 100
        errors1 = rng.standard_normal(n) * 0.1
        errors2 = rng.standard_normal(n) * 0.1

        # Add large outlier at position 50
        errors1[50] = 10.0  # Huge outlier

        result = compute_dm_influence(errors1, errors2)

        # Observation 50 should have high influence
        assert result.observation_high_mask[50], (
            "Outlier observation should be flagged as high influence"
        )
        assert result.n_high_influence_obs >= 1

    def test_outlier_detection_block_level(self) -> None:
        """
        Block containing outlier should have high influence.
        """
        rng = np.random.default_rng(42)
        n = 100
        errors1 = rng.standard_normal(n) * 0.1
        errors2 = rng.standard_normal(n) * 0.1

        # Add outlier
        errors1[50] = 10.0

        result = compute_dm_influence(errors1, errors2, h=1)

        # At least one block should have high influence
        assert result.n_high_influence_blocks >= 1

    def test_no_outliers_few_flags(self) -> None:
        """
        Random normal errors should have few high-influence points.
        """
        rng = np.random.default_rng(42)
        n = 100
        errors1 = rng.standard_normal(n)
        errors2 = rng.standard_normal(n)

        result = compute_dm_influence(errors1, errors2)

        # With threshold=2σ, expect ~5% flagged at most
        flag_rate = result.n_high_influence_obs / n
        assert flag_rate < 0.15, (
            f"Too many flags ({flag_rate:.1%}) for random normal errors"
        )


class TestInfluenceHorizonParameter:
    """Tests for horizon (h) parameter effects."""

    def test_h1_creates_single_obs_blocks(self) -> None:
        """With h=1, blocks should be single observations."""
        rng = np.random.default_rng(42)
        n = 50
        errors1 = rng.standard_normal(n)
        errors2 = rng.standard_normal(n)

        result = compute_dm_influence(errors1, errors2, h=1)

        # With h=1, should have n blocks of size 1
        assert len(result.block_indices) == n
        for start, end in result.block_indices:
            assert end - start == 1

    def test_h4_creates_larger_blocks(self) -> None:
        """With h=4, blocks should be size 4."""
        rng = np.random.default_rng(42)
        n = 100
        errors1 = rng.standard_normal(n)
        errors2 = rng.standard_normal(n)

        result = compute_dm_influence(errors1, errors2, h=4)

        # With h=4, blocks should be size 4
        expected_n_blocks = n // 4
        assert len(result.block_indices) == expected_n_blocks

        for start, end in result.block_indices[:-1]:  # All but last
            assert end - start == 4


class TestInfluenceLossFunction:
    """Tests for different loss functions."""

    def test_squared_loss(self) -> None:
        """Squared loss should use squared errors."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(50)
        errors2 = rng.standard_normal(50)

        result = compute_dm_influence(errors1, errors2, loss="squared")

        # Should complete without error
        assert isinstance(result, InfluenceDiagnostic)

    def test_absolute_loss(self) -> None:
        """Absolute loss should use absolute errors."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(50)
        errors2 = rng.standard_normal(50)

        result = compute_dm_influence(errors1, errors2, loss="absolute")

        # Should complete without error
        assert isinstance(result, InfluenceDiagnostic)

    def test_different_losses_different_results(self) -> None:
        """Different loss functions may give different influence patterns."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(100)
        errors2 = rng.standard_normal(100)

        result_sq = compute_dm_influence(errors1, errors2, loss="squared")
        result_abs = compute_dm_influence(errors1, errors2, loss="absolute")

        # Results should differ (not exactly equal)
        # This is a weak test - just checks they're computed differently
        assert not np.allclose(
            result_sq.observation_influence,
            result_abs.observation_influence
        )


class TestInfluenceThreshold:
    """Tests for influence threshold parameter."""

    def test_higher_threshold_fewer_flags(self) -> None:
        """Higher threshold should flag fewer observations."""
        rng = np.random.default_rng(42)
        n = 100
        errors1 = rng.standard_normal(n)
        errors2 = rng.standard_normal(n)

        result_2 = compute_dm_influence(errors1, errors2, influence_threshold=2.0)
        result_3 = compute_dm_influence(errors1, errors2, influence_threshold=3.0)

        assert result_3.n_high_influence_obs <= result_2.n_high_influence_obs
        assert result_3.n_high_influence_blocks <= result_2.n_high_influence_blocks

    def test_threshold_stored_in_result(self) -> None:
        """Threshold value should be stored in result."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(50)
        errors2 = rng.standard_normal(50)

        result = compute_dm_influence(errors1, errors2, influence_threshold=2.5)

        assert result.influence_threshold == 2.5


class TestInfluenceEdgeCases:
    """Edge case tests for influence diagnostics."""

    def test_minimum_sample_size(self) -> None:
        """Should require minimum sample size."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(5)
        errors2 = rng.standard_normal(5)

        with pytest.raises(ValueError, match="at least 10"):
            compute_dm_influence(errors1, errors2)

    def test_mismatched_array_lengths(self) -> None:
        """Should reject arrays of different lengths."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(50)
        errors2 = rng.standard_normal(60)

        with pytest.raises(ValueError, match="same length"):
            compute_dm_influence(errors1, errors2)

    def test_identical_errors(self) -> None:
        """Identical errors should handle gracefully."""
        n = 50
        errors = np.ones(n)

        result = compute_dm_influence(errors, errors)

        # All loss differentials are zero - should handle
        assert isinstance(result, InfluenceDiagnostic)

    def test_perfect_model_vs_baseline(self) -> None:
        """
        Perfect model (errors1=0) vs baseline should work.
        """
        rng = np.random.default_rng(42)
        n = 50
        errors1 = np.zeros(n)  # Perfect predictions
        errors2 = rng.standard_normal(n)

        result = compute_dm_influence(errors1, errors2)

        # Should complete without error
        assert isinstance(result, InfluenceDiagnostic)


class TestInfluenceDataclassProperties:
    """Tests for InfluenceDiagnostic dataclass."""

    def test_frozen_dataclass(self) -> None:
        """InfluenceDiagnostic should be frozen (intent, not enforcement)."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(50)
        errors2 = rng.standard_normal(50)

        result = compute_dm_influence(errors1, errors2)

        # Should raise FrozenInstanceError on attribute assignment
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.influence_threshold = 5.0

    def test_block_indices_format(self) -> None:
        """Block indices should be (start, end) tuples."""
        rng = np.random.default_rng(42)
        errors1 = rng.standard_normal(50)
        errors2 = rng.standard_normal(50)

        result = compute_dm_influence(errors1, errors2, h=5)

        for start, end in result.block_indices:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end
            assert start >= 0
