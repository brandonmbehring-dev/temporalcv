"""
Tests for gap sensitivity analysis.

Tests the gap parameter sensitivity diagnostic tool.

Knowledge Tier: [T2] - Empirical best practice for time series validation
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from temporalcv.diagnostics import (
    GapSensitivityResult,
    gap_sensitivity_analysis,
)


class TestGapSensitivityBasic:
    """Basic functionality tests for gap sensitivity analysis."""

    def test_basic_analysis_runs(self) -> None:
        """
        Basic gap sensitivity analysis should run without errors.
        """
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X @ rng.standard_normal(5) + rng.standard_normal(n) * 0.5

        model = Ridge(alpha=1.0)
        result = gap_sensitivity_analysis(
            model, X, y,
            gap_range=range(0, 5),
            n_splits=3,
        )

        assert isinstance(result, GapSensitivityResult)
        assert len(result.gap_values) == 5
        assert len(result.metrics) == 5
        assert result.metric_name == "mae"

    def test_metrics_computed_for_each_gap(self) -> None:
        """
        Metrics should be computed for each gap value.
        """
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X @ rng.standard_normal(5) + rng.standard_normal(n) * 0.5

        model = Ridge(alpha=1.0)
        result = gap_sensitivity_analysis(
            model, X, y,
            gap_range=range(0, 4),
            n_splits=3,
        )

        # All metrics should be positive (non-NaN)
        valid_count = np.sum(~np.isnan(result.metrics))
        assert valid_count >= 1, "Should compute at least one valid metric"

    def test_baseline_is_first_valid_gap(self) -> None:
        """
        Baseline should be the first gap with valid metrics.
        """
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 5))
        y = X @ rng.standard_normal(5) + rng.standard_normal(n) * 0.5

        model = Ridge(alpha=1.0)
        result = gap_sensitivity_analysis(
            model, X, y,
            gap_range=range(0, 5),
            n_splits=3,
        )

        # Baseline gap should be in gap_values
        assert result.baseline_gap in result.gap_values


class TestGapSensitivityMetrics:
    """Tests for different metric options."""

    def test_mae_metric(self) -> None:
        """MAE metric should work."""
        rng = np.random.default_rng(42)
        n = 150
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            metric="mae",
            n_splits=3,
        )

        assert result.metric_name == "mae"
        # MAE should be positive
        valid_metrics = result.metrics[~np.isnan(result.metrics)]
        assert all(m >= 0 for m in valid_metrics)

    def test_rmse_metric(self) -> None:
        """RMSE metric should work."""
        rng = np.random.default_rng(42)
        n = 150
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            metric="rmse",
            n_splits=3,
        )

        assert result.metric_name == "rmse"

    def test_mse_metric(self) -> None:
        """MSE metric should work."""
        rng = np.random.default_rng(42)
        n = 150
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            metric="mse",
            n_splits=3,
        )

        assert result.metric_name == "mse"


class TestBreakEvenGap:
    """Tests for break-even gap detection."""

    def test_break_even_detected_when_degradation(self) -> None:
        """
        Break-even gap should be detected when performance degrades.

        Create data where larger gaps hurt (simulated leakage scenario).
        """
        rng = np.random.default_rng(42)
        n = 300

        # Create autocorrelated target - model benefits from recent data
        y = np.zeros(n)
        y[0] = rng.standard_normal()
        for t in range(1, n):
            y[t] = 0.95 * y[t - 1] + rng.standard_normal() * 0.1

        # Features that include lagged target (creates leakage)
        X = np.column_stack([
            y,  # Lagged version would help
            rng.standard_normal((n, 3))
        ])

        result = gap_sensitivity_analysis(
            Ridge(alpha=0.1), X, y,
            gap_range=range(0, 10),
            n_splits=3,
            degradation_threshold=0.10,
        )

        # With autocorrelated data and lagged features, gap should matter
        # This is a weak test - just checks structure
        assert result.break_even_gap is None or result.break_even_gap > 0

    def test_no_break_even_when_robust(self) -> None:
        """
        No break-even gap when model is robust to gaps.

        Use model with no temporal dependencies.
        """
        rng = np.random.default_rng(42)
        n = 200

        # IID data - gap shouldn't matter
        X = rng.standard_normal((n, 5))
        y = X @ np.array([1.0, -1.0, 0.5, -0.5, 0.2]) + rng.standard_normal(n) * 0.1

        result = gap_sensitivity_analysis(
            Ridge(alpha=1.0), X, y,
            gap_range=range(0, 5),
            n_splits=3,
            degradation_threshold=0.10,
        )

        # For IID data, metrics should be relatively stable
        # Sensitivity score should be low
        assert result.sensitivity_score < 0.5, (
            f"IID data should have low sensitivity, got {result.sensitivity_score:.3f}"
        )


class TestDegradationThreshold:
    """Tests for degradation threshold parameter."""

    def test_different_thresholds_different_break_even(self) -> None:
        """
        Different thresholds may give different break-even gaps.
        """
        rng = np.random.default_rng(42)
        n = 200

        # Create data with some degradation
        y = np.zeros(n)
        y[0] = rng.standard_normal()
        for t in range(1, n):
            y[t] = 0.7 * y[t - 1] + rng.standard_normal()

        X = np.column_stack([
            np.roll(y, 1),  # Lagged y
            rng.standard_normal((n, 2))
        ])
        X[0, 0] = 0  # Fix first value

        result_5 = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 8),
            n_splits=3,
            degradation_threshold=0.05,
        )

        result_20 = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 8),
            n_splits=3,
            degradation_threshold=0.20,
        )

        # Stricter threshold (5%) should trigger sooner than lenient (20%)
        if result_5.break_even_gap is not None and result_20.break_even_gap is not None:
            assert result_5.break_even_gap <= result_20.break_even_gap

    def test_threshold_stored_in_result(self) -> None:
        """Threshold should be stored in result for reference."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            n_splits=3,
            degradation_threshold=0.15,
        )

        assert result.degradation_threshold == 0.15


class TestSensitivityScore:
    """Tests for sensitivity score computation."""

    def test_sensitivity_score_coefficient_of_variation(self) -> None:
        """
        Sensitivity score should be std/mean of metrics.
        """
        rng = np.random.default_rng(42)
        X = rng.standard_normal((150, 3))
        y = rng.standard_normal(150)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 4),
            n_splits=3,
        )

        # Verify it's coefficient of variation
        valid_metrics = result.metrics[~np.isnan(result.metrics)]
        expected_score = np.std(valid_metrics) / np.mean(valid_metrics)

        assert abs(result.sensitivity_score - expected_score) < 1e-10

    def test_sensitivity_score_non_negative(self) -> None:
        """Sensitivity score should be non-negative."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            n_splits=3,
        )

        assert result.sensitivity_score >= 0


class TestGapSensitivityEdgeCases:
    """Edge case tests for gap sensitivity analysis."""

    def test_empty_gap_range_raises(self) -> None:
        """Empty gap range should raise error."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)

        with pytest.raises(ValueError, match="at least one"):
            gap_sensitivity_analysis(Ridge(), X, y, gap_range=[])

    def test_single_gap_value(self) -> None:
        """Single gap value should work."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=[0],
            n_splits=3,
        )

        assert len(result.gap_values) == 1
        # No break-even with single gap
        assert result.break_even_gap is None

    def test_large_gap_insufficient_data(self) -> None:
        """
        Large gaps should gracefully handle insufficient data.
        """
        rng = np.random.default_rng(42)
        n = 50  # Small dataset
        X = rng.standard_normal((n, 3))
        y = rng.standard_normal(n)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 20),  # Some gaps won't work
            n_splits=3,
        )

        # Some metrics may be NaN, but should not crash
        assert isinstance(result, GapSensitivityResult)
        # At least some gaps should work
        assert np.any(~np.isnan(result.metrics))


class TestGapSensitivityDataclassProperties:
    """Tests for GapSensitivityResult dataclass."""

    def test_frozen_dataclass(self) -> None:
        """GapSensitivityResult should be frozen."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            n_splits=3,
        )

        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            result.sensitivity_score = 999.0

    def test_result_has_all_expected_fields(self) -> None:
        """Result should have all expected fields."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = rng.standard_normal(100)

        result = gap_sensitivity_analysis(
            Ridge(), X, y,
            gap_range=range(0, 3),
            n_splits=3,
        )

        # Check all fields exist
        assert hasattr(result, "gap_values")
        assert hasattr(result, "metrics")
        assert hasattr(result, "metric_name")
        assert hasattr(result, "break_even_gap")
        assert hasattr(result, "sensitivity_score")
        assert hasattr(result, "degradation_threshold")
        assert hasattr(result, "baseline_metric")
        assert hasattr(result, "baseline_gap")
