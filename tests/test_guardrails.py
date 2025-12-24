"""
Tests for unified guardrails suite.

Tests for:
- Individual guardrail functions
- run_all_guardrails() composite validation
- GuardrailResult dataclass
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv import (
    GuardrailResult,
    check_forecast_horizon_consistency,
    check_minimum_sample_size,
    check_residual_autocorrelation,
    check_stratified_sample_size,
    check_suspicious_improvement,
    run_all_guardrails,
)


# =============================================================================
# GuardrailResult Tests
# =============================================================================


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""

    def test_bool_true_when_passed(self) -> None:
        """Result should be truthy when passed=True."""
        result = GuardrailResult(passed=True)
        assert result
        assert bool(result) is True

    def test_bool_false_when_failed(self) -> None:
        """Result should be falsy when passed=False."""
        result = GuardrailResult(passed=False)
        assert not result
        assert bool(result) is False

    def test_summary_includes_status(self) -> None:
        """Summary should include pass/fail status."""
        passed_result = GuardrailResult(passed=True)
        assert "PASSED" in passed_result.summary()

        failed_result = GuardrailResult(passed=False)
        assert "FAILED" in failed_result.summary()

    def test_summary_includes_errors(self) -> None:
        """Summary should include error messages."""
        result = GuardrailResult(
            passed=False,
            errors=["Error 1", "Error 2"],
        )
        summary = result.summary()
        assert "Error 1" in summary
        assert "Error 2" in summary

    def test_summary_includes_warnings(self) -> None:
        """Summary should include warning messages."""
        result = GuardrailResult(
            passed=True,
            warnings=["Warning 1"],
        )
        summary = result.summary()
        assert "Warning 1" in summary

    def test_summary_includes_recommendations(self) -> None:
        """Summary should include recommendations."""
        result = GuardrailResult(
            passed=False,
            errors=["Some error"],
            recommendations=["Try this fix"],
        )
        summary = result.summary()
        assert "Try this fix" in summary


# =============================================================================
# check_suspicious_improvement Tests
# =============================================================================


class TestCheckSuspiciousImprovement:
    """Tests for check_suspicious_improvement."""

    def test_passes_for_small_improvement(self) -> None:
        """Should pass for improvement < threshold."""
        result = check_suspicious_improvement(
            model_metric=0.18,
            baseline_metric=0.20,
            threshold=0.20,
        )
        assert result.passed
        assert len(result.errors) == 0

    def test_fails_for_large_improvement(self) -> None:
        """Should fail for improvement > threshold."""
        result = check_suspicious_improvement(
            model_metric=0.14,
            baseline_metric=0.20,
            threshold=0.20,
        )
        assert not result.passed
        assert len(result.errors) > 0

    def test_warns_near_threshold(self) -> None:
        """Should warn when improvement approaches threshold."""
        # 16% improvement (> 15% which is 75% of 20%)
        result = check_suspicious_improvement(
            model_metric=0.168,
            baseline_metric=0.20,
            threshold=0.20,
        )
        assert result.passed
        assert len(result.warnings) > 0

    def test_skips_for_zero_baseline(self) -> None:
        """Should skip when baseline <= 0."""
        result = check_suspicious_improvement(
            model_metric=0.1,
            baseline_metric=0.0,
        )
        assert result.passed
        assert len(result.skipped) > 0

    def test_custom_threshold(self) -> None:
        """Should respect custom threshold."""
        # 25% improvement should fail with 20% threshold
        result = check_suspicious_improvement(
            model_metric=0.75,
            baseline_metric=1.0,
            threshold=0.20,
        )
        assert not result.passed

        # But pass with 30% threshold
        result = check_suspicious_improvement(
            model_metric=0.75,
            baseline_metric=1.0,
            threshold=0.30,
        )
        assert result.passed

    def test_includes_recommendations_on_failure(self) -> None:
        """Failed check should include recommendations."""
        result = check_suspicious_improvement(
            model_metric=0.10,
            baseline_metric=0.20,
        )
        assert not result.passed
        assert len(result.recommendations) > 0


# =============================================================================
# check_minimum_sample_size Tests
# =============================================================================


class TestCheckMinimumSampleSize:
    """Tests for check_minimum_sample_size."""

    def test_passes_for_sufficient_samples(self) -> None:
        """Should pass when n >= min_n."""
        result = check_minimum_sample_size(n=100, min_n=50)
        assert result.passed
        assert len(result.errors) == 0

    def test_fails_for_insufficient_samples(self) -> None:
        """Should fail when n < min_n."""
        result = check_minimum_sample_size(n=20, min_n=50)
        assert not result.passed
        assert len(result.errors) > 0

    def test_warns_for_marginal_samples(self) -> None:
        """Should warn when n < 2*min_n."""
        result = check_minimum_sample_size(n=75, min_n=50)
        assert result.passed
        assert len(result.warnings) > 0

    def test_no_warning_for_large_samples(self) -> None:
        """Should not warn when n >= 2*min_n."""
        result = check_minimum_sample_size(n=150, min_n=50)
        assert result.passed
        assert len(result.warnings) == 0


# =============================================================================
# check_stratified_sample_size Tests
# =============================================================================


class TestCheckStratifiedSampleSize:
    """Tests for check_stratified_sample_size."""

    def test_passes_for_sufficient_strata(self) -> None:
        """Should pass when both strata >= min_n."""
        result = check_stratified_sample_size(n_up=30, n_down=25, min_n=10)
        assert result.passed
        assert len(result.errors) == 0

    def test_fails_for_insufficient_up(self) -> None:
        """Should fail when n_up < min_n."""
        result = check_stratified_sample_size(n_up=5, n_down=25, min_n=10)
        assert not result.passed
        assert any("UP" in e for e in result.errors)

    def test_fails_for_insufficient_down(self) -> None:
        """Should fail when n_down < min_n."""
        result = check_stratified_sample_size(n_up=25, n_down=5, min_n=10)
        assert not result.passed
        assert any("DOWN" in e for e in result.errors)

    def test_fails_for_both_insufficient(self) -> None:
        """Should fail when both strata < min_n."""
        result = check_stratified_sample_size(n_up=5, n_down=3, min_n=10)
        assert not result.passed
        assert len(result.errors) == 2

    def test_warns_for_imbalanced_strata(self) -> None:
        """Should warn when ratio > 3:1."""
        result = check_stratified_sample_size(n_up=40, n_down=10, min_n=10)
        assert result.passed
        assert len(result.warnings) > 0
        assert "imbalanced" in result.warnings[0].lower()


# =============================================================================
# check_forecast_horizon_consistency Tests
# =============================================================================


class TestCheckForecastHorizonConsistency:
    """Tests for check_forecast_horizon_consistency."""

    def test_passes_for_consistent_horizons(self) -> None:
        """Should pass when horizons are consistent."""
        result = check_forecast_horizon_consistency(
            horizons=[0.10, 0.12, 0.15],  # Gradual increase
        )
        assert result.passed

    def test_fails_for_h1_much_better(self) -> None:
        """Should fail when h=1 is dramatically better than others."""
        result = check_forecast_horizon_consistency(
            horizons=[0.10, 0.30, 0.35],  # h=1 is 3x better
            max_ratio=2.0,
        )
        assert not result.passed

    def test_skips_for_single_horizon(self) -> None:
        """Should skip when only one horizon provided."""
        result = check_forecast_horizon_consistency(horizons=[0.10])
        assert result.passed
        assert len(result.skipped) > 0

    def test_skips_for_zero_h1(self) -> None:
        """Should skip when h=1 metric is zero."""
        result = check_forecast_horizon_consistency(horizons=[0.0, 0.2])
        assert result.passed
        assert len(result.skipped) > 0

    def test_custom_max_ratio(self) -> None:
        """Should respect custom max_ratio."""
        horizons = [0.10, 0.25]  # ratio = 2.5

        # Should fail with max_ratio=2.0
        result = check_forecast_horizon_consistency(horizons=horizons, max_ratio=2.0)
        assert not result.passed

        # Should pass with max_ratio=3.0
        result = check_forecast_horizon_consistency(horizons=horizons, max_ratio=3.0)
        assert result.passed


# =============================================================================
# check_residual_autocorrelation Tests
# =============================================================================


class TestCheckResidualAutocorrelation:
    """Tests for check_residual_autocorrelation."""

    def test_passes_for_white_noise(self) -> None:
        """Should pass for uncorrelated residuals."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 200)  # White noise

        result = check_residual_autocorrelation(residuals=residuals)
        assert result.passed
        # May or may not have warnings depending on random realization

    def test_warns_for_autocorrelated(self) -> None:
        """Should warn for autocorrelated residuals."""
        # Create AR(1) residuals with high autocorrelation
        rng = np.random.default_rng(42)
        n = 200
        residuals = np.zeros(n)
        residuals[0] = rng.normal()
        for t in range(1, n):
            residuals[t] = 0.7 * residuals[t - 1] + rng.normal() * 0.5

        result = check_residual_autocorrelation(residuals=residuals)
        # Should still pass (autocorrelation is warning only)
        assert result.passed
        # But should have warnings
        assert len(result.warnings) > 0

    def test_skips_for_short_series(self) -> None:
        """Should skip when series is too short."""
        residuals = np.array([1.0, 2.0, 1.5])  # Only 3 points
        result = check_residual_autocorrelation(residuals=residuals, max_lag=5)
        assert result.passed
        assert len(result.skipped) > 0


# =============================================================================
# run_all_guardrails Tests
# =============================================================================


class TestRunAllGuardrails:
    """Tests for run_all_guardrails composite validation."""

    def test_passes_all_checks(self) -> None:
        """Should pass when all checks pass."""
        result = run_all_guardrails(
            model_metric=0.18,
            baseline_metric=0.20,
            n_samples=100,
        )
        assert result.passed

    def test_fails_on_suspicious_improvement(self) -> None:
        """Should fail if improvement is suspicious."""
        result = run_all_guardrails(
            model_metric=0.10,
            baseline_metric=0.20,  # 50% improvement!
            n_samples=100,
        )
        assert not result.passed
        assert len(result.errors) > 0

    def test_fails_on_insufficient_samples(self) -> None:
        """Should fail if sample size is insufficient."""
        result = run_all_guardrails(
            model_metric=0.18,
            baseline_metric=0.20,
            n_samples=20,  # Below default 50
        )
        assert not result.passed

    def test_includes_stratified_check(self) -> None:
        """Should run stratified check when n_up/n_down provided."""
        result = run_all_guardrails(
            model_metric=0.18,
            baseline_metric=0.20,
            n_samples=100,
            n_up=5,  # Insufficient
            n_down=30,
        )
        assert not result.passed
        assert any("UP" in e for e in result.errors)

    def test_includes_horizon_check(self) -> None:
        """Should run horizon check when horizon_metrics provided."""
        result = run_all_guardrails(
            model_metric=0.10,
            baseline_metric=0.12,  # OK improvement
            n_samples=100,
            horizon_metrics=[0.10, 0.35, 0.40],  # Bad consistency
        )
        assert not result.passed

    def test_includes_residual_check(self) -> None:
        """Should run residual check when residuals provided."""
        # Create autocorrelated residuals
        rng = np.random.default_rng(42)
        n = 200
        residuals = np.zeros(n)
        residuals[0] = rng.normal()
        for t in range(1, n):
            residuals[t] = 0.7 * residuals[t - 1] + rng.normal() * 0.5

        result = run_all_guardrails(
            model_metric=0.18,
            baseline_metric=0.20,
            n_samples=200,
            residuals=residuals,
        )
        # Still passes (residual check is warning only)
        assert result.passed
        # But has residual details
        assert "residual_autocorrelation" in result.details

    def test_custom_thresholds(self) -> None:
        """Should respect custom thresholds."""
        # 25% improvement fails with default 20%
        result = run_all_guardrails(
            model_metric=0.75,
            baseline_metric=1.0,
            n_samples=100,
            improvement_threshold=0.20,
        )
        assert not result.passed

        # But passes with 30% threshold
        result = run_all_guardrails(
            model_metric=0.75,
            baseline_metric=1.0,
            n_samples=100,
            improvement_threshold=0.30,
        )
        assert result.passed

    def test_aggregates_all_warnings(self) -> None:
        """Should aggregate warnings from all checks."""
        result = run_all_guardrails(
            model_metric=0.163,  # 18.5% improvement - exceeds 15% (75% of 20%)
            baseline_metric=0.20,
            n_samples=75,  # Marginal sample size
        )
        assert result.passed
        # Should have warnings from both checks
        assert len(result.warnings) >= 2

    def test_deduplicates_recommendations(self) -> None:
        """Should not have duplicate recommendations."""
        result = run_all_guardrails(
            model_metric=0.10,
            baseline_metric=0.20,
            n_samples=20,
            n_up=3,
            n_down=3,
        )
        # Multiple failures, but recommendations should be unique
        unique_recs = set(result.recommendations)
        assert len(unique_recs) == len(result.recommendations)

    def test_summary_is_readable(self) -> None:
        """Summary should be human-readable."""
        result = run_all_guardrails(
            model_metric=0.18,
            baseline_metric=0.20,
            n_samples=100,
        )
        summary = result.summary()
        assert "PASSED" in summary or "FAILED" in summary
        assert len(summary) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestGuardrailsIntegration:
    """Integration tests for guardrails with realistic scenarios."""

    def test_honest_model_passes(self) -> None:
        """Realistic honest model should pass all guardrails."""
        result = run_all_guardrails(
            model_metric=0.092,  # 8% improvement over baseline
            baseline_metric=0.10,
            n_samples=250,
            n_up=120,
            n_down=130,
            horizon_metrics=[0.092, 0.105, 0.115],  # Gradual increase
        )
        assert result.passed

    def test_suspicious_model_fails(self) -> None:
        """Model with suspicious performance should fail."""
        result = run_all_guardrails(
            model_metric=0.05,  # 50% improvement - very suspicious
            baseline_metric=0.10,
            n_samples=100,
        )
        assert not result.passed
        assert "leakage" in result.summary().lower() or "improvement" in result.summary().lower()

    def test_underpowered_study_fails(self) -> None:
        """Study with insufficient data should fail."""
        result = run_all_guardrails(
            model_metric=0.092,
            baseline_metric=0.10,
            n_samples=15,  # Way too small
            n_up=5,
            n_down=10,
        )
        assert not result.passed
