"""
Validation tests for residual diagnostics gate.

Tests the Ljung-Box, Jarque-Bera, and mean-zero diagnostic tests
that check residual quality after model fitting.

Knowledge Tier: [T1] - Based on standard statistical test methodology
References:
- Ljung & Box (1978). Biometrika 65(2), 297-303.
- Jarque & Bera (1987). International Statistical Review 55(2).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from temporalcv.gates import (
    GateStatus,
    gate_residual_diagnostics,
)


class TestResidualDiagnosticsBasic:
    """Basic functionality tests for residual diagnostics gate."""

    def test_white_noise_passes(self) -> None:
        """
        White noise residuals should PASS all tests.

        This is the baseline case: well-specified model with
        normally distributed, uncorrelated residuals.
        """
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(100)

        result = gate_residual_diagnostics(residuals)

        assert result.status == GateStatus.PASS, (
            f"White noise should pass, got {result.status}"
        )
        assert result.details["failing_tests"] == []

    def test_autocorrelated_residuals_detected(self) -> None:
        """
        AR(1) residuals should trigger Ljung-Box failure.

        If a model leaves temporal structure in residuals,
        the Ljung-Box test should detect it.
        """
        rng = np.random.default_rng(42)
        n = 200
        phi = 0.8  # Strong autocorrelation

        # Generate AR(1) residuals
        residuals = np.zeros(n)
        residuals[0] = rng.standard_normal()
        for t in range(1, n):
            residuals[t] = phi * residuals[t - 1] + rng.standard_normal()

        result = gate_residual_diagnostics(residuals, max_lag=10)

        # Should detect autocorrelation
        assert "ljung_box" in result.details["failing_tests"], (
            "Ljung-Box should detect AR(1) residuals"
        )

    def test_biased_residuals_detected(self) -> None:
        """
        Residuals with non-zero mean should trigger mean-zero failure.

        This indicates systematic prediction bias.
        """
        rng = np.random.default_rng(42)
        n = 100
        bias = 2.0  # Significant bias

        residuals = rng.standard_normal(n) + bias

        result = gate_residual_diagnostics(residuals)

        # Should detect bias and HALT (mean_zero triggers HALT)
        assert result.status == GateStatus.HALT
        assert "mean_zero" in result.details["failing_tests"]
        assert "bias" in result.message.lower()

    def test_non_normal_residuals_detected(self) -> None:
        """
        Heavy-tailed residuals should trigger Jarque-Bera failure.
        """
        rng = np.random.default_rng(42)
        n = 200

        # Generate heavy-tailed residuals (t-distribution)
        residuals = stats.t.rvs(df=3, size=n, random_state=42)

        result = gate_residual_diagnostics(residuals)

        # Should detect non-normality (WARN by default)
        assert "jarque_bera" in result.details["failing_tests"], (
            "Jarque-Bera should detect heavy-tailed residuals"
        )

    def test_insufficient_data_skips(self) -> None:
        """Gate should SKIP with insufficient data."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(20)  # Too few samples

        result = gate_residual_diagnostics(residuals)

        assert result.status == GateStatus.SKIP
        assert "n_samples" in result.details
        assert result.details["n_samples"] == 20


class TestResidualDiagnosticsHaltBehavior:
    """Tests for HALT vs WARN behavior with configuration flags."""

    def test_halt_on_autocorr_flag(self) -> None:
        """
        halt_on_autocorr=True should HALT on Ljung-Box failure.
        """
        rng = np.random.default_rng(42)
        n = 200
        phi = 0.8

        # Generate AR(1) residuals
        residuals = np.zeros(n)
        residuals[0] = rng.standard_normal()
        for t in range(1, n):
            residuals[t] = phi * residuals[t - 1] + rng.standard_normal()

        # Default: WARN on autocorrelation
        result_default = gate_residual_diagnostics(residuals)
        assert result_default.status == GateStatus.WARN

        # With flag: HALT on autocorrelation
        result_halt = gate_residual_diagnostics(
            residuals, halt_on_autocorr=True
        )
        assert result_halt.status == GateStatus.HALT
        assert "autocorrelation" in result_halt.message

    def test_halt_on_normality_flag(self) -> None:
        """
        halt_on_normality=True should HALT on Jarque-Bera failure.
        """
        # Generate heavily skewed residuals
        rng = np.random.default_rng(42)
        residuals = stats.t.rvs(df=3, size=200, random_state=42)

        # Default: WARN on non-normality
        result_default = gate_residual_diagnostics(residuals)
        assert "jarque_bera" in result_default.details["failing_tests"]

        # With flag: HALT on non-normality
        result_halt = gate_residual_diagnostics(
            residuals, halt_on_normality=True
        )
        assert result_halt.status == GateStatus.HALT
        assert "non-normality" in result_halt.message

    def test_bias_always_halts(self) -> None:
        """
        Mean-zero test failure should always trigger HALT.

        Biased predictions are a fundamental problem.
        """
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(100) + 2.0  # Add bias

        result = gate_residual_diagnostics(residuals)

        # Should HALT regardless of other flags
        assert result.status == GateStatus.HALT
        assert "bias" in result.message.lower()


class TestLjungBoxImplementation:
    """Tests for custom Ljung-Box implementation correctness."""

    def test_ljung_box_known_answer(self) -> None:
        """
        Test Ljung-Box Q statistic against hand-calculated value.

        For a simple series where we can compute ACF manually.
        """
        # Simple series with known autocorrelation
        # x = [1, -1, 1, -1, 1, -1, ...]  has ACF(1) = -1 for even length
        n = 50
        x = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])

        # ACF(1) for this series should be close to -1
        # Q = n(n+2) * sum(rho_k^2 / (n-k))
        # With max_lag=1 and rho_1 ≈ -1:
        # Q ≈ 50 * 52 * (1 / 49) ≈ 53.06

        from temporalcv.gates import _ljung_box_test

        Q, p = _ljung_box_test(x, max_lag=1)

        # Q should be large (significant autocorrelation)
        assert Q > 40, f"Expected Q > 40 for alternating series, got {Q}"
        assert p < 0.001, "Should be highly significant"

    def test_ljung_box_white_noise_calibration(self) -> None:
        """
        Ljung-Box on white noise should not reject too often.

        Under H₀, p-values should be uniform(0,1).
        With alpha=0.05, we expect ~5% rejection rate.
        """
        from temporalcv.gates import _ljung_box_test

        rng = np.random.default_rng(42)
        n_sims = 200
        n = 100
        max_lag = 10
        alpha = 0.05

        rejections = 0
        for _ in range(n_sims):
            x = rng.standard_normal(n)
            _, p = _ljung_box_test(x, max_lag)
            if p < alpha:
                rejections += 1

        rejection_rate = rejections / n_sims

        # Should be around 5% ± reasonable tolerance
        # Using 2-15% as acceptable range
        assert 0.02 < rejection_rate < 0.15, (
            f"Rejection rate {rejection_rate:.2%} outside expected range for white noise"
        )


class TestResidualDiagnosticsEdgeCases:
    """Edge case tests for residual diagnostics gate."""

    def test_constant_residuals(self) -> None:
        """Constant residuals should handle gracefully."""
        residuals = np.ones(100)  # All same value

        result = gate_residual_diagnostics(residuals)

        # Should complete without error
        # Constant residuals have zero variance, which may affect tests
        assert result.status in (
            GateStatus.PASS,
            GateStatus.WARN,
            GateStatus.HALT,
        )

    def test_max_lag_clamped(self) -> None:
        """max_lag should be clamped to n/3."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(60)

        # Request 30 lags, but should be clamped to 60/3 = 20
        result = gate_residual_diagnostics(residuals, max_lag=30)

        actual_lag = result.details["tests"]["ljung_box"]["max_lag"]
        assert actual_lag == 20, (
            f"max_lag should be clamped to 20, got {actual_lag}"
        )

    def test_very_small_valid_sample(self) -> None:
        """Test with minimum valid sample size (30)."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(30)

        result = gate_residual_diagnostics(residuals)

        assert result.status != GateStatus.SKIP
        assert result.details["n_samples"] == 30

    def test_significance_parameter(self) -> None:
        """Custom significance level should be respected."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(100)

        result = gate_residual_diagnostics(residuals, significance=0.01)

        assert result.details["significance"] == 0.01

    def test_result_details_structure(self) -> None:
        """Result details should have expected structure."""
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(100)

        result = gate_residual_diagnostics(residuals)

        # Check structure
        assert "n_samples" in result.details
        assert "significance" in result.details
        assert "tests" in result.details
        assert "failing_tests" in result.details

        # Check test results
        tests = result.details["tests"]
        assert "ljung_box" in tests
        assert "jarque_bera" in tests
        assert "mean_zero" in tests

        # Check each test has expected fields
        for test_name, test_result in tests.items():
            assert "statistic" in test_result, f"{test_name} missing statistic"
            assert "p_value" in test_result, f"{test_name} missing p_value"
            assert "significant" in test_result, f"{test_name} missing significant"
