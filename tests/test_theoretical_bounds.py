"""
Tests for theoretical bounds validation.

Tests for:
- AR(1) MSE/MAE bounds computation
- AR(2) MSE bounds computation
- Bounds checking gate
- AR series generation
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv import (
    GateStatus,
    check_against_ar1_bounds,
    generate_ar1_series,
    generate_ar2_series,
    theoretical_ar1_mae_bound,
    theoretical_ar1_mse_bound,
    theoretical_ar2_mse_bound,
)


# =============================================================================
# AR(1) MSE Bounds Tests
# =============================================================================


class TestTheoreticalAR1MSEBound:
    """Tests for theoretical_ar1_mse_bound."""

    def test_h1_equals_sigma_sq(self) -> None:
        """For h=1, MSE = σ² (innovation variance is irreducible)."""
        # Any phi value, h=1 should give sigma_sq
        assert theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=1) == pytest.approx(1.0)
        assert theoretical_ar1_mse_bound(phi=0.5, sigma_sq=4.0, h=1) == pytest.approx(4.0)
        assert theoretical_ar1_mse_bound(phi=-0.7, sigma_sq=2.0, h=1) == pytest.approx(2.0)

    def test_phi_zero_gives_h_times_sigma_sq(self) -> None:
        """For φ=0 (white noise), MSE = h·σ²."""
        # White noise: errors are independent, variance adds
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=1) == pytest.approx(1.0)
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=5) == pytest.approx(5.0)
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=2.0, h=10) == pytest.approx(20.0)

    def test_mse_increases_with_horizon(self) -> None:
        """MSE should increase with forecast horizon."""
        mse_h1 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=1)
        mse_h2 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=2)
        mse_h5 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=5)
        mse_h10 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=10)

        assert mse_h1 < mse_h2 < mse_h5 < mse_h10

    def test_mse_converges_to_unconditional_variance(self) -> None:
        """For large h, MSE → Var(y) = σ²/(1-φ²)."""
        phi, sigma_sq = 0.9, 1.0
        unconditional_var = sigma_sq / (1 - phi ** 2)  # 5.263...

        # Large horizon should approach unconditional variance
        mse_h100 = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma_sq, h=100)
        assert mse_h100 == pytest.approx(unconditional_var, rel=1e-3)

    def test_negative_phi(self) -> None:
        """Should work with negative AR coefficient."""
        # φ = -0.8 (oscillating AR(1))
        mse = theoretical_ar1_mse_bound(phi=-0.8, sigma_sq=1.0, h=3)
        assert mse > 1.0  # Must be > h=1 MSE
        assert mse < 10.0  # But bounded

    def test_rejects_nonstationary(self) -> None:
        """Should reject |φ| >= 1."""
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar1_mse_bound(phi=1.0, sigma_sq=1.0, h=1)
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar1_mse_bound(phi=-1.0, sigma_sq=1.0, h=1)
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar1_mse_bound(phi=1.5, sigma_sq=1.0, h=1)

    def test_rejects_nonpositive_sigma_sq(self) -> None:
        """Should reject σ² <= 0."""
        with pytest.raises(ValueError, match="positive"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=0.0, h=1)
        with pytest.raises(ValueError, match="positive"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=-1.0, h=1)

    def test_rejects_invalid_horizon(self) -> None:
        """Should reject h < 1."""
        with pytest.raises(ValueError, match="h must"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=1.0, h=0)
        with pytest.raises(ValueError, match="h must"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=1.0, h=-1)


class TestTheoreticalAR1MAEBound:
    """Tests for theoretical_ar1_mae_bound."""

    def test_h1_equals_sigma_sqrt_2_over_pi(self) -> None:
        """For h=1, MAE = σ·√(2/π) ≈ 0.798σ."""
        expected = np.sqrt(2.0 / np.pi)  # ≈ 0.798

        assert theoretical_ar1_mae_bound(sigma=1.0, phi=0.0, h=1) == pytest.approx(expected)
        assert theoretical_ar1_mae_bound(sigma=2.0, phi=0.0, h=1) == pytest.approx(2.0 * expected)

    def test_mae_less_than_rmse(self) -> None:
        """MAE should be less than RMSE for Gaussian errors."""
        for phi in [0.0, 0.5, 0.9, -0.7]:
            mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=1.0, h=5)
            rmse = np.sqrt(mse)
            mae = theoretical_ar1_mae_bound(sigma=1.0, phi=phi, h=5)
            assert mae < rmse

    def test_mae_ratio_to_rmse(self) -> None:
        """MAE/RMSE should be √(2/π) ≈ 0.798."""
        expected_ratio = np.sqrt(2.0 / np.pi)

        for phi in [0.0, 0.5, 0.9]:
            mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=1.0, h=3)
            rmse = np.sqrt(mse)
            mae = theoretical_ar1_mae_bound(sigma=1.0, phi=phi, h=3)
            assert mae / rmse == pytest.approx(expected_ratio, rel=1e-6)


# =============================================================================
# AR(2) MSE Bounds Tests
# =============================================================================


class TestTheoreticalAR2MSEBound:
    """Tests for theoretical_ar2_mse_bound."""

    def test_h1_equals_sigma_sq(self) -> None:
        """For h=1, MSE = σ² (innovation variance)."""
        assert theoretical_ar2_mse_bound(phi1=0.5, phi2=0.3, sigma_sq=1.0, h=1) == pytest.approx(
            1.0
        )

    def test_ar2_reduces_to_ar1(self) -> None:
        """When φ₂=0, should match AR(1) bounds."""
        phi1 = 0.7
        for h in [1, 2, 5]:
            ar2_mse = theoretical_ar2_mse_bound(phi1=phi1, phi2=0.0, sigma_sq=1.0, h=h)
            ar1_mse = theoretical_ar1_mse_bound(phi=phi1, sigma_sq=1.0, h=h)
            assert ar2_mse == pytest.approx(ar1_mse, rel=1e-6)

    def test_mse_increases_with_horizon(self) -> None:
        """MSE should increase with forecast horizon."""
        mse_h1 = theoretical_ar2_mse_bound(phi1=0.5, phi2=0.3, sigma_sq=1.0, h=1)
        mse_h3 = theoretical_ar2_mse_bound(phi1=0.5, phi2=0.3, sigma_sq=1.0, h=3)
        mse_h5 = theoretical_ar2_mse_bound(phi1=0.5, phi2=0.3, sigma_sq=1.0, h=5)

        assert mse_h1 < mse_h3 < mse_h5

    def test_rejects_nonstationary(self) -> None:
        """Should reject coefficients violating stationarity."""
        # φ₁ + φ₂ >= 1 violates stationarity
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar2_mse_bound(phi1=0.6, phi2=0.6, sigma_sq=1.0, h=1)

        # |φ₂| >= 1 violates stationarity
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar2_mse_bound(phi1=0.5, phi2=1.0, sigma_sq=1.0, h=1)


# =============================================================================
# Bounds Checking Gate Tests
# =============================================================================


class TestCheckAgainstAR1Bounds:
    """Tests for check_against_ar1_bounds gate."""

    def test_halt_when_beating_bounds(self) -> None:
        """HALT when model MSE < theoretical/tolerance."""
        # Theoretical MSE for h=1 is 1.0
        # With tolerance=1.5, threshold is 0.667
        # Model MSE of 0.5 should HALT
        result = check_against_ar1_bounds(model_mse=0.5, phi=0.9, sigma_sq=1.0, h=1)
        assert result.status == GateStatus.HALT
        assert "leakage" in result.message.lower()

    def test_warn_when_close_to_bounds(self) -> None:
        """WARN when model MSE is suspiciously close to theoretical."""
        # Model MSE of 1.0 exactly at theoretical minimum
        result = check_against_ar1_bounds(model_mse=1.0, phi=0.9, sigma_sq=1.0, h=1)
        assert result.status == GateStatus.WARN

    def test_pass_when_within_range(self) -> None:
        """PASS when model MSE is reasonably above theoretical."""
        # Model MSE of 1.5 is 50% above theoretical minimum
        result = check_against_ar1_bounds(model_mse=1.5, phi=0.9, sigma_sq=1.0, h=1)
        assert result.status == GateStatus.PASS

    def test_skip_when_invalid_params(self) -> None:
        """SKIP when AR parameters are invalid."""
        # Non-stationary phi
        result = check_against_ar1_bounds(model_mse=1.0, phi=1.5, sigma_sq=1.0, h=1)
        assert result.status == GateStatus.SKIP

    def test_custom_tolerance(self) -> None:
        """Should respect custom tolerance factor."""
        # With tolerance=2.0, threshold is 0.5
        # Model MSE of 0.6 should PASS (0.6 > 0.5)
        result = check_against_ar1_bounds(
            model_mse=0.6, phi=0.9, sigma_sq=1.0, h=1, tolerance=2.0
        )
        assert result.status != GateStatus.HALT

    def test_details_contain_metadata(self) -> None:
        """Result details should contain useful metadata."""
        result = check_against_ar1_bounds(model_mse=1.5, phi=0.9, sigma_sq=1.0, h=1)

        assert result.details is not None
        assert "model_mse" in result.details
        assert "theoretical_mse" in result.details
        assert "phi" in result.details
        assert "ratio" in result.details

    def test_recommendation_for_halt(self) -> None:
        """HALT result should include actionable recommendation."""
        result = check_against_ar1_bounds(model_mse=0.3, phi=0.9, sigma_sq=1.0, h=1)
        assert result.status == GateStatus.HALT
        assert result.recommendation is not None
        assert len(result.recommendation) > 0


# =============================================================================
# AR Series Generation Tests
# =============================================================================


class TestGenerateAR1Series:
    """Tests for generate_ar1_series."""

    def test_returns_correct_length(self) -> None:
        """Should return series of requested length."""
        for n in [10, 100, 500]:
            series = generate_ar1_series(phi=0.9, sigma=1.0, n=n, random_state=42)
            assert len(series) == n

    def test_reproducible_with_seed(self) -> None:
        """Same seed should give same series."""
        series1 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
        series2 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
        np.testing.assert_array_equal(series1, series2)

    def test_different_seeds_give_different_series(self) -> None:
        """Different seeds should give different series."""
        series1 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
        series2 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=123)
        assert not np.allclose(series1, series2)

    def test_autocorrelation_matches_phi(self) -> None:
        """Generated series should have autocorrelation ≈ φ."""
        phi = 0.8
        series = generate_ar1_series(phi=phi, sigma=1.0, n=10000, random_state=42)

        # Compute lag-1 autocorrelation
        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
        assert autocorr == pytest.approx(phi, abs=0.05)  # Allow sampling variation

    def test_variance_matches_theory(self) -> None:
        """Generated series variance should ≈ σ²/(1-φ²)."""
        phi, sigma = 0.9, 1.0
        theoretical_var = sigma ** 2 / (1 - phi ** 2)

        series = generate_ar1_series(phi=phi, sigma=sigma, n=50000, random_state=42)
        empirical_var = np.var(series)

        assert empirical_var == pytest.approx(theoretical_var, rel=0.1)

    def test_rejects_nonstationary(self) -> None:
        """Should reject |φ| >= 1."""
        with pytest.raises(ValueError, match="stationarity"):
            generate_ar1_series(phi=1.0, sigma=1.0, n=100)

    def test_rejects_invalid_sigma(self) -> None:
        """Should reject σ <= 0."""
        with pytest.raises(ValueError, match="positive"):
            generate_ar1_series(phi=0.5, sigma=0.0, n=100)
        with pytest.raises(ValueError, match="positive"):
            generate_ar1_series(phi=0.5, sigma=-1.0, n=100)


class TestGenerateAR2Series:
    """Tests for generate_ar2_series."""

    def test_returns_correct_length(self) -> None:
        """Should return series of requested length."""
        series = generate_ar2_series(phi1=0.5, phi2=0.3, sigma=1.0, n=100, random_state=42)
        assert len(series) == 100

    def test_reproducible_with_seed(self) -> None:
        """Same seed should give same series."""
        series1 = generate_ar2_series(phi1=0.5, phi2=0.3, sigma=1.0, n=100, random_state=42)
        series2 = generate_ar2_series(phi1=0.5, phi2=0.3, sigma=1.0, n=100, random_state=42)
        np.testing.assert_array_equal(series1, series2)

    def test_rejects_nonstationary(self) -> None:
        """Should reject coefficients violating stationarity."""
        with pytest.raises(ValueError, match="stationarity"):
            generate_ar2_series(phi1=0.7, phi2=0.7, sigma=1.0, n=100)

    def test_reduces_to_ar1_when_phi2_zero(self) -> None:
        """When φ₂=0, variance should match AR(1)."""
        phi1, sigma = 0.7, 1.0
        ar2_series = generate_ar2_series(
            phi1=phi1, phi2=0.0, sigma=sigma, n=50000, random_state=42
        )
        ar1_series = generate_ar1_series(phi=phi1, sigma=sigma, n=50000, random_state=42)

        # Variances should match (both are σ²/(1-φ₁²))
        ar2_var = np.var(ar2_series)
        ar1_var = np.var(ar1_series)
        assert ar2_var == pytest.approx(ar1_var, rel=0.15)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTheoreticalBoundsIntegration:
    """Integration tests verifying bounds match empirical results."""

    def test_ar1_empirical_mse_exceeds_theoretical(self) -> None:
        """Empirical MSE from optimal forecasts should ≈ theoretical MSE."""
        phi, sigma = 0.9, 1.0
        n = 10000

        # Generate AR(1) series
        series = generate_ar1_series(phi=phi, sigma=sigma, n=n, random_state=42)

        # Optimal 1-step forecast: y_hat[t+1] = phi * y[t]
        forecasts = phi * series[:-1]
        actuals = series[1:]

        # Compute empirical MSE
        empirical_mse = np.mean((forecasts - actuals) ** 2)

        # Theoretical MSE
        theoretical_mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma ** 2, h=1)

        # Empirical should be close to theoretical (allowing sampling variation)
        assert empirical_mse == pytest.approx(theoretical_mse, rel=0.1)

    def test_gate_passes_for_honest_model(self) -> None:
        """Gate should PASS for model with realistic performance."""
        phi, sigma = 0.9, 1.0
        n = 5000

        series = generate_ar1_series(phi=phi, sigma=sigma, n=n, random_state=42)

        # Optimal forecast
        forecasts = phi * series[:-1]
        actuals = series[1:]
        model_mse = np.mean((forecasts - actuals) ** 2)

        # Check gate - should not HALT
        result = check_against_ar1_bounds(
            model_mse=model_mse, phi=phi, sigma_sq=sigma ** 2, h=1
        )
        assert result.status != GateStatus.HALT

    def test_gate_halts_for_impossible_performance(self) -> None:
        """Gate should HALT when model has impossibly good MSE."""
        phi, sigma = 0.9, 1.0

        # Pretend model has MSE much better than theoretical minimum
        theoretical_mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma ** 2, h=1)
        impossible_mse = theoretical_mse * 0.3  # 30% of minimum

        result = check_against_ar1_bounds(
            model_mse=impossible_mse, phi=phi, sigma_sq=sigma ** 2, h=1
        )
        assert result.status == GateStatus.HALT
