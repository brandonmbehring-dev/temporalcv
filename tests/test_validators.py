"""
Tests for the validators module.

Tests theoretical bounds validation for AR processes, including:
- AR(1) and AR(2) MSE/MAE bounds
- Gate checks against theoretical limits
- Synthetic series generation
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.gates import GateStatus
from temporalcv.validators import (
    check_against_ar1_bounds,
    generate_ar1_series,
    generate_ar2_series,
    theoretical_ar1_mae_bound,
    theoretical_ar1_mse_bound,
    theoretical_ar2_mse_bound,
)


class TestTheoreticalAR1MSEBound:
    """Tests for theoretical_ar1_mse_bound function."""

    def test_h1_returns_sigma_sq(self) -> None:
        """For h=1, MSE should equal sigma_sq regardless of phi."""
        # [T1] 1-step ahead MSE = σ² (innovation variance is irreducible)
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=1) == 1.0
        assert theoretical_ar1_mse_bound(phi=0.5, sigma_sq=1.0, h=1) == 1.0
        assert theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=1) == 1.0
        assert theoretical_ar1_mse_bound(phi=-0.5, sigma_sq=2.0, h=1) == 2.0

    def test_phi_zero_gives_linear_growth(self) -> None:
        """For phi=0 (white noise), MSE = h * sigma_sq."""
        # [T1] White noise has no predictability, so MSE grows linearly
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=1) == 1.0
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=5) == 5.0
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=10) == 10.0
        assert theoretical_ar1_mse_bound(phi=0.0, sigma_sq=2.0, h=3) == 6.0

    def test_mse_increases_with_horizon(self) -> None:
        """MSE should increase with forecast horizon."""
        mse_1 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=1)
        mse_5 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=5)
        mse_10 = theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=10)

        assert mse_1 < mse_5 < mse_10

    def test_mse_converges_to_unconditional_variance(self) -> None:
        """As h→∞, MSE → σ²/(1-φ²) = Var(y)."""
        phi = 0.9
        sigma_sq = 1.0
        unconditional_var = sigma_sq / (1 - phi ** 2)

        # Large h should approach unconditional variance
        mse_large_h = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma_sq, h=100)
        assert np.isclose(mse_large_h, unconditional_var, rtol=0.01)

    def test_negative_phi(self) -> None:
        """Negative phi should work correctly."""
        # MSE formula should work for negative phi
        mse_pos = theoretical_ar1_mse_bound(phi=0.5, sigma_sq=1.0, h=3)
        mse_neg = theoretical_ar1_mse_bound(phi=-0.5, sigma_sq=1.0, h=3)
        # Due to φ² in formula, both should be equal
        assert np.isclose(mse_pos, mse_neg)

    def test_invalid_phi_raises(self) -> None:
        """Non-stationary phi (|phi| >= 1) should raise ValueError."""
        with pytest.raises(ValueError, match="phi must satisfy"):
            theoretical_ar1_mse_bound(phi=1.0, sigma_sq=1.0)
        with pytest.raises(ValueError, match="phi must satisfy"):
            theoretical_ar1_mse_bound(phi=-1.0, sigma_sq=1.0)
        with pytest.raises(ValueError, match="phi must satisfy"):
            theoretical_ar1_mse_bound(phi=1.5, sigma_sq=1.0)

    def test_invalid_sigma_sq_raises(self) -> None:
        """Non-positive sigma_sq should raise ValueError."""
        with pytest.raises(ValueError, match="sigma_sq must be positive"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=0.0)
        with pytest.raises(ValueError, match="sigma_sq must be positive"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=-1.0)

    def test_invalid_horizon_raises(self) -> None:
        """Horizon h < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="horizon h must be >= 1"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=1.0, h=0)
        with pytest.raises(ValueError, match="horizon h must be >= 1"):
            theoretical_ar1_mse_bound(phi=0.5, sigma_sq=1.0, h=-1)


class TestTheoreticalAR1MAEBound:
    """Tests for theoretical_ar1_mae_bound function."""

    def test_mae_is_sqrt_2_over_pi_times_rmse(self) -> None:
        """MAE = √(2/π) · RMSE for Gaussian errors."""
        # [T1] For N(0, σ²), E[|X|] = σ·√(2/π) ≈ 0.7979σ
        sigma = 1.0
        expected_mae = sigma * np.sqrt(2 / np.pi)
        actual_mae = theoretical_ar1_mae_bound(sigma=sigma, phi=0.0, h=1)
        assert np.isclose(actual_mae, expected_mae, rtol=1e-6)

    def test_mae_scales_with_sigma(self) -> None:
        """MAE should scale linearly with sigma."""
        mae_1 = theoretical_ar1_mae_bound(sigma=1.0, phi=0.5, h=1)
        mae_2 = theoretical_ar1_mae_bound(sigma=2.0, phi=0.5, h=1)
        assert np.isclose(mae_2, 2 * mae_1)

    def test_mae_increases_with_horizon(self) -> None:
        """MAE should increase with forecast horizon."""
        mae_1 = theoretical_ar1_mae_bound(sigma=1.0, phi=0.9, h=1)
        mae_5 = theoretical_ar1_mae_bound(sigma=1.0, phi=0.9, h=5)
        assert mae_1 < mae_5

    def test_consistency_with_mse(self) -> None:
        """MAE should be consistent with MSE via √(2/π) factor."""
        phi, sigma, h = 0.7, 1.5, 3
        mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma ** 2, h=h)
        rmse = np.sqrt(mse)
        expected_mae = rmse * np.sqrt(2 / np.pi)
        actual_mae = theoretical_ar1_mae_bound(sigma=sigma, phi=phi, h=h)
        assert np.isclose(actual_mae, expected_mae, rtol=1e-6)


class TestTheoreticalAR2MSEBound:
    """Tests for theoretical_ar2_mse_bound function."""

    def test_h1_returns_sigma_sq(self) -> None:
        """For h=1, AR(2) MSE should equal sigma_sq."""
        assert theoretical_ar2_mse_bound(phi1=0.5, phi2=0.2, sigma_sq=1.0, h=1) == 1.0
        assert theoretical_ar2_mse_bound(phi1=0.3, phi2=-0.1, sigma_sq=2.0, h=1) == 2.0

    def test_ar1_special_case(self) -> None:
        """AR(2) with phi2=0 should equal AR(1)."""
        phi1, sigma_sq, h = 0.7, 1.0, 5
        ar1_mse = theoretical_ar1_mse_bound(phi=phi1, sigma_sq=sigma_sq, h=h)
        ar2_mse = theoretical_ar2_mse_bound(phi1=phi1, phi2=0.0, sigma_sq=sigma_sq, h=h)
        assert np.isclose(ar1_mse, ar2_mse, rtol=1e-6)

    def test_mse_increases_with_horizon(self) -> None:
        """MSE should increase with forecast horizon."""
        mse_1 = theoretical_ar2_mse_bound(phi1=0.5, phi2=0.2, sigma_sq=1.0, h=1)
        mse_5 = theoretical_ar2_mse_bound(phi1=0.5, phi2=0.2, sigma_sq=1.0, h=5)
        assert mse_1 < mse_5

    def test_stationarity_violation_raises(self) -> None:
        """Non-stationary AR(2) coefficients should raise ValueError."""
        # Violates phi1 + phi2 < 1
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar2_mse_bound(phi1=0.8, phi2=0.5, sigma_sq=1.0)

        # Violates |phi2| < 1
        with pytest.raises(ValueError, match="stationarity"):
            theoretical_ar2_mse_bound(phi1=0.0, phi2=1.0, sigma_sq=1.0)

    def test_invalid_sigma_sq_raises(self) -> None:
        """Non-positive sigma_sq should raise ValueError."""
        with pytest.raises(ValueError, match="sigma_sq must be positive"):
            theoretical_ar2_mse_bound(phi1=0.5, phi2=0.2, sigma_sq=0.0)

    def test_invalid_horizon_raises(self) -> None:
        """Horizon h < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="horizon h must be >= 1"):
            theoretical_ar2_mse_bound(phi1=0.5, phi2=0.2, sigma_sq=1.0, h=0)


class TestCheckAgainstAR1Bounds:
    """Tests for check_against_ar1_bounds gate function."""

    def test_halt_when_beating_bounds(self) -> None:
        """Should HALT when model MSE is below theoretical minimum."""
        # Theoretical MSE = 1.0 for h=1
        # With tolerance=1.5, threshold = 1/1.5 ≈ 0.667
        # MSE=0.5 is 50% of theoretical, which is < 66.7%, so HALT
        result = check_against_ar1_bounds(
            model_mse=0.5, phi=0.9, sigma_sq=1.0, h=1, tolerance=1.5
        )
        assert result.status == GateStatus.HALT
        assert "leakage" in result.message.lower()

    def test_warn_when_suspiciously_close(self) -> None:
        """Should WARN when model MSE is unusually good but not impossible."""
        # MSE = 1.1 is 110% of theoretical (1.0), which is > 66.7% but < 120%
        result = check_against_ar1_bounds(
            model_mse=1.1, phi=0.9, sigma_sq=1.0, h=1, tolerance=1.5
        )
        assert result.status == GateStatus.WARN
        assert "unusually good" in result.message.lower()

    def test_pass_when_within_expected_range(self) -> None:
        """Should PASS when model MSE is within expected range."""
        # MSE = 1.5 is 150% of theoretical, well within expected range
        result = check_against_ar1_bounds(
            model_mse=1.5, phi=0.9, sigma_sq=1.0, h=1, tolerance=1.5
        )
        assert result.status == GateStatus.PASS
        assert "within expected range" in result.message.lower()

    def test_skip_on_invalid_inputs(self) -> None:
        """Should SKIP with message when inputs prevent bound computation."""
        # Non-stationary phi causes ValueError internally
        result = check_against_ar1_bounds(model_mse=1.0, phi=1.5, sigma_sq=1.0)
        assert result.status == GateStatus.SKIP
        assert "cannot compute" in result.message.lower()

    def test_details_contain_expected_fields(self) -> None:
        """Gate result details should contain all diagnostic info."""
        result = check_against_ar1_bounds(
            model_mse=1.5, phi=0.8, sigma_sq=1.0, h=2, tolerance=1.5
        )
        assert result.details is not None
        assert "model_mse" in result.details
        assert "theoretical_mse" in result.details
        assert "phi" in result.details
        assert "sigma_sq" in result.details
        assert "horizon" in result.details
        assert "ratio" in result.details

    def test_metric_name_customization(self) -> None:
        """Metric name should appear in message."""
        result = check_against_ar1_bounds(
            model_mse=1.5, phi=0.9, sigma_sq=1.0, metric_name="RMSE"
        )
        assert "RMSE" in result.message

    def test_tolerance_affects_threshold(self) -> None:
        """Different tolerance values should change HALT threshold."""
        # With tolerance=2.0, threshold = 0.5
        # MSE=0.6 is 60% of theoretical, which is > 50%, so WARN not HALT
        result_strict = check_against_ar1_bounds(
            model_mse=0.6, phi=0.9, sigma_sq=1.0, tolerance=1.0
        )
        result_lenient = check_against_ar1_bounds(
            model_mse=0.6, phi=0.9, sigma_sq=1.0, tolerance=2.0
        )

        # With tolerance=1.0, threshold is 100% so MSE=0.6 halts
        assert result_strict.status == GateStatus.HALT
        # With tolerance=2.0, threshold is 50% so MSE=0.6 doesn't halt
        assert result_lenient.status != GateStatus.HALT


class TestGenerateAR1Series:
    """Tests for generate_ar1_series function."""

    def test_returns_correct_length(self) -> None:
        """Generated series should have requested length."""
        series = generate_ar1_series(phi=0.9, sigma=1.0, n=100)
        assert len(series) == 100

        series = generate_ar1_series(phi=0.5, sigma=1.0, n=50)
        assert len(series) == 50

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical series."""
        series1 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
        series2 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
        np.testing.assert_array_equal(series1, series2)

    def test_different_seeds_produce_different_series(self) -> None:
        """Different seeds should produce different series."""
        series1 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
        series2 = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=43)
        assert not np.allclose(series1, series2)

    def test_autocorrelation_matches_phi(self) -> None:
        """Generated series should have ACF(1) approximately equal to phi."""
        phi = 0.9
        series = generate_ar1_series(phi=phi, sigma=1.0, n=5000, random_state=42)

        # Compute empirical ACF(1)
        centered = series - np.mean(series)
        acf1 = np.corrcoef(centered[:-1], centered[1:])[0, 1]

        # Should be close to phi (allow for sampling variance)
        assert np.isclose(acf1, phi, atol=0.05)

    def test_variance_matches_theory(self) -> None:
        """Generated series variance should match theoretical σ²/(1-φ²)."""
        phi, sigma = 0.8, 1.0
        theoretical_var = sigma ** 2 / (1 - phi ** 2)

        series = generate_ar1_series(phi=phi, sigma=sigma, n=10000, random_state=42)
        empirical_var = np.var(series)

        # Allow 10% tolerance for finite sample
        assert np.isclose(empirical_var, theoretical_var, rtol=0.1)

    def test_invalid_phi_raises(self) -> None:
        """Non-stationary phi should raise ValueError."""
        with pytest.raises(ValueError, match="phi must satisfy"):
            generate_ar1_series(phi=1.0, sigma=1.0, n=100)

    def test_invalid_sigma_raises(self) -> None:
        """Non-positive sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            generate_ar1_series(phi=0.5, sigma=0.0, n=100)
        with pytest.raises(ValueError, match="sigma must be positive"):
            generate_ar1_series(phi=0.5, sigma=-1.0, n=100)

    def test_invalid_n_raises(self) -> None:
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            generate_ar1_series(phi=0.5, sigma=1.0, n=0)


class TestGenerateAR2Series:
    """Tests for generate_ar2_series function."""

    def test_returns_correct_length(self) -> None:
        """Generated series should have requested length."""
        series = generate_ar2_series(phi1=0.5, phi2=0.2, sigma=1.0, n=100)
        assert len(series) == 100

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical series."""
        series1 = generate_ar2_series(
            phi1=0.5, phi2=0.2, sigma=1.0, n=100, random_state=42
        )
        series2 = generate_ar2_series(
            phi1=0.5, phi2=0.2, sigma=1.0, n=100, random_state=42
        )
        np.testing.assert_array_equal(series1, series2)

    def test_ar1_special_case(self) -> None:
        """AR(2) with phi2=0 should behave like AR(1)."""
        # Generate both
        ar1 = generate_ar1_series(phi=0.7, sigma=1.0, n=5000, random_state=42)
        ar2 = generate_ar2_series(phi1=0.7, phi2=0.0, sigma=1.0, n=5000, random_state=42)

        # Won't be identical (different initialization), but ACF should match
        acf1_ar1 = np.corrcoef(ar1[:-1], ar1[1:])[0, 1]
        acf1_ar2 = np.corrcoef(ar2[:-1], ar2[1:])[0, 1]

        assert np.isclose(acf1_ar1, acf1_ar2, atol=0.1)

    def test_stationarity_violation_raises(self) -> None:
        """Non-stationary coefficients should raise ValueError."""
        with pytest.raises(ValueError, match="stationarity"):
            generate_ar2_series(phi1=0.8, phi2=0.5, sigma=1.0, n=100)

    def test_invalid_sigma_raises(self) -> None:
        """Non-positive sigma should raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            generate_ar2_series(phi1=0.5, phi2=0.2, sigma=0.0, n=100)

    def test_invalid_n_raises(self) -> None:
        """n < 2 should raise ValueError (need 2 initial values)."""
        with pytest.raises(ValueError, match="n must be >= 2"):
            generate_ar2_series(phi1=0.5, phi2=0.2, sigma=1.0, n=1)


class TestIntegration:
    """Integration tests combining validators with gates."""

    def test_generated_series_passes_bounds_check(self) -> None:
        """Model trained on AR(1) data should pass theoretical bounds check."""
        # Generate AR(1) data
        phi, sigma = 0.8, 1.0
        series = generate_ar1_series(phi=phi, sigma=sigma, n=1000, random_state=42)

        # Compute "model" errors (just use persistence as baseline)
        y_true = series[1:]
        y_pred = series[:-1] * phi  # Optimal AR(1) forecast
        model_mse = np.mean((y_true - y_pred) ** 2)

        # Should pass bounds check
        result = check_against_ar1_bounds(
            model_mse=model_mse, phi=phi, sigma_sq=sigma ** 2
        )

        # Optimal AR(1) predictor should pass (MSE ≈ σ²)
        assert result.status in (GateStatus.PASS, GateStatus.WARN)

    def test_leaky_model_halts(self) -> None:
        """Model with 'impossible' performance should HALT."""
        # Generate series
        phi, sigma = 0.8, 1.0
        series = generate_ar1_series(phi=phi, sigma=sigma, n=1000, random_state=42)

        # Simulate leakage: predict using future information
        y_true = series[1:]
        y_pred = y_true * 0.99  # Almost perfect (leaky)
        model_mse = np.mean((y_true - y_pred) ** 2)

        # Should HALT - this is impossibly good
        result = check_against_ar1_bounds(
            model_mse=model_mse, phi=phi, sigma_sq=sigma ** 2
        )
        assert result.status == GateStatus.HALT
