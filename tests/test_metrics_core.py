"""
Tests for core forecast evaluation metrics.

Tests for:
- Point forecast metrics (MAE, MSE, RMSE, MAPE, SMAPE, bias)
- Scale-invariant metrics (MASE, MRAE, Theil's U)
- Correlation metrics (Pearson, Spearman, R²)
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.metrics.core import (
    compute_bias,
    compute_forecast_correlation,
    compute_mae,
    compute_mape,
    compute_mase,
    compute_mrae,
    compute_mse,
    compute_naive_error,
    compute_r_squared,
    compute_rmse,
    compute_smape,
    compute_theils_u,
)


# =============================================================================
# Point Forecast Metrics Tests
# =============================================================================


class TestComputeMAE:
    """Tests for compute_mae."""

    def test_perfect_predictions(self) -> None:
        """MAE should be 0 for perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_mae(preds, actuals) == 0.0

    def test_constant_error(self) -> None:
        """MAE for constant error."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([0.9, 1.9, 2.9])
        assert compute_mae(preds, actuals) == pytest.approx(0.1, rel=1e-6)

    def test_mixed_signs(self) -> None:
        """MAE with over and under predictions."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 1.9, 3.2])
        expected = np.mean([0.1, 0.1, 0.2])
        assert compute_mae(preds, actuals) == pytest.approx(expected, rel=1e-6)

    def test_rejects_nan(self) -> None:
        """MAE should reject NaN inputs."""
        with pytest.raises(ValueError, match="NaN"):
            compute_mae([1.0, np.nan], [1.0, 2.0])

    def test_rejects_length_mismatch(self) -> None:
        """MAE should reject mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            compute_mae([1.0, 2.0], [1.0])


class TestComputeMSE:
    """Tests for compute_mse."""

    def test_perfect_predictions(self) -> None:
        """MSE should be 0 for perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_mse(preds, actuals) == 0.0

    def test_known_value(self) -> None:
        """MSE for known error pattern."""
        preds = np.array([1.0, 2.0])
        actuals = np.array([0.0, 0.0])
        # errors: 1, 2 -> squared: 1, 4 -> mean: 2.5
        assert compute_mse(preds, actuals) == pytest.approx(2.5, rel=1e-6)


class TestComputeRMSE:
    """Tests for compute_rmse."""

    def test_rmse_geq_mae(self) -> None:
        """RMSE should be >= MAE (Cauchy-Schwarz)."""
        preds = np.array([1.0, 2.0, 5.0, 1.0])
        actuals = np.array([1.5, 1.5, 4.0, 2.0])
        mae = compute_mae(preds, actuals)
        rmse = compute_rmse(preds, actuals)
        assert rmse >= mae

    def test_rmse_equals_mae_constant_error(self) -> None:
        """RMSE = MAE when all errors are equal."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.1, 3.1])
        mae = compute_mae(preds, actuals)
        rmse = compute_rmse(preds, actuals)
        assert rmse == pytest.approx(mae, rel=1e-6)


class TestComputeMAPE:
    """Tests for compute_mape."""

    def test_percentage_scale(self) -> None:
        """MAPE should be in percentage scale."""
        preds = np.array([1.1, 2.2])
        actuals = np.array([1.0, 2.0])
        # errors: 10%, 10% -> mean: 10%
        assert compute_mape(preds, actuals) == pytest.approx(10.0, rel=1e-2)

    def test_handles_small_actuals(self) -> None:
        """MAPE should handle near-zero actuals with epsilon."""
        preds = np.array([0.1])
        actuals = np.array([0.0])  # Would cause div by zero without epsilon
        # Should not raise, returns large but finite value
        result = compute_mape(preds, actuals)
        assert np.isfinite(result)


class TestComputeSMAPE:
    """Tests for compute_smape."""

    def test_symmetric(self) -> None:
        """SMAPE should be symmetric around zero."""
        preds = np.array([2.0])
        actuals = np.array([1.0])
        smape1 = compute_smape(preds, actuals)

        preds = np.array([1.0])
        actuals = np.array([2.0])
        smape2 = compute_smape(preds, actuals)

        assert smape1 == pytest.approx(smape2, rel=1e-6)

    def test_bounded(self) -> None:
        """SMAPE should be bounded 0-200%."""
        preds = np.array([100.0, 0.0])
        actuals = np.array([0.0, 100.0])
        # Maximum theoretical SMAPE
        result = compute_smape(preds, actuals)
        assert 0 <= result <= 200

    def test_handles_zeros(self) -> None:
        """SMAPE should handle zero predictions and actuals."""
        preds = np.array([0.0])
        actuals = np.array([0.0])
        result = compute_smape(preds, actuals)
        assert result == 0.0


class TestComputeBias:
    """Tests for compute_bias."""

    def test_over_prediction(self) -> None:
        """Positive bias for over-prediction."""
        preds = np.array([1.1, 2.1, 3.1])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_bias(preds, actuals) == pytest.approx(0.1, rel=1e-6)

    def test_under_prediction(self) -> None:
        """Negative bias for under-prediction."""
        preds = np.array([0.9, 1.9, 2.9])
        actuals = np.array([1.0, 2.0, 3.0])
        assert compute_bias(preds, actuals) == pytest.approx(-0.1, rel=1e-6)

    def test_zero_bias(self) -> None:
        """Zero bias when errors cancel."""
        preds = np.array([0.9, 1.1])
        actuals = np.array([1.0, 1.0])
        assert compute_bias(preds, actuals) == pytest.approx(0.0, rel=1e-6)


# =============================================================================
# Scale-Invariant Metrics Tests
# =============================================================================


class TestComputeNaiveError:
    """Tests for compute_naive_error."""

    def test_persistence_method(self) -> None:
        """Persistence naive error should be mean of differences."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        # Differences: 1, 1, 1 -> MAE: 1
        result = compute_naive_error(values, method="persistence")
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_mean_method(self) -> None:
        """Mean naive error should be MAE from mean."""
        values = np.array([1.0, 2.0, 3.0])
        # Mean = 2, deviations: 1, 0, 1 -> MAE: 2/3
        result = compute_naive_error(values, method="mean")
        assert result == pytest.approx(2 / 3, rel=1e-6)

    def test_minimum_length(self) -> None:
        """Should require at least 2 values."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_naive_error([1.0])


class TestComputeMASE:
    """Tests for compute_mase."""

    def test_beats_naive(self) -> None:
        """MASE < 1 when model beats naive."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 1.9, 3.1])
        naive_mae = 0.5  # Pretend naive has larger errors
        mase = compute_mase(preds, actuals, naive_mae)
        assert mase < 1.0

    def test_equals_naive(self) -> None:
        """MASE = 1 when model equals naive."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 1.9, 3.1])
        mae = compute_mae(preds, actuals)
        mase = compute_mase(preds, actuals, mae)  # Same as naive
        assert mase == pytest.approx(1.0, rel=1e-6)

    def test_rejects_zero_naive(self) -> None:
        """Should reject zero naive_mae."""
        with pytest.raises(ValueError, match="positive"):
            compute_mase([1.0], [1.0], naive_mae=0.0)


class TestComputeTheilsU:
    """Tests for compute_theils_u."""

    def test_beats_naive(self) -> None:
        """U < 1 when model beats naive."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = np.array([1.1, 1.9, 3.1, 3.9])
        u = compute_theils_u(preds, actuals)
        assert u < 1.0  # Good model should beat persistence

    def test_with_explicit_naive(self) -> None:
        """Should work with explicit naive predictions."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 1.9, 3.1])
        naive = np.array([0.0, 0.0, 0.0])  # Terrible naive
        u = compute_theils_u(preds, actuals, naive_predictions=naive)
        assert u < 1.0  # Model beats bad naive


class TestComputeMRAE:
    """Tests for compute_mrae."""

    def test_beats_naive(self) -> None:
        """MRAE < 1 when model beats naive at each point."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.1, 3.1])
        naive = np.array([0.0, 0.0, 0.0])  # Much worse
        mrae = compute_mrae(preds, actuals, naive)
        assert mrae < 1.0


# =============================================================================
# Correlation Metrics Tests
# =============================================================================


class TestComputeForecastCorrelation:
    """Tests for compute_forecast_correlation."""

    def test_perfect_correlation(self) -> None:
        """Perfect positive correlation."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = np.array([1.0, 2.0, 3.0, 4.0])
        r = compute_forecast_correlation(preds, actuals)
        assert r == pytest.approx(1.0, rel=1e-6)

    def test_perfect_negative_correlation(self) -> None:
        """Perfect negative correlation."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = np.array([4.0, 3.0, 2.0, 1.0])
        r = compute_forecast_correlation(preds, actuals)
        assert r == pytest.approx(-1.0, rel=1e-6)

    def test_high_correlation_different_scale(self) -> None:
        """High correlation despite scale difference."""
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        actuals = np.array([100.0, 200.0, 300.0, 400.0])  # Same pattern, 100x
        r = compute_forecast_correlation(preds, actuals)
        assert r == pytest.approx(1.0, rel=1e-6)


class TestComputeRSquared:
    """Tests for compute_r_squared."""

    def test_perfect_fit(self) -> None:
        """R² = 1 for perfect predictions."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])
        r2 = compute_r_squared(preds, actuals)
        assert r2 == pytest.approx(1.0, rel=1e-6)

    def test_worse_than_mean(self) -> None:
        """R² < 0 when model is worse than mean."""
        preds = np.array([100.0, 100.0, 100.0])  # Terrible predictions
        actuals = np.array([1.0, 2.0, 3.0])  # Mean = 2
        r2 = compute_r_squared(preds, actuals)
        assert r2 < 0

    def test_equals_mean(self) -> None:
        """R² = 0 when model predicts mean."""
        actuals = np.array([1.0, 2.0, 3.0])
        mean = np.mean(actuals)
        preds = np.full_like(actuals, mean)
        r2 = compute_r_squared(preds, actuals)
        assert r2 == pytest.approx(0.0, rel=1e-6)


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation across all metrics."""

    def test_list_inputs_accepted(self) -> None:
        """Should accept Python lists."""
        result = compute_mae([1.0, 2.0], [1.0, 2.0])
        assert result == 0.0

    def test_tuple_inputs_accepted(self) -> None:
        """Should accept tuples."""
        result = compute_mae((1.0, 2.0), (1.0, 2.0))
        assert result == 0.0

    def test_mixed_precision(self) -> None:
        """Should handle mixed float precision."""
        preds = np.array([1.0, 2.0], dtype=np.float32)
        actuals = np.array([1.0, 2.0], dtype=np.float64)
        result = compute_mae(preds, actuals)
        assert result == 0.0
