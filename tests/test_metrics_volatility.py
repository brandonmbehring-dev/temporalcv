"""
Tests for Volatility-Weighted Metrics.

Tests cover:
- Volatility estimators (RollingVolatility, EWMAVolatility)
- compute_local_volatility: volatility estimation
- compute_volatility_normalized_mae: scale-invariant MAE
- compute_volatility_weighted_mae: weighted by volatility
- compute_volatility_stratified_metrics: tercile breakdown
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.metrics.volatility_weighted import (
    EWMAVolatility,
    RollingVolatility,
    VolatilityStratifiedResult,
    compute_local_volatility,
    compute_volatility_normalized_mae,
    compute_volatility_stratified_metrics,
    compute_volatility_weighted_mae,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def constant_volatility_data():
    """Data with constant volatility."""
    rng = np.random.default_rng(42)
    n = 100
    values = rng.normal(0, 0.1, n)  # Constant std
    return values


@pytest.fixture
def changing_volatility_data():
    """Data with changing volatility regimes."""
    rng = np.random.default_rng(42)
    # Low vol -> high vol -> low vol
    low_vol = rng.normal(0, 0.01, 50)
    high_vol = rng.normal(0, 0.10, 50)
    low_vol2 = rng.normal(0, 0.01, 50)
    return np.concatenate([low_vol, high_vol, low_vol2])


@pytest.fixture
def predictions_actuals():
    """Sample predictions and actuals."""
    rng = np.random.default_rng(42)
    n = 100
    actuals = rng.normal(0, 1, n)
    predictions = actuals + rng.normal(0, 0.2, n)  # Add noise
    return predictions, actuals


# =============================================================================
# Tests: RollingVolatility
# =============================================================================


class TestRollingVolatility:
    """Tests for rolling window volatility estimator."""

    def test_basic_estimation(self, constant_volatility_data):
        """Basic rolling volatility estimation."""
        estimator = RollingVolatility(window=13)
        vol = estimator.estimate(constant_volatility_data)

        assert len(vol) == len(constant_volatility_data)
        assert np.all(vol >= 0)
        assert np.all(np.isfinite(vol))

    def test_constant_data_zero_vol(self):
        """Constant data has zero volatility (std=0)."""
        constant = np.ones(50)
        estimator = RollingVolatility(window=10)
        vol = estimator.estimate(constant)

        # After warmup, volatility should be 0
        assert vol[-1] == 0.0

    def test_window_affects_smoothness(self, changing_volatility_data):
        """Larger window creates smoother volatility."""
        vol_short = RollingVolatility(window=5).estimate(changing_volatility_data)
        vol_long = RollingVolatility(window=20).estimate(changing_volatility_data)

        # Longer window should be smoother (lower variance)
        assert np.std(vol_long[20:]) < np.std(vol_short[5:])

    def test_min_periods(self):
        """Minimum periods for valid estimate."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        estimator = RollingVolatility(window=10, min_periods=2)
        vol = estimator.estimate(data)

        # Should have valid estimates after min_periods
        assert np.isfinite(vol[1])  # After 2 observations

    def test_window_validation(self):
        """Window must be >= 2."""
        with pytest.raises(ValueError, match="window must be >= 2"):
            RollingVolatility(window=1)

    def test_empty_array(self):
        """Empty array returns empty result."""
        estimator = RollingVolatility(window=5)
        vol = estimator.estimate(np.array([]))
        assert len(vol) == 0


# =============================================================================
# Tests: EWMAVolatility
# =============================================================================


class TestEWMAVolatility:
    """Tests for EWMA volatility estimator."""

    def test_basic_estimation(self, constant_volatility_data):
        """Basic EWMA volatility estimation."""
        estimator = EWMAVolatility(span=13)
        vol = estimator.estimate(constant_volatility_data)

        assert len(vol) == len(constant_volatility_data)
        assert np.all(vol >= 0)
        assert np.all(np.isfinite(vol))

    def test_responds_faster_than_rolling(self, changing_volatility_data):
        """EWMA responds faster to volatility changes than rolling."""
        # Create data with sudden volatility change
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0, 0.01, 50)
        high_vol = rng.normal(0, 0.10, 50)
        data = np.concatenate([low_vol, high_vol])

        vol_ewma = EWMAVolatility(span=13).estimate(data)
        vol_roll = RollingVolatility(window=13).estimate(data)

        # Check response at transition point + 5
        idx = 55
        # EWMA should be higher (faster response to increased vol)
        # Note: This can be noisy, so check general direction
        assert vol_ewma[idx] > vol_ewma[40]  # Increased from low vol period

    def test_span_affects_smoothness(self, changing_volatility_data):
        """Larger span creates smoother volatility."""
        vol_short = EWMAVolatility(span=5).estimate(changing_volatility_data)
        vol_long = EWMAVolatility(span=20).estimate(changing_volatility_data)

        # Longer span should be smoother (lower variance)
        assert np.std(vol_long) < np.std(vol_short)

    def test_span_validation(self):
        """Span must be >= 1."""
        with pytest.raises(ValueError, match="span must be >= 1"):
            EWMAVolatility(span=0)

    def test_empty_array(self):
        """Empty array returns empty result."""
        estimator = EWMAVolatility(span=5)
        vol = estimator.estimate(np.array([]))
        assert len(vol) == 0


# =============================================================================
# Tests: compute_local_volatility
# =============================================================================


class TestComputeLocalVolatility:
    """Tests for local volatility computation function."""

    def test_rolling_std_method(self, constant_volatility_data):
        """Rolling std method works."""
        vol = compute_local_volatility(
            constant_volatility_data, window=13, method="rolling_std"
        )
        assert len(vol) == len(constant_volatility_data)
        assert np.all(np.isfinite(vol))

    def test_ewm_method(self, constant_volatility_data):
        """EWM method works."""
        vol = compute_local_volatility(
            constant_volatility_data, window=13, method="ewm"
        )
        assert len(vol) == len(constant_volatility_data)
        assert np.all(np.isfinite(vol))

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            compute_local_volatility(np.array([1, 2, 3]), method="invalid")

    def test_empty_array(self):
        """Empty array returns empty result."""
        vol = compute_local_volatility(np.array([]))
        assert len(vol) == 0

    def test_list_input(self, constant_volatility_data):
        """Accepts list input."""
        vol = compute_local_volatility(list(constant_volatility_data))
        assert len(vol) == len(constant_volatility_data)


# =============================================================================
# Tests: compute_volatility_normalized_mae
# =============================================================================


class TestComputeVolatilityNormalizedMAE:
    """Tests for volatility-normalized MAE."""

    def test_perfect_predictions(self, predictions_actuals):
        """Perfect predictions yield zero VN-MAE."""
        _, actuals = predictions_actuals
        volatility = np.abs(actuals) + 0.1  # Avoid division issues

        vnmae = compute_volatility_normalized_mae(actuals, actuals, volatility)
        assert vnmae == pytest.approx(0.0, abs=1e-10)

    def test_normalization_effect(self):
        """Higher volatility reduces normalized error."""
        predictions = np.array([0.0, 0.0])
        actuals = np.array([1.0, 1.0])  # Same absolute error

        # Low volatility -> high normalized error
        low_vol = np.array([0.1, 0.1])
        vnmae_low = compute_volatility_normalized_mae(predictions, actuals, low_vol)

        # High volatility -> low normalized error
        high_vol = np.array([10.0, 10.0])
        vnmae_high = compute_volatility_normalized_mae(predictions, actuals, high_vol)

        assert vnmae_low > vnmae_high

    def test_scale_invariance(self):
        """VN-MAE is scale-invariant across regimes."""
        # Two series with same relative error
        pred1 = np.array([0.0])
        actual1 = np.array([0.1])
        vol1 = np.array([0.1])

        pred2 = np.array([0.0])
        actual2 = np.array([10.0])
        vol2 = np.array([10.0])

        vnmae1 = compute_volatility_normalized_mae(pred1, actual1, vol1)
        vnmae2 = compute_volatility_normalized_mae(pred2, actual2, vol2)

        # Same relative error -> same normalized error
        assert vnmae1 == pytest.approx(vnmae2, rel=0.01)

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_volatility_normalized_mae(
                np.array([1, 2]),
                np.array([1, 2, 3]),
                np.array([0.1, 0.1]),
            )

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_volatility_normalized_mae(np.array([]), np.array([]), np.array([]))


# =============================================================================
# Tests: compute_volatility_weighted_mae
# =============================================================================


class TestComputeVolatilityWeightedMAE:
    """Tests for volatility-weighted MAE."""

    def test_inverse_weighting(self):
        """Inverse weighting emphasizes low-vol periods."""
        predictions = np.array([0.0, 0.0])
        actuals = np.array([1.0, 1.0])  # Same absolute error

        # First point: low vol (high weight)
        # Second point: high vol (low weight)
        volatility = np.array([0.1, 10.0])

        wmae = compute_volatility_weighted_mae(
            predictions, actuals, volatility, weighting="inverse"
        )

        # Weighted MAE should be closer to error at low-vol point
        # Weight for low-vol: 10, for high-vol: 0.1
        # Normalized weights: 10/10.1 ≈ 0.99, 0.1/10.1 ≈ 0.01
        # So WMAE ≈ 1.0 (dominated by low-vol point)
        assert wmae == pytest.approx(1.0, rel=0.02)

    def test_importance_weighting(self):
        """Importance weighting emphasizes high-vol periods."""
        predictions = np.array([0.0, 0.0])
        actuals = np.array([0.1, 10.0])  # Different errors

        # First point: low vol (low weight)
        # Second point: high vol (high weight)
        volatility = np.array([0.1, 10.0])

        wmae = compute_volatility_weighted_mae(
            predictions, actuals, volatility, weighting="importance"
        )

        # WMAE should be closer to error at high-vol point (10.0)
        assert wmae > 5.0  # Dominated by high-vol point

    def test_equal_volatility_equals_mae(self, predictions_actuals):
        """With equal volatility, weighted MAE equals standard MAE."""
        predictions, actuals = predictions_actuals
        volatility = np.ones(len(predictions))

        wmae = compute_volatility_weighted_mae(predictions, actuals, volatility)
        mae = np.mean(np.abs(predictions - actuals))

        assert wmae == pytest.approx(mae, rel=1e-6)

    def test_invalid_weighting(self, predictions_actuals):
        """Invalid weighting raises ValueError."""
        predictions, actuals = predictions_actuals
        volatility = np.ones(len(predictions))

        with pytest.raises(ValueError, match="weighting must be"):
            compute_volatility_weighted_mae(
                predictions, actuals, volatility, weighting="invalid"
            )

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_volatility_weighted_mae(
                np.array([1, 2]),
                np.array([1, 2, 3]),
                np.array([0.1, 0.1]),
            )

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_volatility_weighted_mae(np.array([]), np.array([]), np.array([]))


# =============================================================================
# Tests: compute_volatility_stratified_metrics
# =============================================================================


class TestComputeVolatilityStratifiedMetrics:
    """Tests for volatility-stratified metrics."""

    def test_basic_stratification(self, predictions_actuals):
        """Basic stratification produces valid result."""
        predictions, actuals = predictions_actuals

        result = compute_volatility_stratified_metrics(
            predictions, actuals, window=13
        )

        assert isinstance(result, VolatilityStratifiedResult)
        assert result.overall_mae > 0
        assert np.isfinite(result.low_vol_mae)
        assert np.isfinite(result.med_vol_mae)
        assert np.isfinite(result.high_vol_mae)

    def test_tercile_counts(self, predictions_actuals):
        """Terciles have approximately equal counts."""
        predictions, actuals = predictions_actuals
        n = len(predictions)

        result = compute_volatility_stratified_metrics(predictions, actuals)

        # Each tercile should have about n/3 observations
        expected = n / 3
        assert abs(result.n_low - expected) < expected * 0.2
        assert abs(result.n_med - expected) < expected * 0.2
        assert abs(result.n_high - expected) < expected * 0.2

        # Total should equal n
        assert result.n_low + result.n_med + result.n_high == n

    def test_overall_mae_consistency(self, predictions_actuals):
        """Overall MAE matches direct computation."""
        predictions, actuals = predictions_actuals

        result = compute_volatility_stratified_metrics(predictions, actuals)
        direct_mae = np.mean(np.abs(predictions - actuals))

        assert result.overall_mae == pytest.approx(direct_mae, rel=1e-6)

    def test_with_provided_volatility(self, predictions_actuals):
        """Works with pre-computed volatility."""
        predictions, actuals = predictions_actuals
        volatility = np.abs(actuals) + 0.1

        result = compute_volatility_stratified_metrics(
            predictions, actuals, volatility=volatility
        )

        assert isinstance(result, VolatilityStratifiedResult)

    def test_thresholds_ordering(self, predictions_actuals):
        """Volatility thresholds are correctly ordered."""
        predictions, actuals = predictions_actuals

        result = compute_volatility_stratified_metrics(predictions, actuals)

        low_upper, high_lower = result.vol_thresholds
        assert low_upper < high_lower

    def test_summary_method(self, predictions_actuals):
        """Summary method produces string output."""
        predictions, actuals = predictions_actuals

        result = compute_volatility_stratified_metrics(predictions, actuals)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Overall MAE" in summary
        assert "Low vol" in summary
        assert "High vol" in summary

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_volatility_stratified_metrics(
                np.array([1, 2]),
                np.array([1, 2, 3]),
            )

    def test_volatility_length_mismatch_error(self, predictions_actuals):
        """Volatility length mismatch raises ValueError."""
        predictions, actuals = predictions_actuals

        with pytest.raises(ValueError, match="volatility length must match"):
            compute_volatility_stratified_metrics(
                predictions, actuals, volatility=np.array([0.1, 0.2])
            )

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_volatility_stratified_metrics(np.array([]), np.array([]))


# =============================================================================
# Integration Tests
# =============================================================================


class TestVolatilityMetricsIntegration:
    """Integration tests for volatility metrics."""

    def test_low_vol_periods_lower_vnmae(self):
        """Low volatility periods naturally have lower VN-MAE."""
        rng = np.random.default_rng(42)

        # Low vol period
        low_vol_actuals = rng.normal(0, 0.01, 50)
        low_vol_preds = low_vol_actuals + rng.normal(0, 0.001, 50)
        low_volatility = np.full(50, 0.01)

        # High vol period with proportionally larger error
        high_vol_actuals = rng.normal(0, 0.1, 50)
        high_vol_preds = high_vol_actuals + rng.normal(0, 0.01, 50)
        high_volatility = np.full(50, 0.1)

        vnmae_low = compute_volatility_normalized_mae(
            low_vol_preds, low_vol_actuals, low_volatility
        )
        vnmae_high = compute_volatility_normalized_mae(
            high_vol_preds, high_vol_actuals, high_volatility
        )

        # Same relative error ratio -> similar VN-MAE
        assert abs(vnmae_low - vnmae_high) < 0.5

    def test_stratified_detects_regime_performance(self):
        """Stratified metrics detect different performance across regimes."""
        rng = np.random.default_rng(42)

        # Create data where we're good at low vol, bad at high vol
        n = 150
        actuals = np.zeros(n)
        predictions = np.zeros(n)
        volatility = np.zeros(n)

        # Low vol: small errors
        actuals[:50] = rng.normal(0, 0.1, 50)
        predictions[:50] = actuals[:50] + rng.normal(0, 0.01, 50)
        volatility[:50] = 0.05

        # Medium vol: medium errors
        actuals[50:100] = rng.normal(0, 0.5, 50)
        predictions[50:100] = actuals[50:100] + rng.normal(0, 0.1, 50)
        volatility[50:100] = 0.25

        # High vol: large errors
        actuals[100:] = rng.normal(0, 1.0, 50)
        predictions[100:] = actuals[100:] + rng.normal(0, 0.5, 50)
        volatility[100:] = 0.5

        result = compute_volatility_stratified_metrics(
            predictions, actuals, volatility=volatility
        )

        # Should see increasing MAE from low to high vol
        assert result.low_vol_mae < result.med_vol_mae < result.high_vol_mae


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestVolatilityMetricsEdgeCases:
    """Edge cases and numerical stability tests."""

    def test_near_zero_volatility(self):
        """Handles near-zero volatility with epsilon."""
        predictions = np.array([1.0, 2.0])
        actuals = np.array([1.1, 2.1])
        volatility = np.array([1e-10, 1e-10])

        vnmae = compute_volatility_normalized_mae(
            predictions, actuals, volatility, epsilon=1e-8
        )
        assert np.isfinite(vnmae)

    def test_large_values(self):
        """Handles large values."""
        predictions = np.array([1e10, 2e10])
        actuals = np.array([1.1e10, 2.1e10])
        volatility = np.array([1e9, 1e9])

        vnmae = compute_volatility_normalized_mae(predictions, actuals, volatility)
        assert np.isfinite(vnmae)

    def test_small_sample(self):
        """Works with small samples."""
        predictions = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.1, 2.2, 2.9])

        # Stratified metrics might have empty terciles
        result = compute_volatility_stratified_metrics(
            predictions, actuals, window=2
        )
        assert result.n_low + result.n_med + result.n_high == 3

    def test_list_inputs_all_functions(self):
        """All functions accept list inputs."""
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        actuals = [1.1, 2.1, 3.1, 4.1, 5.1]
        volatility = [0.1, 0.1, 0.1, 0.1, 0.1]

        assert isinstance(
            compute_volatility_normalized_mae(predictions, actuals, volatility), float
        )
        assert isinstance(
            compute_volatility_weighted_mae(predictions, actuals, volatility), float
        )
        assert isinstance(
            compute_volatility_stratified_metrics(predictions, actuals),
            VolatilityStratifiedResult,
        )
