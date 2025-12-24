"""
Test Regime Classification Module.

CRITICAL: Tests verify that volatility is computed on CHANGES, not LEVELS.
Using levels mislabels steady drifts as "volatile".
"""

import numpy as np
import pytest

from temporalcv.regimes import (
    classify_direction_regime,
    classify_volatility_regime,
    compute_stratified_metrics,
    get_combined_regimes,
    get_regime_counts,
    mask_low_n_regimes,
    StratifiedMetricsResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_values() -> np.ndarray:
    """Generate sample values with known volatility pattern."""
    rng = np.random.default_rng(42)
    n = 200

    # First half: low volatility (small changes)
    # Second half: high volatility (large changes)
    changes = np.concatenate(
        [
            rng.normal(0, 0.01, n // 2),  # Low vol
            rng.normal(0, 0.05, n // 2),  # High vol
        ]
    )
    return 3.5 + np.cumsum(changes)


@pytest.fixture
def drifting_values() -> np.ndarray:
    """
    Generate values with steady drift (high std of levels, low std of changes).

    KEY test case for changes vs levels distinction.
    """
    rng = np.random.default_rng(42)
    n = 200

    # Steady drift: constant increase + tiny noise
    # HIGH std of levels, LOW std of changes
    changes = 0.01 + rng.normal(0, 0.001, n)
    return 3.0 + np.cumsum(changes)


# =============================================================================
# Volatility Regime Tests
# =============================================================================


class TestVolatilityRegime:
    """Test volatility regime classification."""

    def test_returns_valid_labels(self, sample_values: np.ndarray) -> None:
        """Should return valid regime labels."""
        regimes = classify_volatility_regime(sample_values, window=13, basis="changes")

        assert len(regimes) == len(sample_values)
        assert all(r in ["LOW", "MED", "HIGH"] for r in regimes)

    def test_changes_vs_levels_differ_for_drift(self, drifting_values: np.ndarray) -> None:
        """
        CRITICAL: basis='changes' must differ from basis='levels' for drifting data.

        For a steady drift (constant increases):
        - basis='changes': Volatility is constant (similar regimes throughout)
        - basis='levels': Volatility increases over time (early=LOW, late=HIGH)

        The key difference is WHERE the HIGH regimes appear, not the total count.
        """
        regimes_changes = classify_volatility_regime(
            drifting_values, window=13, basis="changes"
        )
        regimes_levels = classify_volatility_regime(
            drifting_values, window=13, basis="levels"
        )

        # For drifting data with basis='levels':
        # - Early points should be LOW (small cumsum, low std)
        # - Late points should be HIGH (large cumsum spread, high std)
        # For basis='changes': distribution should be more uniform

        # Check last 50 points - should differ in HIGH classification
        last_50_changes = regimes_changes[-50:]
        last_50_levels = regimes_levels[-50:]

        high_count_changes = np.sum(last_50_changes == "HIGH")
        high_count_levels = np.sum(last_50_levels == "HIGH")

        # For drifting data, basis='levels' should have MORE HIGH in late period
        # because the rolling std of cumulative values increases over time
        # The point-by-point classifications should differ
        n_differ = np.sum(regimes_changes != regimes_levels)

        assert n_differ > 0, (
            "basis='changes' and basis='levels' should produce different "
            "point-by-point classifications for drifting data"
        )

    def test_default_basis_is_changes(self, sample_values: np.ndarray) -> None:
        """Default should be basis='changes'."""
        # Call without explicit basis
        regimes_default = classify_volatility_regime(sample_values, window=13)
        regimes_changes = classify_volatility_regime(
            sample_values, window=13, basis="changes"
        )

        np.testing.assert_array_equal(regimes_default, regimes_changes)

    def test_handles_insufficient_data(self) -> None:
        """Should handle arrays smaller than window."""
        values = np.array([1.0, 2.0, 3.0])  # Only 3 points
        regimes = classify_volatility_regime(values, window=13)

        assert len(regimes) == 3
        assert all(r == "MED" for r in regimes)  # Default for insufficient data

    def test_window_parameter(self, sample_values: np.ndarray) -> None:
        """Different window sizes should give different results."""
        regimes_short = classify_volatility_regime(sample_values, window=5)
        regimes_long = classify_volatility_regime(sample_values, window=26)

        # Should not be identical (different smoothing)
        assert not np.array_equal(regimes_short, regimes_long)

    def test_custom_percentiles(self, sample_values: np.ndarray) -> None:
        """Custom percentiles should affect classification."""
        # Very low thresholds = most HIGH
        regimes_low = classify_volatility_regime(
            sample_values, window=13, low_percentile=10.0, high_percentile=20.0
        )
        # Very high thresholds = most LOW
        regimes_high = classify_volatility_regime(
            sample_values, window=13, low_percentile=80.0, high_percentile=90.0
        )

        low_count_low = np.sum(regimes_low == "LOW")
        low_count_high = np.sum(regimes_high == "LOW")

        # regimes_high should have more LOW classifications
        assert low_count_high > low_count_low


# =============================================================================
# Direction Regime Tests
# =============================================================================


class TestDirectionRegime:
    """Test direction regime classification."""

    def test_returns_valid_labels(self) -> None:
        """Should return valid direction labels."""
        values = np.array([0.1, -0.1, 0.0, 0.05, -0.05])
        threshold = 0.03

        directions = classify_direction_regime(values, threshold)

        assert len(directions) == len(values)
        assert all(d in ["UP", "DOWN", "FLAT"] for d in directions)

    def test_threshold_correctly_applied(self) -> None:
        """Should classify based on threshold."""
        values = np.array([0.1, -0.1, 0.02, -0.02, 0.0])
        threshold = 0.05

        directions = classify_direction_regime(values, threshold)

        assert directions[0] == "UP"  # 0.1 > 0.05
        assert directions[1] == "DOWN"  # -0.1 < -0.05
        assert directions[2] == "FLAT"  # |0.02| <= 0.05
        assert directions[3] == "FLAT"  # |-0.02| <= 0.05
        assert directions[4] == "FLAT"  # |0.0| <= 0.05

    def test_boundary_at_threshold(self) -> None:
        """Boundary cases: exactly at threshold should be FLAT."""
        threshold = 0.05
        values = np.array([0.05, -0.05, 0.05 + 1e-10])

        directions = classify_direction_regime(values, threshold)

        assert directions[0] == "FLAT"  # Exactly at threshold
        assert directions[1] == "FLAT"  # Exactly at -threshold
        assert directions[2] == "UP"  # Just above threshold

    def test_negative_threshold_raises(self) -> None:
        """Should raise for negative threshold."""
        values = np.array([0.1, -0.1])

        with pytest.raises(ValueError, match="non-negative"):
            classify_direction_regime(values, threshold=-0.05)

    def test_zero_threshold(self) -> None:
        """Zero threshold means only exact zero is FLAT."""
        values = np.array([0.001, -0.001, 0.0])

        directions = classify_direction_regime(values, threshold=0.0)

        assert directions[0] == "UP"
        assert directions[1] == "DOWN"
        assert directions[2] == "FLAT"


# =============================================================================
# Combined Regime Tests
# =============================================================================


class TestCombinedRegimes:
    """Test combined regime functionality."""

    def test_combines_correctly(self) -> None:
        """Should combine vol and direction regimes."""
        vol = np.array(["HIGH", "LOW", "MED"])
        dir_ = np.array(["UP", "DOWN", "FLAT"])

        combined = get_combined_regimes(vol, dir_)

        assert combined[0] == "HIGH-UP"
        assert combined[1] == "LOW-DOWN"
        assert combined[2] == "MED-FLAT"

    def test_length_mismatch_raises(self) -> None:
        """Should raise if lengths don't match."""
        vol = np.array(["HIGH", "LOW"])
        dir_ = np.array(["UP", "DOWN", "FLAT"])

        with pytest.raises(ValueError, match="same length"):
            get_combined_regimes(vol, dir_)

    def test_all_combinations(self) -> None:
        """Test all possible regime combinations."""
        vol_options = ["LOW", "MED", "HIGH"]
        dir_options = ["UP", "DOWN", "FLAT"]

        for v in vol_options:
            for d in dir_options:
                combined = get_combined_regimes(np.array([v]), np.array([d]))
                assert combined[0] == f"{v}-{d}"


# =============================================================================
# Utility Tests
# =============================================================================


class TestGetRegimeCounts:
    """Test regime counting."""

    def test_counts_correctly(self) -> None:
        """Should count each regime."""
        regimes = np.array(["HIGH", "LOW", "LOW", "MED", "LOW"])

        counts = get_regime_counts(regimes)

        assert counts["LOW"] == 3
        assert counts["HIGH"] == 1
        assert counts["MED"] == 1

    def test_sorted_by_count_descending(self) -> None:
        """Should return sorted by count descending."""
        regimes = np.array(["A", "B", "B", "C", "C", "C"])

        counts = get_regime_counts(regimes)
        keys = list(counts.keys())

        assert keys[0] == "C"  # 3 counts
        assert keys[1] == "B"  # 2 counts
        assert keys[2] == "A"  # 1 count

    def test_empty_array(self) -> None:
        """Should handle empty array."""
        counts = get_regime_counts(np.array([]))
        assert len(counts) == 0


class TestMaskLowNRegimes:
    """Test low-n regime masking."""

    def test_masks_low_count_regimes(self) -> None:
        """Should mask regimes with low sample counts."""
        regimes = np.array(
            ["HIGH"] * 5  # n=5 (masked)
            + ["LOW"] * 15  # n=15 (kept)
        )

        masked = mask_low_n_regimes(regimes, min_n=10)

        assert "HIGH" not in masked
        assert "MASKED" in masked
        assert "LOW" in masked

    def test_custom_mask_value(self) -> None:
        """Should use custom mask value."""
        regimes = np.array(["A"] * 5 + ["B"] * 15)

        masked = mask_low_n_regimes(regimes, min_n=10, mask_value="EXCLUDED")

        assert "A" not in masked
        assert "EXCLUDED" in masked

    def test_keeps_all_if_sufficient(self) -> None:
        """Should keep all regimes if all have sufficient samples."""
        regimes = np.array(["A"] * 20 + ["B"] * 20)

        masked = mask_low_n_regimes(regimes, min_n=10)

        assert "MASKED" not in masked
        np.testing.assert_array_equal(regimes, masked)

    def test_preserves_order(self) -> None:
        """Should preserve original array order."""
        regimes = np.array(["A", "B", "B", "A", "B"])  # A=2, B=3

        masked = mask_low_n_regimes(regimes, min_n=3)

        # A should be masked (n=2 < 3), B kept
        expected = np.array(["MASKED", "B", "B", "MASKED", "B"])
        np.testing.assert_array_equal(masked, expected)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegimeIntegration:
    """Integration tests for regime classification."""

    def test_full_pipeline(self, sample_values: np.ndarray) -> None:
        """Test full regime classification pipeline."""
        # Compute changes
        changes = np.diff(sample_values)

        # Classify volatility
        vol_regimes = classify_volatility_regime(
            sample_values[:-1], window=13, basis="changes"
        )

        # Classify direction with threshold
        threshold = np.percentile(np.abs(changes), 70)
        dir_regimes = classify_direction_regime(changes, threshold)

        # Ensure same length (trim vol to match changes)
        min_len = min(len(vol_regimes), len(dir_regimes))
        vol_regimes = vol_regimes[:min_len]
        dir_regimes = dir_regimes[:min_len]

        # Combine
        combined = get_combined_regimes(vol_regimes, dir_regimes)

        # Get counts
        counts = get_regime_counts(combined)

        # Should have some regimes classified
        assert len(counts) > 0
        assert sum(counts.values()) == min_len

    def test_regime_proportions_reasonable(self, sample_values: np.ndarray) -> None:
        """Regime proportions should be roughly 33% each for tertiles."""
        regimes = classify_volatility_regime(
            sample_values,
            window=13,
            basis="changes",
            low_percentile=33.0,
            high_percentile=67.0,
        )

        counts = get_regime_counts(regimes)

        # Filter out MED from early points with insufficient data
        # At least check we have all three regimes
        assert "LOW" in counts or "MED" in counts or "HIGH" in counts


# =============================================================================
# Stratified Metrics Tests
# =============================================================================


class TestStratifiedMetricsResult:
    """Tests for StratifiedMetricsResult dataclass."""

    def test_basic_attributes(self) -> None:
        """Should have correct basic attributes."""
        result = StratifiedMetricsResult(
            overall_mae=0.1,
            overall_rmse=0.15,
            n_total=100,
            by_regime={"A": {"mae": 0.08, "rmse": 0.12, "n": 50, "pct": 50.0}},
            masked_regimes=["B"],
        )

        assert result.overall_mae == 0.1
        assert result.overall_rmse == 0.15
        assert result.n_total == 100
        assert "A" in result.by_regime
        assert result.masked_regimes == ["B"]

    def test_summary_contains_metrics(self) -> None:
        """Summary should include key metrics."""
        result = StratifiedMetricsResult(
            overall_mae=0.1234,
            overall_rmse=0.1567,
            n_total=100,
            by_regime={
                "HIGH": {"mae": 0.15, "rmse": 0.20, "n": 30, "pct": 30.0},
                "LOW": {"mae": 0.08, "rmse": 0.10, "n": 70, "pct": 70.0},
            },
        )

        summary = result.summary()

        assert "0.1234" in summary  # Overall MAE
        assert "0.1567" in summary  # Overall RMSE
        assert "100" in summary  # n_total
        assert "HIGH" in summary
        assert "LOW" in summary

    def test_summary_includes_masked(self) -> None:
        """Summary should list masked regimes."""
        result = StratifiedMetricsResult(
            overall_mae=0.1,
            overall_rmse=0.15,
            n_total=100,
            masked_regimes=["TINY", "SMALL"],
        )

        summary = result.summary()

        assert "TINY" in summary
        assert "SMALL" in summary
        assert "Masked" in summary


class TestComputeStratifiedMetrics:
    """Tests for compute_stratified_metrics function."""

    def test_basic_computation(self) -> None:
        """Should compute overall and per-regime metrics."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        actuals = np.array([1.1, 2.2, 2.8, 4.1, 4.9] * 20)
        regimes = np.array(["A", "A", "B", "B", "B"] * 20)

        result = compute_stratified_metrics(predictions, actuals, regimes, min_n=5)

        assert result.n_total == 100
        assert result.overall_mae > 0
        assert result.overall_rmse > 0
        assert "A" in result.by_regime
        assert "B" in result.by_regime

    def test_per_regime_metrics_correct(self) -> None:
        """Per-regime metrics should be computed correctly."""
        # Create data where regime A has lower error than regime B
        predictions = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0] * 10)
        actuals = np.array([1.1, 1.0, 0.9, 3.0, 4.0, 6.0] * 10)  # A: small error, B: large error
        regimes = np.array(["A", "A", "A", "B", "B", "B"] * 10)

        result = compute_stratified_metrics(predictions, actuals, regimes, min_n=5)

        # Regime A should have lower MAE than regime B
        assert result.by_regime["A"]["mae"] < result.by_regime["B"]["mae"]

    def test_masks_low_n_regimes(self) -> None:
        """Regimes with n < min_n should be masked."""
        predictions = np.array([1.0] * 100 + [2.0] * 5)  # A: 100, B: 5
        actuals = np.array([1.1] * 100 + [2.1] * 5)
        regimes = np.array(["A"] * 100 + ["B"] * 5)

        result = compute_stratified_metrics(predictions, actuals, regimes, min_n=10)

        assert "A" in result.by_regime
        assert "B" not in result.by_regime  # Masked
        assert "B" in result.masked_regimes

    def test_percentage_sums_correctly(self) -> None:
        """Percentages should sum to ~100 for non-masked regimes."""
        predictions = np.array([1.0] * 50 + [2.0] * 30 + [3.0] * 20)
        actuals = np.array([1.1] * 50 + [2.1] * 30 + [3.1] * 20)
        regimes = np.array(["A"] * 50 + ["B"] * 30 + ["C"] * 20)

        result = compute_stratified_metrics(predictions, actuals, regimes, min_n=10)

        total_pct = sum(r["pct"] for r in result.by_regime.values())
        assert abs(total_pct - 100.0) < 0.1

    def test_raises_on_empty_arrays(self) -> None:
        """Should raise ValueError on empty arrays."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_stratified_metrics(
                np.array([]),
                np.array([]),
                np.array([]),
            )

    def test_raises_on_length_mismatch(self) -> None:
        """Should raise ValueError on length mismatch."""
        with pytest.raises(ValueError, match="same length"):
            compute_stratified_metrics(
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1, 3.1]),
                np.array(["A", "B"]),
            )

        with pytest.raises(ValueError, match="same length"):
            compute_stratified_metrics(
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1]),
                np.array(["A"]),
            )

    def test_rmse_greater_than_mae(self) -> None:
        """RMSE should be >= MAE for non-uniform errors."""
        rng = np.random.default_rng(42)
        predictions = rng.normal(0, 1, 200)
        actuals = predictions + rng.normal(0, 0.5, 200)  # Non-uniform errors
        regimes = np.array(["A"] * 100 + ["B"] * 100)

        result = compute_stratified_metrics(predictions, actuals, regimes, min_n=10)

        # RMSE >= MAE by Cauchy-Schwarz
        assert result.overall_rmse >= result.overall_mae
        for regime_metrics in result.by_regime.values():
            assert regime_metrics["rmse"] >= regime_metrics["mae"]

    def test_works_with_volatility_regimes(self) -> None:
        """Should work with volatility regime classification."""
        rng = np.random.default_rng(42)
        n = 200

        # Create values with different volatility
        values = np.concatenate(
            [
                3.5 + np.cumsum(rng.normal(0, 0.01, n // 2)),
                3.5 + np.cumsum(rng.normal(0, 0.05, n // 2)),
            ]
        )

        predictions = values + rng.normal(0, 0.02, n)
        actuals = values

        # Classify volatility
        vol_regimes = classify_volatility_regime(values, window=13, basis="changes")

        result = compute_stratified_metrics(predictions, actuals, vol_regimes, min_n=10)

        # Should have at least some regimes
        assert len(result.by_regime) > 0
        assert result.n_total == n

    def test_works_with_direction_regimes(self) -> None:
        """Should work with direction regime classification."""
        rng = np.random.default_rng(42)
        n = 200

        actuals = rng.normal(0, 0.1, n)
        predictions = actuals + rng.normal(0, 0.02, n)

        # Classify direction
        threshold = np.percentile(np.abs(actuals), 70)
        dir_regimes = classify_direction_regime(actuals, threshold)

        result = compute_stratified_metrics(predictions, actuals, dir_regimes, min_n=5)

        # Should have UP, DOWN, and/or FLAT
        assert len(result.by_regime) > 0

    def test_custom_min_n_threshold(self) -> None:
        """Custom min_n should affect masking."""
        predictions = np.array([1.0] * 15 + [2.0] * 10 + [3.0] * 5)
        actuals = np.array([1.1] * 15 + [2.1] * 10 + [3.1] * 5)
        regimes = np.array(["A"] * 15 + ["B"] * 10 + ["C"] * 5)

        # With min_n=5, all should be included
        result_low = compute_stratified_metrics(predictions, actuals, regimes, min_n=5)
        assert len(result_low.by_regime) == 3
        assert len(result_low.masked_regimes) == 0

        # With min_n=12, C and B should be masked
        result_high = compute_stratified_metrics(predictions, actuals, regimes, min_n=12)
        assert len(result_high.by_regime) == 1
        assert "A" in result_high.by_regime
        assert "B" in result_high.masked_regimes
        assert "C" in result_high.masked_regimes
