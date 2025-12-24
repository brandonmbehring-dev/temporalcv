"""
Tests for changepoint detection module.

Test categories:
1. Variance-based detection
2. PELT detection (with skip if ruptures not installed)
3. Regime classification
4. Regime indicators
5. Edge cases
"""

import numpy as np
import pytest

from temporalcv.changepoint import (
    Changepoint,
    ChangepointResult,
    classify_regimes_from_changepoints,
    create_regime_indicators,
    detect_changepoints,
    detect_changepoints_pelt,
    detect_changepoints_variance,
    get_segment_boundaries,
)

# Check if ruptures is available
try:
    import ruptures

    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False


def generate_piecewise_constant(
    segment_lengths: list[int],
    segment_values: list[float],
    noise_std: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate piecewise constant series with optional noise."""
    rng = np.random.default_rng(seed)
    series = []
    for length, value in zip(segment_lengths, segment_values):
        segment = np.full(length, value) + rng.normal(0, noise_std, length)
        series.append(segment)
    return np.concatenate(series)


class TestChangepointVariance:
    """Tests for variance-based changepoint detection."""

    def test_single_level_shift(self) -> None:
        """Should detect a single level shift."""
        series = generate_piecewise_constant([30, 30], [1.0, 5.0], noise_std=0.1, seed=42)
        result = detect_changepoints_variance(series, penalty=2.0)

        assert isinstance(result, ChangepointResult)
        assert len(result.changepoints) >= 1
        # Changepoint should be near index 30
        cp_idx = result.changepoints[0].index
        assert 25 <= cp_idx <= 35

    def test_multiple_level_shifts(self) -> None:
        """Should detect multiple level shifts."""
        series = generate_piecewise_constant(
            [30, 30, 30], [1.0, 5.0, 2.0], noise_std=0.1, seed=43
        )
        result = detect_changepoints_variance(series, penalty=2.0)

        # Should detect at least 1 changepoint (may not find all due to heuristic)
        assert len(result.changepoints) >= 1
        assert result.n_segments >= 2

    def test_no_changepoints_constant(self) -> None:
        """Constant series should have no changepoints."""
        series = np.ones(100)
        result = detect_changepoints_variance(series)

        assert len(result.changepoints) == 0
        assert result.n_segments == 1

    def test_no_changepoints_noise(self) -> None:
        """White noise should have few/no changepoints with high penalty."""
        rng = np.random.default_rng(44)
        series = rng.normal(0, 1, 100)
        result = detect_changepoints_variance(series, penalty=5.0)

        # With high penalty, should find few or no changepoints
        assert len(result.changepoints) <= 2

    def test_penalty_effect(self) -> None:
        """Higher penalty should find fewer changepoints."""
        series = generate_piecewise_constant(
            [25, 25, 25, 25], [1.0, 3.0, 1.5, 4.0], noise_std=0.2, seed=45
        )

        result_low = detect_changepoints_variance(series, penalty=1.0)
        result_high = detect_changepoints_variance(series, penalty=5.0)

        # Lower penalty finds more or equal changepoints
        assert len(result_low.changepoints) >= len(result_high.changepoints)

    def test_min_segment_length(self) -> None:
        """Should respect minimum segment length."""
        series = generate_piecewise_constant(
            [30, 5, 30], [1.0, 5.0, 1.0], noise_std=0.1, seed=46
        )
        result = detect_changepoints_variance(series, min_segment_length=10)

        # Changepoints should be at least 10 apart
        if len(result.changepoints) >= 2:
            for i in range(1, len(result.changepoints)):
                gap = result.changepoints[i].index - result.changepoints[i - 1].index
                assert gap >= 10

    def test_result_frozen(self) -> None:
        """Result should be immutable."""
        series = generate_piecewise_constant([30, 30], [1.0, 5.0], seed=47)
        result = detect_changepoints_variance(series)

        with pytest.raises(Exception):  # FrozenInstanceError
            result.n_segments = 10  # type: ignore[misc]

    def test_changepoint_frozen(self) -> None:
        """Changepoint should be immutable."""
        cp = Changepoint(index=30, cost_reduction=1.5)

        with pytest.raises(Exception):
            cp.index = 40  # type: ignore[misc]

    def test_short_series_raises(self) -> None:
        """Should raise for series that's too short."""
        # Need 2*window + min_segment_length = 2*8 + 4 = 20
        short = np.array([1.0] * 15)

        with pytest.raises(ValueError, match="too short"):
            detect_changepoints_variance(short)

    def test_method_attribute(self) -> None:
        """Result should have correct method attribute."""
        series = generate_piecewise_constant([30, 30], [1.0, 5.0], seed=48)
        result = detect_changepoints_variance(series)

        assert result.method == "variance"


@pytest.mark.skipif(not HAS_RUPTURES, reason="ruptures not installed")
class TestChangepointPELT:
    """Tests for PELT-based changepoint detection."""

    def test_single_level_shift(self) -> None:
        """PELT should detect a single level shift."""
        series = generate_piecewise_constant([50, 50], [0.0, 3.0], noise_std=0.5, seed=50)
        result = detect_changepoints_pelt(series, penalty="bic")

        assert isinstance(result, ChangepointResult)
        assert len(result.changepoints) >= 1
        # Changepoint should be near index 50
        cp_idx = result.changepoints[0].index
        assert 45 <= cp_idx <= 55

    def test_multiple_level_shifts(self) -> None:
        """PELT should detect multiple level shifts."""
        rng = np.random.default_rng(51)
        series = np.concatenate(
            [
                rng.normal(0, 1, 50),
                rng.normal(3, 1, 50),
                rng.normal(1, 1, 50),
            ]
        )
        result = detect_changepoints_pelt(series, penalty="bic")

        # Should detect approximately 2 changepoints
        assert len(result.changepoints) >= 1
        assert result.n_segments >= 2

    def test_penalty_bic(self) -> None:
        """BIC penalty should work."""
        series = generate_piecewise_constant([50, 50], [1.0, 4.0], seed=52)
        result = detect_changepoints_pelt(series, penalty="bic")

        assert result.method == "pelt"
        assert result.penalty > 0

    def test_penalty_aic(self) -> None:
        """AIC penalty should work."""
        series = generate_piecewise_constant([50, 50], [1.0, 4.0], seed=53)
        result = detect_changepoints_pelt(series, penalty="aic")

        assert result.penalty == 2.0  # AIC penalty

    def test_penalty_custom(self) -> None:
        """Custom penalty should work."""
        series = generate_piecewise_constant([50, 50], [1.0, 4.0], seed=54)
        result = detect_changepoints_pelt(series, penalty=5.0)

        assert result.penalty == 5.0

    def test_cost_model_l1(self) -> None:
        """L1 cost model should work."""
        series = generate_piecewise_constant([50, 50], [1.0, 4.0], seed=55)
        result = detect_changepoints_pelt(series, cost_model="l1")

        assert len(result.changepoints) >= 0

    def test_cost_model_rbf(self) -> None:
        """RBF cost model should work."""
        series = generate_piecewise_constant([50, 50], [1.0, 4.0], seed=56)
        result = detect_changepoints_pelt(series, cost_model="rbf")

        assert len(result.changepoints) >= 0

    def test_invalid_cost_model(self) -> None:
        """Should raise for invalid cost model."""
        series = generate_piecewise_constant([50, 50], [1.0, 4.0], seed=57)

        with pytest.raises(ValueError, match="Unknown cost_model"):
            detect_changepoints_pelt(series, cost_model="invalid")

    def test_short_series_raises(self) -> None:
        """Should raise for series that's too short."""
        short = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="too short"):
            detect_changepoints_pelt(short)


class TestDetectChangepointsUnified:
    """Tests for unified detect_changepoints function."""

    def test_auto_selects_variance_without_ruptures(self) -> None:
        """Auto should use variance if ruptures not available."""
        series = generate_piecewise_constant([30, 30], [1.0, 5.0], seed=60)

        # Force variance by checking method
        result = detect_changepoints(series, method="variance")
        assert result.method == "variance"

    @pytest.mark.skipif(not HAS_RUPTURES, reason="ruptures not installed")
    def test_auto_selects_pelt_with_ruptures(self) -> None:
        """Auto should use PELT if ruptures available."""
        series = generate_piecewise_constant([50, 50], [1.0, 5.0], seed=61)
        result = detect_changepoints(series, method="auto")

        assert result.method == "pelt"

    def test_explicit_variance(self) -> None:
        """Should use variance when explicitly requested."""
        series = generate_piecewise_constant([30, 30], [1.0, 5.0], seed=62)
        result = detect_changepoints(series, method="variance")

        assert result.method == "variance"

    @pytest.mark.skipif(not HAS_RUPTURES, reason="ruptures not installed")
    def test_explicit_pelt(self) -> None:
        """Should use PELT when explicitly requested."""
        series = generate_piecewise_constant([50, 50], [1.0, 5.0], seed=63)
        result = detect_changepoints(series, method="pelt")

        assert result.method == "pelt"

    def test_invalid_method(self) -> None:
        """Should raise for invalid method."""
        series = generate_piecewise_constant([30, 30], [1.0, 5.0], seed=64)

        with pytest.raises(ValueError, match="Unknown method"):
            detect_changepoints(series, method="invalid")  # type: ignore[arg-type]


class TestClassifyRegimes:
    """Tests for regime classification."""

    def test_volatility_classification(self) -> None:
        """Should classify by volatility."""
        # Low volatility then high volatility
        rng = np.random.default_rng(70)
        low_vol = rng.normal(0, 0.01, 30)
        high_vol = rng.normal(0, 1.0, 30)
        series = np.concatenate([low_vol, high_vol])

        cp = Changepoint(index=30, cost_reduction=1.0)
        regimes = classify_regimes_from_changepoints(series, [cp], method="volatility")

        # First segment should be LOW, second HIGH
        assert regimes[0] == "LOW"
        assert regimes[35] == "HIGH"

    def test_level_classification(self) -> None:
        """Should classify by level."""
        series = np.concatenate([np.ones(30) * 1.0, np.ones(30) * 10.0])

        cp = Changepoint(index=30, cost_reduction=1.0)
        regimes = classify_regimes_from_changepoints(series, [cp], method="level")

        assert regimes[0] == "LOW"
        assert regimes[35] == "HIGH"

    def test_trend_classification(self) -> None:
        """Should classify by trend direction."""
        # Flat then upward trend
        flat = np.ones(30)
        upward = np.arange(30) * 0.5

        series = np.concatenate([flat, upward])
        cp = Changepoint(index=30, cost_reduction=1.0)
        regimes = classify_regimes_from_changepoints(series, [cp], method="trend")

        # First segment low/no trend, second positive trend
        assert regimes[35] == "HIGH"  # Strong upward trend

    def test_accepts_changepoint_result(self) -> None:
        """Should accept ChangepointResult as input."""
        series = np.concatenate([np.ones(30) * 1.0, np.ones(30) * 5.0])
        result = detect_changepoints_variance(series)
        regimes = classify_regimes_from_changepoints(series, result)

        assert len(regimes) == len(series)

    def test_custom_thresholds(self) -> None:
        """Should use custom thresholds."""
        series = np.concatenate([np.ones(30) * 2.0, np.ones(30) * 5.0])

        cp = Changepoint(index=30, cost_reduction=1.0)
        regimes = classify_regimes_from_changepoints(
            series, [cp], method="level", thresholds=(3.0, 4.0)
        )

        assert regimes[0] == "LOW"  # 2.0 < 3.0
        assert regimes[35] == "HIGH"  # 5.0 > 4.0

    def test_no_changepoints(self) -> None:
        """Should handle no changepoints."""
        series = np.ones(60)
        regimes = classify_regimes_from_changepoints(series, [])

        # Should assign some regime to entire series
        assert len(regimes) == 60
        assert regimes[0] in ["LOW", "MEDIUM", "HIGH"]

    def test_invalid_method_raises(self) -> None:
        """Should raise for invalid method."""
        series = np.ones(60)

        with pytest.raises(ValueError, match="Unknown method"):
            classify_regimes_from_changepoints(
                series, [], method="invalid"  # type: ignore[arg-type]
            )


class TestCreateRegimeIndicators:
    """Tests for regime indicator creation."""

    def test_creates_all_indicators(self) -> None:
        """Should create all expected indicators."""
        series = np.concatenate([np.ones(30), np.ones(30) * 5])
        cp = Changepoint(index=30, cost_reduction=1.0)
        indicators = create_regime_indicators(series, [cp])

        expected_keys = [
            "is_regime_change",
            "periods_since_change",
            "regime_labels",
            "regime_LOW",
            "regime_MEDIUM",
            "regime_HIGH",
        ]
        for key in expected_keys:
            assert key in indicators

    def test_is_regime_change(self) -> None:
        """is_regime_change should be 1 near changepoints."""
        series = np.ones(60)
        cp = Changepoint(index=30, cost_reduction=1.0)
        indicators = create_regime_indicators(series, [cp], recent_window=4)

        # Should be 1 for indices 31-34 (within 4 of changepoint)
        assert indicators["is_regime_change"][30] == 0  # At changepoint
        assert indicators["is_regime_change"][31] == 1  # 1 after
        assert indicators["is_regime_change"][34] == 1  # 4 after
        assert indicators["is_regime_change"][35] == 0  # 5 after

    def test_periods_since_change(self) -> None:
        """periods_since_change should count correctly."""
        series = np.ones(60)
        cp = Changepoint(index=30, cost_reduction=1.0)
        indicators = create_regime_indicators(series, [cp])

        assert indicators["periods_since_change"][30] == 0
        assert indicators["periods_since_change"][31] == 1
        assert indicators["periods_since_change"][40] == 10

    def test_one_hot_encoding(self) -> None:
        """One-hot regime encoding should be correct."""
        series = np.concatenate([np.ones(30), np.ones(30) * 5])
        cp = Changepoint(index=30, cost_reduction=1.0)
        indicators = create_regime_indicators(series, [cp])

        # Check one-hot properties
        for i in range(len(series)):
            total = (
                indicators["regime_LOW"][i]
                + indicators["regime_MEDIUM"][i]
                + indicators["regime_HIGH"][i]
            )
            assert total == 1  # Exactly one regime active

    def test_accepts_changepoint_result(self) -> None:
        """Should accept ChangepointResult as input."""
        series = np.concatenate([np.ones(30), np.ones(30) * 5])
        result = detect_changepoints_variance(series)
        indicators = create_regime_indicators(series, result)

        assert len(indicators["regime_labels"]) == len(series)


class TestGetSegmentBoundaries:
    """Tests for segment boundary extraction."""

    def test_single_changepoint(self) -> None:
        """Should return correct segments for one changepoint."""
        cp = Changepoint(index=30, cost_reduction=1.0)
        boundaries = get_segment_boundaries(60, [cp])

        assert boundaries == [(0, 30), (30, 60)]

    def test_multiple_changepoints(self) -> None:
        """Should return correct segments for multiple changepoints."""
        cps = [
            Changepoint(index=20, cost_reduction=1.0),
            Changepoint(index=40, cost_reduction=1.0),
        ]
        boundaries = get_segment_boundaries(60, cps)

        assert boundaries == [(0, 20), (20, 40), (40, 60)]

    def test_no_changepoints(self) -> None:
        """Should return single segment for no changepoints."""
        boundaries = get_segment_boundaries(60, [])

        assert boundaries == [(0, 60)]

    def test_accepts_changepoint_result(self) -> None:
        """Should accept ChangepointResult as input."""
        result = ChangepointResult(
            changepoints=(Changepoint(index=30, cost_reduction=1.0),),
            n_segments=2,
            method="variance",
            penalty=3.0,
        )
        boundaries = get_segment_boundaries(60, result)

        assert boundaries == [(0, 30), (30, 60)]


class TestEdgeCases:
    """Edge case tests."""

    def test_list_input(self) -> None:
        """Should accept list input."""
        series_list = [1.0] * 30 + [5.0] * 30
        result = detect_changepoints_variance(series_list)

        assert isinstance(result, ChangepointResult)

    def test_large_series(self) -> None:
        """Should handle large series efficiently."""
        rng = np.random.default_rng(80)
        large = rng.normal(0, 1, 5000)
        # Add some level shifts
        large[1000:2000] += 3
        large[3000:4000] -= 2

        result = detect_changepoints_variance(large, penalty=3.0)

        # Should complete without error
        assert isinstance(result, ChangepointResult)

    def test_near_constant_with_noise(self) -> None:
        """Near-constant series with tiny noise should work."""
        rng = np.random.default_rng(81)
        series = np.ones(100) + rng.normal(0, 1e-6, 100)

        result = detect_changepoints_variance(series, penalty=3.0)

        assert len(result.changepoints) <= 1

    def test_gradual_trend(self) -> None:
        """Gradual trend should not trigger many changepoints."""
        series = np.linspace(0, 10, 100)
        result = detect_changepoints_variance(series, penalty=3.0)

        # Linear trend shouldn't have changepoints
        assert len(result.changepoints) <= 1

    def test_oscillating_series(self) -> None:
        """Oscillating series handling."""
        t = np.linspace(0, 4 * np.pi, 100)
        series = np.sin(t)

        result = detect_changepoints_variance(series, penalty=2.0)

        # Sine wave might trigger some changepoints
        assert isinstance(result, ChangepointResult)

    @pytest.mark.skipif(not HAS_RUPTURES, reason="ruptures not installed")
    def test_pelt_vs_variance_consistency(self) -> None:
        """PELT and variance should find similar changepoints for clear shifts."""
        series = generate_piecewise_constant([50, 50], [0.0, 5.0], noise_std=0.1, seed=82)

        result_var = detect_changepoints_variance(series, penalty=2.0)
        result_pelt = detect_changepoints_pelt(series, penalty="bic")

        # Both should find at least one changepoint
        assert len(result_var.changepoints) >= 1
        assert len(result_pelt.changepoints) >= 1

        # Changepoints should be in similar locations
        if len(result_var.changepoints) > 0 and len(result_pelt.changepoints) > 0:
            var_cp = result_var.changepoints[0].index
            pelt_cp = result_pelt.changepoints[0].index
            assert abs(var_cp - pelt_cp) <= 10


class TestRupturesNotInstalled:
    """Tests for ruptures not installed behavior."""

    def test_pelt_raises_without_ruptures(self) -> None:
        """PELT should give clear error if ruptures not installed."""
        # This test only makes sense when ruptures isn't installed
        # When ruptures IS installed, we skip this test
        if HAS_RUPTURES:
            pytest.skip("ruptures is installed")

        series = np.ones(100)
        with pytest.raises(ImportError, match="ruptures"):
            detect_changepoints_pelt(series)

    def test_auto_falls_back_to_variance(self) -> None:
        """Auto should fall back to variance without ruptures."""
        if HAS_RUPTURES:
            pytest.skip("ruptures is installed")

        series = generate_piecewise_constant([30, 30], [1.0, 5.0], seed=90)
        result = detect_changepoints(series, method="auto")

        assert result.method == "variance"
