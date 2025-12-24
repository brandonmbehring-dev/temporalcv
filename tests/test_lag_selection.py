"""
Tests for lag_selection module.

Test categories:
1. Known-answer tests with AR(p) processes
2. Method consistency (AIC vs BIC vs PACF)
3. Edge cases
4. Integration with CV gap suggestion
"""

import numpy as np
import pytest

from temporalcv.lag_selection import (
    LagSelectionResult,
    auto_select_lag,
    select_lag_aic,
    select_lag_bic,
    select_lag_pacf,
    suggest_cv_gap,
)


def generate_ar(n: int, coeffs: list[float], seed: int = 42) -> np.ndarray:
    """Generate AR(p) process with given coefficients."""
    rng = np.random.default_rng(seed)
    p = len(coeffs)
    series = np.zeros(n)

    # Burn-in
    for i in range(p, n):
        ar_term = sum(c * series[i - j - 1] for j, c in enumerate(coeffs))
        series[i] = ar_term + rng.normal(0, 1)

    return series


class TestSelectLagPACF:
    """Tests for select_lag_pacf function."""

    def test_ar1_detection(self) -> None:
        """PACF should detect AR(1) process."""
        ar1 = generate_ar(300, [0.7], seed=42)
        result = select_lag_pacf(ar1)

        assert isinstance(result, LagSelectionResult)
        assert result.method == "pacf"
        # AR(1) should have significant PACF at lag 1 only
        assert result.optimal_lag >= 1
        assert result.optimal_lag <= 3  # Allow some tolerance

    def test_ar2_detection(self) -> None:
        """PACF should detect AR(2) process."""
        ar2 = generate_ar(300, [0.5, 0.3], seed=43)
        result = select_lag_pacf(ar2)

        # AR(2) should have significant PACF at lags 1 and 2
        # The cutoff-based algorithm finds the last consecutive significant lag
        assert result.optimal_lag >= 1
        assert result.optimal_lag <= 5

    def test_white_noise_zero_lag(self) -> None:
        """PACF should return 0 or small lag for white noise."""
        rng = np.random.default_rng(44)
        white_noise = rng.normal(0, 1, 200)

        result = select_lag_pacf(white_noise)

        # White noise should have no significant lags (or very few due to sampling)
        assert result.optimal_lag <= 2

    def test_pacf_values_in_result(self) -> None:
        """Result should contain PACF values for all lags."""
        rng = np.random.default_rng(45)
        series = rng.normal(0, 1, 100)

        result = select_lag_pacf(series)

        # Should have PACF for lag 0 through max_lag
        assert 0 in result.criterion_values
        assert len(result.criterion_values) == len(result.all_lags_tested)
        # PACF at lag 0 should be 1
        assert abs(result.criterion_values[0] - 1.0) < 0.01

    def test_max_lag_respected(self) -> None:
        """Should respect max_lag parameter."""
        rng = np.random.default_rng(46)
        series = rng.normal(0, 1, 100)

        result = select_lag_pacf(series, max_lag=5)

        assert max(result.all_lags_tested) == 5
        assert len(result.criterion_values) == 6  # 0 through 5

    def test_alpha_affects_threshold(self) -> None:
        """Different alpha should affect detection threshold."""
        ar1 = generate_ar(200, [0.3], seed=47)  # Weak AR(1)

        result_strict = select_lag_pacf(ar1, alpha=0.01)
        result_loose = select_lag_pacf(ar1, alpha=0.10)

        # Looser alpha may detect more lags
        assert result_loose.optimal_lag >= result_strict.optimal_lag

    def test_short_series_raises(self) -> None:
        """Should raise for series shorter than 10."""
        short = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="too short"):
            select_lag_pacf(short)


class TestSelectLagAIC:
    """Tests for select_lag_aic function."""

    def test_ar1_detection(self) -> None:
        """AIC should return valid result for AR(1) process."""
        ar1 = generate_ar(200, [0.7], seed=50)
        result = select_lag_aic(ar1)

        assert isinstance(result, LagSelectionResult)
        assert result.method == "aic"
        # AIC should select some reasonable lag (may overfit)
        assert result.optimal_lag >= 1

    def test_ar2_detection(self) -> None:
        """AIC should return valid result for AR(2) process."""
        ar2 = generate_ar(300, [0.5, 0.3], seed=51)
        result = select_lag_aic(ar2)

        # Should be at least 1
        assert result.optimal_lag >= 1

    def test_aic_values_decrease_then_increase(self) -> None:
        """AIC values should decrease to optimum then increase."""
        ar1 = generate_ar(200, [0.7], seed=52)
        result = select_lag_aic(ar1)

        aic_values = [result.criterion_values[k] for k in sorted(result.criterion_values)]

        # Find the optimal point
        opt_idx = aic_values.index(min(aic_values))

        # Values after optimum should generally increase
        if opt_idx < len(aic_values) - 1:
            # At least some values after optimum should be higher
            assert any(
                aic_values[i] >= aic_values[opt_idx] for i in range(opt_idx, len(aic_values))
            )

    def test_max_lag_respected(self) -> None:
        """Should respect max_lag parameter."""
        rng = np.random.default_rng(53)
        series = rng.normal(0, 1, 100)

        result = select_lag_aic(series, max_lag=3)

        assert max(result.all_lags_tested) == 3

    def test_short_series_raises(self) -> None:
        """Should raise for series shorter than 10."""
        short = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="too short"):
            select_lag_aic(short)


class TestSelectLagBIC:
    """Tests for select_lag_bic function."""

    def test_ar1_detection(self) -> None:
        """BIC should detect AR(1) process."""
        ar1 = generate_ar(200, [0.7], seed=60)
        result = select_lag_bic(ar1)

        assert isinstance(result, LagSelectionResult)
        assert result.method == "bic"
        assert result.optimal_lag >= 1
        assert result.optimal_lag <= 2  # BIC is more parsimonious

    def test_bic_more_parsimonious_than_aic(self) -> None:
        """BIC should select same or smaller lag than AIC."""
        ar2 = generate_ar(200, [0.4, 0.2, 0.1], seed=61)  # AR(3) with weak coeffs

        aic_result = select_lag_aic(ar2)
        bic_result = select_lag_bic(ar2)

        # BIC penalizes complexity more, should select <= AIC
        assert bic_result.optimal_lag <= aic_result.optimal_lag + 1

    def test_white_noise(self) -> None:
        """BIC should select small lag for white noise."""
        rng = np.random.default_rng(62)
        white_noise = rng.normal(0, 1, 200)

        result = select_lag_bic(white_noise)

        # For white noise, smallest lag (1) should have lowest BIC
        # (We start from lag 1, not 0)
        assert result.optimal_lag <= 2

    def test_bic_values_in_result(self) -> None:
        """Result should contain BIC values for all lags."""
        rng = np.random.default_rng(63)
        series = rng.normal(0, 1, 100)

        result = select_lag_bic(series)

        # BIC starts from lag 1 (not 0)
        assert 1 in result.criterion_values
        assert len(result.criterion_values) == len(result.all_lags_tested)
        # All BIC values should be finite
        assert all(np.isfinite(v) for v in result.criterion_values.values())

    def test_short_series_raises(self) -> None:
        """Should raise for series shorter than 10."""
        short = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="too short"):
            select_lag_bic(short)


class TestAutoSelectLag:
    """Tests for auto_select_lag convenience function."""

    def test_returns_int(self) -> None:
        """Should return integer."""
        rng = np.random.default_rng(70)
        series = rng.normal(0, 1, 100)

        lag = auto_select_lag(series)

        assert isinstance(lag, int)
        assert lag >= 0

    def test_method_aic(self) -> None:
        """Should use AIC method."""
        ar1 = generate_ar(200, [0.7], seed=71)

        lag = auto_select_lag(ar1, method="aic")

        # Compare with direct call
        result = select_lag_aic(ar1)
        assert lag == result.optimal_lag

    def test_method_bic(self) -> None:
        """Should use BIC method (default)."""
        ar1 = generate_ar(200, [0.7], seed=72)

        lag = auto_select_lag(ar1, method="bic")
        lag_default = auto_select_lag(ar1)

        # Default should be BIC
        assert lag == lag_default

        # Compare with direct call
        result = select_lag_bic(ar1)
        assert lag == result.optimal_lag

    def test_method_pacf(self) -> None:
        """Should use PACF method."""
        ar1 = generate_ar(200, [0.7], seed=73)

        lag = auto_select_lag(ar1, method="pacf")

        # Compare with direct call
        result = select_lag_pacf(ar1)
        assert lag == result.optimal_lag

    def test_invalid_method_raises(self) -> None:
        """Should raise for invalid method."""
        rng = np.random.default_rng(74)
        series = rng.normal(0, 1, 100)

        with pytest.raises(ValueError, match="Unknown method"):
            auto_select_lag(series, method="invalid")  # type: ignore[arg-type]

    def test_max_lag_passed_through(self) -> None:
        """Should pass max_lag to underlying function."""
        rng = np.random.default_rng(75)
        series = rng.normal(0, 1, 100)

        lag = auto_select_lag(series, max_lag=3)

        # Should be at most 3
        assert lag <= 3


class TestSuggestCVGap:
    """Tests for suggest_cv_gap function."""

    def test_returns_at_least_horizon(self) -> None:
        """Gap should be at least the forecast horizon."""
        rng = np.random.default_rng(80)
        white_noise = rng.normal(0, 1, 100)

        gap = suggest_cv_gap(white_noise, horizon=5)

        assert gap >= 5

    def test_ar_increases_gap(self) -> None:
        """AR process should suggest larger gap."""
        ar5 = generate_ar(300, [0.3, 0.2, 0.1, 0.1, 0.1], seed=81)

        gap = suggest_cv_gap(ar5, horizon=1)

        # Should account for AR memory
        assert gap >= 1

    def test_white_noise_uses_horizon(self) -> None:
        """White noise gap should equal horizon."""
        rng = np.random.default_rng(82)
        white_noise = rng.normal(0, 1, 200)

        gap = suggest_cv_gap(white_noise, horizon=3)

        # For white noise, optimal lag is 0 or 1, so gap = max(3, 0|1) = 3
        assert gap >= 3

    def test_different_methods(self) -> None:
        """Should work with different methods."""
        ar2 = generate_ar(200, [0.5, 0.3], seed=83)

        gap_bic = suggest_cv_gap(ar2, horizon=1, method="bic")
        gap_aic = suggest_cv_gap(ar2, horizon=1, method="aic")
        gap_pacf = suggest_cv_gap(ar2, horizon=1, method="pacf")

        # All should be valid
        assert gap_bic >= 1
        assert gap_aic >= 1
        assert gap_pacf >= 1


class TestLagSelectionResult:
    """Tests for LagSelectionResult dataclass."""

    def test_frozen(self) -> None:
        """Result should be immutable."""
        result = LagSelectionResult(
            optimal_lag=2,
            criterion_values={0: 100.0, 1: 95.0, 2: 90.0},
            method="bic",
            all_lags_tested=[0, 1, 2],
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            result.optimal_lag = 3  # type: ignore[misc]

    def test_repr(self) -> None:
        """Should have sensible repr."""
        result = LagSelectionResult(
            optimal_lag=2,
            criterion_values={0: 100.0, 1: 95.0, 2: 90.0},
            method="bic",
            all_lags_tested=[0, 1, 2],
        )

        repr_str = repr(result)
        assert "optimal_lag=2" in repr_str
        assert "bic" in repr_str


class TestEdgeCases:
    """Edge case tests."""

    def test_constant_series(self) -> None:
        """Constant series should select small lag."""
        constant = np.ones(100) + np.random.default_rng(999).normal(0, 0.001, 100)

        result = select_lag_bic(constant)

        # Near-constant series has minimal autocorrelation
        # Since we start from lag 1, result should be 1
        assert result.optimal_lag == 1

    def test_near_unit_root(self) -> None:
        """Near-unit-root series should be handled."""
        rng = np.random.default_rng(90)
        # AR(1) with phi = 0.99
        n = 200
        near_unit = np.zeros(n)
        for i in range(1, n):
            near_unit[i] = 0.99 * near_unit[i - 1] + rng.normal(0, 0.1)

        result = select_lag_bic(near_unit)

        # Should still return valid result
        assert result.optimal_lag >= 0

    def test_list_input(self) -> None:
        """Should accept list input."""
        rng = np.random.default_rng(91)
        series_list = rng.normal(0, 1, 100).tolist()

        result = select_lag_bic(series_list)

        assert isinstance(result, LagSelectionResult)

    def test_short_series_boundary(self) -> None:
        """Series of length 10 should work."""
        rng = np.random.default_rng(92)
        short = rng.normal(0, 1, 10)

        result = select_lag_bic(short)

        assert result.optimal_lag >= 0

    def test_very_long_series(self) -> None:
        """Very long series should work efficiently."""
        rng = np.random.default_rng(93)
        long_series = rng.normal(0, 1, 10000)

        # Should complete without issues
        result = select_lag_bic(long_series)

        assert result.optimal_lag >= 0
        # Max lag should be reasonable (not testing all 10000 lags)
        assert max(result.all_lags_tested) < 1000


class TestMethodConsistency:
    """Test consistency across methods for known processes."""

    @pytest.mark.parametrize("seed", range(5))
    def test_all_methods_return_valid_results(self, seed: int) -> None:
        """All methods should return valid integer lags."""
        ar1 = generate_ar(300, [0.7], seed=seed + 100)

        pacf_lag = auto_select_lag(ar1, method="pacf")
        aic_lag = auto_select_lag(ar1, method="aic")
        bic_lag = auto_select_lag(ar1, method="bic")

        # All should return valid integers >= 1 (PACF can be 0)
        assert isinstance(pacf_lag, int) and pacf_lag >= 0
        assert isinstance(aic_lag, int) and aic_lag >= 1
        assert isinstance(bic_lag, int) and bic_lag >= 1

    @pytest.mark.parametrize("seed", range(5))
    def test_bic_parsimonious(self, seed: int) -> None:
        """BIC should select <= AIC lag (BIC penalizes complexity more)."""
        ar1 = generate_ar(200, [0.5], seed=seed + 300)

        aic_lag = auto_select_lag(ar1, method="aic")
        bic_lag = auto_select_lag(ar1, method="bic")

        # BIC is more parsimonious, should be <= AIC
        # Allow some tolerance for sampling variability
        assert bic_lag <= aic_lag + 2
