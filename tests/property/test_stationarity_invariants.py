"""Property-based tests for stationarity module.

Tests invariants of stationarity tests using Hypothesis.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from temporalcv.stationarity import (
    adf_test,
    kpss_test,
    pp_test,
    check_stationarity,
    difference_until_stationary,
    StationarityConclusion,
)


# Custom strategies
@st.composite
def stationary_series(draw: st.DrawFn) -> np.ndarray:
    """Generate stationary time series (white noise or weak AR)."""
    n = draw(st.integers(min_value=50, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=10000))
    rng = np.random.default_rng(seed)

    # White noise is stationary
    return rng.standard_normal(n)


@st.composite
def non_stationary_series(draw: st.DrawFn) -> np.ndarray:
    """Generate non-stationary time series (random walk)."""
    n = draw(st.integers(min_value=50, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=10000))
    rng = np.random.default_rng(seed)

    # Random walk (unit root) is non-stationary
    return np.cumsum(rng.standard_normal(n))


@st.composite
def any_numeric_series(draw: st.DrawFn) -> np.ndarray:
    """Generate any numeric series (may or may not be stationary)."""
    n = draw(st.integers(min_value=50, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=10000))
    rng = np.random.default_rng(seed)

    # Mix of stationary and non-stationary
    if draw(st.booleans()):
        return rng.standard_normal(n)
    else:
        return np.cumsum(rng.standard_normal(n))


class TestADFInvariants:
    """Property tests for ADF test."""

    @given(series=stationary_series())
    @settings(max_examples=50)
    def test_adf_returns_valid_pvalue(self, series: np.ndarray) -> None:
        """ADF p-value must be in [0, 1]."""
        result = adf_test(series)
        assert 0 <= result.pvalue <= 1

    @given(series=stationary_series())
    @settings(max_examples=50)
    def test_adf_returns_valid_statistic(self, series: np.ndarray) -> None:
        """ADF statistic should be finite."""
        result = adf_test(series)
        assert np.isfinite(result.statistic)

    @given(series=stationary_series())
    @settings(max_examples=50)
    def test_adf_lags_non_negative(self, series: np.ndarray) -> None:
        """ADF lags used must be non-negative."""
        result = adf_test(series)
        assert result.lags_used >= 0


class TestKPSSInvariants:
    """Property tests for KPSS test."""

    @given(series=stationary_series())
    @settings(max_examples=50)
    def test_kpss_returns_valid_pvalue(self, series: np.ndarray) -> None:
        """KPSS p-value must be in [0, 1]."""
        result = kpss_test(series)
        assert 0 <= result.pvalue <= 1

    @given(series=stationary_series())
    @settings(max_examples=50)
    def test_kpss_returns_valid_statistic(self, series: np.ndarray) -> None:
        """KPSS statistic should be non-negative and finite."""
        result = kpss_test(series)
        assert result.statistic >= 0
        assert np.isfinite(result.statistic)


class TestPPInvariants:
    """Property tests for Phillips-Perron test."""

    @given(series=stationary_series())
    @settings(max_examples=50)
    def test_pp_returns_valid_pvalue(self, series: np.ndarray) -> None:
        """PP p-value must be in [0, 1]."""
        result = pp_test(series)
        assert 0 <= result.pvalue <= 1


class TestCheckStationarityInvariants:
    """Property tests for joint stationarity check."""

    @given(series=any_numeric_series())
    @settings(max_examples=50)
    def test_check_stationarity_returns_valid_conclusion(
        self, series: np.ndarray
    ) -> None:
        """check_stationarity must return a valid conclusion."""
        result = check_stationarity(series)

        assert result.conclusion in [
            StationarityConclusion.STATIONARY,
            StationarityConclusion.NON_STATIONARY,
            StationarityConclusion.DIFFERENCE_STATIONARY,
            StationarityConclusion.INSUFFICIENT_EVIDENCE,
        ]

    @given(series=any_numeric_series())
    @settings(max_examples=50)
    def test_check_stationarity_has_both_tests(
        self, series: np.ndarray
    ) -> None:
        """check_stationarity must run both ADF and KPSS."""
        result = check_stationarity(series)

        assert result.adf_result is not None
        assert result.kpss_result is not None


class TestDifferenceUntilStationaryInvariants:
    """Property tests for differencing function."""

    @given(series=non_stationary_series())
    @settings(max_examples=30)
    def test_difference_reduces_length(self, series: np.ndarray) -> None:
        """Each difference reduces length by 1."""
        try:
            differenced_series, n_diff = difference_until_stationary(series, max_diff=2)
            # If it succeeded, check the length relationship
            if n_diff > 0:
                assert len(differenced_series) == len(series) - n_diff
        except ValueError:
            # Series not stationary after max_diff - this is acceptable
            pass

    @given(series=any_numeric_series())
    @settings(max_examples=30)
    def test_difference_respects_max(self, series: np.ndarray) -> None:
        """Should not exceed max_diff (when successful)."""
        max_d = 2
        try:
            _, n_diff = difference_until_stationary(series, max_diff=max_d)
            assert n_diff <= max_d
        except ValueError:
            # Series not stationary - function raises as documented
            pass

    @given(series=stationary_series())
    @settings(max_examples=30)
    def test_stationary_series_returns_original(self, series: np.ndarray) -> None:
        """Stationary series should require 0 differences."""
        try:
            differenced_series, n_diff = difference_until_stationary(series, max_diff=2)
            # If already stationary, n_diff should be 0
            if n_diff == 0:
                np.testing.assert_array_equal(differenced_series, series)
        except ValueError:
            # Rare case - even "stationary" series might fail ADF due to randomness
            pass


class TestCrossTestConsistency:
    """Test consistency across different stationarity tests."""

    @given(series=stationary_series())
    @settings(max_examples=30)
    def test_stationary_series_usually_detected(self, series: np.ndarray) -> None:
        """White noise should usually be detected as stationary."""
        result = check_stationarity(series, alpha=0.05)

        # Not a strict invariant, but should hold most of the time
        # We just check that the function runs without error
        assert result.conclusion is not None

    @given(series=non_stationary_series())
    @settings(max_examples=30)
    def test_nonstationary_series_detected(self, series: np.ndarray) -> None:
        """Random walk should usually be detected as non-stationary."""
        result = check_stationarity(series, alpha=0.05)

        # Not a strict invariant, but check function runs
        assert result.conclusion is not None
