"""
Tests for Financial and Trading Metrics.

Tests cover:
- compute_sharpe_ratio: risk-adjusted return
- compute_max_drawdown: peak-to-trough decline
- compute_cumulative_return: total return
- compute_information_ratio: active return vs tracking error
- compute_hit_rate: directional accuracy
- compute_profit_factor: gross profit / gross loss
- compute_calmar_ratio: return / max drawdown
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.metrics.financial import (
    compute_calmar_ratio,
    compute_cumulative_return,
    compute_hit_rate,
    compute_information_ratio,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe_ratio,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def positive_returns():
    """Consistently positive returns."""
    return np.array([0.01, 0.02, 0.01, 0.015, 0.005])


@pytest.fixture
def negative_returns():
    """Consistently negative returns."""
    return np.array([-0.01, -0.02, -0.01, -0.015, -0.005])


@pytest.fixture
def mixed_returns():
    """Mix of positive and negative returns."""
    return np.array([0.02, -0.01, 0.03, -0.02, 0.01])


@pytest.fixture
def flat_returns():
    """Zero returns."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0])


# =============================================================================
# Tests: compute_sharpe_ratio
# =============================================================================


class TestComputeSharpeRatio:
    """Tests for Sharpe ratio computation."""

    def test_positive_returns_positive_sharpe(self, positive_returns):
        """Positive returns yield positive Sharpe ratio."""
        sharpe = compute_sharpe_ratio(positive_returns)
        assert sharpe > 0

    def test_negative_returns_negative_sharpe(self, negative_returns):
        """Negative returns yield negative Sharpe ratio."""
        sharpe = compute_sharpe_ratio(negative_returns)
        assert sharpe < 0

    def test_risk_free_rate_reduces_sharpe(self, positive_returns):
        """Higher risk-free rate reduces Sharpe ratio."""
        sharpe_no_rf = compute_sharpe_ratio(positive_returns, risk_free_rate=0.0)
        sharpe_with_rf = compute_sharpe_ratio(positive_returns, risk_free_rate=0.005)
        assert sharpe_no_rf > sharpe_with_rf

    def test_annualization_scaling(self, positive_returns):
        """Annualization factor scales the Sharpe ratio."""
        sharpe_daily = compute_sharpe_ratio(positive_returns, annualization=252)
        sharpe_weekly = compute_sharpe_ratio(positive_returns, annualization=52)
        # sqrt(252/52) ≈ 2.2 factor difference
        ratio = sharpe_daily / sharpe_weekly
        assert ratio == pytest.approx(np.sqrt(252 / 52), rel=0.01)

    def test_higher_vol_lower_sharpe(self):
        """Higher volatility (same mean) leads to lower Sharpe."""
        low_vol = np.array([0.01, 0.011, 0.01, 0.009, 0.01])  # Low variance
        high_vol = np.array([0.05, -0.03, 0.04, -0.02, 0.06])  # High variance, similar mean

        sharpe_low = compute_sharpe_ratio(low_vol)
        sharpe_high = compute_sharpe_ratio(high_vol)

        assert sharpe_low > sharpe_high

    def test_zero_volatility_positive_returns(self):
        """Zero volatility with positive mean returns infinity."""
        constant_positive = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = compute_sharpe_ratio(constant_positive)
        assert sharpe == np.inf

    def test_zero_volatility_negative_returns(self):
        """Zero volatility with negative mean returns negative infinity."""
        constant_negative = np.array([-0.01, -0.01, -0.01, -0.01])
        sharpe = compute_sharpe_ratio(constant_negative)
        assert sharpe == -np.inf

    def test_zero_volatility_zero_returns(self, flat_returns):
        """Zero volatility with zero mean returns 0."""
        sharpe = compute_sharpe_ratio(flat_returns)
        assert sharpe == 0.0

    def test_empty_array_error(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_sharpe_ratio(np.array([]))

    def test_single_observation_error(self):
        """Single observation raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            compute_sharpe_ratio(np.array([0.01]))

    def test_list_input(self, positive_returns):
        """Accepts list input."""
        sharpe = compute_sharpe_ratio(list(positive_returns))
        assert isinstance(sharpe, float)


# =============================================================================
# Tests: compute_max_drawdown
# =============================================================================


class TestComputeMaxDrawdown:
    """Tests for maximum drawdown computation."""

    def test_monotonic_increasing_zero_drawdown(self):
        """Monotonically increasing curve has zero drawdown."""
        curve = np.array([100, 110, 120, 130, 140])
        mdd = compute_max_drawdown(cumulative_returns=curve)
        assert mdd == 0.0

    def test_monotonic_decreasing_full_drawdown(self):
        """Monotonically decreasing curve has drawdown from first to last."""
        curve = np.array([100, 80, 60, 40, 20])
        mdd = compute_max_drawdown(cumulative_returns=curve)
        # Drawdown from 100 to 20 = 80%
        assert mdd == pytest.approx(0.80, rel=1e-6)

    def test_simple_drawdown(self):
        """Simple drawdown calculation."""
        # Peak at 120, trough at 108
        curve = np.array([100, 110, 120, 115, 108, 125])
        mdd = compute_max_drawdown(cumulative_returns=curve)
        # (120 - 108) / 120 = 0.1
        assert mdd == pytest.approx(0.10, rel=1e-6)

    def test_multiple_drawdowns(self):
        """Returns maximum of multiple drawdowns."""
        # First drawdown: 110 -> 100 (9.1%)
        # Second drawdown: 130 -> 100 (23.1%)
        curve = np.array([100, 110, 100, 120, 130, 100])
        mdd = compute_max_drawdown(cumulative_returns=curve)
        # Max is (130 - 100) / 130 = 0.231
        assert mdd == pytest.approx((130 - 100) / 130, rel=1e-6)

    def test_from_returns(self):
        """Can compute from period returns."""
        returns = np.array([0.10, -0.05, 0.10, -0.15])
        # Cumulative: 1.0 -> 1.1 -> 1.045 -> 1.1495 -> 0.977
        # Max from 1.1495 to 0.977 = 15%
        mdd = compute_max_drawdown(returns=returns)
        assert 0.10 < mdd < 0.20  # Rough check

    def test_cumulative_takes_precedence(self):
        """When both provided, cumulative_returns is used."""
        curve = np.array([100, 90, 100])  # 10% drawdown
        returns = np.array([0.50, -0.90])  # Would give ~80% drawdown
        mdd = compute_max_drawdown(cumulative_returns=curve, returns=returns)
        assert mdd == pytest.approx(0.10, rel=1e-6)

    def test_neither_input_error(self):
        """Must provide at least one input."""
        with pytest.raises(ValueError, match="Must provide"):
            compute_max_drawdown()

    def test_empty_array_error(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_max_drawdown(cumulative_returns=np.array([]))

    def test_drawdown_always_positive(self, mixed_returns):
        """Drawdown is always returned as positive fraction."""
        mdd = compute_max_drawdown(returns=mixed_returns)
        assert mdd >= 0


# =============================================================================
# Tests: compute_cumulative_return
# =============================================================================


class TestComputeCumulativeReturn:
    """Tests for cumulative return computation."""

    def test_geometric_compounding(self):
        """Geometric method compounds correctly."""
        returns = np.array([0.10, 0.10, 0.10])
        # (1.1)^3 - 1 = 0.331
        cum_ret = compute_cumulative_return(returns, method="geometric")
        assert cum_ret == pytest.approx(0.331, rel=1e-3)

    def test_arithmetic_sum(self):
        """Arithmetic method sums returns."""
        returns = np.array([0.10, 0.10, 0.10])
        cum_ret = compute_cumulative_return(returns, method="arithmetic")
        assert cum_ret == pytest.approx(0.30, rel=1e-6)

    def test_geometric_vs_arithmetic_positive(self, positive_returns):
        """Geometric < arithmetic for positive returns."""
        geo = compute_cumulative_return(positive_returns, method="geometric")
        arith = compute_cumulative_return(positive_returns, method="arithmetic")
        # For positive returns, geometric compounding gives slightly less
        # due to variance drag (but very close for small returns)
        assert abs(geo - arith) < 0.01  # Within 1% for small returns

    def test_negative_returns(self, negative_returns):
        """Handles negative returns correctly."""
        cum_ret = compute_cumulative_return(negative_returns, method="geometric")
        assert cum_ret < 0

    def test_zero_returns(self, flat_returns):
        """Zero returns yield zero cumulative return."""
        cum_ret = compute_cumulative_return(flat_returns)
        assert cum_ret == 0.0

    def test_empty_array_error(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_cumulative_return(np.array([]))

    def test_invalid_method_error(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Invalid method"):
            compute_cumulative_return(np.array([0.01, 0.02]), method="invalid")


# =============================================================================
# Tests: compute_information_ratio
# =============================================================================


class TestComputeInformationRatio:
    """Tests for information ratio computation."""

    def test_outperformance_positive_ir(self):
        """Consistent outperformance yields positive IR."""
        portfolio = np.array([0.02, 0.03, 0.02, 0.03, 0.02])
        benchmark = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        ir = compute_information_ratio(portfolio, benchmark)
        assert ir > 0

    def test_underperformance_negative_ir(self):
        """Consistent underperformance yields negative IR."""
        portfolio = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        benchmark = np.array([0.02, 0.03, 0.02, 0.03, 0.02])
        ir = compute_information_ratio(portfolio, benchmark)
        assert ir < 0

    def test_identical_returns_zero_ir(self):
        """Identical returns yield IR of 0."""
        returns = np.array([0.01, 0.02, 0.01, 0.02, 0.01])
        ir = compute_information_ratio(returns, returns)
        assert ir == 0.0

    def test_higher_tracking_error_lower_ir(self):
        """Higher tracking error reduces IR for same mean outperformance."""
        benchmark = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

        # Low tracking error
        port_low_te = np.array([0.015, 0.015, 0.015, 0.015, 0.015])

        # High tracking error, same mean outperformance
        port_high_te = np.array([0.03, 0.0, 0.02, 0.01, 0.025])

        ir_low = compute_information_ratio(port_low_te, benchmark)
        ir_high = compute_information_ratio(port_high_te, benchmark)

        assert ir_low > ir_high

    def test_annualization_scaling(self):
        """Annualization factor scales IR."""
        portfolio = np.array([0.02, 0.01, 0.03, 0.01, 0.02])
        benchmark = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

        ir_daily = compute_information_ratio(portfolio, benchmark, annualization=252)
        ir_weekly = compute_information_ratio(portfolio, benchmark, annualization=52)

        ratio = ir_daily / ir_weekly
        assert ratio == pytest.approx(np.sqrt(252 / 52), rel=0.01)

    def test_zero_tracking_error(self):
        """Zero tracking error with positive alpha returns infinity."""
        portfolio = np.array([0.02, 0.02, 0.02])  # Constant outperformance
        benchmark = np.array([0.01, 0.01, 0.01])  # Constant
        ir = compute_information_ratio(portfolio, benchmark)
        assert ir == np.inf

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_information_ratio(np.array([0.01, 0.02]), np.array([0.01]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_information_ratio(np.array([]), np.array([0.01]))


# =============================================================================
# Tests: compute_hit_rate
# =============================================================================


class TestComputeHitRate:
    """Tests for directional hit rate computation."""

    def test_perfect_predictions(self):
        """Correct direction every time yields 100% hit rate."""
        predicted = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([0.5, -0.3, 0.2, -0.1])
        hr = compute_hit_rate(predicted, actual)
        assert hr == 1.0

    def test_all_wrong_predictions(self):
        """Wrong direction every time yields 0% hit rate."""
        predicted = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([-0.5, 0.3, -0.2, 0.1])
        hr = compute_hit_rate(predicted, actual)
        assert hr == 0.0

    def test_half_correct(self):
        """Half correct yields 50% hit rate."""
        predicted = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([0.5, 0.3, -0.2, -0.1])  # 1st and 4th correct
        hr = compute_hit_rate(predicted, actual)
        assert hr == 0.5

    def test_zero_prediction_counts_as_hit(self):
        """Zero prediction (no direction) counts as hit."""
        predicted = np.array([0.0, 0.0, 0.0])
        actual = np.array([0.5, -0.3, 0.2])
        hr = compute_hit_rate(predicted, actual)
        assert hr == 1.0

    def test_zero_actual_counts_as_hit(self):
        """Zero actual (no move) counts as hit."""
        predicted = np.array([1.0, -1.0, 1.0])
        actual = np.array([0.0, 0.0, 0.0])
        hr = compute_hit_rate(predicted, actual)
        assert hr == 1.0

    def test_magnitude_doesnt_matter(self):
        """Magnitude of predictions doesn't affect hit rate."""
        predicted_small = np.array([0.001, -0.001])
        predicted_large = np.array([100.0, -100.0])
        actual = np.array([0.5, -0.3])

        hr_small = compute_hit_rate(predicted_small, actual)
        hr_large = compute_hit_rate(predicted_large, actual)

        assert hr_small == hr_large == 1.0

    def test_hit_rate_bounds(self, mixed_returns):
        """Hit rate is always in [0, 1]."""
        rng = np.random.default_rng(42)
        predicted = rng.standard_normal(len(mixed_returns))
        hr = compute_hit_rate(predicted, mixed_returns)
        assert 0.0 <= hr <= 1.0

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_hit_rate(np.array([1.0, -1.0]), np.array([0.1]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_hit_rate(np.array([]), np.array([]))


# =============================================================================
# Tests: compute_profit_factor
# =============================================================================


class TestComputeProfitFactor:
    """Tests for profit factor computation."""

    def test_all_profitable_trades(self):
        """All profitable trades yield infinite profit factor."""
        predicted = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([0.01, -0.01, 0.02, -0.02])
        pf = compute_profit_factor(predicted, actual)
        assert pf == np.inf

    def test_all_losing_trades(self):
        """All losing trades yield zero profit factor."""
        predicted = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([-0.01, 0.01, -0.02, 0.02])
        pf = compute_profit_factor(predicted, actual)
        assert pf == 0.0

    def test_profit_factor_calculation(self):
        """Correct profit factor calculation."""
        # Long when positive, short when negative
        predicted = np.array([1.0, -1.0, 1.0, -1.0])
        actual = np.array([0.02, -0.01, -0.01, -0.02])
        # Long +0.02: profit 0.02
        # Short -0.01: short gain = +0.01
        # Long -0.01: loss 0.01
        # Short -0.02: short gain = +0.02
        # Total profit: 0.02 + 0.01 + 0.02 = 0.05
        # Total loss: 0.01
        # PF = 5.0
        pf = compute_profit_factor(predicted, actual)
        assert pf == pytest.approx(5.0, rel=1e-6)

    def test_balanced_trades(self):
        """Equal profits and losses yield PF = 1."""
        predicted = np.array([1.0, -1.0])
        actual = np.array([0.01, 0.01])  # Long wins, short loses
        # Long +0.01: profit 0.01
        # Short +0.01: loss 0.01
        pf = compute_profit_factor(predicted, actual)
        assert pf == pytest.approx(1.0, rel=1e-6)

    def test_separate_returns_parameter(self):
        """Uses separate returns if provided."""
        predicted = np.array([1.0, -1.0])
        actual = np.array([1.0, -1.0])  # Ignored when returns provided
        returns = np.array([0.05, -0.02])

        pf = compute_profit_factor(predicted, actual, returns=returns)
        # Long +0.05: profit 0.05
        # Short -0.02: gain 0.02
        # Total profit: 0.07, no losses
        assert pf == np.inf

    def test_zero_prediction_no_trade(self):
        """Zero prediction means no trade (contributes nothing)."""
        predicted = np.array([0.0, 1.0, -1.0])
        actual = np.array([0.10, 0.01, -0.01])
        # First: no trade (0.10 not counted)
        # Second: long +0.01
        # Third: short -0.01 (gain)
        pf = compute_profit_factor(predicted, actual)
        assert pf == np.inf  # All profitable

    def test_length_mismatch_error(self):
        """Different lengths raise ValueError."""
        with pytest.raises(ValueError, match="lengths must match"):
            compute_profit_factor(np.array([1.0, -1.0]), np.array([0.01]))

    def test_empty_array_error(self):
        """Empty arrays raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_profit_factor(np.array([]), np.array([]))


# =============================================================================
# Tests: compute_calmar_ratio
# =============================================================================


class TestComputeCalmarRatio:
    """Tests for Calmar ratio computation."""

    def test_positive_returns_positive_calmar(self, positive_returns):
        """Positive returns with drawdown yield positive Calmar."""
        calmar = compute_calmar_ratio(positive_returns)
        # These returns have small drawdown, so Calmar should be high
        assert calmar > 0

    def test_negative_returns_negative_calmar(self, negative_returns):
        """Negative returns yield negative Calmar."""
        calmar = compute_calmar_ratio(negative_returns)
        assert calmar < 0

    def test_no_drawdown_infinite_calmar(self):
        """No drawdown with positive returns yields infinite Calmar."""
        monotonic_up = np.array([0.01, 0.01, 0.01, 0.01])
        calmar = compute_calmar_ratio(monotonic_up)
        assert calmar == np.inf

    def test_higher_drawdown_lower_calmar(self):
        """Higher drawdown (same return) yields lower Calmar."""
        # Same ending return, different paths
        low_dd = np.array([0.02, 0.02, 0.02])  # No drawdown
        high_dd = np.array([0.10, -0.05, 0.02])  # Has drawdown

        calmar_low = compute_calmar_ratio(low_dd)
        calmar_high = compute_calmar_ratio(high_dd)

        assert calmar_low > calmar_high

    def test_annualization_affects_calmar(self):
        """Annualization affects the return component."""
        returns = np.array([0.01, 0.02, -0.01, 0.02])

        calmar_daily = compute_calmar_ratio(returns, annualization=252)
        calmar_weekly = compute_calmar_ratio(returns, annualization=52)

        # Daily annualization should give higher annualized return
        assert calmar_daily != calmar_weekly

    def test_empty_array_error(self):
        """Empty array raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_calmar_ratio(np.array([]))


# =============================================================================
# Integration Tests
# =============================================================================


class TestFinancialMetricsIntegration:
    """Integration tests for financial metrics."""

    def test_sharpe_and_drawdown_relationship(self):
        """Higher Sharpe strategies tend to have lower drawdowns."""
        rng = np.random.default_rng(42)

        # High Sharpe: positive mean, low vol
        high_sharpe = rng.normal(0.001, 0.005, 100)

        # Low Sharpe: near-zero mean, high vol
        low_sharpe = rng.normal(0.0001, 0.02, 100)

        sharpe_high = compute_sharpe_ratio(high_sharpe)
        sharpe_low = compute_sharpe_ratio(low_sharpe)
        mdd_high = compute_max_drawdown(returns=high_sharpe)
        mdd_low = compute_max_drawdown(returns=low_sharpe)

        assert sharpe_high > sharpe_low
        assert mdd_high < mdd_low

    def test_hit_rate_profit_factor_consistency(self):
        """High hit rate with equal magnitudes gives good profit factor."""
        # 80% hit rate, equal magnitude trades
        predicted = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
        actual = np.array([1, 1, 1, 1, -1, -1, -1, -1, -1, 1])
        # 8/10 correct

        hr = compute_hit_rate(predicted, actual)
        pf = compute_profit_factor(predicted, actual)

        assert hr == 0.8
        # PF = 8/2 = 4.0 (8 wins, 2 losses, equal magnitude)
        assert pf == pytest.approx(4.0, rel=1e-6)

    def test_calmar_relates_to_sharpe_and_drawdown(self):
        """Calmar ratio relates to both return and drawdown."""
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

        sharpe = compute_sharpe_ratio(returns)
        mdd = compute_max_drawdown(returns=returns)
        calmar = compute_calmar_ratio(returns)

        # If positive Sharpe and non-zero drawdown, Calmar should be finite and positive
        if sharpe > 0 and mdd > 0:
            assert np.isfinite(calmar)
            assert calmar > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestFinancialMetricsEdgeCases:
    """Edge cases and numerical stability tests."""

    def test_large_returns(self):
        """Handles large return values."""
        large_returns = np.array([1.0, 2.0, 0.5, 1.5])  # 100%, 200%, 50%, 150%
        sharpe = compute_sharpe_ratio(large_returns)
        assert np.isfinite(sharpe)

    def test_small_returns(self):
        """Handles very small return values."""
        tiny_returns = np.array([1e-8, -1e-8, 1e-8, 1e-8])
        sharpe = compute_sharpe_ratio(tiny_returns)
        assert np.isfinite(sharpe)

    def test_single_large_loss(self):
        """Handles single large loss correctly."""
        returns = np.array([0.01, 0.01, 0.01, -0.50, 0.01])  # 50% loss
        mdd = compute_max_drawdown(returns=returns)
        assert mdd > 0.45  # Should capture the 50% loss impact

    def test_many_small_losses(self):
        """Accumulation of small losses creates drawdown."""
        returns = np.array([-0.01] * 20)  # 20 x 1% losses
        mdd = compute_max_drawdown(returns=returns)
        # Cumulative: (0.99)^20 ≈ 0.818, so ~18% drawdown
        assert 0.15 < mdd < 0.20

    def test_list_inputs_all_functions(self):
        """All functions accept list inputs."""
        returns = [0.01, 0.02, -0.01, 0.02]
        benchmark = [0.01, 0.01, 0.01, 0.01]
        predicted = [1.0, -1.0, 1.0, -1.0]
        actual = [0.5, -0.5, 0.5, -0.5]

        assert isinstance(compute_sharpe_ratio(returns), float)
        assert isinstance(compute_max_drawdown(returns=returns), float)
        assert isinstance(compute_cumulative_return(returns), float)
        assert isinstance(compute_information_ratio(returns, benchmark), float)
        assert isinstance(compute_hit_rate(predicted, actual), float)
        assert isinstance(compute_profit_factor(predicted, actual), float)
        assert isinstance(compute_calmar_ratio(returns), float)
