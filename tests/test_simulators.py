"""Tests for the generic AR/ARMA simulators (issue #11).

Covers: shape/determinism contracts, statistical sanity against closed-form
moments (AR(1)/MA(1)/ARMA(1,1) variance and lag-1 autocorrelation), fail-loud
validation, and the equality-pinned delegation from
``validators/theoretical.py``'s ``generate_ar*_series`` helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv import simulate_ar, simulate_arma
from temporalcv.validators import generate_ar1_series, generate_ar2_series
from temporalcv.validators.theoretical import _stationary_burn_in


def _lag_autocorr(x: np.ndarray, k: int) -> float:
    x = x - x.mean()
    return float(np.dot(x[k:], x[:-k]) / np.dot(x, x))


def _lag1_autocorr(x: np.ndarray) -> float:
    return _lag_autocorr(x, 1)


# ---------------------------------------------------------------------------
# Shape and determinism contracts
# ---------------------------------------------------------------------------


class TestShapeContract:
    def test_always_2d_single_path(self) -> None:
        out = simulate_arma([0.5], [0.2], n=50, rng=0)
        assert out.shape == (1, 50)

    def test_matrix_out_multi_path(self) -> None:
        out = simulate_arma([0.5], [], n=80, n_paths=7, rng=0)
        assert out.shape == (7, 80)

    def test_simulate_ar_matches_arma_no_ma(self) -> None:
        a = simulate_ar([0.6, -0.2], n=100, n_paths=2, rng=123)
        b = simulate_arma([0.6, -0.2], [], n=100, n_paths=2, rng=123)
        np.testing.assert_array_equal(a, b)

    def test_white_noise_empty_coefficients(self) -> None:
        out = simulate_arma([], [], n=1000, sigma=2.0, rng=5)
        assert out.shape == (1, 1000)
        # White noise: empirical std close to sigma.
        assert np.std(out[0]) == pytest.approx(2.0, rel=0.1)

    def test_n_one_works(self) -> None:
        assert simulate_arma([0.5], [], n=1, rng=0).shape == (1, 1)

    def test_dtype_float(self) -> None:
        out = simulate_arma([0.5], [0.1], n=10, rng=0)
        assert out.dtype == np.float64


class TestDeterminism:
    def test_same_seed_identical(self) -> None:
        a = simulate_arma([0.7], [0.3], n=200, n_paths=3, rng=42)
        b = simulate_arma([0.7], [0.3], n=200, n_paths=3, rng=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        a = simulate_arma([0.7], [0.3], n=200, rng=1)
        b = simulate_arma([0.7], [0.3], n=200, rng=2)
        assert not np.allclose(a, b)

    def test_generator_instance_accepted(self) -> None:
        a = simulate_ar([0.5], n=50, rng=np.random.default_rng(7))
        b = simulate_ar([0.5], n=50, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_paths_mutually_independent(self) -> None:
        out = simulate_ar([0.5], n=5000, n_paths=2, rng=11)
        corr = np.corrcoef(out[0], out[1])[0, 1]
        assert abs(corr) < 0.1


# ---------------------------------------------------------------------------
# Statistical sanity vs closed-form moments
# ---------------------------------------------------------------------------


class TestStatisticalSanity:
    def test_ar1_variance(self) -> None:
        phi, sigma = 0.8, 1.5
        path = simulate_ar([phi], n=20_000, sigma=sigma, rng=42)[0]
        theoretical = sigma**2 / (1 - phi**2)
        assert np.var(path) == pytest.approx(theoretical, rel=0.1)

    def test_ar1_lag1_autocorrelation(self) -> None:
        phi = 0.7
        path = simulate_ar([phi], n=20_000, rng=42)[0]
        assert _lag1_autocorr(path) == pytest.approx(phi, abs=0.03)

    def test_ma1_variance_and_acf(self) -> None:
        theta, sigma = 0.6, 1.0
        path = simulate_arma([], [theta], n=20_000, sigma=sigma, rng=42)[0]
        # gamma_0 = sigma^2 (1 + theta^2); rho_1 = theta / (1 + theta^2)
        assert np.var(path) == pytest.approx(sigma**2 * (1 + theta**2), rel=0.1)
        assert _lag1_autocorr(path) == pytest.approx(theta / (1 + theta**2), abs=0.03)

    def test_arma11_variance(self) -> None:
        phi, theta, sigma = 0.5, 0.3, 1.0
        path = simulate_arma([phi], [theta], n=20_000, sigma=sigma, rng=42)[0]
        # gamma_0 = sigma^2 (1 + 2*phi*theta + theta^2) / (1 - phi^2)
        theoretical = sigma**2 * (1 + 2 * phi * theta + theta**2) / (1 - phi**2)
        assert np.var(path) == pytest.approx(theoretical, rel=0.1)

    def test_ar2_variance(self) -> None:
        phi1, phi2, sigma = 0.5, 0.3, 1.0
        path = simulate_ar([phi1, phi2], n=20_000, sigma=sigma, rng=42)[0]
        # gamma_0 = sigma^2 (1 - phi2) / ((1 + phi2) ((1 - phi2)^2 - phi1^2))
        theoretical = sigma**2 * (1 - phi2) / ((1 + phi2) * ((1 - phi2) ** 2 - phi1**2))
        assert np.var(path) == pytest.approx(theoretical, rel=0.1)

    def test_burn_in_zero_keeps_transient(self) -> None:
        # With burn_in=0 the zero-init transient is visible: first obs is the
        # bare innovation (no history), so across many paths its variance is
        # sigma^2, well below the stationary variance.
        phi = 0.9
        out = simulate_ar([phi], n=2, n_paths=4000, burn_in=0, rng=3)
        first_var = float(np.var(out[:, 0]))
        stationary = 1.0 / (1 - phi**2)
        assert first_var == pytest.approx(1.0, rel=0.15)
        assert first_var < 0.5 * stationary

    def test_default_burn_in_reaches_stationarity(self) -> None:
        # Mirror of the burn_in=0 test: with the DEFAULT burn-in the returned
        # segment must START at (approximately) the stationary distribution —
        # ensemble variance of the first column ~ sigma^2/(1-phi^2), not
        # sigma^2. Catches "burn-in silently not discarded" regressions
        # (review finding: this mutation previously survived the suite).
        phi = 0.9
        out = simulate_ar([phi], n=2, n_paths=4000, rng=7)
        first_var = float(np.var(out[:, 0]))
        stationary = 1.0 / (1 - phi**2)
        assert first_var == pytest.approx(stationary, rel=0.15)

    def test_ar2_lag1_acf_pins_coefficient_order(self) -> None:
        # Yule-Walker: rho_1 = phi1/(1-phi2). With asymmetric mixed-sign
        # coefficients (0.6, -0.3): rho_1 = 0.4615; REVERSED coefficient
        # order would give -0.75. Pins that ar[0] is the lag-1 coefficient
        # (review finding: a reversed-order mutation previously survived).
        phi1, phi2 = 0.6, -0.3
        path = simulate_ar([phi1, phi2], n=20_000, rng=42)[0]
        rho1 = phi1 / (1 - phi2)
        assert _lag_autocorr(path, 1) == pytest.approx(rho1, abs=0.05)

    def test_ma2_acf_pins_coefficient_order(self) -> None:
        # MA(2): rho_1 = theta1(1+theta2)/(1+theta1^2+theta2^2),
        # rho_2 = theta2/(1+theta1^2+theta2^2). With (0.7, -0.4):
        # rho_1 = 0.2545, rho_2 = -0.2424; reversed order gives
        # (-0.412, +0.424). Pins that ma[0] is the lag-1 coefficient.
        theta1, theta2 = 0.7, -0.4
        path = simulate_arma([], [theta1, theta2], n=20_000, rng=42)[0]
        denom = 1 + theta1**2 + theta2**2
        assert _lag_autocorr(path, 1) == pytest.approx(theta1 * (1 + theta2) / denom, abs=0.04)
        assert _lag_autocorr(path, 2) == pytest.approx(theta2 / denom, abs=0.04)


# ---------------------------------------------------------------------------
# Fail-loud validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_nonstationary_ar1_raises(self) -> None:
        with pytest.raises(ValueError, match="non-stationary"):
            simulate_ar([1.0], n=10)

    def test_explosive_ar1_raises(self) -> None:
        with pytest.raises(ValueError, match="non-stationary"):
            simulate_ar([1.05], n=10)

    def test_nonstationary_ar2_raises(self) -> None:
        # phi1 + phi2 >= 1 violates stationarity.
        with pytest.raises(ValueError, match="non-stationary"):
            simulate_ar([0.6, 0.5], n=10)

    def test_ma_never_checked_for_stationarity(self) -> None:
        # MA(q) is stationary for any finite coefficients.
        out = simulate_arma([], [5.0, -3.0], n=50, rng=0)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("bad_sigma", [0.0, -1.0, np.nan, np.inf])
    def test_bad_sigma_raises(self, bad_sigma: float) -> None:
        with pytest.raises(ValueError, match="sigma"):
            simulate_arma([0.5], [], n=10, sigma=bad_sigma)

    def test_n_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            simulate_arma([0.5], [], n=0)

    def test_n_paths_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_paths"):
            simulate_arma([0.5], [], n=10, n_paths=0)

    def test_negative_burn_in_raises(self) -> None:
        with pytest.raises(ValueError, match="burn_in"):
            simulate_arma([0.5], [], n=10, burn_in=-1)

    def test_non_finite_coefficient_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            simulate_arma([np.nan], [], n=10)

    def test_2d_coefficients_raise(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            simulate_arma([[0.5]], [], n=10)

    def test_complex_coefficients_raise(self) -> None:
        # np.asarray(..., dtype=float) would silently DISCARD imaginary parts
        # (review finding: complex ar=[0.5+0.3j] previously simulated [0.5]).
        with pytest.raises(ValueError, match="complex"):
            simulate_arma(np.array([0.5 + 0.3j]), [], n=10)
        with pytest.raises(ValueError, match="complex"):
            simulate_arma([], np.array([0.5 + 0.3j]), n=10)

    def test_none_coefficients_raise(self) -> None:
        # np.asarray(None, dtype=float) is array(nan) — would surface as a
        # misleading "non-finite coefficients: [nan]" message.
        with pytest.raises(ValueError, match="None"):
            simulate_arma(None, [], n=10)
        with pytest.raises(ValueError, match="None"):
            simulate_arma([0.5], None, n=10)

    @pytest.mark.parametrize("bad_n", [10.5, 10.0, np.float64(10.0), True, "100"])
    def test_non_integer_n_raises_typeerror(self, bad_n: object) -> None:
        # Named TypeError instead of a nameless one from numpy internals;
        # bools and integral floats are rejected, never truncated.
        with pytest.raises(TypeError, match="n must be an integer"):
            simulate_arma([0.5], [], n=bad_n)  # type: ignore[arg-type]

    def test_non_integer_n_paths_and_burn_in_raise_typeerror(self) -> None:
        with pytest.raises(TypeError, match="n_paths must be an integer"):
            simulate_arma([0.5], [], n=10, n_paths=2.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="burn_in must be an integer"):
            simulate_arma([0.5], [], n=10, burn_in=5.5)  # type: ignore[arg-type]

    def test_overflow_raises_instead_of_returning_nonfinite(self) -> None:
        # Review finding: huge-but-finite MA coefficients or sigma previously
        # returned paths contaminated with inf/NaN, silently.
        with pytest.raises(ValueError, match="non-finite"):
            simulate_arma([], [1e308], n=1000, rng=0)
        with pytest.raises(ValueError, match="non-finite"):
            simulate_arma([0.5], [], n=50, sigma=1e308, rng=0)


# ---------------------------------------------------------------------------
# Equality-pinned delegation from validators/theoretical.py
# ---------------------------------------------------------------------------


class TestDelegationPins:
    """generate_ar*_series must be the SAME implementation as simulate_ar."""

    def test_generate_ar1_series_equals_simulate_ar(self) -> None:
        phi, sigma, n, seed = 0.9, 1.3, 250, 42
        old_api = generate_ar1_series(phi=phi, sigma=sigma, n=n, random_state=seed)
        new_api = simulate_ar([phi], n, sigma=sigma, burn_in=_stationary_burn_in(phi), rng=seed)[0]
        np.testing.assert_array_equal(old_api, new_api)

    def test_generate_ar1_series_high_phi_equals_simulate_ar(self) -> None:
        # phi=0.99 needs burn_in 688 > the simulate_ar default of 100 — pins
        # that the delegation really passes the persistence-aware burn-in.
        phi, sigma, n, seed = 0.99, 1.0, 100, 11
        old_api = generate_ar1_series(phi=phi, sigma=sigma, n=n, random_state=seed)
        new_api = simulate_ar([phi], n, sigma=sigma, burn_in=_stationary_burn_in(phi), rng=seed)[0]
        np.testing.assert_array_equal(old_api, new_api)
        assert _stationary_burn_in(phi) > 100

    def test_generate_ar2_series_equals_simulate_ar(self) -> None:
        phi1, phi2, sigma, n, seed = 0.5, 0.2, 0.8, 250, 7
        old_api = generate_ar2_series(phi1=phi1, phi2=phi2, sigma=sigma, n=n, random_state=seed)
        dominant = float(np.max(np.abs(np.roots([1.0, -phi1, -phi2]))))
        new_api = simulate_ar(
            [phi1, phi2], n, sigma=sigma, burn_in=_stationary_burn_in(dominant), rng=seed
        )[0]
        np.testing.assert_array_equal(old_api, new_api)

    def test_generate_ar1_series_near_unit_root_is_stationary(self) -> None:
        # Review finding: with a fixed burn-in of 100, phi=0.99 draws started
        # at ~87% of stationary variance (and phi=0.999 at ~18%). The
        # persistence-aware burn-in must restore approximate stationary init.
        phi = 0.99
        stationary_var = 1.0 / (1 - phi**2)
        first_obs = np.array(
            [generate_ar1_series(phi=phi, sigma=1.0, n=1, random_state=s)[0] for s in range(800)]
        )
        assert float(np.var(first_obs)) == pytest.approx(stationary_var, rel=0.2)

    def test_stationary_burn_in_helper(self) -> None:
        assert _stationary_burn_in(0.0) == 100
        assert _stationary_burn_in(0.9) == 100  # needed 66 < floor 100
        assert _stationary_burn_in(0.99) == 688
        assert _stationary_burn_in(0.99999) == 100_000  # capped

    def test_generate_ar1_series_validation_preserved(self) -> None:
        with pytest.raises(ValueError, match="phi"):
            generate_ar1_series(phi=1.0, sigma=1.0, n=100)
        with pytest.raises(ValueError, match="sigma"):
            generate_ar1_series(phi=0.5, sigma=0.0, n=100)
        with pytest.raises(ValueError, match="n must be"):
            generate_ar1_series(phi=0.5, sigma=1.0, n=0)

    def test_generate_ar2_series_validation_preserved(self) -> None:
        with pytest.raises(ValueError, match="stationarity"):
            generate_ar2_series(phi1=0.6, phi2=0.5, sigma=1.0, n=100)
        with pytest.raises(ValueError, match="n must be"):
            generate_ar2_series(phi1=0.3, phi2=0.2, sigma=1.0, n=1)
