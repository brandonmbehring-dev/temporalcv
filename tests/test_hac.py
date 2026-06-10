"""Tests for the HAC module (issue #9) — port of dml_ts/dml/hac.py.

Four layers:

1. **Golden parity** — reference values captured from the LIVE
   ``dml_ts/dml/hac.py`` at port time (2026-06-10) on deterministic inputs
   reproduced exactly below, across a kernel x bandwidth x mode grid
   (``prewhiten=False`` only: dml_ts's prewhitening path was broken — the
   X-mode crashed on a length mismatch and mean-mode never recolored).
2. **Correctness** — closed-form/statistical checks the port must satisfy
   independently of dml_ts (long-run variance of AR(1), prewhitening
   recoloring, matrix/scalar consistency).
3. **Semantics pins** — the dml_ts issue #7 lesson: ``se`` is the SE of the
   mean (sd/sqrt(n) scale, never sd/n).
4. **Fail-loud validation + HACResult contract.**
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from temporalcv import (
    HACResult,
    bartlett_kernel,
    long_run_covariance,
    newey_west_covariance,
    newey_west_se,
    optimal_bandwidth,
    parzen_kernel,
    quadratic_spectral_kernel,
)

# ---------------------------------------------------------------------------
# Deterministic inputs shared with the golden capture (do not change)
# ---------------------------------------------------------------------------


def _reference_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260610)
    raw = rng.standard_normal(120)
    e = raw.copy()
    for t in range(1, 120):
        e[t] = 0.6 * e[t - 1] + raw[t]
    x = rng.standard_normal((120, 3))
    x[:, 0] = 1.0
    return e, x


# Captured from live dml_ts.dml.hac (2026-06-10, prewhiten=False paths).
MEAN_SE = {
    ("bartlett", 0): 0.10032791004187357,
    ("bartlett", 3): 0.13660142828892133,
    ("bartlett", 8): 0.12977839788626092,
    ("bartlett", "auto"): 0.1378442009662154,
    ("parzen", 0): 0.10032791004187357,
    ("parzen", 3): 0.13225835167712013,
    ("parzen", 8): 0.1404982764734066,
    ("parzen", "auto"): 0.13750385651955688,
    ("quadratic_spectral", 0): 0.10032791004187357,
    ("quadratic_spectral", 3): 0.14422043055056347,
    ("quadratic_spectral", 8): 0.12751612349020516,
    ("quadratic_spectral", "auto"): 0.14375098129562086,
}

SANDWICH = {
    ("bartlett", 3): (0.01947759516991687, 0.010304215620899178, 0.0010993099345878432),
    ("bartlett", 8): (0.01866561767087748, 0.008974488125181397, 0.0006749068331288481),
    ("bartlett", "auto"): (0.020021395957173428, 0.009633666365067994, 0.001117480092628435),
    ("parzen", 3): (0.01810821319967461, 0.010437935525936424, 0.0010653429708289378),
    ("parzen", 8): (0.021121158880964965, 0.009157022511774679, 0.0010909094585914957),
    ("parzen", "auto"): (0.019677177388756302, 0.010260844445381405, 0.0011308363407253692),
    ("quadratic_spectral", 3): (
        0.021822718583634368,
        0.009997620908121358,
        0.0012571326770125118,
    ),
    ("quadratic_spectral", 8): (
        0.018571876348847084,
        0.00855421042876355,
        0.0004372936079772233,
    ),
    ("quadratic_spectral", "auto"): (
        0.021950242535339025,
        0.008988800930481938,
        0.0012255750094552129,
    ),
}

BANDWIDTHS = {
    ("newey_west", "bartlett"): 4,
    ("newey_west", "parzen"): 4,
    ("newey_west", "quadratic_spectral"): 4,
    # andrews+bartlett is a DELIBERATE deviation from dml_ts (which gave 11):
    # Andrews (1991) order-1 kernels use alpha(1) = 4 rho^2/(1-rho^2)^2, not
    # the alpha(2) dml_ts applied to all kernels (module deviation #6).
    ("andrews", "bartlett"): 6,
    ("andrews", "parzen"): 10,
    ("andrews", "quadratic_spectral"): 5,
}

KERNEL_VALUES = {
    ("bartlett", 0, 10): 1.0,
    ("bartlett", 3, 10): 0.7272727272727273,
    ("bartlett", 5, 10): 0.5454545454545454,
    ("bartlett", 8, 10): 0.2727272727272727,
    ("bartlett", 12, 10): 0.0,
    ("bartlett", 2, 0): 0.0,
    ("parzen", 0, 10): 1.0,
    ("parzen", 3, 10): 0.6754320060105184,
    ("parzen", 5, 10): 0.32381667918858004,
    ("parzen", 8, 10): 0.04057099924868519,
    ("parzen", 12, 10): 0.0,
    ("parzen", 2, 0): 0.0,
    ("quadratic_spectral", 0, 10): 1.0,
    ("quadratic_spectral", 3, 10): 0.8982029895904742,
    ("quadratic_spectral", 5, 10): 0.7355336415145157,
    ("quadratic_spectral", 8, 10): 0.4242671609882777,
    ("quadratic_spectral", 12, 10): 0.06451450295420841,
    ("quadratic_spectral", 2, 0): 0.0,
}

FULL_COV_B5_BARTLETT = [
    [0.01986973265296556, 0.006031868943986624, 0.0010288830301255047],
    [0.006031868943986624, 0.009294935917413706, -0.001827449848995503],
    [0.0010288830301255047, -0.001827449848995503, 0.009775964231249736],
]

_KERNEL_FUNCS = {
    "bartlett": bartlett_kernel,
    "parzen": parzen_kernel,
    "quadratic_spectral": quadratic_spectral_kernel,
}


class TestGoldenParityVsDmlTs:
    """Value parity with the live dml_ts implementation at port time."""

    @pytest.mark.parametrize(("kernel", "bw"), sorted(MEAN_SE, key=str))
    def test_mean_mode_se(self, kernel: str, bw: int | str) -> None:
        e, _ = _reference_inputs()
        result = newey_west_se(e, bandwidth=bw, kernel=kernel)  # type: ignore[arg-type]
        assert result.se == pytest.approx(MEAN_SE[(kernel, bw)], rel=1e-10)

    @pytest.mark.parametrize(("kernel", "bw"), sorted(SANDWICH, key=str))
    def test_sandwich_covariance(self, kernel: str, bw: int | str) -> None:
        e, x = _reference_inputs()
        result = newey_west_covariance(e, x, bandwidth=bw, kernel=kernel)  # type: ignore[arg-type]
        cov = result.covariance
        assert cov is not None
        v00, v11, v02 = SANDWICH[(kernel, bw)]
        assert cov[0, 0] == pytest.approx(v00, rel=1e-10)
        assert cov[1, 1] == pytest.approx(v11, rel=1e-10)
        assert cov[0, 2] == pytest.approx(v02, rel=1e-10)

    def test_full_covariance_matrix(self) -> None:
        e, x = _reference_inputs()
        result = newey_west_covariance(e, x, bandwidth=5, kernel="bartlett")
        np.testing.assert_allclose(result.covariance, FULL_COV_B5_BARTLETT, rtol=1e-10)

    @pytest.mark.parametrize(("method", "kernel"), sorted(BANDWIDTHS))
    def test_optimal_bandwidth(self, method: str, kernel: str) -> None:
        e, _ = _reference_inputs()
        assert optimal_bandwidth(e, method=method, kernel=kernel) == BANDWIDTHS[(method, kernel)]  # type: ignore[arg-type]

    @pytest.mark.parametrize(("kernel", "lag", "bw"), sorted(KERNEL_VALUES))
    def test_kernel_weights(self, kernel: str, lag: int, bw: int) -> None:
        assert _KERNEL_FUNCS[kernel](lag, bw) == pytest.approx(
            KERNEL_VALUES[(kernel, lag, bw)], abs=1e-15
        )


class TestCorrectness:
    def test_iid_long_run_variance_close_to_variance(self) -> None:
        rng = np.random.default_rng(1)
        e = rng.standard_normal(20_000)
        omega = long_run_covariance(e, bandwidth=5)[0, 0]
        assert omega == pytest.approx(1.0, rel=0.05)

    def test_ar1_long_run_variance(self) -> None:
        # AR(1): true long-run variance = sigma^2 / (1 - phi)^2.
        phi, n = 0.5, 50_000
        rng = np.random.default_rng(2)
        raw = rng.standard_normal(n)
        e = raw.copy()
        for t in range(1, n):
            e[t] = phi * e[t - 1] + raw[t]
        true_omega = 1.0 / (1 - phi) ** 2  # = 4.0
        omega = long_run_covariance(e, bandwidth=60)[0, 0]
        assert omega == pytest.approx(true_omega, rel=0.15)

    def test_matrix_diag_matches_scalar_paths(self) -> None:
        # The (k, k) core's diagonal must equal the per-series scalar
        # long-run variances at the same int bandwidth.
        rng = np.random.default_rng(3)
        u = rng.standard_normal((500, 2))
        u[:, 1] = 0.7 * u[:, 0] + u[:, 1]
        omega = long_run_covariance(u, bandwidth=6)
        assert omega.shape == (2, 2)
        for col in range(2):
            scalar = long_run_covariance(u[:, col], bandwidth=6)[0, 0]
            assert omega[col, col] == pytest.approx(scalar, rel=1e-12)
        # symmetric
        assert omega[0, 1] == pytest.approx(omega[1, 0], abs=1e-15)

    def test_long_run_covariance_always_2d(self) -> None:
        e = np.arange(10.0)
        assert long_run_covariance(e, bandwidth=2).shape == (1, 1)

    def test_mean_se_consistency_with_core(self) -> None:
        # mean-mode result: variance * n == long_run_variance == core's (1,1).
        e, _ = _reference_inputs()
        result = newey_west_se(e, bandwidth=4)
        omega_core = long_run_covariance(e, bandwidth=4)[0, 0]
        assert result.long_run_variance == pytest.approx(omega_core, rel=1e-12)
        assert result.variance * result.n_samples == pytest.approx(
            result.long_run_variance, rel=1e-12
        )
        assert result.se == pytest.approx(np.sqrt(result.variance), rel=1e-15)

    def test_positive_autocorrelation_inflates_se(self) -> None:
        e, _ = _reference_inputs()
        hac_se = newey_west_se(e, bandwidth=8).se
        naive_se = float(e.std(ddof=0) / np.sqrt(e.size))
        assert hac_se > naive_se


class TestSemanticsPins:
    """The dml_ts issue #7 lesson: se is sd/sqrt(n) scale, NEVER sd/n."""

    def test_iid_se_is_sd_over_sqrt_n(self) -> None:
        rng = np.random.default_rng(0)
        psi = rng.standard_normal(400)
        result = newey_west_se(psi, bandwidth=0)
        correct = float(psi.std(ddof=0) / np.sqrt(psi.size))
        assert result.se == pytest.approx(correct, rel=1e-10)
        # The double-division mistake would be off by a factor sqrt(n) = 20.
        assert result.se > 10 * (correct / np.sqrt(psi.size))

    def test_sandwich_se_matches_leading_covariance(self) -> None:
        e, x = _reference_inputs()
        result = newey_west_se(e, x, bandwidth=5)
        cov_result = newey_west_covariance(e, x, bandwidth=5)
        assert result.covariance is not None
        assert result.se == pytest.approx(np.sqrt(cov_result.covariance[0, 0]), rel=1e-12)
        assert result.long_run_variance is None


class TestPrewhitening:
    """Correctness-owned (dml_ts's prewhiten path was broken — no parity)."""

    def test_recoloring_recovers_ar1_long_run_variance(self) -> None:
        # Strong AR(1) with a deliberately tiny bandwidth: the plain
        # estimator badly underestimates the long-run variance; the
        # prewhitened+recolored one recovers it.
        phi, n = 0.9, 4000
        rng = np.random.default_rng(7)
        raw = rng.standard_normal(n)
        e = raw.copy()
        for t in range(1, n):
            e[t] = phi * e[t - 1] + raw[t]
        true_omega = 1.0 / (1 - phi) ** 2  # = 100

        plain = newey_west_se(e, bandwidth=2, prewhiten=False)
        white = newey_west_se(e, bandwidth=2, prewhiten=True)
        assert plain.long_run_variance is not None
        assert white.long_run_variance is not None
        # Plain underestimates by far more than the prewhitened estimate errs.
        assert plain.long_run_variance < 0.3 * true_omega
        assert white.long_run_variance == pytest.approx(true_omega, rel=0.25)
        assert white.prewhitened and white.ar_coef == pytest.approx(phi, abs=0.05)

    def test_iid_prewhitening_is_nearly_neutral(self) -> None:
        rng = np.random.default_rng(8)
        e = rng.standard_normal(5000)
        plain = newey_west_se(e, bandwidth=3, prewhiten=False)
        white = newey_west_se(e, bandwidth=3, prewhiten=True)
        assert white.se == pytest.approx(plain.se, rel=0.05)

    def test_prewhiten_with_design_matrix_works(self) -> None:
        # Regression vs dml_ts, whose prewhiten+X path ALWAYS crashed with
        # "residuals length (n-1) != X rows (n)".
        e, x = _reference_inputs()
        result = newey_west_covariance(e, x, bandwidth=4, prewhiten=True)
        assert result.covariance is not None
        assert result.covariance.shape == (3, 3)
        assert result.prewhitened and result.ar_coef is not None
        assert np.isfinite(result.se) and result.se > 0

    def test_prewhiten_needs_three_observations(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            newey_west_se([1.0, 2.0], prewhiten=True)

    def test_prewhiten_constant_series_raises(self) -> None:
        # Both the all-zero and the constant-NONZERO series must raise:
        # before demeaning was added to the phi fit, a constant-5 series
        # sailed through with a fabricated ar_coef=0.99 (review finding).
        with pytest.raises(ValueError, match="constant series"):
            newey_west_se(np.zeros(10), prewhiten=True)
        with pytest.raises(ValueError, match="constant series"):
            newey_west_se(np.full(50, 5.0), prewhiten=True)

    def test_prewhiten_neutral_on_nonzero_mean_iid_series(self) -> None:
        # Review CRITICAL finding: an un-demeaned phi fit on mean-10 iid
        # noise gave phi=0.989 and inflated the SE ~50x silently. With
        # demeaning, prewhitening must be nearly neutral — the mean is the
        # ESTIMAND for influence scores, not autocorrelation.
        rng = np.random.default_rng(11)
        e = 10.0 + rng.standard_normal(500)
        plain = newey_west_se(e, bandwidth="auto", prewhiten=False)
        white = newey_west_se(e, bandwidth="auto", prewhiten=True)
        assert white.se == pytest.approx(plain.se, rel=0.15)
        assert abs(white.ar_coef) < 0.2  # type: ignore[arg-type]

    def test_prewhiten_sandwich_recovers_ar1_variance(self) -> None:
        # Value-level pin for the prewhitened SANDWICH path (a wrong
        # recolor factor here previously survived the suite — mutation
        # finding). Intercept-only design: leading coefficient is the
        # mean, true SE = sqrt(Omega/n) with Omega = 1/(1-phi)^2.
        phi, n = 0.9, 4000
        rng = np.random.default_rng(7)
        raw = rng.standard_normal(n)
        e = raw.copy()
        for t in range(1, n):
            e[t] = phi * e[t - 1] + raw[t]
        true_se = float(np.sqrt((1.0 / (1 - phi) ** 2) / n))
        result = newey_west_covariance(e - e.mean(), np.ones((n, 1)), bandwidth=2, prewhiten=True)
        assert result.se == pytest.approx(true_se, rel=0.25)

    def test_prewhiten_auto_bandwidth_uses_whitened_series(self) -> None:
        # Documented deviation (Andrews-Monahan 1992): auto bandwidth is
        # selected on the WHITENED series. Selecting on the ORIGINAL
        # phi=0.9 series would give a bandwidth in the hundreds.
        phi, n = 0.9, 4000
        rng = np.random.default_rng(7)
        raw = rng.standard_normal(n)
        e = raw.copy()
        for t in range(1, n):
            e[t] = phi * e[t - 1] + raw[t]
        result = newey_west_se(e, bandwidth="andrews", prewhiten=True)
        assert result.bandwidth <= 3


class TestValidation:
    def test_series_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            newey_west_se(np.ones((10, 2)))
        with pytest.raises(ValueError, match="1-D"):
            optimal_bandwidth(np.ones((10, 2)))

    def test_complex_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            newey_west_se(np.array([1 + 1j, 2.0]))
        with pytest.raises(ValueError, match="complex"):
            long_run_covariance(np.array([1 + 1j, 2.0]))
        e, _ = _reference_inputs()
        with pytest.raises(ValueError, match="complex"):
            newey_west_covariance(e, np.array([[1 + 1j]] * 120))

    def test_none_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="None"):
            newey_west_se(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="None"):
            long_run_covariance(None)  # type: ignore[arg-type]
        e, _ = _reference_inputs()
        with pytest.raises(ValueError, match="None"):
            newey_west_covariance(e, None)  # type: ignore[arg-type]

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            newey_west_se([1.0, np.nan, 2.0])

    def test_too_few_observations_raise(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            newey_west_se([1.0])
        with pytest.raises(ValueError, match="at least 2"):
            long_run_covariance([1.0])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="!= X rows"):
            newey_west_covariance([1.0, 2.0, 3.0], np.ones((4, 1)))

    def test_n_less_than_k_raises(self) -> None:
        with pytest.raises(ValueError, match="n >= k"):
            newey_west_covariance([1.0, 2.0], np.ones((2, 3)))

    def test_singular_xtx_raises(self) -> None:
        # Duplicate columns -> rank-deficient X'X. dml_ts silently fell
        # back to a pseudo-inverse here; we refuse.
        rng = np.random.default_rng(5)
        e = rng.standard_normal(50)
        col = rng.standard_normal(50)
        x = np.column_stack([col, col])
        with pytest.raises(ValueError, match="rank-deficient"):
            newey_west_covariance(e, x, bandwidth=3)

    def test_unknown_kernel_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown kernel"):
            newey_west_se([1.0, 2.0, 3.0], kernel="gaussian")  # type: ignore[arg-type]

    def test_unknown_bandwidth_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown bandwidth method"):
            newey_west_se([1.0, 2.0, 3.0], bandwidth="plugin")

    def test_bool_or_float_bandwidth_raises(self) -> None:
        with pytest.raises(TypeError, match="bandwidth"):
            newey_west_se([1.0, 2.0, 3.0], bandwidth=2.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="bandwidth"):
            newey_west_se([1.0, 2.0, 3.0], bandwidth=True)  # type: ignore[arg-type]

    def test_negative_bandwidth_raises(self) -> None:
        with pytest.raises(ValueError, match="bandwidth must be >= 0"):
            newey_west_se([1.0, 2.0, 3.0], bandwidth=-1)

    def test_bandwidth_clamped_to_n_minus_1(self) -> None:
        # Parity with dml_ts: oversized int bandwidths clamp silently.
        result = newey_west_se([1.0, 2.0, 3.0, 4.0], bandwidth=99)
        assert result.bandwidth == 3

    def test_andrews_on_matrix_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="andrews"):
            long_run_covariance(np.ones((10, 2)) + np.eye(10, 2), bandwidth="andrews")

    def test_andrews_on_constant_series_raises(self) -> None:
        # Review finding: np.corrcoef of a zero-variance series is NaN and
        # previously crashed with a context-free numpy error.
        with pytest.raises(ValueError, match="constant series"):
            optimal_bandwidth(np.full(50, 5.0), method="andrews")

    def test_optimal_bandwidth_validates_kernel_for_every_method(self) -> None:
        # Review finding: a kernel typo silently passed for newey_west
        # (kernel only affects andrews constants, but typos must be loud).
        with pytest.raises(ValueError, match="Unknown kernel"):
            optimal_bandwidth([1.0, 2.0, 3.0], method="newey_west", kernel="barlett")  # type: ignore[arg-type]


class TestFailLoudOnNumericGarbage:
    """Review CRITICAL findings: overflow and negative estimates must raise,
    never launder into a plausible se=0.0 or a NaN matrix."""

    def test_scalar_overflow_raises(self) -> None:
        # max(0.0, nan) is 0.0 — the old dml_ts clamp silently turned
        # float64 overflow into se=0.0 (infinite t-stats downstream).
        e = np.array([1e200, -1e200] * 10)
        with pytest.raises(ValueError, match="non-finite"):
            newey_west_se(e, bandwidth=4)

    def test_long_run_covariance_overflow_raises(self) -> None:
        scores = np.array([1e200, -1e200] * 10)
        with pytest.raises(ValueError, match="non-finite"):
            long_run_covariance(scores, bandwidth=4)
        mixed = np.column_stack([np.array([1e200, -1e200] * 10), np.ones(20)])
        with pytest.raises(ValueError, match="non-finite"):
            long_run_covariance(mixed, bandwidth=4)

    def test_sandwich_overflow_raises_with_named_cause(self) -> None:
        rng = np.random.default_rng(9)
        e = np.array([1e200, -1e200] * 25)
        x = rng.standard_normal((50, 2))
        with pytest.raises(ValueError, match="overflow"):
            newey_west_covariance(e, x, bandwidth=3)

    def test_qs_negative_scalar_variance_raises(self) -> None:
        # Alternating series + QS: the truncated weight sequence fails
        # PSD-ness (raw omega < 0). dml_ts clamped this to se=0.0 silently.
        e = np.array([(-1.0) ** t for t in range(12)])
        with pytest.raises(ValueError, match="negative"):
            newey_west_se(e, bandwidth=1, kernel="quadratic_spectral")

    def test_qs_negative_sandwich_diagonal_raises(self) -> None:
        # Exercises the documented deviation #4 raise branch (previously
        # untested). dml_ts returned se=NaN with only a RuntimeWarning.
        e = np.array([(-1.0) ** t for t in range(12)])
        with pytest.raises(ValueError, match="negative diagonal"):
            newey_west_covariance(e, np.ones((12, 1)), bandwidth=1, kernel="quadratic_spectral")

    def test_long_run_covariance_shift_invariant(self) -> None:
        # Demeaning pin by property: Omega must be invariant to adding a
        # constant (redundant guard for the single test that catches a
        # dropped-demeaning mutation).
        rng = np.random.default_rng(12)
        e = rng.standard_normal(300)
        np.testing.assert_allclose(
            long_run_covariance(e, bandwidth=5),
            long_run_covariance(e + 100.0, bandwidth=5),
            rtol=1e-8,
        )


class TestHACResultContract:
    def _result(self) -> HACResult:
        e, _ = _reference_inputs()
        return newey_west_se(e, bandwidth=4)

    def test_frozen(self) -> None:
        result = self._result()
        with pytest.raises(FrozenInstanceError):
            result.se = 1.0  # type: ignore[misc]

    def test_schema_version_and_to_dict_json_serializable(self) -> None:
        result = self._result()
        assert HACResult.SCHEMA_VERSION == 1
        payload = result.to_dict()
        json.dumps(payload)  # must not raise
        assert payload["schema_version"] == 1
        assert payload["se"] == result.se

    def test_eq_is_identity(self) -> None:
        a = self._result()
        b = self._result()
        assert a == a
        assert a != b  # eq=False: identity semantics (array-bearing object)
        hash(a)  # hashable

    def test_post_init_guards(self) -> None:
        kwargs: dict = {
            "se": 0.1,
            "variance": 0.01,
            "long_run_variance": 1.0,
            "covariance": None,
            "bandwidth": 4,
            "kernel": "bartlett",
            "n_samples": 100,
            "effective_dof": 96.0,
            "prewhitened": False,
            "ar_coef": None,
        }
        HACResult(**kwargs)  # valid
        with pytest.raises(ValueError, match="se must be"):
            HACResult(**{**kwargs, "se": -0.1})
        with pytest.raises(ValueError, match="bandwidth"):
            HACResult(**{**kwargs, "bandwidth": -1})
        with pytest.raises(ValueError, match="n_samples"):
            HACResult(**{**kwargs, "n_samples": 1})
        with pytest.raises(ValueError, match="ar_coef must be set iff"):
            HACResult(**{**kwargs, "prewhitened": True})
        with pytest.raises(ValueError, match="square"):
            HACResult(**{**kwargs, "covariance": np.ones((2, 3))})
        # Review finding: NaN covariance entries previously slipped through
        # (only ndim/squareness was checked) and produced RFC-invalid JSON.
        with pytest.raises(ValueError, match="non-finite"):
            HACResult(**{**kwargs, "covariance": np.array([[1.0, np.nan], [np.nan, 1.0]])})
