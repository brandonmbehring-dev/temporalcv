"""Tests for the numeric output validators (issue #10).

Every guard is tested both ways: valid input passes through unchanged
(validate-and-return), and each impossible-arithmetic class raises
``ValueError`` with an informative message.
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv import ci_ordered, coverage_in_unit, finite_se, psd

# ---------------------------------------------------------------------------
# finite_se
# ---------------------------------------------------------------------------


class TestFiniteSE:
    def test_scalar_passes_through(self) -> None:
        out = finite_se(0.25)
        assert float(out) == 0.25
        assert isinstance(out, np.ndarray)

    def test_array_passes_through_unchanged(self) -> None:
        se = [0.1, 0.5, 2.0]
        out = finite_se(se)
        np.testing.assert_array_equal(out, np.asarray(se, dtype=float))

    @pytest.mark.parametrize("bad", [0.0, -0.1])
    def test_non_positive_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-positive"):
            finite_se([0.5, bad])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            finite_se([0.5, bad])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            finite_se([])

    def test_name_in_message(self) -> None:
        with pytest.raises(ValueError, match="theta_se"):
            finite_se(-1.0, name="theta_se")


# ---------------------------------------------------------------------------
# psd
# ---------------------------------------------------------------------------


class TestPSD:
    def test_valid_covariance_passes_through(self) -> None:
        cov = [[1.0, 0.2], [0.2, 1.0]]
        out = psd(cov)
        np.testing.assert_array_equal(out, np.asarray(cov, dtype=float))

    def test_psd_but_singular_passes(self) -> None:
        # Rank-deficient is still PSD (zero eigenvalue allowed).
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        psd(cov)

    def test_barely_negative_eigenvalue_within_tol_passes(self) -> None:
        cov = np.array([[1.0, 0.0], [0.0, -1e-12]])
        psd(cov, tol=1e-8)

    def test_negative_definite_raises(self) -> None:
        with pytest.raises(ValueError, match="positive semi-definite"):
            psd([[1.0, 0.0], [0.0, -0.5]])

    def test_indefinite_raises(self) -> None:
        # Correlation > 1 in magnitude -> indefinite "covariance".
        with pytest.raises(ValueError, match="positive semi-definite"):
            psd([[1.0, 2.0], [2.0, 1.0]])

    def test_asymmetric_raises(self) -> None:
        with pytest.raises(ValueError, match="not symmetric"):
            psd([[1.0, 0.5], [0.1, 1.0]])

    def test_asymmetry_within_tol_passes(self) -> None:
        cov = np.array([[1.0, 0.2 + 1e-12], [0.2, 1.0]])
        psd(cov, tol=1e-8)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            psd([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            psd([1.0, 2.0])

    def test_non_finite_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            psd([[1.0, np.nan], [np.nan, 1.0]])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            psd(np.empty((0, 0)))

    @pytest.mark.parametrize("bad_tol", [-1e-8, np.nan, np.inf])
    def test_bad_tol_raises(self, bad_tol: float) -> None:
        with pytest.raises(ValueError, match="tol"):
            psd([[1.0]], tol=bad_tol)

    def test_tol_is_wired_to_eigenvalue_threshold(self) -> None:
        # Review finding: hardcoding the eigenvalue threshold to the default
        # previously survived the suite — pin that tol= actually controls it.
        cov = np.array([[1.0, 0.0], [0.0, -1e-5]])
        psd(cov, tol=1e-4)  # within the loose tolerance
        with pytest.raises(ValueError, match="positive semi-definite"):
            psd(cov)  # default tol=1e-8 must reject the same matrix

    def test_name_in_message(self) -> None:
        with pytest.raises(ValueError, match="theta_cov"):
            psd([[-1.0]], name="theta_cov")

    @staticmethod
    def _large_scale_sandwich() -> np.ndarray:
        """Large-scale SPD matrix with roundoff-scale asymmetry (the #33 fixture).

        Models a sandwich covariance ``G_inv @ meat @ G_inv.T`` at
        ``|cov| ~ 1e10``: one off-diagonal element carries a 1e-15 relative
        (a few ULPs, ~1e-5 absolute) asymmetry — above the 1e-8 atol but
        far below rtol. Injected deterministically rather than relying on
        real dgemm roundoff, which is BLAS/platform-dependent and would
        make the fixture's discrimination property flaky across CI runners.
        """
        base = (
            np.array(
                [
                    [4.0, 1.2, -0.7, 0.3],
                    [1.2, 3.5, 0.9, -0.4],
                    [-0.7, 0.9, 5.1, 1.0],
                    [0.3, -0.4, 1.0, 2.8],
                ]
            )
            * 1e10
        )  # diagonally dominant with positive diagonal -> SPD
        cov = base.copy()
        cov[0, 1] *= 1.0 + 1e-15  # roundoff-scale relative asymmetry
        return cov

    def test_large_scale_sandwich_passes_by_default(self) -> None:
        """A healthy large-scale sandwich passes the scale-aware default (#33)."""
        cov = self._large_scale_sandwich()
        # The fixture must actually exercise the fix: its asymmetry
        # exceeds the absolute tolerance on its own.
        assert float(np.max(np.abs(cov - cov.T))) > 1e-8

        psd(cov)  # must not raise

    def test_rtol_zero_restores_absolute_strictness(self) -> None:
        """rtol=0 reproduces the pre-#33 false positive on the same matrix."""
        with pytest.raises(ValueError, match="not symmetric"):
            psd(self._large_scale_sandwich(), rtol=0.0)

    def test_genuine_relative_asymmetry_still_raises(self) -> None:
        """Real asymmetry (~1e-6 relative) is caught at any scale (#33)."""
        cov = np.array([[1.0, 0.2], [0.2 + 1e-6, 1.0]]) * 1e10
        with pytest.raises(ValueError, match="not symmetric"):
            psd(cov)

    def test_eigen_floor_scales_with_magnitude(self) -> None:
        """The eigenvalue floor is -(tol + rtol * max|eig|) (#33)."""
        # -1e-4 against max|eig| = 1e10: relative -1e-14, jitter at scale.
        psd(np.diag([1e10, -1e-4]))
        # -1.0 against the same scale exceeds the floor (~ -1e-2): real.
        with pytest.raises(ValueError, match="positive semi-definite"):
            psd(np.diag([1e10, -1.0]))

    def test_eigen_floor_condition_number_tradeoff_is_pinned(self) -> None:
        """Beyond condition number ~1/rtol the floor absorbs real negatives (#33).

        Documented trade-off, pinned so it stays a visible decision: at
        max|eig| = 1e12 the default floor is ~1.0, so min_eig = -0.5
        (-5e-13 relative — inside the jitter band at this tolerance) passes.
        A caller validating such ill-conditioned matrices tightens rtol,
        which restores the rejection.
        """
        ill_conditioned = np.diag([1e12, -0.5])
        psd(ill_conditioned)  # default: within rtol of zero at this scale
        with pytest.raises(ValueError, match="positive semi-definite"):
            psd(ill_conditioned, rtol=1e-14)

    @pytest.mark.parametrize("bad_rtol", [-1e-12, np.nan, np.inf])
    def test_bad_rtol_raises(self, bad_rtol: float) -> None:
        with pytest.raises(ValueError, match="rtol"):
            psd([[1.0]], rtol=bad_rtol)


# ---------------------------------------------------------------------------
# ci_ordered
# ---------------------------------------------------------------------------


class TestCIOrdered:
    def test_ordered_passes_through(self) -> None:
        lo, hi = ci_ordered([-1.0, 0.0], [1.0, 2.0])
        np.testing.assert_array_equal(lo, [-1.0, 0.0])
        np.testing.assert_array_equal(hi, [1.0, 2.0])

    def test_scalar_bounds(self) -> None:
        lo, hi = ci_ordered(-1.96, 1.96)
        assert float(lo) == -1.96
        assert float(hi) == 1.96

    def test_degenerate_equal_bounds_pass(self) -> None:
        ci_ordered(1.0, 1.0)

    def test_one_sided_infinite_bounds_pass(self) -> None:
        ci_ordered(-np.inf, 1.96)
        ci_ordered(0.0, np.inf)

    def test_inverted_raises_with_location(self) -> None:
        with pytest.raises(ValueError, match="inverted.*index 1"):
            ci_ordered([0.0, 5.0], [1.0, 2.0])

    def test_inverted_2d_reports_correct_flat_index_and_values(self) -> None:
        # Review finding: np.nonzero(...)[0][0] on a 2-D mask returned the
        # ROW index (1) and read the wrong elements; the true violation here
        # is at flat index 3 (lower=5 > upper=2).
        with pytest.raises(ValueError, match=r"flat index 3: lower = 5 > upper = 2"):
            ci_ordered([[0.0, 0.0], [0.0, 5.0]], [[1.0, 1.0], [1.0, 2.0]])

    def test_name_in_message(self) -> None:
        with pytest.raises(ValueError, match="theta_ci"):
            ci_ordered(2.0, 1.0, name="theta_ci")

    def test_inverted_scalar_raises(self) -> None:
        with pytest.raises(ValueError, match="inverted"):
            ci_ordered(2.0, 1.0)

    def test_nan_bound_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            ci_ordered([0.0, np.nan], [1.0, 2.0])
        with pytest.raises(ValueError, match="NaN"):
            ci_ordered([0.0], [np.nan])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mismatched shapes"):
            ci_ordered([0.0, 1.0], [1.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ci_ordered([], [])


# ---------------------------------------------------------------------------
# coverage_in_unit
# ---------------------------------------------------------------------------


class TestCoverageInUnit:
    @pytest.mark.parametrize("ok", [0.0, 0.5, 0.95, 1.0])
    def test_boundary_inclusive_passes(self, ok: float) -> None:
        assert float(coverage_in_unit(ok)) == ok

    def test_array_passes_through(self) -> None:
        cov = [0.9, 0.95, 0.99]
        np.testing.assert_array_equal(coverage_in_unit(cov), cov)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
    def test_outside_unit_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            coverage_in_unit([0.5, bad])

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            coverage_in_unit(bad)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            coverage_in_unit([])

    def test_name_in_message(self) -> None:
        with pytest.raises(ValueError, match="pi_coverage"):
            coverage_in_unit(1.5, name="pi_coverage")

    def test_message_shows_boundary_violation_precisely(self) -> None:
        """The message must not round 1.0000001 down to "1" (#33 nit).

        %g formatting produced the self-contradictory "outside [0, 1]:
        min = 1, max = 1"; repr keeps the violating digits visible.
        """
        with pytest.raises(ValueError, match="1.0000001"):
            coverage_in_unit(1.0000001)


# ---------------------------------------------------------------------------
# Cross-guard input hygiene (complex / None rejection)
# ---------------------------------------------------------------------------


class TestComplexAndNoneRejection:
    """np.asarray(x, dtype=float) silently discards imaginary parts and turns
    None into NaN — both are upstream-bug signatures these guards exist to
    catch (review finding: a complex 'standard error' previously PASSED
    finite_se with only a ComplexWarning)."""

    def test_finite_se_complex_raises(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            finite_se(np.array([0.5 + 1j]))

    def test_psd_complex_raises(self) -> None:
        # Even a valid complex Hermitian PSD matrix must be refused: the cast
        # would silently return a DIFFERENT (real-truncated) matrix.
        with pytest.raises(ValueError, match="complex"):
            psd(np.array([[2.0, 1j], [-1j, 2.0]]))

    def test_ci_ordered_complex_raises_either_bound(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            ci_ordered(np.array([1 + 5j]), np.array([2.0]))
        with pytest.raises(ValueError, match="complex"):
            ci_ordered(np.array([1.0]), np.array([2 - 3j]))

    def test_coverage_in_unit_complex_raises(self) -> None:
        with pytest.raises(ValueError, match="complex"):
            coverage_in_unit(np.array([0.5 + 0.5j]))

    def test_none_raises_with_clear_message(self) -> None:
        # Not "contains NaN" — the user passed None, say so.
        with pytest.raises(ValueError, match="None"):
            finite_se(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="None"):
            coverage_in_unit(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="None"):
            psd(None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="None"):
            ci_ordered(None, [1.0])  # type: ignore[arg-type]
