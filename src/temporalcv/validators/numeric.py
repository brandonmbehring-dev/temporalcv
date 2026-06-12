"""
Numeric Output Validators (hard guards).

Validate-and-return guards that RAISE stdlib ``ValueError`` on arithmetically
impossible statistical outputs: non-positive or non-finite standard errors,
non-PSD covariance matrices, inverted confidence bounds, and coverage rates
outside [0, 1]. On success each guard returns the normalized
(``np.asarray``-ed) input, so it composes as a pass-through at the arithmetic
boundary of an inference pipeline::

    se = finite_se(se)
    cov = psd(cov)
    lo, hi = ci_ordered(lo, hi)

How these relate to the library's other "checking" vocabularies:

- **Conformance** ``check_*`` functions (``conformance.py``) assert seam
  CONTRACTS for developers and raise ``AssertionError``.
- **Gates** (``gates.py``) return *investigation signals*
  (PASS/WARN/HALT, or SKIP when a gate cannot run) for methodology
  questions where the right answer may be "look closer".
- **These guards** police impossible ARITHMETIC: if one fires, the upstream
  computation is wrong — there is nothing to investigate, so they raise
  immediately (``ValueError``).

All guards reject empty input: an empty statistic arriving at an inference
boundary is an upstream bug, not a vacuous pass.

Knowledge Tiers
---------------
[T1] A standard error is a square root of a positive variance: finite, > 0.
[T1] A covariance matrix is symmetric positive semi-definite.
[T1] A confidence interval satisfies lower <= upper (infinite bounds are
     legitimate for one-sided intervals; NaN bounds are not).
[T1] A coverage rate is a proportion in [0, 1].
"""

from __future__ import annotations

import numpy as np

from temporalcv._typing import ArrayLike

__all__ = ["ci_ordered", "coverage_in_unit", "finite_se", "psd"]


def _as_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """Cast to a float array, refusing None and complex input loudly.

    ``np.asarray(x, dtype=float)`` would silently turn None into NaN and
    silently DISCARD imaginary parts of complex input — both are exactly the
    upstream-bug signatures these guards exist to catch.
    """
    if x is None:
        raise ValueError(f"{name} is None — expected a numeric value or array")
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError(
            f"{name} has complex dtype {raw.dtype} — refusing to silently "
            f"discard imaginary parts (complex statistics indicate an upstream bug)"
        )
    return np.asarray(raw, dtype=float)


def _as_nonempty_float_array(x: ArrayLike, name: str) -> np.ndarray:
    arr = _as_float_array(x, name)
    if arr.size == 0:
        raise ValueError(f"{name} is empty — nothing to validate (upstream bug?)")
    return arr


def finite_se(se: ArrayLike, *, name: str = "se") -> np.ndarray:
    """
    Validate that standard errors are finite and strictly positive.

    Parameters
    ----------
    se : ArrayLike
        Standard error(s) — scalar, sequence, or array.
    name : str, default "se"
        Label used in error messages (e.g. ``"theta_se"``).

    Returns
    -------
    np.ndarray
        ``np.asarray(se, dtype=float)`` unchanged (0-d for scalar input).

    Raises
    ------
    ValueError
        If empty, any value is non-finite (NaN/inf), or any value is <= 0.

    Examples
    --------
    >>> float(finite_se(0.25))
    0.25
    >>> finite_se([0.1, -0.2])
    Traceback (most recent call last):
        ...
    ValueError: se contains non-positive values: min = -0.2 (a standard error is a square root of a positive variance)
    """
    arr = _as_nonempty_float_array(se, name)
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} contains non-finite values "
            f"(NaN: {int(np.isnan(arr).sum())}, inf: {int(np.isinf(arr).sum())})"
        )
    if np.any(arr <= 0):
        raise ValueError(
            f"{name} contains non-positive values: min = {arr.min():g} "
            f"(a standard error is a square root of a positive variance)"
        )
    return arr


def psd(cov: ArrayLike, *, tol: float = 1e-8, rtol: float = 1e-12, name: str = "cov") -> np.ndarray:
    """
    Validate that a covariance matrix is symmetric positive semi-definite.

    Both checks are scale-aware: a sandwich covariance
    ``G_inv @ meat @ G_inv.T`` is symmetric by construction but carries
    float roundoff proportional to its magnitude (relative asymmetry
    ~1e-19 on healthy fits), so a purely absolute tolerance false-positives
    at ``|cov| >~ 1e8`` (#33).

    Parameters
    ----------
    cov : ArrayLike
        Square 2-D covariance matrix.
    tol : float, default 1e-8
        Absolute tolerance for the symmetry check and the minimum-eigenvalue
        floor. Must be finite and >= 0.
    rtol : float, default 1e-12
        Relative tolerance. Symmetry passes when
        ``np.allclose(cov, cov.T, atol=tol, rtol=rtol)``; the eigenvalue
        floor is ``-(tol + rtol * max|eigenvalue|)``. The default sits
        orders of magnitude above healthy float roundoff (~1e-19 relative)
        and below genuine asymmetry bugs (~1e-6 relative and up). Pass
        ``rtol=0`` for the pre-#33 strictly absolute behavior. Must be
        finite and >= 0.
    name : str, default "cov"
        Label used in error messages.

    Returns
    -------
    np.ndarray
        ``np.asarray(cov, dtype=float)`` unchanged.

    Raises
    ------
    ValueError
        If empty, not 2-D square, non-finite, asymmetric beyond the
        combined tolerance, or its minimum eigenvalue is below the floor.

    Examples
    --------
    >>> psd([[1.0, 0.2], [0.2, 1.0]]).shape
    (2, 2)
    """
    if not np.isfinite(tol) or tol < 0:
        raise ValueError(f"tol must be finite and >= 0, got {tol}")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError(f"rtol must be finite and >= 0, got {rtol}")
    arr = _as_nonempty_float_array(cov, name)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(arr, arr.T, atol=tol, rtol=rtol):
        max_asym = float(np.max(np.abs(arr - arr.T)))
        scale = float(np.max(np.abs(arr)))
        rel_asym = max_asym / scale if scale > 0 else 0.0
        raise ValueError(
            f"{name} is not symmetric: max |cov - cov.T| = {max_asym:.3e} "
            f"(relative {rel_asym:.3e}) exceeds tol = {tol:g} with rtol = {rtol:g}"
        )
    # Symmetrize before eigvalsh to absorb within-tol asymmetry.
    eigs = np.linalg.eigvalsh((arr + arr.T) / 2.0)
    min_eig = float(eigs.min())
    eig_floor = tol + rtol * float(np.max(np.abs(eigs)))
    if min_eig < -eig_floor:
        raise ValueError(
            f"{name} is not positive semi-definite: min eigenvalue = "
            f"{min_eig:.3e} < -(tol + rtol * max|eig|) = {-eig_floor:.3e}"
        )
    return arr


def ci_ordered(
    lower: ArrayLike, upper: ArrayLike, *, name: str = "ci"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate that confidence-interval bounds are ordered (lower <= upper).

    Infinite bounds are allowed (one-sided intervals are legitimate);
    NaN bounds are not.

    Parameters
    ----------
    lower, upper : ArrayLike
        Interval bounds; must have identical shapes after ``np.asarray``.
    name : str, default "ci"
        Label used in error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(lower, upper)`` as float arrays, unchanged.

    Raises
    ------
    ValueError
        If empty, shapes differ, any bound is NaN, or any ``lower > upper``.

    Examples
    --------
    >>> lo, hi = ci_ordered([-1.0, 0.0], [1.0, 2.0])
    >>> lo, hi = ci_ordered(-np.inf, 1.96)  # one-sided is fine
    """
    lo = _as_nonempty_float_array(lower, f"{name} lower")
    hi = _as_float_array(upper, f"{name} upper")
    if lo.shape != hi.shape:
        raise ValueError(
            f"{name} bounds have mismatched shapes: lower {lo.shape} vs upper {hi.shape}"
        )
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError(f"{name} bounds contain NaN")
    if np.any(lo > hi):
        bad = int(np.flatnonzero(np.atleast_1d(lo > hi))[0])
        lo_1d, hi_1d = np.atleast_1d(lo), np.atleast_1d(hi)
        raise ValueError(
            f"{name} bounds are inverted (lower > upper), first at flat index "
            f"{bad}: lower = {lo_1d.ravel()[bad]:g} > upper = {hi_1d.ravel()[bad]:g}"
        )
    return lo, hi


def coverage_in_unit(coverage: ArrayLike, *, name: str = "coverage") -> np.ndarray:
    """
    Validate that coverage rates lie in [0, 1].

    Parameters
    ----------
    coverage : ArrayLike
        Coverage proportion(s) — scalar, sequence, or array.
    name : str, default "coverage"
        Label used in error messages.

    Returns
    -------
    np.ndarray
        ``np.asarray(coverage, dtype=float)`` unchanged.

    Raises
    ------
    ValueError
        If empty, any value is non-finite, or any value is outside [0, 1].

    Examples
    --------
    >>> float(coverage_in_unit(0.95))
    0.95
    """
    arr = _as_nonempty_float_array(coverage, name)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < 0) or np.any(arr > 1):
        # repr (shortest round-trip) instead of %g: %g rendered 1.0000001
        # as "1", making the message contradict itself (#33).
        raise ValueError(
            f"{name} outside [0, 1]: min = {float(arr.min())!r}, "
            f"max = {float(arr.max())!r} (a coverage rate is a proportion)"
        )
    return arr
