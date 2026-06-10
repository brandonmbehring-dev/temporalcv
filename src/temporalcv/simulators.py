"""
Generic AR/ARMA Time-Series Simulators.

Plain stochastic path generators for testing, benchmarking, and building
downstream test DGPs. These are mechanism-level simulators only: no
treatment effects and no causal structure (causal known-theta DGPs are a
consumer concern and live with the consumer, e.g. dml_ts).

Conventions
-----------
- Coefficients use the **recursion convention**::

      y_t = ar[0]*y_{t-1} + ... + ar[p-1]*y_{t-p}
            + e_t + ma[0]*e_{t-1} + ... + ma[q-1]*e_{t-q}

  This is NOT the statsmodels lag-polynomial convention (which expects a
  leading 1 and negated AR signs).
- Innovations are Gaussian: ``e_t ~ N(0, sigma**2)``.
- Paths start from zero initial conditions and a burn-in prefix is
  discarded, so returned paths are *approximately* stationary draws (the
  zero-init transient decays geometrically; lengthen ``burn_in`` to tighten).
- Output is always a 2-D ``(n_paths, n)`` matrix — panel-consistent, no
  shape magic for the single-path case.

Knowledge Tiers
---------------
[T1] ARMA simulation by recursion + burn-in: Hamilton (1994, Ch. 1-4);
     Box & Jenkins (1970).
[T1] An AR(p) process is stationary iff all roots of
     ``z**p - ar[0]*z**(p-1) - ... - ar[p-1]`` lie strictly inside the unit
     circle (equivalently: roots of the AR lag polynomial lie outside it).
[T1] MA(q) processes are stationary for any finite coefficients;
     invertibility matters for estimation, not simulation, and is not
     checked here.

References
----------
[T1] Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
[T1] Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis: Forecasting
     and Control. Holden-Day.
"""

from __future__ import annotations

import numpy as np

from temporalcv._typing import ArrayLike

__all__ = ["simulate_ar", "simulate_arma"]

#: Minimum default burn-in; scaled up with model order (10 per lag).
_MIN_BURN_IN = 100


def _validate_coefficients(coef: ArrayLike, name: str) -> np.ndarray:
    """Normalize a coefficient sequence to a finite 1-D float array."""
    arr = np.asarray(coef, dtype=float)
    if arr.ndim > 1:
        raise ValueError(f"{name} must be a 1-D coefficient sequence, got ndim={arr.ndim}")
    arr = np.atleast_1d(arr)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite coefficients: {arr}")
    return arr


def _check_ar_stationarity(ar: np.ndarray) -> None:
    """Raise if AR coefficients define a non-stationary recursion.

    Stationarity holds iff every root of the characteristic polynomial
    ``z**p - ar[0]*z**(p-1) - ... - ar[p-1]`` lies strictly inside the
    unit circle.
    """
    if ar.size == 0:
        return
    roots = np.roots(np.concatenate(([1.0], -ar)))
    max_modulus = float(np.max(np.abs(roots))) if roots.size else 0.0
    if max_modulus >= 1.0:
        raise ValueError(
            f"AR coefficients are non-stationary: characteristic root with "
            f"|root| = {max_modulus:.6f} >= 1 (ar = {ar.tolist()})"
        )


def simulate_arma(
    ar: ArrayLike,
    ma: ArrayLike,
    n: int,
    *,
    n_paths: int = 1,
    sigma: float = 1.0,
    burn_in: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Simulate paths from a Gaussian ARMA(p, q) process.

    Generates ``n_paths`` independent paths by direct recursion from zero
    initial conditions, discarding a burn-in prefix so the returned segment
    is approximately stationary.

    Parameters
    ----------
    ar : ArrayLike
        AR coefficients ``[phi_1, ..., phi_p]`` in the recursion convention
        (``y_t = phi_1*y_{t-1} + ...``). Pass ``[]`` for no AR part.
        Must define a stationary recursion (all characteristic roots
        strictly inside the unit circle).
    ma : ArrayLike
        MA coefficients ``[theta_1, ..., theta_q]``
        (``... + e_t + theta_1*e_{t-1} + ...``). Pass ``[]`` for no MA part.
    n : int
        Number of observations per returned path. Must be >= 1.
    n_paths : int, default 1
        Number of independent paths. Must be >= 1.
    sigma : float, default 1.0
        Innovation standard deviation. Must be finite and > 0.
    burn_in : int, optional
        Number of leading observations to generate and discard. Defaults to
        ``max(100, 10 * (p + q))``. Must be >= 0 if given (``0`` means the
        zero-init transient is NOT discarded).
    rng : np.random.Generator | int | None, optional
        Random generator, integer seed, or None (fresh entropy). Routed
        through ``np.random.default_rng``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n)`` — always 2-D, including for
        ``n_paths=1`` (index ``[0]`` for a single series).

    Raises
    ------
    ValueError
        If ``n < 1``, ``n_paths < 1``, ``sigma <= 0`` or non-finite,
        ``burn_in < 0``, coefficients are non-finite or not 1-D, or the AR
        part is non-stationary.

    Knowledge Tier: [T1] Standard ARMA generation (Hamilton 1994).

    Examples
    --------
    >>> paths = simulate_arma([0.5], [0.3], n=200, n_paths=3, rng=42)
    >>> paths.shape
    (3, 200)
    >>> white_noise = simulate_arma([], [], n=100, rng=0)
    >>> white_noise.shape
    (1, 100)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be finite and positive, got {sigma}")

    ar_arr = _validate_coefficients(ar, "ar")
    ma_arr = _validate_coefficients(ma, "ma")
    _check_ar_stationarity(ar_arr)

    p, q = ar_arr.size, ma_arr.size
    if burn_in is None:
        burn_in = max(_MIN_BURN_IN, 10 * (p + q))
    elif burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}")

    generator = np.random.default_rng(rng)
    total = burn_in + n
    eps = generator.normal(0.0, sigma, size=(n_paths, total))
    y = np.zeros((n_paths, total))

    for t in range(total):
        acc = eps[:, t].copy()
        for i in range(1, min(p, t) + 1):
            acc += ar_arr[i - 1] * y[:, t - i]
        for j in range(1, min(q, t) + 1):
            acc += ma_arr[j - 1] * eps[:, t - j]
        y[:, t] = acc

    return y[:, burn_in:]


def simulate_ar(
    ar: ArrayLike,
    n: int,
    *,
    n_paths: int = 1,
    sigma: float = 1.0,
    burn_in: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Simulate paths from a Gaussian AR(p) process.

    Convenience wrapper over :func:`simulate_arma` with no MA part; see it
    for conventions, parameters, and raises.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n)`` — always 2-D.

    Knowledge Tier: [T1] Standard AR generation.

    Examples
    --------
    >>> path = simulate_ar([0.9], n=500, rng=42)[0]
    >>> path.shape
    (500,)
    """
    return simulate_arma(ar, [], n, n_paths=n_paths, sigma=sigma, burn_in=burn_in, rng=rng)
