"""
Theoretical Bounds Validation for Time Series Forecasts.

Provides theoretical minimum error bounds for AR(p) processes.
A model that beats these bounds is likely suffering from data leakage.

Knowledge Tiers
---------------
[T1] AR(1) 1-step MSE = σ² (innovation variance is irreducible)
[T1] AR(1) h-step MSE = σ² · Σ(φ^(2i) for i=0..h-1) = σ² · (1 - φ^(2h)) / (1 - φ²)
[T1] AR(1) 1-step MAE = σ · √(2/π) ≈ 0.798σ (half-normal mean)
[T1] AR(2) theory: Hamilton (1994), stationary if roots outside unit circle
[T3] Tolerance factor 1.5 allows for finite-sample variation (empirical choice)

Theory
------
For an AR(1) process: y_t = φ·y_{t-1} + ε_t where ε_t ~ N(0, σ²)

The optimal h-step ahead forecast E[y_{t+h} | y_t, y_{t-1}, ...] = φ^h · y_t

The h-step ahead forecast error is:
  e_{t+h} = y_{t+h} - E[y_{t+h} | y_t] = Σ(φ^i · ε_{t+h-i} for i=0..h-1)

Since ε terms are independent:
  Var(e_{t+h}) = σ² · Σ(φ^(2i) for i=0..h-1) = σ² · (1 - φ^(2h)) / (1 - φ²)

For h=1: MSE = σ² (just the innovation variance)
For h→∞: MSE → σ² / (1 - φ²) = Var(y) (unconditional variance)

References
----------
[T1] Hamilton, J.D. (1994). Time Series Analysis. Princeton University Press.
     Chapter 4: Linear Stationary Time Series Models.
[T1] Box, G.E.P. & Jenkins, G.M. (1970). Time Series Analysis.
"""

from __future__ import annotations

from typing import Optional, cast

import numpy as np

from temporalcv.gates import GateResult, GateStatus


def theoretical_ar1_mse_bound(
    phi: float,
    sigma_sq: float,
    h: int = 1,
) -> float:
    """
    Compute theoretical minimum MSE for AR(1) process at horizon h.

    For AR(1): y_t = φ·y_{t-1} + ε_t, ε_t ~ N(0, σ²)
    The h-step ahead MSE = σ² · (1 - φ^(2h)) / (1 - φ²)

    Parameters
    ----------
    phi : float
        AR(1) coefficient. Must satisfy |φ| < 1 for stationarity.
    sigma_sq : float
        Innovation variance σ².
    h : int, default=1
        Forecast horizon.

    Returns
    -------
    float
        Theoretical minimum MSE for h-step ahead forecast.

    Raises
    ------
    ValueError
        If |phi| >= 1 (non-stationary) or sigma_sq <= 0 or h < 1.

    Knowledge Tier: [T1] Standard time series theory (Hamilton 1994).

    Examples
    --------
    >>> # Random walk (phi=0): MSE = sigma_sq for all h
    >>> theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=1)
    1.0
    >>> theoretical_ar1_mse_bound(phi=0.0, sigma_sq=1.0, h=10)
    10.0

    >>> # High persistence (phi=0.9): MSE grows with horizon
    >>> theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=1)
    1.0
    >>> theoretical_ar1_mse_bound(phi=0.9, sigma_sq=1.0, h=5)  # doctest: +ELLIPSIS
    4.0...
    """
    # Validation
    if abs(phi) >= 1.0:
        raise ValueError(f"phi must satisfy |phi| < 1 for stationarity, got {phi}")
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be positive, got {sigma_sq}")
    if h < 1:
        raise ValueError(f"horizon h must be >= 1, got {h}")

    # Special case: phi = 0 (white noise)
    if phi == 0:
        return float(sigma_sq * h)

    # General case: geometric sum
    # MSE = σ² · Σ(φ^(2i) for i=0..h-1) = σ² · (1 - φ^(2h)) / (1 - φ²)
    phi_sq = phi ** 2
    mse = sigma_sq * (1.0 - phi_sq ** h) / (1.0 - phi_sq)
    return float(mse)


def theoretical_ar1_mae_bound(
    sigma: float,
    phi: float = 0.0,
    h: int = 1,
) -> float:
    """
    Compute theoretical minimum MAE for AR(1) process at horizon h.

    For Gaussian innovations, MAE = √(2/π) · RMSE ≈ 0.798 · RMSE.

    Parameters
    ----------
    sigma : float
        Innovation standard deviation σ (not variance).
    phi : float, default=0.0
        AR(1) coefficient. Must satisfy |φ| < 1.
    h : int, default=1
        Forecast horizon.

    Returns
    -------
    float
        Theoretical minimum MAE for h-step ahead forecast.

    Knowledge Tier: [T1] For N(0, σ²), E[|X|] = σ·√(2/π) (half-normal mean).

    Examples
    --------
    >>> # For h=1, MAE = sigma * sqrt(2/pi)
    >>> import numpy as np
    >>> theoretical_ar1_mae_bound(sigma=1.0, phi=0.0, h=1)  # doctest: +ELLIPSIS
    0.797...
    """
    mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma ** 2, h=h)
    rmse = np.sqrt(mse)
    # E[|X|] = σ·√(2/π) for X ~ N(0, σ²)
    mae = rmse * np.sqrt(2.0 / np.pi)
    return float(mae)


def theoretical_ar2_mse_bound(
    phi1: float,
    phi2: float,
    sigma_sq: float,
    h: int = 1,
) -> float:
    """
    Compute theoretical minimum MSE for AR(2) process at horizon h.

    For AR(2): y_t = φ₁·y_{t-1} + φ₂·y_{t-2} + ε_t

    Parameters
    ----------
    phi1 : float
        First AR coefficient.
    phi2 : float
        Second AR coefficient.
    sigma_sq : float
        Innovation variance σ².
    h : int, default=1
        Forecast horizon.

    Returns
    -------
    float
        Theoretical minimum MSE for h-step ahead forecast.

    Raises
    ------
    ValueError
        If process is non-stationary or inputs invalid.

    Notes
    -----
    Stationarity conditions for AR(2):
    - φ₁ + φ₂ < 1
    - φ₂ - φ₁ < 1
    - |φ₂| < 1

    Knowledge Tier: [T1] AR(2) MSE computation (Hamilton 1994, Ch. 4).
    """
    if sigma_sq <= 0:
        raise ValueError(f"sigma_sq must be positive, got {sigma_sq}")
    if h < 1:
        raise ValueError(f"horizon h must be >= 1, got {h}")

    # Check stationarity conditions
    if phi1 + phi2 >= 1 or phi2 - phi1 >= 1 or abs(phi2) >= 1:
        raise ValueError(
            f"AR(2) coefficients violate stationarity: "
            f"φ₁={phi1}, φ₂={phi2}. "
            f"Need: φ₁+φ₂<1, φ₂-φ₁<1, |φ₂|<1"
        )

    # Compute MSE via recursion for ψ weights
    # y_t = Σ(ψⱼ·ε_{t-j}) where ψ₀=1, ψ₁=φ₁, ψⱼ=φ₁ψⱼ₋₁+φ₂ψⱼ₋₂
    # MSE(h) = σ² · Σ(ψⱼ² for j=0..h-1)

    psi = np.zeros(h)
    psi[0] = 1.0
    if h > 1:
        psi[1] = phi1
    for j in range(2, h):
        psi[j] = phi1 * psi[j - 1] + phi2 * psi[j - 2]

    mse = sigma_sq * np.sum(psi ** 2)
    return float(mse)


def check_against_ar1_bounds(
    model_mse: float,
    phi: float,
    sigma_sq: float,
    h: int = 1,
    tolerance: float = 1.5,
    metric_name: str = "MSE",
) -> GateResult:
    """
    Check if model MSE beats theoretical AR(1) minimum.

    If model_mse < theoretical_mse / tolerance, this indicates
    the model is "impossibly good" — likely due to data leakage.

    Parameters
    ----------
    model_mse : float
        Observed model MSE.
    phi : float
        Estimated AR(1) coefficient.
    sigma_sq : float
        Estimated innovation variance.
    h : int, default=1
        Forecast horizon.
    tolerance : float, default=1.5
        Factor to account for finite-sample variation.
        HALT if model_mse < theoretical_mse / tolerance.
    metric_name : str, default="MSE"
        Name for the metric in messages.

    Returns
    -------
    GateResult
        HALT if model beats bounds (leakage likely)
        WARN if model is suspiciously close to bounds
        PASS if model is within expected range

    Knowledge Tier: [T3] Tolerance factor 1.5 is empirical heuristic.

    Examples
    --------
    >>> # Model with MSE = 0.5 vs theoretical minimum of 1.0
    >>> result = check_against_ar1_bounds(model_mse=0.5, phi=0.9, sigma_sq=1.0)
    >>> result.status.value
    'HALT'

    >>> # Model with MSE = 1.2 vs theoretical minimum of 1.0
    >>> result = check_against_ar1_bounds(model_mse=1.2, phi=0.9, sigma_sq=1.0)
    >>> result.status.value
    'PASS'
    """
    try:
        theoretical_mse = theoretical_ar1_mse_bound(phi=phi, sigma_sq=sigma_sq, h=h)
    except ValueError as e:
        return GateResult(
            name="ar1_theoretical_bounds",
            status=GateStatus.SKIP,
            message=f"Cannot compute bounds: {e}",
            recommendation="Verify AR(1) coefficient and variance estimates",
        )

    ratio = model_mse / theoretical_mse
    threshold = 1.0 / tolerance  # e.g., 0.667 for tolerance=1.5

    # HALT: Beating theoretical bounds (impossible without leakage)
    if ratio < threshold:
        return GateResult(
            name="ar1_theoretical_bounds",
            status=GateStatus.HALT,
            message=(
                f"Model {metric_name} ({model_mse:.4f}) is {ratio:.1%} of theoretical "
                f"minimum ({theoretical_mse:.4f}). This is below the {threshold:.1%} "
                f"threshold, indicating likely data leakage."
            ),
            metric_value=ratio,
            threshold=threshold,
            details={
                "model_mse": model_mse,
                "theoretical_mse": theoretical_mse,
                "phi": phi,
                "sigma_sq": sigma_sq,
                "horizon": h,
                "ratio": ratio,
            },
            recommendation=(
                "Investigate for lookahead bias: "
                "1) Check feature computation for future leakage "
                "2) Verify train/test split respects temporal order "
                "3) Examine threshold/parameter computations"
            ),
        )

    # WARN: Suspiciously close to theoretical bounds (ratio < 1.2)
    if ratio < 1.2:
        return GateResult(
            name="ar1_theoretical_bounds",
            status=GateStatus.WARN,
            message=(
                f"Model {metric_name} ({model_mse:.4f}) is only {ratio:.1%} of "
                f"theoretical minimum ({theoretical_mse:.4f}). This is unusually "
                f"good - verify no subtle leakage."
            ),
            metric_value=ratio,
            threshold=threshold,
            details={
                "model_mse": model_mse,
                "theoretical_mse": theoretical_mse,
                "phi": phi,
                "sigma_sq": sigma_sq,
                "horizon": h,
                "ratio": ratio,
            },
            recommendation="Verify feature engineering does not use future information",
        )

    # PASS: Model is within expected range
    return GateResult(
        name="ar1_theoretical_bounds",
        status=GateStatus.PASS,
        message=(
            f"Model {metric_name} ({model_mse:.4f}) is {ratio:.1%} of theoretical "
            f"minimum ({theoretical_mse:.4f}). Within expected range."
        ),
        metric_value=ratio,
        threshold=threshold,
        details={
            "model_mse": model_mse,
            "theoretical_mse": theoretical_mse,
            "phi": phi,
            "sigma_sq": sigma_sq,
            "horizon": h,
            "ratio": ratio,
        },
    )


def generate_ar1_series(
    phi: float,
    sigma: float,
    n: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic AR(1) time series.

    y_t = φ·y_{t-1} + ε_t, where ε_t ~ N(0, σ²)

    Parameters
    ----------
    phi : float
        AR(1) coefficient. Must satisfy |φ| < 1 for stationarity.
    sigma : float
        Innovation standard deviation.
    n : int
        Number of observations to generate.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Generated AR(1) series of length n.

    Raises
    ------
    ValueError
        If |phi| >= 1 or n < 1 or sigma <= 0.

    Knowledge Tier: [T1] Standard AR(1) generation.

    Examples
    --------
    >>> series = generate_ar1_series(phi=0.9, sigma=1.0, n=100, random_state=42)
    >>> len(series)
    100
    """
    if abs(phi) >= 1.0:
        raise ValueError(f"phi must satisfy |phi| < 1 for stationarity, got {phi}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    rng = np.random.default_rng(random_state)

    # Initialize from stationary distribution
    # Var(y) = σ² / (1 - φ²) for stationary AR(1)
    y0_std = sigma / np.sqrt(1.0 - phi ** 2) if phi != 0 else sigma
    y = np.zeros(n)
    y[0] = rng.normal(0, y0_std)

    # Generate innovations
    innovations = rng.normal(0, sigma, size=n)

    # AR(1) recursion
    for t in range(1, n):
        y[t] = phi * y[t - 1] + innovations[t]

    return cast(np.ndarray, y)


def generate_ar2_series(
    phi1: float,
    phi2: float,
    sigma: float,
    n: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a synthetic AR(2) time series.

    y_t = φ₁·y_{t-1} + φ₂·y_{t-2} + ε_t, where ε_t ~ N(0, σ²)

    Parameters
    ----------
    phi1 : float
        First AR coefficient.
    phi2 : float
        Second AR coefficient.
    sigma : float
        Innovation standard deviation.
    n : int
        Number of observations to generate.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Generated AR(2) series of length n.

    Raises
    ------
    ValueError
        If stationarity conditions violated, sigma <= 0, or n < 2.

    Knowledge Tier: [T1] Standard AR(2) generation.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n < 2:
        raise ValueError(f"n must be >= 2 for AR(2), got {n}")

    # Check stationarity conditions
    if phi1 + phi2 >= 1 or phi2 - phi1 >= 1 or abs(phi2) >= 1:
        raise ValueError(
            f"AR(2) coefficients violate stationarity: "
            f"φ₁={phi1}, φ₂={phi2}. "
            f"Need: φ₁+φ₂<1, φ₂-φ₁<1, |φ₂|<1"
        )

    rng = np.random.default_rng(random_state)

    # Compute unconditional variance for initialization
    # γ₀ = σ² / ((1 - φ₂) · ((1 + φ₂)² - φ₁²))
    try:
        gamma0 = sigma ** 2 / ((1 - phi2) * ((1 + phi2) ** 2 - phi1 ** 2))
        y0_std = np.sqrt(max(gamma0, sigma ** 2))
    except (ZeroDivisionError, ValueError):
        y0_std = sigma

    y = np.zeros(n)
    y[0] = rng.normal(0, y0_std)
    y[1] = rng.normal(0, y0_std)

    # Generate innovations
    innovations = rng.normal(0, sigma, size=n)

    # AR(2) recursion
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + innovations[t]

    return cast(np.ndarray, y)


# Type-only export for public API
__all__ = [
    "theoretical_ar1_mse_bound",
    "theoretical_ar1_mae_bound",
    "theoretical_ar2_mse_bound",
    "check_against_ar1_bounds",
    "generate_ar1_series",
    "generate_ar2_series",
]
