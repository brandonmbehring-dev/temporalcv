"""
Pytest configuration and shared fixtures for temporalcv tests.

Provides:
- Custom markers for slow/monte_carlo tests
- Monte Carlo validation utilities
- Data generating process (DGP) functions
- Shared test fixtures
"""

from typing import Tuple

import numpy as np
import pytest


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "monte_carlo: Monte Carlo calibration tests (run nightly)"
    )


# =============================================================================
# Monte Carlo Validation Utilities
# =============================================================================


def compute_mc_bias(estimates: np.ndarray, true_value: float) -> float:
    """
    Compute absolute bias from Monte Carlo estimates.

    Parameters
    ----------
    estimates : np.ndarray
        Array of estimates from MC simulations
    true_value : float
        True parameter value

    Returns
    -------
    float
        Absolute bias: |mean(estimates) - true_value|
    """
    return abs(np.mean(estimates) - true_value)


def compute_mc_coverage(
    ci_lower: np.ndarray, ci_upper: np.ndarray, true_value: float
) -> float:
    """
    Compute coverage rate from Monte Carlo confidence intervals.

    Parameters
    ----------
    ci_lower : np.ndarray
        Lower bounds of confidence intervals
    ci_upper : np.ndarray
        Upper bounds of confidence intervals
    true_value : float
        True parameter value

    Returns
    -------
    float
        Coverage rate: proportion of CIs containing true_value
        Target: 93-97% for nominal 95% CI
    """
    contains = (np.asarray(ci_lower) <= true_value) & (true_value <= np.asarray(ci_upper))
    return float(np.mean(contains))


def compute_se_accuracy(estimates: np.ndarray, standard_errors: np.ndarray) -> float:
    """
    Compute SE accuracy: relative error between empirical SD and mean SE.

    Parameters
    ----------
    estimates : np.ndarray
        Array of estimates from MC simulations
    standard_errors : np.ndarray
        Array of standard errors from MC simulations

    Returns
    -------
    float
        Relative error: |std(estimates) - mean(SE)| / std(estimates)
        Target: < 10%
    """
    empirical_sd = np.std(estimates, ddof=1)
    mean_se = np.mean(standard_errors)
    if empirical_sd == 0:
        return 0.0 if mean_se == 0 else float("inf")
    return abs(empirical_sd - mean_se) / empirical_sd


def validate_mc_results(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    ci_lower: np.ndarray,
    ci_upper: np.ndarray,
    true_value: float,
    bias_threshold: float = 0.05,
    coverage_range: Tuple[float, float] = (0.93, 0.97),
    se_accuracy_threshold: float = 0.10,
) -> dict:
    """
    Validate Monte Carlo simulation results.

    Parameters
    ----------
    estimates : np.ndarray
        Array of estimates
    standard_errors : np.ndarray
        Array of standard errors
    ci_lower : np.ndarray
        Lower CI bounds
    ci_upper : np.ndarray
        Upper CI bounds
    true_value : float
        True parameter value
    bias_threshold : float, default=0.05
        Maximum acceptable bias
    coverage_range : tuple, default=(0.93, 0.97)
        Acceptable coverage range for 95% CI
    se_accuracy_threshold : float, default=0.10
        Maximum relative SE error

    Returns
    -------
    dict
        Validation results with metrics and pass/fail flags
    """
    bias = compute_mc_bias(estimates, true_value)
    coverage = compute_mc_coverage(ci_lower, ci_upper, true_value)
    se_accuracy = compute_se_accuracy(estimates, standard_errors)

    return {
        "bias": bias,
        "bias_ok": bias < bias_threshold,
        "coverage": coverage,
        "coverage_ok": coverage_range[0] <= coverage <= coverage_range[1],
        "se_accuracy": se_accuracy,
        "se_accuracy_ok": se_accuracy < se_accuracy_threshold,
        "all_pass": (
            bias < bias_threshold
            and coverage_range[0] <= coverage <= coverage_range[1]
            and se_accuracy < se_accuracy_threshold
        ),
    }


# =============================================================================
# Data Generating Processes (DGPs)
# =============================================================================


def dgp_ar1(
    n: int, phi: float, sigma: float = 1.0, random_state: int | None = None
) -> np.ndarray:
    """
    Generate AR(1) process with known parameters.

    y[t] = phi * y[t-1] + epsilon[t], epsilon ~ N(0, sigma^2)

    Parameters
    ----------
    n : int
        Number of observations
    phi : float
        AR(1) coefficient (must be in (-1, 1) for stationarity)
    sigma : float, default=1.0
        Innovation standard deviation
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        AR(1) time series of length n
    """
    if not -1 < phi < 1:
        raise ValueError(f"phi must be in (-1, 1) for stationarity, got {phi}")

    rng = np.random.RandomState(random_state)
    y = np.zeros(n)

    # Initialize from stationary distribution
    stationary_sd = sigma / np.sqrt(1 - phi**2)
    y[0] = rng.normal(0, stationary_sd)

    # Generate series
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal(0, sigma)

    return y


def dgp_ar2(
    n: int,
    phi1: float,
    phi2: float,
    sigma: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate AR(2) process with known parameters.

    y[t] = phi1 * y[t-1] + phi2 * y[t-2] + epsilon[t]

    Parameters
    ----------
    n : int
        Number of observations
    phi1 : float
        AR(1) coefficient
    phi2 : float
        AR(2) coefficient
    sigma : float, default=1.0
        Innovation standard deviation
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        AR(2) time series of length n
    """
    rng = np.random.RandomState(random_state)
    y = np.zeros(n)

    # Simple initialization
    y[0] = rng.normal(0, sigma)
    y[1] = rng.normal(0, sigma)

    # Generate series
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + rng.normal(0, sigma)

    return y


def dgp_white_noise(
    n: int, sigma: float = 1.0, random_state: int | None = None
) -> np.ndarray:
    """
    Generate white noise (IID Gaussian).

    Parameters
    ----------
    n : int
        Number of observations
    sigma : float, default=1.0
        Standard deviation
    random_state : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        White noise series
    """
    rng = np.random.RandomState(random_state)
    return rng.normal(0, sigma, n)


def dgp_heavy_tailed(
    n: int, df: float = 3.0, random_state: int | None = None
) -> np.ndarray:
    """
    Generate heavy-tailed noise (Student-t distribution).

    Parameters
    ----------
    n : int
        Number of observations
    df : float, default=3.0
        Degrees of freedom (lower = heavier tails)
    random_state : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        Heavy-tailed series
    """
    rng = np.random.RandomState(random_state)
    return rng.standard_t(df, n)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Fixed RNG for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_ar1_data():
    """AR(1) data with known parameters for testing."""
    return {
        "y": dgp_ar1(n=200, phi=0.6, sigma=1.0, random_state=42),
        "phi": 0.6,
        "sigma": 1.0,
        "n": 200,
        "theoretical_mae": 1.0 * np.sqrt(2 / np.pi),  # ~0.798
    }


@pytest.fixture
def simple_white_noise():
    """White noise data for residual tests."""
    return dgp_white_noise(n=100, sigma=1.0, random_state=42)


@pytest.fixture
def autocorrelated_residuals():
    """AR(1) residuals that should trigger autocorrelation detection."""
    return dgp_ar1(n=100, phi=0.7, sigma=1.0, random_state=42)


@pytest.fixture
def heavy_tailed_residuals():
    """Heavy-tailed residuals that should trigger normality warnings."""
    return dgp_heavy_tailed(n=100, df=3.0, random_state=42)
