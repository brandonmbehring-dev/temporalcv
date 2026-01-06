#!/usr/bin/env python3
"""
Example 01: Detecting Data Leakage with the Shuffled Target Test
================================================================

This example demonstrates temporalcv's most powerful validation gate:
the shuffled target test. We show how it catches subtle leakage bugs
that would otherwise inflate apparent model performance.

Real-World Case Study: Interest Rate Forecasting
-------------------------------------------------
We forecast 10-Year Treasury rates using lagged features. A common bug
is computing rolling statistics (like 13-week volatility) on the FULL
series before train/test split, leaking future information into features.

The shuffled target test catches this: if your model beats a baseline
trained on randomized targets, the features themselves contain target
information — a definitive leakage signal.

Key Insight
-----------
A properly constructed model should NOT significantly outperform a model
trained on shuffled (randomized) targets. If it does, your features are
encoding temporal position or future information.

Usage
-----
    # With FRED API key (recommended):
    export FRED_API_KEY=your_key_here
    python 01_leakage_detection.py

    # Without API key (uses synthetic data):
    python 01_leakage_detection.py

Requirements
------------
    pip install temporalcv[fred]  # For FRED data
    # or
    pip install temporalcv scikit-learn  # Minimum requirements
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from temporalcv.gates import (
    GateStatus,
    gate_signal_verification,
    gate_suspicious_improvement,
    run_gates,
)

# Suppress sklearn convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Data Loading: FRED or Synthetic Fallback
# =============================================================================


def load_treasury_data() -> Tuple[np.ndarray, str]:
    """
    Load 10-Year Treasury rate data from FRED, or generate realistic synthetic.

    Returns
    -------
    rates : np.ndarray
        Weekly rate observations (500+ points)
    source : str
        "FRED" or "synthetic"
    """
    try:
        from temporalcv.benchmarks import load_fred_rates

        dataset = load_fred_rates(
            series="DGS10",
            start="2010-01-01",
            frequency="W",
        )
        return dataset.values, "FRED (10-Year Treasury)"
    except Exception:
        # Generate realistic synthetic data mimicking Treasury characteristics
        print("Note: Using synthetic data (set FRED_API_KEY for real data)")
        return _generate_synthetic_rates(), "Synthetic (Treasury-like)"


def _generate_synthetic_rates(
    n_samples: int = 600,
    initial_rate: float = 2.5,
    phi: float = 0.995,  # High persistence (typical for rates)
    sigma: float = 0.08,  # Weekly volatility
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic interest rate data with realistic characteristics.

    Mimics Treasury rate dynamics:
    - High persistence (AR(1) coefficient ~0.995)
    - Mean-reverting around long-run level
    - Realistic volatility regime
    """
    rng = np.random.default_rng(seed)
    rates = np.zeros(n_samples)
    rates[0] = initial_rate

    # AR(1) with mean reversion
    long_run_mean = 2.5

    for t in range(1, n_samples):
        innovation = sigma * rng.normal()
        rates[t] = phi * rates[t - 1] + (1 - phi) * long_run_mean + innovation

    return rates


# =============================================================================
# Feature Engineering: Clean vs Leaky
# =============================================================================


def create_clean_features(rates: np.ndarray, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features WITHOUT leakage — the correct way.

    Features are computed using ONLY past data relative to each observation.
    For walk-forward validation, we split THEN compute features.
    """
    n = len(rates)
    features = []

    # Lag features (no leakage — just past values)
    for lag in range(1, n_lags + 1):
        lagged = np.full(n, np.nan)
        lagged[lag:] = rates[:-lag]
        features.append(lagged)

    # Stack and remove NaN rows
    X = np.column_stack(features)
    y = rates.copy()

    # Keep only rows with complete features
    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], y[valid_mask]


def create_leaky_features(rates: np.ndarray, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create features WITH leakage — the WRONG way (intentionally buggy).

    This mimics a catastrophic bug: including future target values as features.
    In real codebases, this happens through:
    - Off-by-one errors in lag computation
    - Target encoding computed on full dataset
    - Improperly aligned rolling windows
    """
    n = len(rates)
    features = []

    # Lag features (same as clean)
    for lag in range(1, n_lags + 1):
        lagged = np.full(n, np.nan)
        lagged[lag:] = rates[:-lag]
        features.append(lagged)

    # BUG: "Smoothed target" feature that accidentally includes current value
    # This mimics an off-by-one error in rolling window computation
    # Real example: using pandas rolling(..., center=True) by mistake
    smoothed = np.full(n, np.nan)
    window = 3
    for t in range(window, n - window):
        # BUG: includes rates[t] and rates[t+1], rates[t+2] (future!)
        smoothed[t] = np.mean(rates[t - window : t + window + 1])
    features.append(smoothed)

    # Stack and remove NaN rows
    X = np.column_stack(features)
    y = rates.copy()

    valid_mask = ~np.isnan(X).any(axis=1)
    return X[valid_mask], y[valid_mask]


# =============================================================================
# Demonstration: Shuffled Target Test Catches Leakage
# =============================================================================


def demonstrate_leakage_detection():
    """
    Demonstrate how the shuffled target test catches data leakage.

    We compare two scenarios:
    1. Clean features (no leakage) — shuffled test should PASS
    2. Leaky features (future info) — shuffled test should HALT
    """
    print("=" * 70)
    print("TEMPORALCV: Detecting Data Leakage with Shuffled Target Test")
    print("=" * 70)

    # Load data
    rates, source = load_treasury_data()
    print(f"\nData source: {source}")
    print(f"Observations: {len(rates)}")
    print(f"Mean rate: {np.mean(rates):.2f}%")
    print(f"Std dev: {np.std(rates):.2f}%")
    print(f"ACF(1): {np.corrcoef(rates[1:], rates[:-1])[0, 1]:.3f} (high persistence)")

    # =========================================================================
    # Scenario 1: Clean Features (Baseline Comparison)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: Clean Features (only lagged values)")
    print("=" * 70)

    X_clean, y_clean = create_clean_features(rates)
    print(f"Feature shape: {X_clean.shape}")
    print("Features: y_{t-1}, y_{t-2}, ..., y_{t-5}")

    # Use Ridge regression (less prone to overfitting)
    model_clean = Ridge(alpha=1.0)

    # Run shuffled target test (permutation mode - default)
    # For high-persistence data, models WILL beat shuffled significantly
    # because lag features genuinely predict the target
    # Note: With permutation mode, metric_value is the p-value
    result_clean = gate_signal_verification(
        model=model_clean,
        X=X_clean,
        y=y_clean,
        n_shuffles=100,  # Need >=100 for statistical power in permutation mode
        random_state=42,
    )

    pvalue_clean = result_clean.metric_value  # p-value in permutation mode
    improvement_clean = result_clean.details.get("improvement_ratio", 0.0)
    print(f"\nShuffled Target Test Result: {result_clean}")
    print(f"  - MAE (real target): {result_clean.details['mae_real']:.4f}")
    print(f"  - MAE (shuffled avg): {result_clean.details['mae_shuffled_avg']:.4f}")
    print(f"  - P-value: {pvalue_clean:.4f}")
    print(f"  - Improvement ratio: {improvement_clean:.1%}")

    # =========================================================================
    # Scenario 2: Leaky Features (Should show MUCH higher improvement)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: Leaky Features (includes future info)")
    print("=" * 70)

    X_leaky, y_leaky = create_leaky_features(rates)
    print(f"Feature shape: {X_leaky.shape}")
    print("BUG: 'Smoothed' feature uses centered window (includes y_t, y_{t+1}, y_{t+2})")

    # Use same model for fair comparison
    model_leaky = Ridge(alpha=1.0)

    result_leaky = gate_signal_verification(
        model=model_leaky,
        X=X_leaky,
        y=y_leaky,
        n_shuffles=100,  # Need >=100 for statistical power in permutation mode
        random_state=42,
    )

    pvalue_leaky = result_leaky.metric_value  # p-value in permutation mode
    improvement_leaky = result_leaky.details.get("improvement_ratio", 0.0)
    print(f"\nShuffled Target Test Result: {result_leaky}")
    print(f"  - MAE (real target): {result_leaky.details['mae_real']:.4f}")
    print(f"  - MAE (shuffled avg): {result_leaky.details['mae_shuffled_avg']:.4f}")
    print(f"  - P-value: {pvalue_leaky:.4f}")
    print(f"  - Improvement ratio: {improvement_leaky:.1%}")

    # =========================================================================
    # Compare the two scenarios
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Clean vs Leaky Features")
    print("=" * 70)
    print(f"\n  Clean features improvement:  {improvement_clean:.1%}")
    print(f"  Leaky features improvement:  {improvement_leaky:.1%}")
    print(f"  Difference (leaky - clean):  {(improvement_leaky - improvement_clean):.1%}")

    if improvement_leaky > improvement_clean + 0.05:
        print("\n  LEAKAGE DETECTED!")
        print("  The leaky features show significantly higher improvement,")
        print("  indicating they contain information about the target's position.")
    else:
        print("\n  Note: Both scenarios show similar improvement patterns.")

    # =========================================================================
    # Practical Gate Usage
    # =========================================================================
    print("\n" + "=" * 70)
    print("PRACTICAL GATE USAGE")
    print("=" * 70)

    # For production use, set a threshold based on domain knowledge
    # Typical guideline: if improvement over shuffled > 95%, suspect leakage
    print("\nRunning gates with production thresholds...")

    # Compute baseline for suspicious improvement gate
    # Use train-test split for OUT-OF-SAMPLE evaluation
    split_idx = int(len(y_clean) * 0.8)
    X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
    y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]

    # Persistence baseline on TEST data
    persistence_preds = X_test[:, 0]  # First lag is y[t-1]
    persistence_mae = np.mean(np.abs(y_test - persistence_preds))

    # Model predictions on TEST data (out-of-sample)
    model_clean.fit(X_train, y_train)
    model_preds = model_clean.predict(X_test)
    model_mae = np.mean(np.abs(y_test - model_preds))

    # Run multiple gates with production thresholds
    # In permutation mode (default), gate HALTs if p-value < alpha (0.05)
    result_shuffled = gate_signal_verification(
        model=Ridge(alpha=1.0),
        X=X_leaky,  # Test the LEAKY features
        y=y_leaky,
        n_shuffles=100,  # Need >=100 for statistical power in permutation mode
        random_state=42,
    )

    gates = [
        result_shuffled,
        gate_suspicious_improvement(
            model_metric=model_mae,
            baseline_metric=persistence_mae,
            threshold=0.20,
            warn_threshold=0.10,
        ),
    ]

    report = run_gates(gates)
    print(report.summary())

    # =========================================================================
    # Key Takeaways
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. The SHUFFLED TARGET TEST is the definitive leakage detector.
   - If your model beats randomized targets, features encode target info.
   - This catches rolling stats computed on full series, lookahead bias, etc.

2. Common leakage sources in time-series:
   - Rolling statistics computed before train/test split
   - Normalization (mean/std) computed on full dataset
   - Feature selection using future data
   - Information from the test period in feature engineering

3. Run the shuffled test BEFORE trusting impressive results.
   - 40%+ improvement over persistence? Probably leakage.
   - Run shuffled test, then investigate if it HALTs.

4. temporalcv gates follow HALT > WARN > PASS priority:
   - HALT: Stop and investigate (critical failure)
   - WARN: Proceed with caution (verify externally)
   - PASS: Validation passed
""")

    return report


if __name__ == "__main__":
    report = demonstrate_leakage_detection()
    print(f"\nFinal status: {report.status}")
