#!/usr/bin/env python3
"""
Example 05: Conformal Prediction for Time Series
================================================

This example demonstrates temporalcv's conformal prediction framework for
constructing distribution-free prediction intervals with coverage guarantees.

Real-World Case Study: Quantifying Forecast Uncertainty
-------------------------------------------------------
Point predictions are insufficient for decision-making. Stakeholders need to
know: "How confident are you?" Conformal prediction provides prediction
intervals with finite-sample coverage guarantees — no distributional
assumptions required.

The Challenge for Time Series:
- Standard conformal assumes exchangeability (i.i.d. data)
- Time series violate this assumption (autocorrelation)
- Solution: Adaptive conformal + walk-forward calibration

Methods Covered:
1. **Split Conformal**: Simple, uses holdout calibration set
2. **Adaptive Conformal**: Adjusts to distribution shift (Gibbs & Candes 2021)
3. **Bootstrap**: Comparison baseline (no coverage guarantee)
4. **Walk-Forward Conformal**: Temporal-aware calibration

Key Insight
-----------
Coverage must be evaluated on data NOT used for calibration. Using calibration
data inflates reported coverage — a common but critical mistake.

Usage
-----
    python 05_conformal_prediction.py

Requirements
------------
    pip install temporalcv scikit-learn
"""

from __future__ import annotations

# sphinx_gallery_thumbnail_number = 1

import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from temporalcv.conformal import (
    AdaptiveConformalPredictor,
    BootstrapUncertainty,
    PredictionInterval,
    SplitConformalPredictor,
    evaluate_interval_quality,
    walk_forward_conformal,
)
from temporalcv.cv import WalkForwardCV

warnings.filterwarnings("ignore")


# =============================================================================
# Generate Time Series Data
# =============================================================================


def generate_ar1_data(
    n: int = 500,
    phi: float = 0.8,
    sigma: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate AR(1) process for demonstration."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2))

    for t in range(1, n):
        y[t] = phi * y[t - 1] + sigma * rng.normal()

    return y


def generate_regime_change_data(
    n: int = 500,
    change_point: int = 350,
    sigma_before: float = 1.0,
    sigma_after: float = 2.5,  # Volatility regime change
    seed: int = 42,
) -> np.ndarray:
    """Generate data with volatility regime change (distribution shift)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    phi = 0.8

    for t in range(1, n):
        sigma = sigma_before if t < change_point else sigma_after
        y[t] = phi * y[t - 1] + sigma * rng.normal()

    return y


def create_features(series: np.ndarray, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Create lagged features for forecasting."""
    n = len(series)
    features = []
    for lag in range(1, n_lags + 1):
        lagged = np.full(n, np.nan)
        lagged[lag:] = series[:-lag]
        features.append(lagged)

    X = np.column_stack(features)
    y = series.copy()
    valid = ~np.isnan(X).any(axis=1)
    return X[valid], y[valid]


# =============================================================================
# Walk-Forward Prediction Pipeline
# =============================================================================


def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    test_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate walk-forward predictions using Ridge regression."""
    cv = WalkForwardCV(
        n_splits=n_splits,
        window_type="expanding",
        test_size=test_size,
        extra_gap=0,
    )

    all_preds = []
    all_actuals = []

    for train_idx, test_idx in cv.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        all_preds.extend(preds)
        all_actuals.extend(y[test_idx])

    return np.array(all_preds), np.array(all_actuals)


# =============================================================================
# Demonstration
# =============================================================================


def demonstrate_conformal_prediction():
    """Demonstrate conformal prediction methods for time series."""
    print("=" * 70)
    print("TEMPORALCV: Conformal Prediction for Time Series")
    print("=" * 70)

    # =========================================================================
    # Part 1: The Problem with Standard Prediction Intervals
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Why Standard Methods Fail for Time Series")
    print("=" * 70)

    print("""
    Standard prediction intervals assume:
    - Gaussian residuals (violated in practice)
    - i.i.d. data (violated by autocorrelation)
    - Known variance (often estimated poorly)

    Conformal prediction provides:
    - Distribution-free coverage guarantee
    - Finite-sample validity (not just asymptotic)
    - No parametric assumptions

    BUT: Standard conformal assumes exchangeability!
    For time series: use walk-forward calibration or adaptive methods.
    """)

    # =========================================================================
    # Part 2: Walk-Forward Conformal Prediction
    # =========================================================================
    print("=" * 70)
    print("PART 2: Walk-Forward Conformal Prediction")
    print("=" * 70)

    # Generate data
    series = generate_ar1_data(n=500, phi=0.8)
    X, y = create_features(series)
    print(f"\nData: {len(y)} observations, AR(1) with phi=0.8")

    # Walk-forward predictions
    predictions, actuals = walk_forward_predict(X, y, n_splits=5, test_size=50)
    print(f"Walk-forward predictions: {len(predictions)} points")

    # Apply conformal with walk-forward calibration
    intervals, quality = walk_forward_conformal(
        predictions=predictions,
        actuals=actuals,
        calibration_fraction=0.3,  # First 30% for calibration
        alpha=0.10,  # 90% intervals
    )

    print(f"\n--- Split Conformal (Walk-Forward Calibration) ---")
    print(f"  Calibration size: {quality['calibration_size']}")
    print(f"  Holdout size: {quality['holdout_size']}")
    print(f"  Calibrated quantile: {quality['quantile']:.4f}")
    print(f"\n  Coverage (90% target):")
    print(f"    Empirical: {quality['coverage']:.1%}")
    print(f"    Gap: {quality['coverage_gap']:+.1%}")
    print(f"  Mean interval width: {quality['mean_width']:.4f}")
    print(f"  Interval score: {quality['interval_score']:.4f} (lower is better)")

    # Check conditional coverage
    if not np.isnan(quality['conditional_gap']):
        print(f"\n  Conditional coverage (low/high prediction magnitude):")
        print(f"    Low predictions: {quality['low_coverage']:.1%}")
        print(f"    High predictions: {quality['high_coverage']:.1%}")
        print(f"    Gap: {quality['conditional_gap']:.1%}")

    # =========================================================================
    # Part 3: Comparing Methods
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Comparing Conformal vs Bootstrap")
    print("=" * 70)

    # Split into calibration and holdout
    cal_size = int(0.3 * len(predictions))
    cal_preds, cal_actuals = predictions[:cal_size], actuals[:cal_size]
    holdout_preds, holdout_actuals = predictions[cal_size:], actuals[cal_size:]

    alpha = 0.10  # 90% intervals

    # Method 1: Split Conformal
    conformal = SplitConformalPredictor(alpha=alpha)
    conformal.calibrate(cal_preds, cal_actuals)
    conformal_intervals = conformal.predict_interval(holdout_preds)
    conformal_quality = evaluate_interval_quality(conformal_intervals, holdout_actuals)

    # Method 2: Bootstrap
    bootstrap = BootstrapUncertainty(n_bootstrap=100, alpha=alpha, random_state=42)
    bootstrap.fit(cal_preds, cal_actuals)
    bootstrap_intervals = bootstrap.predict_interval(holdout_preds)
    bootstrap_quality = evaluate_interval_quality(bootstrap_intervals, holdout_actuals)

    print(f"\n{'Method':<20} {'Coverage':<12} {'Width':<10} {'Int Score':<12}")
    print("-" * 54)
    print(
        f"{'Split Conformal':<20} {conformal_quality['coverage']:.1%}"
        f"        {conformal_quality['mean_width']:.4f}"
        f"    {conformal_quality['interval_score']:.4f}"
    )
    print(
        f"{'Bootstrap':<20} {bootstrap_quality['coverage']:.1%}"
        f"        {bootstrap_quality['mean_width']:.4f}"
        f"    {bootstrap_quality['interval_score']:.4f}"
    )
    print(f"\nTarget coverage: {1 - alpha:.0%}")

    # Interpretation
    if conformal_quality['coverage'] >= 1 - alpha - 0.05:
        print("\n  ✓ Conformal achieves valid coverage")
    else:
        print("\n  ✗ Conformal undercoverage (may indicate distribution shift)")

    # =========================================================================
    # Part 4: Adaptive Conformal for Distribution Shift
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: Adaptive Conformal for Distribution Shift")
    print("=" * 70)

    # Generate data with regime change
    series_shift = generate_regime_change_data(n=500, change_point=350)
    X_shift, y_shift = create_features(series_shift)
    print("\nData: Volatility regime change at t=350")
    print("  Before change: σ = 1.0")
    print("  After change:  σ = 2.5 (150% increase)")

    # Walk-forward predictions
    preds_shift, actuals_shift = walk_forward_predict(X_shift, y_shift, n_splits=5, test_size=50)

    # Static conformal (doesn't adapt)
    cal_size = int(0.3 * len(preds_shift))
    conformal_static = SplitConformalPredictor(alpha=0.10)
    conformal_static.calibrate(preds_shift[:cal_size], actuals_shift[:cal_size])
    static_intervals = conformal_static.predict_interval(preds_shift[cal_size:])
    static_quality = evaluate_interval_quality(static_intervals, actuals_shift[cal_size:])

    # Adaptive conformal (adjusts to shift)
    adaptive = AdaptiveConformalPredictor(alpha=0.10, gamma=0.05)
    adaptive.initialize(preds_shift[:cal_size], actuals_shift[:cal_size])

    # Online updates
    adaptive_lowers = []
    adaptive_uppers = []
    for pred, actual in zip(preds_shift[cal_size:], actuals_shift[cal_size:]):
        lower, upper = adaptive.predict_interval(pred)
        adaptive_lowers.append(lower)
        adaptive_uppers.append(upper)
        adaptive.update(pred, actual)

    # Evaluate adaptive
    adaptive_intervals = PredictionInterval(
        point=preds_shift[cal_size:],
        lower=np.array(adaptive_lowers),
        upper=np.array(adaptive_uppers),
        confidence=0.90,
        method="adaptive_conformal",
    )
    adaptive_quality = evaluate_interval_quality(adaptive_intervals, actuals_shift[cal_size:])

    print(f"\n--- Comparison Under Distribution Shift ---")
    print(f"{'Method':<20} {'Coverage':<12} {'Width':<10}")
    print("-" * 42)
    print(f"{'Static Conformal':<20} {static_quality['coverage']:.1%}        {static_quality['mean_width']:.4f}")
    print(f"{'Adaptive Conformal':<20} {adaptive_quality['coverage']:.1%}        {adaptive_quality['mean_width']:.4f}")

    # Show adaptation trajectory
    print(f"\n  Adaptive quantile trajectory:")
    print(f"    Initial: {adaptive.quantile_history[0]:.4f}")
    print(f"    Final:   {adaptive.quantile_history[-1]:.4f}")
    print(f"    Change:  {adaptive.quantile_history[-1] - adaptive.quantile_history[0]:+.4f}")

    if adaptive_quality['coverage'] > static_quality['coverage']:
        print("\n  ✓ Adaptive conformal maintains better coverage under shift")
    else:
        print("\n  Note: Both methods affected by distribution shift")

    # =========================================================================
    # Part 5: Best Practices
    # =========================================================================
    print("\n" + "=" * 70)
    print("BEST PRACTICES FOR PREDICTION INTERVALS")
    print("=" * 70)
    print("""
1. NEVER use calibration data for coverage evaluation
   - Split data: calibration → calibrate, holdout → evaluate
   - Using calibration data inflates coverage

2. CHECK conditional coverage
   - Coverage should be uniform across prediction magnitudes
   - Large conditional gap → intervals are miscalibrated

3. USE adaptive conformal for non-stationary data
   - Static conformal assumes stable distribution
   - Adaptive adjusts to regime changes

4. REPORT interval score, not just coverage
   - Coverage alone can be gamed (wide intervals = 100% coverage)
   - Interval score penalizes both undercoverage and width

5. CHOOSE alpha based on decision context
   - Conservative (α=0.01, 99%): High-stakes decisions
   - Standard (α=0.10, 90%): Typical use case
   - Aggressive (α=0.20, 80%): When uncertainty tolerance is high

6. COMPARE to bootstrap baseline
   - Bootstrap has no coverage guarantee
   - But useful calibration check for conformal
    """)

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("=" * 70)
    print("METHOD SELECTION GUIDE")
    print("=" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Scenario                        │ Recommended Method               │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Stationary time series          │ Split Conformal + Walk-Forward   │
    │ Distribution shift expected     │ Adaptive Conformal               │
    │ Comparison baseline             │ Bootstrap (no guarantee)         │
    │ Very limited data (<100)        │ Adaptive with small gamma        │
    │ Coverage guarantee required     │ Conformal (not bootstrap)        │
    └─────────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    demonstrate_conformal_prediction()

# %%
# Visualization: Prediction Intervals with Coverage
# --------------------------------------------------
# This plot shows conformal prediction intervals with actual values overlaid,
# demonstrating coverage and interval width for time series forecasting.

# Generate data and predictions for visualization
series = generate_ar1_data(n=300, phi=0.8)
X, y = create_features(series)
predictions, actuals = walk_forward_predict(X, y, n_splits=4, test_size=30)

# Split and calibrate
cal_size = int(0.3 * len(predictions))
cal_preds, cal_actuals = predictions[:cal_size], actuals[:cal_size]
holdout_preds, holdout_actuals = predictions[cal_size:], actuals[cal_size:]

# Fit conformal predictor
conformal = SplitConformalPredictor(alpha=0.10)
conformal.calibrate(cal_preds, cal_actuals)
intervals = conformal.predict_interval(holdout_preds)

# Compute coverage
covered = (holdout_actuals >= intervals.lower) & (holdout_actuals <= intervals.upper)
coverage = covered.mean()

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top: Prediction intervals with actuals
ax1 = axes[0]
x_range = np.arange(len(holdout_preds))
ax1.fill_between(x_range, intervals.lower, intervals.upper,
                 alpha=0.3, color='#1f77b4', label='90% Prediction Interval')
ax1.plot(x_range, holdout_preds, 'b-', linewidth=1.5, label='Predictions')
ax1.scatter(x_range[covered], holdout_actuals[covered],
            c='#2ca02c', s=30, zorder=5, label=f'Covered ({coverage:.1%})')
ax1.scatter(x_range[~covered], holdout_actuals[~covered],
            c='#d62728', s=50, marker='x', zorder=5, label='Not Covered')
ax1.set_xlabel('Test Index')
ax1.set_ylabel('Value')
ax1.set_title(f'Conformal Prediction Intervals (90% Target Coverage, {coverage:.1%} Achieved)')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Bottom: Interval width over time
ax2 = axes[1]
widths = intervals.upper - intervals.lower
ax2.bar(x_range, widths, color='#1f77b4', alpha=0.7, edgecolor='none')
ax2.axhline(widths.mean(), color='#d62728', linestyle='--', linewidth=2,
            label=f'Mean Width: {widths.mean():.3f}')
ax2.set_xlabel('Test Index')
ax2.set_ylabel('Interval Width')
ax2.set_title('Prediction Interval Width (Constant for Split Conformal)')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Conformal Prediction for Time Series Forecasting', y=1.02, fontsize=14)
plt.show()
