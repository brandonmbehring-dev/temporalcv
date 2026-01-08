"""
Example 08: Regime-Stratified Evaluation
========================================

Real-World Case Study: Model Performance Varies by Market Regime
----------------------------------------------------------------
A model that looks great on average may fail catastrophically during market
stress. This is one of the most expensive mistakes in production forecasting:

1. **Hidden Failures**: A model with 3% average MAE might have 8% MAE during
   high volatility—exactly when accuracy matters most.

2. **False Confidence**: Aggregate metrics hide regime-dependent performance,
   leading to deployment of models that fail during market stress.

3. **Wrong Model Selection**: The "best" model by global metrics may be the
   worst in the regime that matters (e.g., high volatility for risk management).

This example demonstrates how to detect these issues using temporalcv's
regime-stratified evaluation.

Key Concepts
------------
- Volatility regimes: LOW, MED, HIGH based on rolling volatility
- Stratified metrics: MAE/RMSE computed per-regime
- run_gates_stratified(): Validation gates checked per-regime
- Critical insight: Compute volatility on CHANGES, not levels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# temporalcv imports
from temporalcv import WalkForwardCV
from temporalcv.regimes import (
    classify_volatility_regime,
    compute_stratified_metrics,
    get_regime_counts,
)
from temporalcv.gates import (
    gate_signal_verification,
    gate_suspicious_improvement,
    run_gates_stratified,
)
from temporalcv.persistence import compute_persistence_mae

# =============================================================================
# PART 1: Generate Synthetic Data with Known Volatility Regimes
# =============================================================================


def generate_regime_data(
    n_samples: int = 600,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic data with distinct volatility regimes.

    The data has three phases:
    - LOW volatility: Stable market (first third)
    - HIGH volatility: Crisis period (middle third)
    - MED volatility: Recovery/normal (final third)

    This structure lets us verify that regime classification works and
    that per-regime metrics expose hidden model weaknesses.

    Parameters
    ----------
    n_samples : int
        Total number of samples (divided into thirds by regime).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with target values, features, and true regime labels.
    """
    rng = np.random.default_rng(seed)

    n_per_regime = n_samples // 3

    # Generate target with different volatility per regime
    # LOW regime: small noise
    y_low = np.cumsum(rng.normal(0.01, 0.02, n_per_regime))

    # HIGH regime: large noise with occasional jumps
    y_high_changes = rng.normal(0.0, 0.08, n_per_regime)
    # Add some jumps to make it clearly "crisis-like"
    jump_indices = rng.choice(n_per_regime, size=5, replace=False)
    y_high_changes[jump_indices] = rng.choice([-1, 1], size=5) * rng.uniform(0.1, 0.2, 5)
    y_high = y_low[-1] + np.cumsum(y_high_changes)

    # MED regime: moderate noise
    y_med = y_high[-1] + np.cumsum(rng.normal(0.005, 0.04, n_per_regime))

    # Combine
    y = np.concatenate([y_low, y_high, y_med])
    true_regimes = (
        ["LOW"] * n_per_regime + ["HIGH"] * n_per_regime + ["MED"] * n_per_regime
    )

    # Generate features (lagged values + noise to create predictable signal)
    # In LOW regime: features are more predictive
    # In HIGH regime: features are less predictive (regime shift)
    X = np.zeros((n_samples, 3))

    # Feature 1: Lagged target (with varying noise by regime)
    noise_scale = np.concatenate([
        np.full(n_per_regime, 0.01),   # LOW: low noise, predictable
        np.full(n_per_regime, 0.05),   # HIGH: high noise, unpredictable
        np.full(n_per_regime, 0.025),  # MED: medium noise
    ])
    X[1:, 0] = y[:-1] + rng.normal(0, noise_scale[1:])

    # Feature 2: Lagged change
    changes = np.diff(y)
    X[2:, 1] = changes[:-1]

    # Feature 3: Momentum (lagged rolling mean of changes)
    momentum = np.convolve(changes, np.ones(5) / 5, mode="valid")
    X[6:, 2] = momentum[:-1]

    # Create DataFrame
    df = pd.DataFrame({
        "y": y,
        "x1_lagged_level": X[:, 0],
        "x2_lagged_change": X[:, 1],
        "x3_momentum": X[:, 2],
        "true_regime": true_regimes,
    })

    # Add index for time reference
    df.index = pd.date_range("2020-01-01", periods=n_samples, freq="D")

    return df


print("=" * 70)
print("EXAMPLE 08: REGIME-STRATIFIED EVALUATION")
print("=" * 70)

# Generate data
df = generate_regime_data(n_samples=600, seed=42)

print("\n📊 Generated data with known volatility regimes:")
print(f"   Total samples: {len(df)}")
print(f"   True regime distribution: {dict(pd.Series(df['true_regime']).value_counts())}")

# =============================================================================
# PART 2: The Problem — Why Aggregate Metrics Hide Critical Failures
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: THE PROBLEM — WHY AGGREGATE METRICS HIDE CRITICAL FAILURES")
print("=" * 70)

print("""
Consider a model deployed for risk management. You need it to work during
market stress (HIGH volatility). But if you only look at aggregate MAE:

   Model A: Global MAE = 0.035
   Model B: Global MAE = 0.038

Model A looks better! But what if:

   Model A:  LOW=0.010, MED=0.025, HIGH=0.070  ← Fails when it matters!
   Model B:  LOW=0.030, MED=0.035, HIGH=0.050  ← Consistent performer

For risk management, Model B is clearly superior. But aggregate metrics
hide this completely.
""")

# =============================================================================
# PART 3: WRONG Approach — Ignore Regimes, Report Global Average
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: WRONG APPROACH — IGNORE REGIMES")
print("=" * 70)

# Prepare data (skip initial rows with NaN features)
valid_mask = ~df[["x1_lagged_level", "x2_lagged_change", "x3_momentum"]].isna().any(axis=1)
df_valid = df[valid_mask].copy()

X = df_valid[["x1_lagged_level", "x2_lagged_change", "x3_momentum"]].values
y = df_valid["y"].values
true_regimes = df_valid["true_regime"].values

# Split into train/test (use last 30% as test)
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
regimes_test = true_regimes[split_idx:]

# Train a model
model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute global metrics only
global_mae = np.mean(np.abs(y_test - y_pred))
global_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("\n❌ WRONG: Only reporting global metrics:")
print(f"   Global MAE:  {global_mae:.4f}")
print(f"   Global RMSE: {global_rmse:.4f}")
print("   → Looks reasonable, right? But this hides critical information...")

# =============================================================================
# PART 4: CORRECT Approach — Stratify by Volatility Regime
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: CORRECT APPROACH — STRATIFY BY VOLATILITY REGIME")
print("=" * 70)

# Step 4.1: Classify regimes using temporalcv
# CRITICAL: Use basis='changes' to compute volatility on first differences
detected_regimes = classify_volatility_regime(
    y_test,
    window=13,  # ~2 weeks of context
    basis="changes",  # CRITICAL: volatility of CHANGES, not levels
    low_percentile=33.0,
    high_percentile=67.0,
)

print("\n📊 Step 4.1: Classify volatility regimes")
print(f"   Method: Rolling std of CHANGES (window=13)")
print(f"   Detected regime distribution: {get_regime_counts(detected_regimes)}")

# Step 4.2: Compute stratified metrics
print("\n📊 Step 4.2: Compute stratified metrics")
stratified_result = compute_stratified_metrics(
    predictions=y_pred,
    actuals=y_test,
    regimes=detected_regimes,
    min_n=10,  # Require at least 10 samples per regime
)

print("\n" + stratified_result.summary())

# Highlight the key insight
regime_maes = {r: m["mae"] for r, m in stratified_result.by_regime.items()}
worst_regime = max(regime_maes, key=regime_maes.get)
best_regime = min(regime_maes, key=regime_maes.get)

print(f"\n⚠️  KEY INSIGHT:")
print(f"   Worst regime ({worst_regime}): MAE = {regime_maes[worst_regime]:.4f}")
print(f"   Best regime ({best_regime}):  MAE = {regime_maes[best_regime]:.4f}")
print(f"   Ratio: {regime_maes[worst_regime] / regime_maes[best_regime]:.1f}x worse in {worst_regime} regime!")

# =============================================================================
# PART 5: Validate Regime Boundaries Don't Leak
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: VALIDATE REGIME BOUNDARIES DON'T LEAK")
print("=" * 70)

print("""
A subtle bug: if regimes are computed on the FULL dataset (train + test),
the test set knows about future volatility patterns. This is a form of
data leakage that inflates performance.

temporalcv's run_gates_stratified() checks for this and other issues.
""")

# First, compute overall gates
# Persistence baseline
persistence_predictions = np.zeros_like(y_test)
persistence_predictions[1:] = y_test[:-1]  # Predict previous value
persistence_mae = np.mean(np.abs(y_test[1:] - persistence_predictions[1:]))

overall_gates = [
    gate_suspicious_improvement(
        model_metric=global_mae,
        baseline_metric=persistence_mae,
        threshold=0.20,  # HALT if >20% improvement
        warn_threshold=0.10,  # WARN if >10% improvement
    ),
]

# Run stratified validation
stratified_report = run_gates_stratified(
    overall_gates=overall_gates,
    actuals=y_test,
    predictions=y_pred,
    regimes="auto",  # Auto-classify volatility regimes
    min_n_per_regime=10,
    volatility_window=13,
    improvement_threshold=0.20,
    warning_threshold=0.10,
)

print(f"\n✅ Stratified Validation Report:")
print(f"   Overall status: {stratified_report.status}")
print(f"   Regime counts: {stratified_report.regime_counts}")

if stratified_report.status == "HALT":
    print(f"\n🛑 HALT detected! Check per-regime results:")
    for regime, report in stratified_report.by_regime.items():
        print(f"   {regime}: {report.status}")
        for gate in report.gates:
            print(f"      - {gate.name}: {gate.status} — {gate.message}")
elif stratified_report.status == "WARN":
    print(f"\n⚠️  WARN detected. Review per-regime results for potential issues.")
else:
    print(f"\n✅ PASS: Model passes validation across all regimes.")

# =============================================================================
# PART 6: Compare Models Across Regimes
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: COMPARE MODELS ACROSS REGIMES")
print("=" * 70)

print("""
Different models may excel in different regimes. A complex model might
overfit in LOW volatility but capture patterns in HIGH volatility.

Let's compare Gradient Boosting vs Ridge Regression across regimes.
""")

# Train a simpler model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

# Compute stratified metrics for both
gb_stratified = compute_stratified_metrics(y_pred, y_test, detected_regimes, min_n=10)
ridge_stratified = compute_stratified_metrics(ridge_pred, y_test, detected_regimes, min_n=10)

print("\n📊 Model Comparison by Regime:")
print("-" * 60)
print(f"{'Regime':<10} {'GradientBoost MAE':<20} {'Ridge MAE':<20} {'Winner':<10}")
print("-" * 60)

for regime in sorted(gb_stratified.by_regime.keys()):
    gb_mae = gb_stratified.by_regime[regime]["mae"]
    ridge_mae = ridge_stratified.by_regime[regime]["mae"]
    winner = "GB" if gb_mae < ridge_mae else "Ridge"
    diff_pct = abs(gb_mae - ridge_mae) / max(gb_mae, ridge_mae) * 100

    print(f"{regime:<10} {gb_mae:<20.4f} {ridge_mae:<20.4f} {winner:<10} ({diff_pct:.1f}% diff)")

print("-" * 60)
print(f"{'OVERALL':<10} {gb_stratified.overall_mae:<20.4f} {ridge_stratified.overall_mae:<20.4f}")

# =============================================================================
# PART 7: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: KEY TAKEAWAYS")
print("=" * 70)

print("""
1. ALWAYS STRATIFY BY REGIME
   - Global metrics hide regime-dependent failures
   - A model can pass overall but fail catastrophically in specific regimes
   - For risk management, performance in HIGH volatility matters most

2. USE VOLATILITY OF CHANGES, NOT LEVELS
   - classify_volatility_regime(values, basis='changes')
   - Using 'levels' mislabels steady drifts as "volatile"
   - This is a common bug that leads to incorrect regime classification

3. CHECK SAMPLE SIZES PER REGIME
   - Metrics from regimes with n < 10-20 samples are unreliable
   - mask_low_n_regimes() helps identify these
   - Don't draw conclusions from small regime subsets

4. VALIDATE PER-REGIME WITH GATES
   - run_gates_stratified() checks validation per-regime
   - A model might pass globally but HALT in a specific regime
   - This exposes issues that aggregate metrics hide

5. PRODUCTION RECOMMENDATION
   - Report per-regime metrics in model cards
   - Set regime-specific thresholds for deployment decisions
   - Consider ensemble strategies: different models for different regimes
""")

print("\n" + "=" * 70)
print("Example 08 complete.")
print("=" * 70)
