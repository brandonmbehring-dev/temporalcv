"""
Example 09: Multi-Horizon Comparison
====================================

Real-World Case Study: Horizon-Dependent Model Performance
----------------------------------------------------------
A model that excels at short-term forecasting may fail at long horizons.
This is one of the most overlooked aspects of forecast evaluation:

1. **Horizon Decay**: Most models' advantage over baselines degrades with
   forecast horizon. A model beating persistence at h=1 may be indistinguishable
   at h=12.

2. **Optimal Horizon**: The "predictability horizon" is where a model's
   statistically significant advantage disappears. Beyond this, use a simpler
   baseline.

3. **Model Selection**: Different models may be optimal at different horizons.
   Complex models often win short-term, simpler models win long-term.

This example demonstrates temporalcv's multi-horizon comparison framework
using Diebold-Mariano tests with horizon-specific HAC adjustment.

Key Concepts
------------
- compare_horizons(): Two-model comparison across multiple horizons
- compare_models_horizons(): Multi-model comparison at each horizon
- Predictability horizon: Where model advantage becomes insignificant
- Degradation patterns: "degrading", "consistent", "irregular"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# temporalcv imports
from temporalcv.statistical_tests import (
    compare_horizons,
    compare_models_horizons,
    dm_test,
)

# =============================================================================
# PART 1: Generate Multi-Horizon Forecast Data
# =============================================================================


def generate_multi_horizon_data(
    n_samples: int = 400,
    ar_coef: float = 0.8,
    ma_coef: float = 0.3,
    seasonality: int = 7,
    noise_std: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate ARMA(1,1) + seasonal data for multi-horizon forecasting.

    The data has:
    - Strong autocorrelation (AR component) → short-term predictable
    - Moving average component → smoothed shocks
    - Weekly seasonality → pattern at h=7

    This structure creates horizon-dependent predictability:
    - h=1: High predictability (AR dominates)
    - h=7: Moderate (seasonality helps)
    - h>10: Low (noise dominates)

    Parameters
    ----------
    n_samples : int
        Total samples to generate.
    ar_coef : float
        AR(1) coefficient (persistence).
    ma_coef : float
        MA(1) coefficient.
    seasonality : int
        Seasonal period (7 = weekly).
    noise_std : float
        Innovation standard deviation.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with target and lagged features.
    """
    rng = np.random.default_rng(seed)

    # Generate innovations
    innovations = rng.normal(0, noise_std, n_samples)

    # Build ARMA(1,1) + seasonal component
    y = np.zeros(n_samples)
    y[0] = innovations[0]

    for t in range(1, n_samples):
        # AR(1) component
        ar_term = ar_coef * y[t - 1]

        # MA(1) component
        ma_term = ma_coef * innovations[t - 1]

        # Seasonal component
        if t >= seasonality:
            seasonal_term = 0.3 * y[t - seasonality]
        else:
            seasonal_term = 0

        y[t] = ar_term + ma_term + seasonal_term + innovations[t]

    # Create DataFrame with lagged features
    df = pd.DataFrame({"y": y})

    # Add lagged features (strictly causal)
    for lag in [1, 2, 3, 7]:
        df[f"y_lag{lag}"] = df["y"].shift(lag)

    # Add time index
    df.index = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    df = df.dropna()

    return df


print("=" * 70)
print("EXAMPLE 09: MULTI-HORIZON COMPARISON")
print("=" * 70)

# Generate data
df = generate_multi_horizon_data(n_samples=500, seed=42)

print(f"\n📊 Generated ARMA(1,1) + seasonal data: {len(df)} samples")
print("   AR coefficient: 0.8 (strong short-term persistence)")
print("   Seasonality: 7 days (weekly pattern)")

# =============================================================================
# PART 2: The Problem — Single-Horizon Evaluation Misleads
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: THE PROBLEM — SINGLE-HORIZON EVALUATION MISLEADS")
print("=" * 70)

print(
    """
Common mistake: Evaluate model only at h=1 (one-step ahead).

   "Our model beats persistence by 15% at h=1!"

But this tells you nothing about h=4, h=7, or h=12. If your business
needs 7-day forecasts, the h=1 result is irrelevant.

Worse: A model tuned for h=1 may be WORSE than baseline at h=7.
"""
)

# =============================================================================
# PART 3: Generate Multi-Step Forecasts
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: GENERATE MULTI-STEP FORECASTS")
print("=" * 70)

# Split data
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

X_train = df_train[["y_lag1", "y_lag2", "y_lag3", "y_lag7"]].values
y_train = df_train["y"].values
X_test = df_test[["y_lag1", "y_lag2", "y_lag3", "y_lag7"]].values
y_test = df_test["y"].values

# Train models
gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
ridge_model = Ridge(alpha=1.0)

gb_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Generate predictions (direct forecasting approach)
gb_pred = gb_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# Persistence baseline: y[t] = y[t-h]
# For multi-horizon, we use the lagged value as the "forecast"
persistence_pred = X_test[:, 0]  # y_lag1 column

# Compute errors
gb_errors = y_test - gb_pred
ridge_errors = y_test - ridge_pred
persistence_errors = y_test - persistence_pred

print(f"✅ Generated forecasts for {len(y_test)} test points")
print("   Models: GradientBoosting, Ridge, Persistence")
print("   Horizons to test: 1, 2, 4, 7, 12")

# =============================================================================
# PART 4: WRONG Approach — Report Only h=1
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: WRONG APPROACH — REPORT ONLY h=1")
print("=" * 70)

# Single-horizon DM test
dm_h1 = dm_test(gb_errors, persistence_errors, h=1)

print("\n❌ Single-horizon result (h=1 only):")
print(f"   GB vs Persistence: DM stat = {dm_h1.statistic:.3f}, p = {dm_h1.pvalue:.4f}")

if dm_h1.pvalue < 0.05:
    print("   → 'GradientBoosting significantly beats persistence!'")
    print("   → But this is ONLY for h=1 forecasts...")

# =============================================================================
# PART 5: CORRECT Approach — compare_horizons()
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: CORRECT APPROACH — compare_horizons()")
print("=" * 70)

print(
    """
compare_horizons() runs DM tests at multiple horizons with:
- Horizon-specific HAC bandwidth (h-1) for MA(h-1) error structure
- Harvey et al. (1997) small-sample correction
- Pattern detection: "degrading", "consistent", or "irregular"
- Predictability horizon: Where advantage becomes insignificant
"""
)

# Compare GB vs Persistence across horizons
horizons = (1, 2, 4, 7, 12)

result_gb = compare_horizons(
    gb_errors,
    persistence_errors,
    horizons=horizons,
    alternative="less",  # Test if GB has lower error
    model_1_name="GradientBoosting",
    model_2_name="Persistence",
)

print("\n📊 GradientBoosting vs Persistence across horizons:")
print(result_gb.to_markdown())

# Key insights
print("\n🔍 Key Insights:")
print(f"   Significant horizons (p < 0.05): {result_gb.significant_horizons}")
print(f"   First insignificant horizon: {result_gb.first_insignificant_horizon}")
print(f"   Degradation pattern: {result_gb.degradation_pattern}")

# Note: If p-values are high (>0.5), it means model1 (GB) is WORSE than baseline
# The DM test with alternative="less" tests if model1 has LOWER error
# High p-values mean we can't reject that model1 is worse or equal
if len(result_gb.significant_horizons) == 0:
    print("\n⚠️  NOTE: No significant improvement at any horizon!")
    print("   This often happens when complex models overfit on simple AR data.")

# =============================================================================
# PART 6: Multi-Model Horizon Comparison
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: MULTI-MODEL HORIZON COMPARISON")
print("=" * 70)

print(
    """
compare_models_horizons() compares multiple models at each horizon:
- Bonferroni-corrected pairwise DM tests
- Identifies best model at each horizon
- Detects if one model consistently dominates
"""
)

# Prepare error dictionary
errors_dict = {
    "GradientBoosting": gb_errors,
    "Ridge": ridge_errors,
    "Persistence": persistence_errors,
}

# Multi-model, multi-horizon comparison
multi_result = compare_models_horizons(
    errors_dict,
    horizons=horizons,
    loss="squared",
    alpha=0.05,
)

print("\n📊 Multi-Model Comparison by Horizon:")
print("-" * 60)
print(f"{'Horizon':<10} {'Best Model':<20} {'Notes':<30}")
print("-" * 60)

for h in horizons:
    horizon_result = multi_result.pairwise_by_horizon[h]
    best = horizon_result.best_model
    n_sig = len(horizon_result.significant_pairs)
    notes = f"{n_sig} significant differences" if n_sig > 0 else "No significant differences"
    print(f"h={h:<8} {best:<20} {notes:<30}")

print("-" * 60)

# Summary
print("\n📈 Summary:")
print(f"   Best models by horizon: {multi_result.best_model_by_horizon}")
if multi_result.consistent_best:
    print(f"   Consistent winner: {multi_result.consistent_best}")
else:
    print("   No consistent winner (model selection depends on horizon)")

# =============================================================================
# PART 7: Visualizing Horizon-Dependent Performance
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: HORIZON-DEPENDENT PERFORMANCE METRICS")
print("=" * 70)

# Compute MAE at each "effective" horizon
# (In practice, you'd re-train for each horizon; here we show the pattern)
print("\n📊 Mean Squared Error by Model:")
print("-" * 50)
print(f"{'Model':<20} {'MSE':<15}")
print("-" * 50)

for name, errors in errors_dict.items():
    mse = np.mean(errors**2)
    print(f"{name:<20} {mse:<15.4f}")

print("-" * 50)

# Show the degradation pattern more explicitly
print("\n📉 p-values by horizon (GB vs Persistence):")
print("-" * 40)
print(f"{'Horizon':<10} {'p-value':<15} {'Significant?':<15}")
print("-" * 40)

for h in horizons:
    p_val = result_gb.dm_results[h].pvalue
    sig = "YES ✅" if p_val < 0.05 else "NO"
    print(f"h={h:<8} {p_val:<15.4f} {sig:<15}")

print("-" * 40)

# =============================================================================
# PART 8: Practical Recommendations
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: PRACTICAL RECOMMENDATIONS")
print("=" * 70)

print(
    """
Based on multi-horizon analysis:

1. IDENTIFY YOUR BUSINESS HORIZON
   - What forecast horizon does your application need?
   - Evaluate models AT THAT HORIZON, not just h=1

2. USE PREDICTABILITY HORIZON FOR MODEL SELECTION
   - Beyond the predictability horizon, switch to simpler models
   - Complex models' advantage fades; simpler models are more robust

3. CONSIDER HORIZON-SPECIFIC MODELS
   - Short-term (h≤4): Complex models (GB, LSTM) may help
   - Long-term (h>7): Simpler models (Ridge, ARIMA) often sufficient
   - Very long (h>12): Consider seasonal naive or external regressors

4. REPORT MULTIPLE HORIZONS
   - Never report just h=1 performance
   - Show degradation curve to stakeholders
   - Be honest about where model advantage ends
"""
)

# =============================================================================
# PART 9: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 9: KEY TAKEAWAYS")
print("=" * 70)

print(
    """
1. SINGLE-HORIZON EVALUATION IS DANGEROUS
   - h=1 results don't generalize to h=7 or h=12
   - Business decisions often need longer horizons
   - Always test at your actual deployment horizon

2. USE compare_horizons() FOR TWO-MODEL COMPARISON
   - Horizon-specific HAC bandwidth (DM test accounts for MA(h-1))
   - Detects predictability horizon automatically
   - Shows degradation pattern

3. USE compare_models_horizons() FOR MULTIPLE MODELS
   - Bonferroni-corrected pairwise comparisons
   - Identifies best model at each horizon
   - Reveals when "best model" changes with horizon

4. PREDICTABILITY HORIZON IS ACTIONABLE
   - First horizon where model is not significantly better than baseline
   - Beyond this, prefer simpler/cheaper models
   - Document this in model cards

5. DEGRADATION PATTERNS INDICATE MODEL QUALITY
   - "degrading": Normal for most models (advantage fades)
   - "consistent": Rare but desirable (model robust across horizons)
   - "irregular": Investigate — may indicate overfitting or data issues
"""
)

print("\n" + "=" * 70)
print("Example 09 complete.")
print("=" * 70)
