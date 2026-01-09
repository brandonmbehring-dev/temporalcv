"""
Example 11: Web Traffic Forecasting
===================================

Real-World Case Study: Website Traffic Prediction
-------------------------------------------------
Web traffic data has unique characteristics that require careful handling:

1. **Strong Weekly Seasonality**: Traffic patterns repeat weekly
   - Weekdays vs weekends differ dramatically
   - Business sites peak Monday-Friday; entertainment peaks weekends

2. **Trend + Level Shifts**: Growth over time with occasional jumps
   - Marketing campaigns cause spikes
   - Product launches shift baseline

3. **High Noise**: Day-to-day variation is large
   - Weather, news events, viral content
   - Makes point forecasts hard; intervals matter

This example demonstrates:
- Weekly seasonality handling with appropriate CV
- MASE metric for seasonal data
- Proper gap enforcement for multi-day forecasts

Key Concepts
------------
- Weekly seasonality (period=7)
- MASE: Mean Absolute Scaled Error (appropriate for seasonal data)
- Seasonal naive baseline: y[t] = y[t-7]
- WalkForwardCV with test_size >= 7 for full-week evaluation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

# temporalcv imports
from temporalcv import WalkForwardCV
from temporalcv.gates import gate_suspicious_improvement

# =============================================================================
# PART 1: Generate Synthetic Web Traffic Data
# =============================================================================


def generate_web_traffic_data(
    n_days: int = 365,
    base_traffic: float = 10000,
    trend: float = 20,
    weekly_pattern: tuple = (1.0, 1.1, 1.15, 1.2, 1.1, 0.7, 0.6),
    noise_std: float = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic web traffic data with weekly seasonality.

    Mimics real website traffic patterns:
    - Monday-Friday: Higher traffic (business hours)
    - Saturday-Sunday: Lower traffic
    - Linear trend: Growth over time
    - Random noise: Day-to-day variation

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    base_traffic : float
        Baseline daily traffic.
    trend : float
        Daily traffic growth.
    weekly_pattern : tuple
        Multipliers for each day of week (Mon=0, Sun=6).
    noise_std : float
        Standard deviation of daily noise.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily traffic and features.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    day_of_week = dates.dayofweek  # 0=Monday, 6=Sunday

    # Components
    trend_component = trend * np.arange(n_days)
    seasonal_component = np.array([weekly_pattern[dow] for dow in day_of_week])
    noise = rng.normal(0, noise_std, n_days)

    # Combine
    traffic = (base_traffic + trend_component) * seasonal_component + noise
    traffic = np.maximum(traffic, 0)  # Traffic can't be negative

    # Create DataFrame
    df = pd.DataFrame(
        {
            "traffic": traffic,
            "day_of_week": day_of_week,
            "is_weekend": (day_of_week >= 5).astype(int),
        },
        index=dates,
    )

    # Add lagged features (strictly causal)
    df["traffic_lag1"] = df["traffic"].shift(1)
    df["traffic_lag7"] = df["traffic"].shift(7)  # Same day last week
    df["traffic_lag14"] = df["traffic"].shift(14)

    # Rolling features (with shift to prevent leakage)
    df["traffic_ma7"] = df["traffic"].shift(1).rolling(7).mean()
    df["traffic_std7"] = df["traffic"].shift(1).rolling(7).std()

    df = df.dropna()

    return df


print("=" * 70)
print("EXAMPLE 11: WEB TRAFFIC FORECASTING")
print("=" * 70)

# Generate data
df = generate_web_traffic_data(n_days=365, seed=42)

print(f"\n📊 Generated web traffic data: {len(df)} days")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Mean traffic: {df['traffic'].mean():.0f} visits/day")
print("   Weekly pattern: Mon-Fri high, Sat-Sun low")

# Show weekly pattern
weekly_avg = df.groupby("day_of_week")["traffic"].mean()
print("\n📈 Average traffic by day of week:")
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
for dow, day_name in enumerate(days):
    print(f"   {day_name}: {weekly_avg[dow]:,.0f}")

# =============================================================================
# PART 2: Why MASE Matters for Seasonal Data
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: WHY MASE MATTERS FOR SEASONAL DATA")
print("=" * 70)

print(
    """
For seasonal data, standard MAE can be misleading:

   MAE = 500 visits/day

Is that good or bad? It depends on the baseline. MASE (Mean Absolute
Scaled Error) provides context by comparing to a seasonal naive forecast:

   MASE = MAE_model / MAE_seasonal_naive

Where seasonal naive predicts y[t] = y[t-season] (e.g., last week's value).

Interpretation:
   MASE < 1: Model beats seasonal naive
   MASE = 1: Model equals seasonal naive
   MASE > 1: Model worse than just using last week's value

For web traffic with period=7, seasonal naive means: predict Monday
with last Monday, predict Tuesday with last Tuesday, etc.
"""
)

# =============================================================================
# PART 3: Baseline — Seasonal Naive
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: BASELINE — SEASONAL NAIVE")
print("=" * 70)

# Split data
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

y_test = df_test["traffic"].values

# Seasonal naive: y[t] = y[t-7] (same day last week)
# Using the lag7 column as our baseline forecast
seasonal_naive_pred = df_test["traffic_lag7"].values

# Compute seasonal naive MAE
seasonal_naive_mae = np.mean(np.abs(y_test - seasonal_naive_pred))

print("📊 Seasonal Naive Baseline:")
print("   Forecast: y[t] = y[t-7] (same day last week)")
print(f"   Test MAE: {seasonal_naive_mae:.2f} visits/day")
print("   This is our reference for MASE calculation.")

# =============================================================================
# PART 4: Train Models
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: TRAIN MODELS")
print("=" * 70)

# Prepare features
feature_cols = [
    "day_of_week",
    "is_weekend",
    "traffic_lag1",
    "traffic_lag7",
    "traffic_lag14",
    "traffic_ma7",
    "traffic_std7",
]

X_train = df_train[feature_cols].values
y_train = df_train["traffic"].values
X_test = df_test[feature_cols].values

# Train models
gb_model = GradientBoostingRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
ridge_model = Ridge(alpha=1.0)

gb_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Generate predictions
gb_pred = gb_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

print("✅ Trained 2 models: GradientBoosting, Ridge")
print(f"   Training samples: {len(y_train)}")
print(f"   Test samples: {len(y_test)}")

# =============================================================================
# PART 5: Evaluate with MASE
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: EVALUATE WITH MASE")
print("=" * 70)

# Compute MAE for each model
gb_mae = np.mean(np.abs(y_test - gb_pred))
ridge_mae = np.mean(np.abs(y_test - ridge_pred))

# Compute MASE
gb_mase = gb_mae / seasonal_naive_mae
ridge_mase = ridge_mae / seasonal_naive_mae

print("\n📊 Model Performance:")
print("-" * 60)
print(f"{'Model':<20} {'MAE':<15} {'MASE':<15} {'vs Baseline':<15}")
print("-" * 60)
print(f"{'Seasonal Naive':<20} {seasonal_naive_mae:<15.2f} {1.0:<15.2f} {'(baseline)':<15}")
print(f"{'GradientBoosting':<20} {gb_mae:<15.2f} {gb_mase:<15.3f} {'+' if gb_mase < 1 else '-'}")
print(f"{'Ridge':<20} {ridge_mae:<15.2f} {ridge_mase:<15.3f} {'+' if ridge_mase < 1 else '-'}")
print("-" * 60)

# Interpretation
print("\n🔍 Interpretation:")
if gb_mase < 1:
    print(f"   GradientBoosting beats seasonal naive by {(1 - gb_mase) * 100:.1f}%")
else:
    print(f"   GradientBoosting is {(gb_mase - 1) * 100:.1f}% WORSE than seasonal naive!")

if ridge_mase < 1:
    print(f"   Ridge beats seasonal naive by {(1 - ridge_mase) * 100:.1f}%")
else:
    print(f"   Ridge is {(ridge_mase - 1) * 100:.1f}% WORSE than seasonal naive!")

# =============================================================================
# PART 6: WalkForward CV with Weekly Test Windows
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: WALKFORWARD CV WITH WEEKLY TEST WINDOWS")
print("=" * 70)

print(
    """
For web traffic, we want to evaluate on FULL WEEKS:
- test_size=7 ensures each fold tests a complete Mon-Sun cycle
- This captures the weekly seasonality properly
- Avoids biased evaluation from partial weeks
"""
)

# Set up WalkForwardCV
wfcv = WalkForwardCV(
    window_type="expanding",
    window_size=180,  # Start with ~6 months of training data
    horizon=1,  # 1-day ahead forecast
    test_size=7,  # Test on full weeks
    n_splits=5,
)

# Manual cross-validation to compute per-fold MASE
print("\n📊 WalkForward CV Results (test_size=7 for full weeks):")
print("-" * 70)
print(f"{'Fold':<8} {'Train Size':<12} {'Test Dates':<25} {'GB MASE':<12} {'Ridge MASE':<12}")
print("-" * 70)

gb_mases = []
ridge_mases = []

for fold_idx, (train_idx, test_idx) in enumerate(wfcv.split(df[feature_cols].values)):
    # Get data
    X_tr = df.iloc[train_idx][feature_cols].values
    y_tr = df.iloc[train_idx]["traffic"].values
    X_te = df.iloc[test_idx][feature_cols].values
    y_te = df.iloc[test_idx]["traffic"].values

    # Seasonal naive for this fold
    sn_te = df.iloc[test_idx]["traffic_lag7"].values
    sn_mae = np.mean(np.abs(y_te - sn_te))

    # Train and predict
    gb_fold = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    ridge_fold = Ridge(alpha=1.0)

    gb_fold.fit(X_tr, y_tr)
    ridge_fold.fit(X_tr, y_tr)

    gb_pred_fold = gb_fold.predict(X_te)
    ridge_pred_fold = ridge_fold.predict(X_te)

    # Compute MASE
    gb_mase_fold = np.mean(np.abs(y_te - gb_pred_fold)) / sn_mae if sn_mae > 0 else np.nan
    ridge_mase_fold = np.mean(np.abs(y_te - ridge_pred_fold)) / sn_mae if sn_mae > 0 else np.nan

    gb_mases.append(gb_mase_fold)
    ridge_mases.append(ridge_mase_fold)

    test_dates = df.index[test_idx]
    print(
        f"{fold_idx + 1:<8} {len(train_idx):<12} {str(test_dates[0].date()) + ' to ' + str(test_dates[-1].date()):<25} {gb_mase_fold:<12.3f} {ridge_mase_fold:<12.3f}"
    )

print("-" * 70)
print(f"{'Mean':<8} {'':<12} {'':<25} {np.mean(gb_mases):<12.3f} {np.mean(ridge_mases):<12.3f}")
print(f"{'Std':<8} {'':<12} {'':<25} {np.std(gb_mases):<12.3f} {np.std(ridge_mases):<12.3f}")

# =============================================================================
# PART 7: Validation Gates
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: VALIDATION GATES")
print("=" * 70)

# Check if improvement is suspiciously large
best_model_mae = min(gb_mae, ridge_mae)
improvement = (seasonal_naive_mae - best_model_mae) / seasonal_naive_mae

gate_result = gate_suspicious_improvement(
    model_metric=best_model_mae,
    baseline_metric=seasonal_naive_mae,
    threshold=0.30,  # HALT if >30% improvement
    warn_threshold=0.15,  # WARN if >15% improvement
)

print("\n📊 Gate: Suspicious Improvement Check")
print(f"   Best model MAE: {best_model_mae:.2f}")
print(f"   Seasonal naive MAE: {seasonal_naive_mae:.2f}")
print(f"   Improvement: {improvement * 100:.1f}%")
print(f"   Status: {gate_result.status}")
print(f"   Message: {gate_result.message}")

if str(gate_result.status) == "GateStatus.HALT":
    print("\n🛑 HALT: Improvement is suspiciously large!")
    print("   Check for data leakage or feature engineering bugs.")
elif str(gate_result.status) == "GateStatus.WARN":
    print("\n⚠️  WARN: Large improvement detected.")
    print("   Verify features are strictly causal (no future leakage).")
else:
    print("\n✅ PASS: Improvement is within reasonable bounds.")

# =============================================================================
# PART 8: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

print(
    """
1. USE MASE FOR SEASONAL DATA
   - MASE = MAE / MAE_seasonal_naive
   - MASE < 1 means beating the seasonal baseline
   - More interpretable than raw MAE

2. SEASONAL NAIVE IS A STRONG BASELINE
   - For weekly data: y[t] = y[t-7]
   - Often hard to beat with ML models
   - If you can't beat it, use it!

3. TEST ON FULL SEASONAL CYCLES
   - test_size=7 for weekly data
   - test_size=30 for monthly patterns
   - Partial cycles give biased estimates

4. FEATURE ENGINEERING FOR SEASONALITY
   - day_of_week as categorical or encoded
   - is_weekend binary flag
   - Lagged values at seasonal period (lag7, lag14)
   - Rolling means over seasonal period

5. WATCH FOR LEAKAGE IN ROLLING FEATURES
   - Always use .shift(1) before .rolling()
   - df['ma7'] = df['y'].shift(1).rolling(7).mean()  # CORRECT
   - df['ma7'] = df['y'].rolling(7).mean()  # WRONG (includes current)

6. DOMAIN KNOWLEDGE MATTERS
   - Weekday vs weekend patterns
   - Holiday effects (not shown here)
   - Marketing campaigns, events
   - External regressors can help
"""
)

print("\n" + "=" * 70)
print("Example 11 complete.")
print("=" * 70)
