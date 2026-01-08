"""
Example 14: Energy Load Forecasting
===================================

Real-World Case Study: Electrical Load Prediction
-------------------------------------------------
Energy load forecasting is a classic time series problem with:

1. **Multiple Seasonalities**: Daily (24h), weekly (168h), annual patterns
   - Peak demand during business hours
   - Lower on weekends
   - Weather-dependent seasonal shifts

2. **Multi-Step Horizons**: Grid operators need forecasts for:
   - Day-ahead (24 hours)
   - Week-ahead (168 hours)
   - Hour-ahead (real-time adjustments)

3. **High Stakes**: Errors cost money
   - Over-forecast: Pay for unused generation capacity
   - Under-forecast: Emergency power purchases at premium

This example demonstrates:
- Handling multiple seasonalities in features
- Multi-step ahead forecasting with proper gap enforcement
- Calendar feature engineering (hour, day of week, holiday proxies)

Key Concepts
------------
- Hourly data with daily + weekly patterns
- WalkForwardCV with horizon > 1 (gap enforcement critical)
- Calendar features: hour_of_day, day_of_week, is_business_hour
- Multi-step MAE evaluation
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
# PART 1: Generate Synthetic Energy Load Data
# =============================================================================


def generate_energy_load_data(
    n_days: int = 60,
    base_load: float = 1000,
    daily_amplitude: float = 300,
    weekly_amplitude: float = 100,
    noise_std: float = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly energy load data.

    Mimics real electrical load patterns:
    - Daily cycle: Peak at 2-6 PM, trough at 3-5 AM
    - Weekly cycle: Lower on weekends
    - Random noise: Weather, demand variation

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    base_load : float
        Baseline load in MW.
    daily_amplitude : float
        Daily cycle amplitude.
    weekly_amplitude : float
        Weekly cycle amplitude.
    noise_std : float
        Standard deviation of noise.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Hourly load data with calendar features.
    """
    rng = np.random.default_rng(seed)

    n_hours = n_days * 24
    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h")

    # Extract calendar features
    hour_of_day = timestamps.hour
    day_of_week = timestamps.dayofweek

    # Daily pattern: peaks at hour 14-18 (2-6 PM), trough at 3-5 AM
    # Using sine wave shifted to peak in afternoon
    daily_pattern = daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Weekly pattern: weekends are ~10% lower
    is_weekend = (day_of_week >= 5).astype(float)
    weekly_pattern = -weekly_amplitude * is_weekend

    # Add some trend (slight growth over time)
    trend = 0.5 * np.arange(n_hours) / 24  # ~0.5 MW growth per day

    # Combine components
    noise = rng.normal(0, noise_std, n_hours)
    load = base_load + daily_pattern + weekly_pattern + trend + noise
    load = np.maximum(load, 0)  # Load can't be negative

    # Create DataFrame
    df = pd.DataFrame({
        "load": load,
        "hour": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend.astype(int),
        "is_business_hour": ((hour_of_day >= 9) & (hour_of_day <= 17) & (day_of_week < 5)).astype(int),
    }, index=timestamps)

    # Add lagged features (strictly causal)
    df["load_lag1"] = df["load"].shift(1)  # Previous hour
    df["load_lag24"] = df["load"].shift(24)  # Same hour yesterday
    df["load_lag168"] = df["load"].shift(168)  # Same hour last week

    # Rolling features (with shift to prevent leakage)
    df["load_ma24"] = df["load"].shift(1).rolling(24).mean()  # Daily rolling avg
    df["load_std24"] = df["load"].shift(1).rolling(24).std()

    df = df.dropna()

    return df


print("=" * 70)
print("EXAMPLE 14: ENERGY LOAD FORECASTING")
print("=" * 70)

# Generate data
df = generate_energy_load_data(n_days=90, seed=42)

print(f"\n📊 Generated hourly energy load data: {len(df)} hours")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")
print(f"   Mean load: {df['load'].mean():.0f} MW")
print(f"   Peak load: {df['load'].max():.0f} MW")
print(f"   Min load: {df['load'].min():.0f} MW")

# Show daily pattern
daily_avg = df.groupby("hour")["load"].mean()
print(f"\n📈 Average load by hour (daily pattern):")
print(f"   Peak hour: {daily_avg.idxmax()}:00 ({daily_avg.max():.0f} MW)")
print(f"   Trough hour: {daily_avg.idxmin()}:00 ({daily_avg.min():.0f} MW)")

# =============================================================================
# PART 2: Why Multi-Step Forecasting Needs Gap Enforcement
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: WHY MULTI-STEP FORECASTING NEEDS GAP ENFORCEMENT")
print("=" * 70)

print("""
For day-ahead forecasting (24-hour horizon):

   At time t, you need to forecast load[t+1], load[t+2], ..., load[t+24]

If your training data goes up to time t, then:
   - load_lag1 at time t+1 = load[t] ← AVAILABLE at time t
   - load_lag24 at time t+1 = load[t-23] ← AVAILABLE at time t

But if you train on data up to t+23 for a 24-step forecast:
   - Your features at t+24 use load[t+23] ← NOT AVAILABLE at time t!

WalkForwardCV enforces the gap: train data ends at t - horizon + 1.
""")

# =============================================================================
# PART 3: Baseline — 24-Hour Lag (Same Hour Yesterday)
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: BASELINE — 24-HOUR LAG (SAME HOUR YESTERDAY)")
print("=" * 70)

# Split data (last 14 days for test)
test_days = 14
test_size = test_days * 24
train_size = len(df) - test_size

df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

y_test = df_test["load"].values

# Baseline: predict same hour yesterday
baseline_pred = df_test["load_lag24"].values
baseline_mae = np.mean(np.abs(y_test - baseline_pred))

print(f"📊 24-Hour Lag Baseline (same hour yesterday):")
print(f"   Test MAE: {baseline_mae:.2f} MW")
print(f"   This is our reference for model improvement.")

# =============================================================================
# PART 4: Train Models with Proper Features
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: TRAIN MODELS WITH CALENDAR FEATURES")
print("=" * 70)

# Features
feature_cols = [
    "hour", "day_of_week", "is_weekend", "is_business_hour",
    "load_lag1", "load_lag24", "load_lag168",
    "load_ma24", "load_std24"
]

X_train = df_train[feature_cols].values
y_train = df_train["load"].values
X_test = df_test[feature_cols].values

# Train models
gb_model = GradientBoostingRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
ridge_model = Ridge(alpha=1.0)

gb_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Predictions
gb_pred = gb_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# MAE
gb_mae = np.mean(np.abs(y_test - gb_pred))
ridge_mae = np.mean(np.abs(y_test - ridge_pred))

print(f"✅ Model Performance (1-hour ahead features):")
print("-" * 50)
print(f"{'Model':<20} {'MAE (MW)':<15} {'vs Baseline':<15}")
print("-" * 50)
print(f"{'24-Hour Lag':<20} {baseline_mae:<15.2f} {'(baseline)':<15}")
print(f"{'GradientBoosting':<20} {gb_mae:<15.2f} {(gb_mae - baseline_mae) / baseline_mae * 100:+.1f}%")
print(f"{'Ridge':<20} {ridge_mae:<15.2f} {(ridge_mae - baseline_mae) / baseline_mae * 100:+.1f}%")
print("-" * 50)

# =============================================================================
# PART 5: Walk-Forward CV for Day-Ahead (24-Hour) Forecasting
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: WALK-FORWARD CV FOR DAY-AHEAD (24-HOUR) FORECASTING")
print("=" * 70)

print("""
For day-ahead forecasting, we set horizon=24:
- This enforces a 24-hour gap between train end and test start
- Ensures features at test time use only available information
- Simulates real-world operational forecasting
""")

# WalkForwardCV with 24-hour horizon
wfcv = WalkForwardCV(
    window_type="expanding",
    window_size=30 * 24,  # Start with 30 days of training
    horizon=24,  # 24-hour ahead forecast
    test_size=7 * 24,  # Test on 1 week at a time
    n_splits=3,
)

print(f"\n📊 Walk-Forward CV with horizon=24 (day-ahead):")
print("-" * 70)
print(f"{'Fold':<8} {'Train Hours':<15} {'Test Period':<25} {'GB MAE':<12} {'Ridge MAE':<12}")
print("-" * 70)

gb_maes = []
ridge_maes = []
baseline_maes = []

for fold_idx, (train_idx, test_idx) in enumerate(wfcv.split(df[feature_cols].values)):
    # Get data
    X_tr = df.iloc[train_idx][feature_cols].values
    y_tr = df.iloc[train_idx]["load"].values
    X_te = df.iloc[test_idx][feature_cols].values
    y_te = df.iloc[test_idx]["load"].values

    # Baseline for this fold
    bl_te = df.iloc[test_idx]["load_lag24"].values
    bl_mae = np.mean(np.abs(y_te - bl_te))
    baseline_maes.append(bl_mae)

    # Train and predict
    gb_fold = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
    )
    ridge_fold = Ridge(alpha=1.0)

    gb_fold.fit(X_tr, y_tr)
    ridge_fold.fit(X_tr, y_tr)

    gb_pred_fold = gb_fold.predict(X_te)
    ridge_pred_fold = ridge_fold.predict(X_te)

    # MAE
    gb_mae_fold = np.mean(np.abs(y_te - gb_pred_fold))
    ridge_mae_fold = np.mean(np.abs(y_te - ridge_pred_fold))

    gb_maes.append(gb_mae_fold)
    ridge_maes.append(ridge_mae_fold)

    # Print
    test_start = df.index[test_idx[0]]
    test_end = df.index[test_idx[-1]]
    print(f"{fold_idx + 1:<8} {len(train_idx):<15} {str(test_start.date()) + ' to ' + str(test_end.date()):<25} {gb_mae_fold:<12.2f} {ridge_mae_fold:<12.2f}")

print("-" * 70)
print(f"{'Mean':<8} {'':<15} {'':<25} {np.mean(gb_maes):<12.2f} {np.mean(ridge_maes):<12.2f}")

# =============================================================================
# PART 6: Feature Importance for Energy Forecasting
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: FEATURE IMPORTANCE")
print("=" * 70)

# Get feature importance from the last GB model
importance = gb_fold.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False)

print(f"\n📊 Feature Importance (GradientBoosting):")
print("-" * 40)
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"{row['feature']:<20} {row['importance']:.3f} {bar}")

# =============================================================================
# PART 7: Validation Gates
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: VALIDATION GATES")
print("=" * 70)

best_model_mae = min(np.mean(gb_maes), np.mean(ridge_maes))
avg_baseline_mae = np.mean(baseline_maes)

gate_result = gate_suspicious_improvement(
    model_metric=best_model_mae,
    baseline_metric=avg_baseline_mae,
    threshold=0.30,
    warn_threshold=0.15,
)

print(f"\n📊 Gate: Suspicious Improvement Check")
print(f"   Best model MAE: {best_model_mae:.2f} MW")
print(f"   Baseline MAE: {avg_baseline_mae:.2f} MW")
print(f"   Improvement: {(avg_baseline_mae - best_model_mae) / avg_baseline_mae * 100:.1f}%")
print(f"   Status: {gate_result.status}")

# =============================================================================
# PART 8: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

print("""
1. MULTI-STEP FORECASTING REQUIRES PROPER GAPS
   - horizon=24 for day-ahead forecasting
   - horizon=168 for week-ahead
   - Gap ensures features use only available information

2. CALENDAR FEATURES ARE CRUCIAL
   - hour_of_day captures daily pattern
   - day_of_week captures weekly pattern
   - is_business_hour helps with demand peaks
   - Consider holidays as special features

3. LAGGED FEATURES AT SEASONAL PERIODS
   - load_lag24: Same hour yesterday
   - load_lag168: Same hour last week
   - These are often the most important features

4. 24-HOUR LAG IS A STRONG BASELINE
   - Simple: predict what happened same hour yesterday
   - Often hard to beat significantly
   - Use as sanity check for your model

5. ROLLING FEATURES NEED .shift()
   - load_ma24 = load.shift(1).rolling(24).mean()
   - Without shift, you include current value → leakage
   - Even small leakage compounds over 24-hour horizon

6. OPERATIONAL CONSIDERATIONS
   - Real-time corrections (hour-ahead) can use more recent data
   - Day-ahead scheduling needs 24-hour gap
   - Week-ahead planning needs 168-hour gap
   - Different horizons may need different models
""")

print("\n" + "=" * 70)
print("Example 14 complete.")
print("=" * 70)
