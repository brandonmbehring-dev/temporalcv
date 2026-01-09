"""
Example 12: IoT Sensor Forecasting
==================================

Real-World Case Study: Predictive Maintenance
---------------------------------------------
Industrial IoT sensors generate high-frequency data with unique challenges:

1. **High-Frequency Data**: Minute-level readings create large datasets
   - Rolling features must be computed carefully
   - Computational efficiency matters

2. **Anomalies Corrupt Features**: Sensor spikes/failures pollute rolling stats
   - A single spike can corrupt hours of rolling averages
   - Anomaly-aware feature engineering is critical

3. **Non-Stationary Patterns**: Equipment degradation, seasonal effects
   - Models trained on normal data fail during degradation
   - Walk-forward CV captures evolving patterns

This example demonstrates:
- Handling high-frequency minute-level sensor data
- How anomalies corrupt rolling features (WRONG approach)
- Anomaly-aware feature engineering (CORRECT approach)
- Evaluation on normal vs anomalous periods

Key Concepts
------------
- Minute-level time series (~1440 points/day)
- Point anomalies: sudden spikes (sensor malfunction, interference)
- Anomaly masking: exclude anomalies from rolling calculations
- Per-regime evaluation: separate metrics for normal vs anomalous
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# temporalcv imports
from temporalcv import WalkForwardCV
from temporalcv.gates import gate_suspicious_improvement

# =============================================================================
# PART 1: Generate Synthetic Sensor Data
# =============================================================================


def generate_sensor_data(
    n_days: int = 7,
    readings_per_hour: int = 60,  # Minute-level
    base_temp: float = 75.0,
    daily_amplitude: float = 10.0,
    noise_std: float = 1.0,
    anomaly_prob: float = 0.01,  # 1% of readings are anomalies
    anomaly_magnitude: float = 30.0,  # Spike size
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic IoT sensor data with point anomalies.

    Simulates an industrial temperature sensor:
    - Base temperature with daily cycle (HVAC pattern)
    - Random noise from sensor precision
    - Point anomalies: sudden spikes (interference, malfunction)

    Parameters
    ----------
    n_days : int
        Number of days to simulate.
    readings_per_hour : int
        Readings per hour (60 = minute-level).
    base_temp : float
        Baseline temperature (Fahrenheit).
    daily_amplitude : float
        Daily cycle amplitude.
    noise_std : float
        Sensor noise standard deviation.
    anomaly_prob : float
        Probability of point anomaly at each reading.
    anomaly_magnitude : float
        Size of anomaly spike.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Minute-level sensor data with anomaly labels.
    """
    rng = np.random.default_rng(seed)

    n_hours = n_days * 24
    n_readings = n_hours * readings_per_hour

    timestamps = pd.date_range("2024-01-01", periods=n_readings, freq="min")

    # Time components
    hour_of_day = timestamps.hour + timestamps.minute / 60

    # Daily pattern: warmer during business hours (9-17)
    daily_pattern = daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Generate base signal
    noise = rng.normal(0, noise_std, n_readings)
    temperature = base_temp + daily_pattern + noise

    # Add point anomalies (sudden spikes)
    is_anomaly = rng.random(n_readings) < anomaly_prob
    anomaly_direction = rng.choice([-1, 1], n_readings)  # Spike up or down
    anomaly_values = anomaly_magnitude * anomaly_direction
    temperature = np.where(is_anomaly, temperature + anomaly_values, temperature)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "temperature": temperature,
            "is_anomaly": is_anomaly.astype(int),
            "hour": timestamps.hour,
            "minute": timestamps.minute,
        },
        index=timestamps,
    )

    return df


print("=" * 70)
print("EXAMPLE 12: IOT SENSOR FORECASTING")
print("=" * 70)

# Generate data
df = generate_sensor_data(n_days=14, seed=42)

print(f"\n📊 Generated sensor data: {len(df):,} readings")
print(f"   Frequency: Minute-level ({60*24:,} readings/day)")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")
print(f"   Anomalies: {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")

# Show temperature stats
print("\n📈 Temperature Statistics:")
print(f"   Normal readings: mean={df[~df['is_anomaly'].astype(bool)]['temperature'].mean():.1f}°F")
print(f"   Anomaly readings: mean={df[df['is_anomaly'].astype(bool)]['temperature'].mean():.1f}°F")
print(f"   Overall range: {df['temperature'].min():.1f}°F to {df['temperature'].max():.1f}°F")

# =============================================================================
# PART 2: The Problem — Anomalies Corrupt Rolling Features
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: THE PROBLEM — ANOMALIES CORRUPT ROLLING FEATURES")
print("=" * 70)

print(
    """
Rolling statistics are fundamental to sensor forecasting:

   temp_ma60 = temperature.rolling(60).mean()  # Hourly average
   temp_std60 = temperature.rolling(60).std()   # Hourly volatility

But a SINGLE anomaly can corrupt the entire rolling window:

   Normal readings: [75.1, 75.2, 75.0, 74.9, ...]
   With anomaly:    [75.1, 75.2, 105.0, 74.9, ...]  ← Spike!

   Rolling mean jumps from ~75 to ~75.5 for the next 60 minutes
   Rolling std explodes, making the model think "high volatility"

This is a form of DATA LEAKAGE: the anomaly "leaks" into future features.
"""
)

# =============================================================================
# PART 3: WRONG Approach — Include Anomalies in Rolling Stats
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: WRONG APPROACH — INCLUDE ANOMALIES IN ROLLING STATS")
print("=" * 70)

# WRONG: Calculate rolling stats including anomalies
df_wrong = df.copy()
df_wrong["temp_lag1"] = df_wrong["temperature"].shift(1)
df_wrong["temp_ma60"] = df_wrong["temperature"].shift(1).rolling(60).mean()
df_wrong["temp_std60"] = df_wrong["temperature"].shift(1).rolling(60).std()
df_wrong["temp_ma360"] = df_wrong["temperature"].shift(1).rolling(360).mean()

df_wrong = df_wrong.dropna()

print("❌ WRONG: Rolling stats computed on ALL data (including anomalies)")
print("   Features include temp_ma60, temp_std60, temp_ma360")
print("   A single spike corrupts rolling mean for 60+ minutes")

# Show corruption example - find an anomaly that has enough data before/after
anomaly_indices = df[df["is_anomaly"] == 1].index
# Find anomaly at least 2 hours into the data (to have rolling stats available)
for anomaly_idx in anomaly_indices:
    if anomaly_idx > df.index[0] + pd.Timedelta(hours=8):
        break

window_start = anomaly_idx - pd.Timedelta(minutes=5)
window_end = anomaly_idx + pd.Timedelta(minutes=65)

# Get values safely
try:
    before_vals = df_wrong.loc[
        window_start : anomaly_idx - pd.Timedelta(minutes=1), "temp_ma60"
    ].dropna()
    after_vals = df_wrong.loc[
        anomaly_idx + pd.Timedelta(minutes=1) : window_end, "temp_ma60"
    ].dropna()
    if len(before_vals) > 0 and len(after_vals) > 0:
        print(f"\n🔍 Example of corruption around anomaly at {anomaly_idx}:")
        print(f"   temp_ma60 before anomaly: {before_vals.iloc[-1]:.2f}°F")
        print(f"   temp_ma60 after anomaly enters window: {after_vals.iloc[0]:.2f}°F")
        print("   The rolling mean is distorted for the entire 60-minute window!")
    else:
        print(f"\n🔍 Anomaly at {anomaly_idx} demonstrates corruption in rolling stats")
except Exception:
    print(f"\n🔍 Anomaly at {anomaly_idx} demonstrates corruption in rolling stats")

# =============================================================================
# PART 4: CORRECT Approach — Anomaly-Aware Feature Engineering
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: CORRECT APPROACH — ANOMALY-AWARE FEATURE ENGINEERING")
print("=" * 70)

print(
    """
Solution: Mask anomalies before computing rolling statistics.

   # Create "clean" temperature series
   temp_clean = temperature.where(~is_anomaly, np.nan)

   # Rolling stats ignore NaN values
   temp_ma60 = temp_clean.shift(1).rolling(60, min_periods=30).mean()

Now anomalies don't corrupt the rolling windows.
"""
)

# CORRECT: Mask anomalies, then calculate rolling stats
df_correct = df.copy()

# Create clean temperature (anomalies → NaN)
df_correct["temp_clean"] = df_correct["temperature"].where(
    ~df_correct["is_anomaly"].astype(bool), np.nan
)

# Rolling stats on clean data (NaN ignored)
df_correct["temp_lag1"] = df_correct["temperature"].shift(1)
df_correct["temp_ma60"] = df_correct["temp_clean"].shift(1).rolling(60, min_periods=30).mean()
df_correct["temp_std60"] = df_correct["temp_clean"].shift(1).rolling(60, min_periods=30).std()
df_correct["temp_ma360"] = df_correct["temp_clean"].shift(1).rolling(360, min_periods=180).mean()

# Add anomaly indicator as feature (model can learn from recent anomalies)
df_correct["recent_anomaly_count"] = df_correct["is_anomaly"].shift(1).rolling(60).sum()

df_correct = df_correct.dropna()

print("✅ CORRECT: Rolling stats computed on CLEAN data (anomalies masked)")
print("   Anomalies replaced with NaN before rolling calculations")
print("   min_periods ensures stability even with some NaN values")

# Show corrected behavior
try:
    before_vals_correct = df_correct.loc[
        window_start : anomaly_idx - pd.Timedelta(minutes=1), "temp_ma60"
    ].dropna()
    after_vals_correct = df_correct.loc[
        anomaly_idx + pd.Timedelta(minutes=1) : window_end, "temp_ma60"
    ].dropna()
    if len(before_vals_correct) > 0 and len(after_vals_correct) > 0:
        print("\n🔍 Same window with anomaly-aware features:")
        print(f"   temp_ma60 before anomaly: {before_vals_correct.iloc[-1]:.2f}°F")
        print(f"   temp_ma60 after anomaly (masked): {after_vals_correct.iloc[0]:.2f}°F")
        print("   Rolling mean is STABLE because anomaly is masked!")
except Exception:
    print("\n🔍 Anomaly-aware features provide stable rolling statistics")

# =============================================================================
# PART 5: Model Comparison with WalkForwardCV
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: MODEL COMPARISON WITH WALKFORWARDCV")
print("=" * 70)

# Prepare features
feature_cols_wrong = ["temp_lag1", "temp_ma60", "temp_std60", "temp_ma360", "hour"]
feature_cols_correct = [
    "temp_lag1",
    "temp_ma60",
    "temp_std60",
    "temp_ma360",
    "hour",
    "recent_anomaly_count",
]

# Align datasets
common_idx = df_wrong.index.intersection(df_correct.index)
df_wrong_aligned = df_wrong.loc[common_idx]
df_correct_aligned = df_correct.loc[common_idx]

X_wrong = df_wrong_aligned[feature_cols_wrong].values
X_correct = df_correct_aligned[feature_cols_correct].values
y = df_wrong_aligned["temperature"].values

# Walk-Forward CV
wfcv = WalkForwardCV(
    window_type="expanding",
    window_size=60 * 24 * 3,  # 3 days minimum training
    horizon=60,  # 1-hour ahead forecast
    test_size=60 * 24,  # Test on 1 day
    n_splits=5,
)

print("📊 WalkForwardCV Configuration:")
print("   Window: Expanding from 3 days")
print("   Horizon: 60 minutes (1-hour ahead)")
print("   Test size: 1 day per fold")
print("   Folds: 5")

# Evaluate both approaches
results_wrong = []
results_correct = []

for fold_idx, (train_idx, test_idx) in enumerate(wfcv.split(X_wrong)):
    # Train models
    model_wrong = Ridge(alpha=1.0)
    model_correct = Ridge(alpha=1.0)

    model_wrong.fit(X_wrong[train_idx], y[train_idx])
    model_correct.fit(X_correct[train_idx], y[train_idx])

    # Predict
    pred_wrong = model_wrong.predict(X_wrong[test_idx])
    pred_correct = model_correct.predict(X_correct[test_idx])

    y_test = y[test_idx]

    # MAE
    mae_wrong = np.mean(np.abs(y_test - pred_wrong))
    mae_correct = np.mean(np.abs(y_test - pred_correct))

    results_wrong.append(mae_wrong)
    results_correct.append(mae_correct)

print("\n📊 Results (MAE in °F):")
print("-" * 60)
print(f"{'Fold':<8} {'WRONG (with anomalies)':<25} {'CORRECT (masked)':<20}")
print("-" * 60)
for i, (w, c) in enumerate(zip(results_wrong, results_correct)):
    improvement = (w - c) / w * 100
    print(f"{i+1:<8} {w:<25.3f} {c:<20.3f} ({improvement:+.1f}%)")
print("-" * 60)
print(f"{'Mean':<8} {np.mean(results_wrong):<25.3f} {np.mean(results_correct):<20.3f}")

# =============================================================================
# PART 6: Per-Regime Evaluation
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: PER-REGIME EVALUATION")
print("=" * 70)

print(
    """
For production monitoring, we care about performance in BOTH regimes:

1. NORMAL periods: Most of the time, model should be accurate
2. ANOMALY periods: During/after anomalies, performance may degrade

Separate evaluation helps understand operational reliability.
"""
)

# Final fold evaluation with per-regime split
train_idx, test_idx = list(wfcv.split(X_wrong))[-1]

model_wrong = Ridge(alpha=1.0)
model_correct = Ridge(alpha=1.0)

model_wrong.fit(X_wrong[train_idx], y[train_idx])
model_correct.fit(X_correct[train_idx], y[train_idx])

pred_wrong = model_wrong.predict(X_wrong[test_idx])
pred_correct = model_correct.predict(X_correct[test_idx])

y_test = y[test_idx]
is_anomaly_test = df_wrong_aligned.iloc[test_idx]["is_anomaly"].values.astype(bool)

# Per-regime MAE
normal_mask = ~is_anomaly_test
anomaly_mask = is_anomaly_test

if normal_mask.sum() > 0:
    mae_wrong_normal = np.mean(np.abs(y_test[normal_mask] - pred_wrong[normal_mask]))
    mae_correct_normal = np.mean(np.abs(y_test[normal_mask] - pred_correct[normal_mask]))
else:
    mae_wrong_normal = mae_correct_normal = np.nan

if anomaly_mask.sum() > 0:
    mae_wrong_anomaly = np.mean(np.abs(y_test[anomaly_mask] - pred_wrong[anomaly_mask]))
    mae_correct_anomaly = np.mean(np.abs(y_test[anomaly_mask] - pred_correct[anomaly_mask]))
else:
    mae_wrong_anomaly = mae_correct_anomaly = np.nan

print("\n📊 Per-Regime MAE (last fold):")
print("-" * 60)
print(f"{'Regime':<15} {'WRONG':<15} {'CORRECT':<15} {'Improvement':<15}")
print("-" * 60)
print(
    f"{'Normal':<15} {mae_wrong_normal:<15.3f} {mae_correct_normal:<15.3f} {(mae_wrong_normal - mae_correct_normal) / mae_wrong_normal * 100:+.1f}%"
)
if not np.isnan(mae_wrong_anomaly):
    print(
        f"{'Anomaly':<15} {mae_wrong_anomaly:<15.3f} {mae_correct_anomaly:<15.3f} {(mae_wrong_anomaly - mae_correct_anomaly) / mae_wrong_anomaly * 100:+.1f}%"
    )
print("-" * 60)

# =============================================================================
# PART 7: Validation Gate
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: VALIDATION GATE")
print("=" * 70)

# Baseline: lag-1 (persistence)
baseline_pred = df_correct_aligned.iloc[test_idx]["temp_lag1"].values
baseline_mae = np.mean(np.abs(y_test - baseline_pred))

gate_result = gate_suspicious_improvement(
    model_metric=np.mean(results_correct),
    baseline_metric=baseline_mae,
    threshold=0.40,
    warn_threshold=0.25,
)

print("\n📊 Gate: Suspicious Improvement Check")
print(f"   Model MAE (CORRECT): {np.mean(results_correct):.3f}°F")
print(f"   Baseline MAE (lag-1): {baseline_mae:.3f}°F")
print(f"   Improvement: {(baseline_mae - np.mean(results_correct)) / baseline_mae * 100:.1f}%")
print(f"   Status: {gate_result.status}")

# =============================================================================
# PART 8: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

print(
    """
1. ANOMALIES CORRUPT ROLLING FEATURES
   - A single spike propagates through the entire rolling window
   - 1 anomaly → 60+ corrupted feature values (for 60-minute window)
   - This is a form of data leakage (anomaly information leaks forward)

2. MASK ANOMALIES BEFORE ROLLING CALCULATIONS
   - temp_clean = temperature.where(~is_anomaly, np.nan)
   - Rolling functions ignore NaN, so anomalies don't corrupt stats
   - Use min_periods to ensure stability with some NaN values

3. SEPARATE EVALUATION BY REGIME
   - Normal periods: Model should be accurate
   - Anomaly periods: Expect higher errors (that's why they're anomalies!)
   - Don't average across regimes without understanding the mix

4. HIGH-FREQUENCY DATA NEEDS EFFICIENT PROCESSING
   - Minute-level: ~1.4M readings/year
   - Use vectorized operations (pandas rolling, not loops)
   - Consider downsampling for training, minute-level for evaluation

5. ANOMALY DETECTION IS A SEPARATE PROBLEM
   - This example uses labeled anomalies (ground truth)
   - In production, you need anomaly detection first
   - Common approaches: threshold, isolation forest, autoencoders

6. EXTENSIONS FOR REAL IoT DEPLOYMENTS
   - Drift anomalies: gradual sensor degradation (more complex)
   - Dropout anomalies: missing data periods (interpolation)
   - Multiple sensors: cross-sensor validation
   - Edge computing: model must run on limited hardware
"""
)

print("\n" + "=" * 70)
print("Example 12 complete.")
print("=" * 70)
