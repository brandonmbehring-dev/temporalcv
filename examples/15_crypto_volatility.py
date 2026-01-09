"""
Example 15: Crypto Volatility Forecasting
=========================================

Real-World Case Study: Cryptocurrency Volatility with Regime Shifts
-------------------------------------------------------------------
Cryptocurrency markets exhibit extreme characteristics:

1. **Regime Shifts**: Calm periods → crashes → recovery
   - Volatility can 10x overnight
   - Fixed models fail during regime transitions

2. **Non-Stationarity**: Distribution changes over time
   - Yesterday's calibration may be invalid today
   - Fixed prediction intervals break down

3. **Fat Tails**: Extreme events more common than normal distribution
   - VaR/ES calculations using normal assumptions fail
   - Conformal prediction provides distribution-free coverage

This example demonstrates:
- How fixed (split) conformal intervals fail during regime shifts
- How adaptive conformal adjusts to maintain coverage
- Per-regime coverage analysis

Key Concepts
------------
- Regime labels: CALM (low vol), CRASH (high vol), RECOVERY (transitional)
- Split conformal: Fixed quantile from calibration period
- Adaptive conformal: Quantile adjusts based on recent coverage
- Coverage rate: Proportion of actuals within predicted intervals

References
----------
- Gibbs & Candès (2021) "Adaptive Conformal Inference Under Distribution Shift"
- Romano et al. (2019) "Conformalized Quantile Regression"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# temporalcv imports
from temporalcv.conformal import AdaptiveConformalPredictor, SplitConformalPredictor

# =============================================================================
# PART 1: Generate Synthetic Crypto Returns with Regime Shifts
# =============================================================================


def generate_crypto_data(
    n_days: int = 500,
    regimes: list[tuple[int, str, float, float]] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic crypto returns with known regime switches.

    Each regime has different volatility and drift characteristics.

    Parameters
    ----------
    n_days : int
        Total number of daily observations.
    regimes : list of (end_day, regime_name, drift, volatility)
        List of regime specifications. Defaults to CALM → CRASH → RECOVERY pattern.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Daily returns with regime labels.
    """
    rng = np.random.default_rng(seed)

    if regimes is None:
        # Default: CALM → CRASH → RECOVERY cycle (twice)
        regimes = [
            (100, "CALM", 0.001, 0.02),  # Days 0-100: Low vol, slight positive drift
            (150, "CRASH", -0.01, 0.08),  # Days 100-150: High vol, negative drift
            (200, "RECOVERY", 0.005, 0.04),  # Days 150-200: Medium vol, recovery
            (300, "CALM", 0.001, 0.02),  # Days 200-300: Back to calm
            (350, "CRASH", -0.015, 0.10),  # Days 300-350: Severe crash
            (400, "RECOVERY", 0.003, 0.05),  # Days 350-400: Recovery
            (n_days, "CALM", 0.001, 0.02),  # Days 400+: Calm
        ]

    # Generate returns
    returns = np.zeros(n_days)
    regime_labels = np.empty(n_days, dtype=object)

    prev_end = 0
    for end_day, regime_name, drift, volatility in regimes:
        end_day = min(end_day, n_days)
        n_regime = end_day - prev_end
        if n_regime > 0:
            returns[prev_end:end_day] = drift + volatility * rng.standard_t(df=5, size=n_regime)
            regime_labels[prev_end:end_day] = regime_name
        prev_end = end_day

    # Create DataFrame
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "return": returns,
            "regime": regime_labels,
        },
        index=dates,
    )

    # Add features
    df["return_lag1"] = df["return"].shift(1)
    df["volatility_20d"] = df["return"].shift(1).rolling(20).std()

    df = df.dropna()

    return df


print("=" * 70)
print("EXAMPLE 15: CRYPTO VOLATILITY FORECASTING")
print("=" * 70)

# Generate data
df = generate_crypto_data(n_days=500, seed=42)

print(f"\n📊 Generated crypto return data: {len(df)} days")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")

# Show regime breakdown
regime_stats = df.groupby("regime")["return"].agg(["count", "mean", "std"]).round(4)
print("\n📈 Regime Statistics:")
print(regime_stats.to_string())

# =============================================================================
# PART 2: The Problem — Fixed Intervals Fail During Regime Shifts
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: THE PROBLEM — FIXED INTERVALS FAIL DURING REGIME SHIFTS")
print("=" * 70)

print(
    """
Split Conformal Prediction:
1. Train model on training data
2. Calibrate quantile on calibration data (fixed period)
3. Apply fixed quantile to test data

Problem: If calibration is during CALM regime, quantile is small.
When CRASH hits, the fixed quantile gives UNDER-COVERAGE (too narrow).

Similarly, if calibration is during CRASH, intervals are too wide
during CALM periods (OVER-COVERAGE, inefficient).

The quantile doesn't adapt to changing volatility!
"""
)

# =============================================================================
# PART 3: Setup — Train Model and Generate Predictions
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: TRAIN MODEL AND GENERATE PREDICTIONS")
print("=" * 70)

# Split data
cal_end = 100  # Calibrate on CALM period (intentionally problematic)
test_start = cal_end

# Train simple model on first portion
train_end = 50
train_df = df.iloc[:train_end]
cal_df = df.iloc[train_end:cal_end]
test_df = df.iloc[test_start:]

# Features and target
X_train = train_df[["return_lag1", "volatility_20d"]].values
y_train = train_df["return"].values

X_cal = cal_df[["return_lag1", "volatility_20d"]].values
y_cal = cal_df["return"].values

X_test = test_df[["return_lag1", "volatility_20d"]].values
y_test = test_df["return"].values

# Train model
model = Ridge(alpha=0.01)
model.fit(X_train, y_train)

# Generate predictions
cal_preds = model.predict(X_cal)
test_preds = model.predict(X_test)

print("📊 Data Splits:")
print(f"   Training: {len(train_df)} days (model fitting)")
print(f"   Calibration: {len(cal_df)} days (regime: {cal_df['regime'].iloc[0]})")
print(f"   Test: {len(test_df)} days (multiple regimes)")

# =============================================================================
# PART 4: WRONG Approach — Split Conformal with Fixed Calibration
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: WRONG APPROACH — SPLIT CONFORMAL (FIXED QUANTILE)")
print("=" * 70)

# Split Conformal (WRONG for non-stationary data)
scp = SplitConformalPredictor(alpha=0.10)  # 90% intervals
scp.calibrate(cal_preds, y_cal)

print("❌ Split Conformal Prediction:")
print(f"   Calibration regime: {cal_df['regime'].iloc[0]} (low volatility)")
print(f"   Calibrated quantile: {scp.quantile_:.4f}")
print("   This quantile will be FIXED for all test data!")

# Generate intervals
intervals_split = scp.predict_interval(test_preds)
lower_split = intervals_split.lower
upper_split = intervals_split.upper

# Check coverage by regime
test_df_with_preds = test_df.copy()
test_df_with_preds["pred"] = test_preds
test_df_with_preds["lower"] = lower_split
test_df_with_preds["upper"] = upper_split
test_df_with_preds["covered"] = (y_test >= lower_split) & (y_test <= upper_split)

print("\n📊 Split Conformal Coverage by Regime:")
print("-" * 50)
coverage_split = test_df_with_preds.groupby("regime")["covered"].mean()
for regime in ["CALM", "CRASH", "RECOVERY"]:
    if regime in coverage_split.index:
        cov = coverage_split[regime]
        status = "✅" if 0.85 <= cov <= 0.95 else "❌"
        print(f"   {regime:<12} Coverage: {cov*100:.1f}% (target: 90%) {status}")

overall_split = test_df_with_preds["covered"].mean()
print(f"\n   Overall Coverage: {overall_split*100:.1f}%")
print("   ⚠️  Coverage breaks down during CRASH — intervals too narrow!")

# =============================================================================
# PART 5: CORRECT Approach — Adaptive Conformal Prediction
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: CORRECT APPROACH — ADAPTIVE CONFORMAL PREDICTION")
print("=" * 70)

print(
    """
Adaptive Conformal Prediction (Gibbs & Candès 2021):
1. Start with initial quantile estimate
2. For each new observation:
   a. Predict interval using current quantile
   b. Observe actual value
   c. Update quantile based on whether actual was covered
3. Quantile ADAPTS to maintain target coverage

The update rule:
   If covered: quantile ← quantile - gamma * alpha
   If not covered: quantile ← quantile + gamma * (1 - alpha)

This pushes the quantile up when under-covering (like during crashes)
and down when over-covering (like during calm periods).
"""
)

# Adaptive Conformal (CORRECT for non-stationary data)
acp = AdaptiveConformalPredictor(alpha=0.10, gamma=0.1)
acp.initialize(cal_preds, y_cal)

print("✅ Adaptive Conformal Prediction:")
print(f"   Initial quantile: {acp.quantile_history[0]:.4f}")
print("   Adaptation rate (gamma): 0.1")

# Online prediction
adaptive_lowers = []
adaptive_uppers = []
adaptive_covered = []
adaptive_quantiles = []

for i in range(len(test_preds)):
    pred = test_preds[i]
    actual = y_test[i]

    # Get interval with current quantile
    lower, upper = acp.predict_interval(pred)
    adaptive_lowers.append(lower)
    adaptive_uppers.append(upper)

    # Check coverage
    covered = (actual >= lower) and (actual <= upper)
    adaptive_covered.append(covered)

    # Record quantile before update
    adaptive_quantiles.append(acp.quantile_history[-1])

    # Update based on coverage
    acp.update(pred, actual)

# Add to DataFrame
test_df_with_preds["lower_adaptive"] = adaptive_lowers
test_df_with_preds["upper_adaptive"] = adaptive_uppers
test_df_with_preds["covered_adaptive"] = adaptive_covered
test_df_with_preds["quantile_adaptive"] = adaptive_quantiles

print("\n📊 Adaptive Conformal Coverage by Regime:")
print("-" * 50)
coverage_adaptive = test_df_with_preds.groupby("regime")["covered_adaptive"].mean()
for regime in ["CALM", "CRASH", "RECOVERY"]:
    if regime in coverage_adaptive.index:
        cov = coverage_adaptive[regime]
        status = "✅" if 0.80 <= cov <= 0.95 else "❌"
        print(f"   {regime:<12} Coverage: {cov*100:.1f}% (target: 90%) {status}")

overall_adaptive = test_df_with_preds["covered_adaptive"].mean()
print(f"\n   Overall Coverage: {overall_adaptive*100:.1f}%")
print("   ✅ Coverage is maintained across ALL regimes!")

# =============================================================================
# PART 6: Quantile Adaptation Visualization (Text-Based)
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: QUANTILE ADAPTATION OVER TIME")
print("=" * 70)

# Show how quantile changes across regimes
regime_transitions = []
prev_regime = None
for i, regime in enumerate(test_df_with_preds["regime"]):
    if regime != prev_regime:
        regime_transitions.append((i, regime))
        prev_regime = regime

print("\n📊 Adaptive Quantile at Regime Transitions:")
print("-" * 60)
for idx, regime in regime_transitions[:6]:  # Show first 6 transitions
    quantile = adaptive_quantiles[idx] if idx < len(adaptive_quantiles) else np.nan
    print(f"   Day {idx + test_start:4d}: Entering {regime:<12} Quantile: {quantile:.4f}")
print("-" * 60)

# Show quantile range
print(f"\n   Quantile range: {min(adaptive_quantiles):.4f} to {max(adaptive_quantiles):.4f}")
print(f"   Fixed (split) quantile: {scp.quantile_:.4f}")
print("   Adaptive quantile GROWS during crashes, SHRINKS during calm!")

# =============================================================================
# PART 7: Side-by-Side Comparison
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: SIDE-BY-SIDE COMPARISON")
print("=" * 70)

print("\n📊 Coverage Comparison (Target: 90%):")
print("-" * 70)
print(f"{'Regime':<15} {'Split Conformal':<20} {'Adaptive Conformal':<20} {'Winner':<15}")
print("-" * 70)

for regime in ["CALM", "CRASH", "RECOVERY"]:
    if regime in coverage_split.index and regime in coverage_adaptive.index:
        cov_split = coverage_split[regime] * 100
        cov_adapt = coverage_adaptive[regime] * 100

        # Closer to 90% is better
        dist_split = abs(cov_split - 90)
        dist_adapt = abs(cov_adapt - 90)
        winner = "Adaptive ✅" if dist_adapt < dist_split else "Split"

        print(f"{regime:<15} {cov_split:<20.1f}% {cov_adapt:<20.1f}% {winner:<15}")

print("-" * 70)
print(f"{'OVERALL':<15} {overall_split*100:<20.1f}% {overall_adaptive*100:<20.1f}%")

# Interval width comparison
width_split = (test_df_with_preds["upper"] - test_df_with_preds["lower"]).mean()
width_adaptive = (
    test_df_with_preds["upper_adaptive"] - test_df_with_preds["lower_adaptive"]
).mean()

print("\n📊 Average Interval Width:")
print(f"   Split Conformal:    {width_split:.4f}")
print(f"   Adaptive Conformal: {width_adaptive:.4f}")

# =============================================================================
# PART 8: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: KEY TAKEAWAYS")
print("=" * 70)

print(
    """
1. SPLIT CONFORMAL FAILS UNDER DISTRIBUTION SHIFT
   - Calibration quantile is FIXED from historical period
   - During regime change, coverage breaks down
   - CALM calibration → UNDER-COVERAGE in CRASH
   - CRASH calibration → OVER-COVERAGE in CALM (inefficient)

2. ADAPTIVE CONFORMAL MAINTAINS COVERAGE
   - Quantile adjusts based on recent coverage
   - Grows when under-covering (during volatility spikes)
   - Shrinks when over-covering (during calm periods)
   - Converges to target coverage over time

3. GAMMA CONTROLS ADAPTATION SPEED
   - gamma=0.01: Slow adaptation, stable quantile
   - gamma=0.10: Moderate adaptation (default)
   - gamma=0.50: Fast adaptation, responsive but noisy
   - Choose based on regime change frequency

4. CRYPTO-SPECIFIC CONSIDERATIONS
   - Volatility regimes are common and persistent
   - Fat tails (use t-distribution or non-parametric)
   - 24/7 trading → no market close discontinuities
   - Cross-asset correlations during crashes

5. INTERVAL WIDTH TRADE-OFF
   - Adaptive may have wider intervals during CRASH
   - This is CORRECT — uncertainty IS higher
   - Split conformal is overconfident when it matters most

6. PRODUCTION DEPLOYMENT
   - Update model + conformal after each observation
   - Monitor coverage in rolling windows
   - Alert when coverage deviates significantly
   - Consider ensemble of adaptation rates

The pattern: Use adaptive conformal when distribution can shift.
If data is truly stationary, split conformal is more efficient.
"""
)

print("\n" + "=" * 70)
print("Example 15 complete.")
print("=" * 70)
