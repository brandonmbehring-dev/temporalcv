"""
Example 13: Macro GDP Forecasting
=================================

Real-World Case Study: Quarterly Economic Forecasting
-----------------------------------------------------
Macroeconomic forecasting presents unique challenges:

1. **Low-Frequency Data**: Quarterly observations = ~80-200 samples
   - Small samples make statistical tests less reliable
   - Harvey correction becomes critical (small-sample adjustment)

2. **Nested Model Comparison**: AR(1) vs AR(1)+Indicators
   - The simpler model is "nested" within the larger model
   - Standard DM test is biased for this comparison
   - Clark-West (CW) test is the correct choice

3. **Limited CV Splits**: With n<100, you can't do 10-fold CV
   - WalkForwardCV with n_splits=3-4 is typical
   - Each split must have enough data for reliable estimation

This example demonstrates:
- Generating GDP-like quarterly time series
- Why DM test fails for nested model comparison
- How CW test provides unbiased comparison
- Walk-forward CV with limited quarterly data

Key Concepts
------------
- Nested models: AR(1) is nested in AR(1)+X (X has coefficient=0 under null)
- DM test bias: Estimating extra parameters adds noise, biasing against larger model
- CW adjustment: Removes the noise penalty from estimated-zero coefficients
- Harvey correction: Small-sample adjustment for finite n

References
----------
- Clark & West (2007) "Approximately Normal Tests for Equal Predictive Accuracy"
  Journal of Econometrics 138(1):291-311
- Diebold & Mariano (1995) "Comparing Predictive Accuracy" JBES 13(3):253-263
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# temporalcv imports
from temporalcv import WalkForwardCV
from temporalcv.statistical_tests import dm_test, cw_test

# =============================================================================
# PART 1: Generate Synthetic Quarterly GDP Data
# =============================================================================


def generate_gdp_data(
    n_quarters: int = 100,
    base_growth: float = 0.02,  # 2% annual growth = 0.5% quarterly
    ar_coef: float = 0.7,  # Persistence in growth
    leading_indicator_coef: float = 0.0,  # True coefficient (0 = no predictive power)
    noise_std: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic quarterly GDP growth data.

    The DGP is:
        growth_t = mu + phi * growth_{t-1} + beta * indicator_{t-1} + epsilon_t

    When beta=0 (default), the indicator has NO predictive power,
    but the AR(1)+X model must estimate beta, which adds noise.

    Parameters
    ----------
    n_quarters : int
        Number of quarterly observations.
    base_growth : float
        Mean growth rate (annualized).
    ar_coef : float
        AR(1) coefficient for growth persistence.
    leading_indicator_coef : float
        True coefficient on leading indicator (0 = no signal).
    noise_std : float
        Standard deviation of innovations.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Quarterly data with growth and indicator.
    """
    rng = np.random.default_rng(seed)

    # Quarterly base growth
    mu = base_growth / 4

    # Generate leading indicator (independent noise)
    indicator = rng.normal(0, 1, n_quarters)

    # Generate GDP growth via AR(1) + X
    growth = np.zeros(n_quarters)
    growth[0] = mu + rng.normal(0, noise_std)

    for t in range(1, n_quarters):
        growth[t] = (
            mu
            + ar_coef * (growth[t - 1] - mu)
            + leading_indicator_coef * indicator[t - 1]
            + rng.normal(0, noise_std)
        )

    # Create quarterly dates
    dates = pd.date_range("1990-01-01", periods=n_quarters, freq="QE")

    df = pd.DataFrame({
        "growth": growth,
        "indicator": indicator,
    }, index=dates)

    # Add lagged features
    df["growth_lag1"] = df["growth"].shift(1)
    df["indicator_lag1"] = df["indicator"].shift(1)

    df = df.dropna()

    return df


print("=" * 70)
print("EXAMPLE 13: MACRO GDP FORECASTING")
print("=" * 70)

# Generate data with NO indicator signal (beta=0)
# This is the null hypothesis: indicator adds NO predictive power
df = generate_gdp_data(n_quarters=100, leading_indicator_coef=0.0, seed=42)

print(f"\n📊 Generated quarterly GDP data: {len(df)} observations")
print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"   Mean growth: {df['growth'].mean()*100:.2f}% per quarter")
print(f"   Growth volatility: {df['growth'].std()*100:.2f}%")
print(f"\n   KEY: Indicator has TRUE coefficient = 0 (no signal)")
print(f"   This means AR(1)+Indicator should NOT beat AR(1)!")

# =============================================================================
# PART 2: The Problem — Comparing Nested Models
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: THE PROBLEM — COMPARING NESTED MODELS")
print("=" * 70)

print("""
Two models to compare:

   AR(1):        growth_t = mu + phi * growth_{t-1} + epsilon_t
   AR(1)+X:      growth_t = mu + phi * growth_{t-1} + beta * indicator_{t-1} + epsilon_t

AR(1) is NESTED within AR(1)+X:
   - Under H0: beta = 0, so AR(1)+X reduces to AR(1)
   - AR(1) is a restricted version of AR(1)+X

The DM test assumes models are NON-NESTED:
   - Both models estimate their "best" parameters
   - Compare which has lower expected loss

For nested models, DM test is BIASED:
   - AR(1)+X must estimate beta even when true beta=0
   - Estimating beta adds noise to forecasts
   - DM test penalizes AR(1)+X for this noise
   - Result: DM test favors AR(1) even when AR(1)+X is correct!
""")

# =============================================================================
# PART 3: Generate Out-of-Sample Forecasts
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: GENERATE OUT-OF-SAMPLE FORECASTS")
print("=" * 70)

# Split data
train_size = int(len(df) * 0.6)
test_size = len(df) - train_size

# Walk-forward to generate forecasts
actuals = []
pred_ar1 = []  # Restricted model (AR(1))
pred_ar1_x = []  # Unrestricted model (AR(1)+X)

# Simple expanding window
for t in range(train_size, len(df)):
    # Training data up to t-1
    train_df = df.iloc[:t]

    # Features
    X_train_ar1 = train_df[["growth_lag1"]].values
    X_train_ar1_x = train_df[["growth_lag1", "indicator_lag1"]].values
    y_train = train_df["growth"].values

    # Test point at t
    X_test_ar1 = df.iloc[t:t+1][["growth_lag1"]].values
    X_test_ar1_x = df.iloc[t:t+1][["growth_lag1", "indicator_lag1"]].values

    # Train models
    model_ar1 = Ridge(alpha=0.01)
    model_ar1_x = Ridge(alpha=0.01)

    model_ar1.fit(X_train_ar1, y_train)
    model_ar1_x.fit(X_train_ar1_x, y_train)

    # Predict
    pred_ar1.append(model_ar1.predict(X_test_ar1)[0])
    pred_ar1_x.append(model_ar1_x.predict(X_test_ar1_x)[0])
    actuals.append(df.iloc[t]["growth"])

actuals = np.array(actuals)
pred_ar1 = np.array(pred_ar1)
pred_ar1_x = np.array(pred_ar1_x)

# Compute errors
errors_ar1 = actuals - pred_ar1  # Restricted model
errors_ar1_x = actuals - pred_ar1_x  # Unrestricted model

# MSE
mse_ar1 = np.mean(errors_ar1**2)
mse_ar1_x = np.mean(errors_ar1_x**2)

print(f"📊 Out-of-Sample Performance ({test_size} quarters):")
print("-" * 50)
print(f"{'Model':<20} {'MSE':<15} {'RMSE':<15}")
print("-" * 50)
print(f"{'AR(1)':<20} {mse_ar1:.6f} {np.sqrt(mse_ar1):.4f}")
print(f"{'AR(1)+Indicator':<20} {mse_ar1_x:.6f} {np.sqrt(mse_ar1_x):.4f}")
print("-" * 50)

# Note which model appears better
if mse_ar1_x < mse_ar1:
    print(f"\n   AR(1)+X appears better by {(mse_ar1 - mse_ar1_x)/mse_ar1*100:.1f}%")
else:
    print(f"\n   AR(1) appears better by {(mse_ar1_x - mse_ar1)/mse_ar1_x*100:.1f}%")

# =============================================================================
# PART 4: WRONG Approach — Using DM Test for Nested Models
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: WRONG APPROACH — USING DM TEST FOR NESTED MODELS")
print("=" * 70)

print("""
The DM test compares loss differentials:
   d_t = L(e_ar1_t) - L(e_ar1x_t)

Under H0: E[d_t] = 0 (equal predictive accuracy)

PROBLEM: For nested models, DM test has wrong size under H0.
   - True beta=0, so models should have equal accuracy
   - But DM test rejects too often (over-sized test)
""")

# Run DM test (WRONG for nested models)
dm_result = dm_test(
    errors_1=errors_ar1,  # AR(1) - restricted
    errors_2=errors_ar1_x,  # AR(1)+X - unrestricted
    h=1,
    loss="squared",
    alternative="two-sided",
    harvey_correction=True,
)

print(f"❌ DM Test Results (WRONG for nested models):")
print(f"   Test statistic: {dm_result.statistic:.3f}")
print(f"   p-value: {dm_result.pvalue:.3f}")
print(f"   Conclusion at α=0.05: {'Reject H0' if dm_result.pvalue < 0.05 else 'Fail to reject H0'}")

if dm_result.pvalue < 0.05:
    print(f"\n   ⚠️  DM test rejects equal accuracy!")
    print(f"   But we KNOW the indicator has no signal (beta=0)")
    print(f"   This is a FALSE REJECTION due to test bias")
else:
    print(f"\n   DM test correctly fails to reject, but this is luck")
    print(f"   On average, DM test over-rejects for nested models")

# =============================================================================
# PART 5: Why DM Test is Biased for Nested Models
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: WHY DM TEST IS BIASED FOR NESTED MODELS")
print("=" * 70)

print("""
The bias comes from parameter estimation uncertainty:

1. AR(1)+X estimates coefficient beta on indicator
2. True beta = 0, but sample estimate beta_hat ≠ 0
3. Using beta_hat in forecasts adds noise (even though true beta=0)
4. AR(1) doesn't have this noise
5. DM test sees AR(1) as having lower loss variance

The key insight from Clark & West (2007):
   - Under H0 (beta=0), the squared prediction difference should be:
     E[(ŷ_ar1_x - ŷ_ar1)²] > 0  (due to estimation noise)
   - This "noise penalty" biases DM against AR(1)+X

Mathematical adjustment:
   d*_t = d_t - (ŷ_ar1 - ŷ_ar1_x)²

The CW test uses d*_t instead of d_t, removing the bias.
""")

# =============================================================================
# PART 6: CORRECT Approach — Using CW Test for Nested Models
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: CORRECT APPROACH — USING CW TEST FOR NESTED MODELS")
print("=" * 70)

print("""
The Clark-West (CW) test adjusts for the parameter estimation bias:

   d*_t = L(e_ar1) - L(e_ar1x) - (ŷ_ar1 - ŷ_ar1x)²

The adjustment term (ŷ_ar1 - ŷ_ar1x)² removes the expected noise
from estimating the extra parameters in the unrestricted model.

Under H0 (models have equal population accuracy), CW test has:
   - Correct size (rejects at α=0.05 about 5% of the time)
   - Good power against alternatives where AR(1)+X truly helps
""")

# Run CW test (CORRECT for nested models)
cw_result = cw_test(
    errors_unrestricted=errors_ar1_x,  # AR(1)+X - larger model
    errors_restricted=errors_ar1,  # AR(1) - nested model
    predictions_unrestricted=pred_ar1_x,
    predictions_restricted=pred_ar1,
    h=1,
    loss="squared",
    alternative="two-sided",
    harvey_correction=True,
)

print(f"✅ CW Test Results (CORRECT for nested models):")
print(f"   Test statistic: {cw_result.statistic:.3f}")
print(f"   p-value: {cw_result.pvalue:.3f}")
print(f"   Conclusion at α=0.05: {'Reject H0' if cw_result.pvalue < 0.05 else 'Fail to reject H0'}")

if cw_result.pvalue >= 0.05:
    print(f"\n   ✅ CW test correctly fails to reject")
    print(f"   The indicator has no true predictive power (beta=0)")
    print(f"   AR(1) and AR(1)+X have equal population accuracy")
else:
    print(f"\n   CW test rejects (sample variation)")
    print(f"   With small samples, this can happen 5% of the time under H0")

# =============================================================================
# PART 7: Comparison Summary
# =============================================================================

print("\n" + "=" * 70)
print("PART 7: COMPARISON SUMMARY")
print("=" * 70)

print(f"\n📊 Test Comparison (TRUE beta = 0, models should be equal):")
print("-" * 60)
print(f"{'Test':<20} {'Statistic':<15} {'p-value':<15} {'Decision':<15}")
print("-" * 60)
print(f"{'DM (WRONG)':<20} {dm_result.statistic:<15.3f} {dm_result.pvalue:<15.3f} {'Reject' if dm_result.pvalue < 0.05 else 'Fail to reject':<15}")
print(f"{'CW (CORRECT)':<20} {cw_result.statistic:<15.3f} {cw_result.pvalue:<15.3f} {'Reject' if cw_result.pvalue < 0.05 else 'Fail to reject':<15}")
print("-" * 60)

print("""
Key difference:
- DM test is biased AGAINST the larger model (AR(1)+X)
- CW test corrects for this bias
- When comparing nested models, ALWAYS use CW test
""")

# =============================================================================
# PART 8: Walk-Forward CV with Limited Data
# =============================================================================

print("\n" + "=" * 70)
print("PART 8: WALK-FORWARD CV WITH LIMITED DATA")
print("=" * 70)

print("""
With quarterly data (n~100), we have limited splits:
- n_splits=3-4 is typical
- Each fold needs enough data for estimation
- Expanding window preferable (use all available history)
""")

# WalkForwardCV with limited splits
wfcv = WalkForwardCV(
    window_type="expanding",
    window_size=40,  # Start with ~10 years of quarterly data
    horizon=1,  # 1-quarter ahead
    test_size=10,  # Test on 2.5 years
    n_splits=3,
)

print(f"\n📊 WalkForwardCV Configuration:")
print(f"   Window: Expanding from 40 quarters")
print(f"   Test size: 10 quarters per fold")
print(f"   Folds: 3")

print(f"\n📊 Per-Fold Results:")
print("-" * 60)
print(f"{'Fold':<8} {'Train':<12} {'AR(1) RMSE':<15} {'AR(1)+X RMSE':<15}")
print("-" * 60)

for fold_idx, (train_idx, test_idx) in enumerate(wfcv.split(df[["growth_lag1"]].values)):
    # Get data
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Features
    X_train_ar1 = train_df[["growth_lag1"]].values
    X_train_ar1_x = train_df[["growth_lag1", "indicator_lag1"]].values
    y_train = train_df["growth"].values

    X_test_ar1 = test_df[["growth_lag1"]].values
    X_test_ar1_x = test_df[["growth_lag1", "indicator_lag1"]].values
    y_test = test_df["growth"].values

    # Train and predict
    model_ar1 = Ridge(alpha=0.01)
    model_ar1_x = Ridge(alpha=0.01)

    model_ar1.fit(X_train_ar1, y_train)
    model_ar1_x.fit(X_train_ar1_x, y_train)

    pred_fold_ar1 = model_ar1.predict(X_test_ar1)
    pred_fold_ar1_x = model_ar1_x.predict(X_test_ar1_x)

    # RMSE
    rmse_ar1 = np.sqrt(np.mean((y_test - pred_fold_ar1)**2))
    rmse_ar1_x = np.sqrt(np.mean((y_test - pred_fold_ar1_x)**2))

    print(f"{fold_idx+1:<8} {len(train_idx):<12} {rmse_ar1:<15.4f} {rmse_ar1_x:<15.4f}")

print("-" * 60)

# =============================================================================
# PART 9: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("PART 9: KEY TAKEAWAYS")
print("=" * 70)

print("""
1. NESTED MODEL COMPARISON REQUIRES CW TEST, NOT DM TEST
   - DM test is biased against larger model
   - CW test corrects for parameter estimation noise
   - Use DM for non-nested, CW for nested comparisons

2. KNOW YOUR NESTING STRUCTURE
   - AR(1) nested in AR(1)+X: use CW test
   - Random Forest vs GBM: use DM test (not nested)
   - ARIMA(1,0,0) vs ARIMA(2,0,1): use DM test (not nested)

3. SMALL SAMPLES NEED HARVEY CORRECTION
   - harvey_correction=True adjusts for finite-sample bias
   - Critical with n < 100 (quarterly macro data)
   - Uses T instead of T-1 in variance estimation

4. LIMITED CV SPLITS WITH LOW-FREQUENCY DATA
   - Quarterly data: n_splits=3-4 is reasonable
   - Monthly data: n_splits=6-12
   - Daily data: n_splits=10-20

5. EXPANDING WINDOW FOR MACRO DATA
   - Use all available history (data is precious)
   - Sliding window wastes early observations
   - exception: structural breaks require sliding

6. PUBLICATION BIAS AWARENESS
   - Many published macro models use DM test incorrectly
   - Results may over-state indicator predictive power
   - Always check if models are nested before choosing test

The pattern: Is model A a special case of model B?
   Yes → CW test
   No  → DM test
""")

print("\n" + "=" * 70)
print("Example 13 complete.")
print("=" * 70)
