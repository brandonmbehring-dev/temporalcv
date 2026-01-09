"""
Example 06: Financial Cross-Validation with PurgedKFold
=======================================================

Real-World Case Study: Predicting Stock Returns
-----------------------------------------------
Financial data has unique challenges that standard cross-validation ignores:

1. **Label Overlap**: If predicting 5-day returns, labels for adjacent days
   share 4 days of price data. Train/test contamination occurs even without
   time overlap.

2. **Lookahead Bias**: Features computed from future prices (rolling means
   that include future data) leak information.

3. **Non-IID Returns**: Returns are serially correlated during market events,
   violating assumptions of standard CV metrics.

This example demonstrates how PurgedKFold handles these challenges.

Key Concepts
------------
- PurgedKFold: Removes train samples that overlap with test labels
- Embargo: Additional gap after each test fold to prevent leakage
- Label overlap detection and quantification
- Comparison with naive (leaky) approaches
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from temporalcv import WalkForwardCV

# temporalcv imports
from temporalcv.cv_financial import PurgedKFold

# =============================================================================
# PART 1: Generate Synthetic Financial Data
# =============================================================================


def generate_synthetic_returns(
    n_days: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily stock data with realistic properties.

    Properties:
    - Daily log returns with slight autocorrelation
    - Volatility clustering (GARCH-like)
    - Multi-day return labels (overlapping)

    Parameters
    ----------
    n_days : int
        Number of trading days to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, price, returns, and features.
    """
    rng = np.random.default_rng(seed)

    # Generate dates (business days only)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

    # Generate returns with volatility clustering
    volatility = np.zeros(n_days)
    volatility[0] = 0.02  # 2% daily vol
    returns = np.zeros(n_days)

    for t in range(1, n_days):
        # GARCH(1,1)-like volatility
        volatility[t] = np.sqrt(0.00001 + 0.1 * returns[t - 1] ** 2 + 0.85 * volatility[t - 1] ** 2)
        returns[t] = rng.normal(0.0003, volatility[t])  # Small drift

    # Generate price from returns
    price = 100 * np.exp(np.cumsum(returns))

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "price": price,
            "returns": returns,
            "volatility": volatility,
        }
    )

    # Generate features (lagged, so no lookahead bias)
    df["returns_lag1"] = df["returns"].shift(1)
    df["returns_lag2"] = df["returns"].shift(2)
    df["returns_lag5"] = df["returns"].shift(5)

    # Rolling features (properly shifted)
    df["vol_5d"] = df["returns"].shift(1).rolling(5).std()
    df["vol_20d"] = df["returns"].shift(1).rolling(20).std()
    df["momentum_5d"] = df["returns"].shift(1).rolling(5).sum()
    df["momentum_20d"] = df["returns"].shift(1).rolling(20).sum()

    # Create multi-day forward return (the target)
    # This is where label overlap occurs
    forward_window = 5
    df["forward_return_5d"] = df["returns"].rolling(forward_window).sum().shift(-forward_window)

    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)

    return df, forward_window


print("=" * 70)
print("PART 1: Data Generation")
print("=" * 70)

df, forward_window = generate_synthetic_returns(n_days=1000, seed=42)
print(f"\nGenerated {len(df)} days of synthetic financial data")
print(f"Forward return window: {forward_window} days")
print(f"\nFeatures: {[c for c in df.columns if c not in ['date', 'price', 'forward_return_5d']]}")
print("\nTarget: forward_return_5d (sum of returns over next 5 days)")

# =============================================================================
# PART 2: The Problem - Label Overlap
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: Understanding Label Overlap")
print("=" * 70)

print(
    """
When predicting 5-day forward returns:

Day 1 label: returns[2] + returns[3] + returns[4] + returns[5] + returns[6]
Day 2 label: returns[3] + returns[4] + returns[5] + returns[6] + returns[7]

These labels share 4 days of return data!

If Day 1 is in training and Day 2 is in test, the model has already
seen most of the information it needs to predict Day 2's label.

This is NOT the same as time overlap. It's information leakage
through shared label components.
"""
)

# Quantify overlap
overlap_days = forward_window - 1
overlap_fraction = overlap_days / forward_window

print(f"For {forward_window}-day returns:")
print(f"  - Adjacent labels share {overlap_days} days of data")
print(f"  - Overlap fraction: {overlap_fraction:.0%}")
print("  - Effective information leakage: HIGH")

# =============================================================================
# PART 3: WRONG Approach - Standard KFold
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: WRONG Approach - Standard KFold (with leakage)")
print("=" * 70)

# Prepare features and target
feature_cols = [
    "returns_lag1",
    "returns_lag2",
    "returns_lag5",
    "vol_5d",
    "vol_20d",
    "momentum_5d",
    "momentum_20d",
]
X = df[feature_cols].values
y = df["forward_return_5d"].values

# Standard KFold - WRONG for financial data
print("\nUsing sklearn KFold (shuffle=True) - THIS IS WRONG:")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
scores_kfold = cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_squared_error")
rmse_kfold = np.sqrt(-scores_kfold.mean())

print(f"  KFold RMSE: {rmse_kfold:.6f}")
print("  WARNING: This score is OPTIMISTICALLY BIASED due to label overlap!")

# =============================================================================
# PART 4: BETTER Approach - WalkForwardCV
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: Better Approach - WalkForwardCV")
print("=" * 70)

print("\nUsing WalkForwardCV with gap (respects time order):")
wf_cv = WalkForwardCV(
    n_splits=5,
    window_type="expanding",
    extra_gap=forward_window,  # Gap matches prediction horizon
    test_size=50,
)

scores_wf = cross_val_score(model, X, y, cv=wf_cv, scoring="neg_mean_squared_error")
rmse_wf = np.sqrt(-scores_wf.mean())

print(f"  WalkForward RMSE: {rmse_wf:.6f}")
print("  This is more realistic but still doesn't handle label overlap")
print("  within each fold.")

# =============================================================================
# PART 5: BEST Approach - PurgedKFold
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: Best Approach - PurgedKFold with Embargo")
print("=" * 70)

print(
    """
PurgedKFold handles label overlap by:

1. PURGING: Remove training samples whose labels overlap with test labels
2. EMBARGO: Add extra gap after test fold to prevent information leakage

This ensures the model cannot exploit label overlap for better scores.
"""
)

# Create PurgedKFold with label information
dates = df["date"].values

purged_cv = PurgedKFold(
    n_splits=5,
    purge_gap=forward_window,  # Purge samples with label overlap
    embargo_pct=0.01,  # 1% of data as additional buffer
)

# Manual CV loop to show what PurgedKFold does
print("\nFold details:")
fold_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(purged_cv.split(X, y)):
    # Fit and predict
    model_fold = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    model_fold.fit(X[train_idx], y[train_idx])
    y_pred = model_fold.predict(X[test_idx])

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y[test_idx], y_pred))
    fold_scores.append(rmse)

    # Show fold info
    train_dates = dates[train_idx]
    test_dates = dates[test_idx]

    print(f"\n  Fold {fold_idx + 1}:")
    print(f"    Train: {len(train_idx)} samples")
    print(f"    Test:  {len(test_idx)} samples")
    print(f"    RMSE:  {rmse:.6f}")

rmse_purged = np.mean(fold_scores)

# =============================================================================
# PART 6: Comparison
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: Method Comparison")
print("=" * 70)

print(
    f"""
Method Comparison (lower RMSE seems better, but context matters):

  KFold (shuffle=True):  RMSE = {rmse_kfold:.6f}  <- LEAKY, not reliable
  WalkForwardCV:         RMSE = {rmse_wf:.6f}  <- Better, but misses overlap
  PurgedKFold:           RMSE = {rmse_purged:.6f}  <- Most conservative, most realistic

The PurgedKFold RMSE is typically HIGHER than KFold because:
1. No information leakage through label overlap
2. Embargo prevents temporal leakage
3. This estimate matches real-world deployment performance

If your KFold RMSE is much lower than PurgedKFold RMSE, you likely have
significant information leakage that will hurt production performance.
"""
)

# Leakage quantification
leakage_ratio = (rmse_purged - rmse_kfold) / rmse_purged
print(f"Estimated leakage impact: {leakage_ratio:.1%} of error hidden by KFold")

# =============================================================================
# PART 7: Key Takeaways
# =============================================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print(
    """
1. LABEL OVERLAP is different from TIME OVERLAP
   - Even with proper time splits, overlapping labels leak information
   - Multi-day returns are the most common case

2. USE PurgedKFold FOR FINANCIAL DATA
   - Removes training samples with label overlap
   - Adds embargo period after test folds
   - Provides realistic performance estimates

3. EXPECT HIGHER ERRORS WITH PROPER CV
   - If PurgedKFold error >> KFold error, you had significant leakage
   - Production performance will match PurgedKFold, not KFold

4. PARAMETERS TO SET
   - embargo_days: At least equal to your prediction horizon
   - pct_embargo: 0.5-2% of data is typical

5. COMBINE WITH VALIDATION GATES
   - gate_signal_verification catches many feature leakage issues
   - PurgedKFold handles label overlap specifically
"""
)

# =============================================================================
# PART 8: Advanced - Quantifying Label Overlap
# =============================================================================

print("\n" + "=" * 70)
print("ADVANCED: Quantifying Label Overlap Impact")
print("=" * 70)


def compute_overlap_matrix(n_samples: int, label_window: int) -> np.ndarray:
    """
    Compute the overlap matrix between sample labels.

    Element (i,j) is the fraction of shared information between
    labels of samples i and j.
    """
    overlap = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            # Labels span [i:i+label_window] and [j:j+label_window]
            start_i, end_i = i, i + label_window
            start_j, end_j = j, j + label_window

            # Compute overlap
            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)
            shared = max(0, overlap_end - overlap_start)

            overlap[i, j] = shared / label_window

    return overlap


# For a smaller example
n_example = 20
overlap_matrix = compute_overlap_matrix(n_example, forward_window)

# Show overlap statistics
print(f"\nFor {forward_window}-day labels with {n_example} consecutive samples:")
print(f"  - Max overlap (adjacent): {overlap_matrix[0, 1]:.0%}")
print(f"  - Overlap at distance 5:  {overlap_matrix[0, 5]:.0%}")
print(f"  - Overlap at distance 10: {overlap_matrix[0, 10]:.0%}")

# PurgedKFold removes samples with overlap > 0
samples_to_purge = np.sum(overlap_matrix[0, :] > 0) - 1  # Exclude self
print(f"  - Samples to purge around sample 0: {samples_to_purge}")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
