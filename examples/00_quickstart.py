#!/usr/bin/env python3
"""
Quickstart: temporalcv in 50 lines
==================================

This is the fastest way to understand temporalcv's core value proposition.

Run this example:
    python examples/00_quickstart.py

What you'll learn:
    1. How validation gates catch data leakage
    2. The HALT/WARN/PASS decision framework
    3. Basic walk-forward cross-validation
"""

import numpy as np
from sklearn.linear_model import Ridge

from temporalcv.cv import WalkForwardCV
from temporalcv.gates import gate_signal_verification, gate_suspicious_improvement, run_gates

# Generate simple time series data
rng = np.random.default_rng(42)
n = 200
y = np.cumsum(rng.standard_normal(n))  # Random walk
X = np.column_stack([np.roll(y, i) for i in range(1, 6)])  # Lag features
X, y = X[5:], y[5:]  # Remove NaN rows

# Step 1: Walk-forward cross-validation (the right way)
cv = WalkForwardCV(n_splits=5, extra_gap=1, test_size=20)
print("Walk-Forward CV splits:")
for i, (train, test) in enumerate(cv.split(X, y)):
    print(f"  Fold {i+1}: train={len(train)}, test={len(test)}, gap enforced")

# Step 2: Fit model and compute errors
model = Ridge(alpha=1.0)
all_errors = []
persistence_errors = []
for train_idx, test_idx in cv.split(X, y):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    all_errors.extend(np.abs(y[test_idx] - preds))
    persistence_errors.extend(np.abs(y[test_idx] - y[test_idx - 1]))

model_mae = np.mean(all_errors)
baseline_mae = np.mean(persistence_errors)
print(f"\nModel MAE: {model_mae:.4f}")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"Improvement: {(1 - model_mae/baseline_mae)*100:.1f}%")

# Step 3: Run validation gates
print("\n--- Validation Gates ---")
gates = [
    gate_signal_verification(model, X, y, n_shuffles=100, random_state=42),
    gate_suspicious_improvement(model_mae, baseline_mae, threshold=0.20),
]
report = run_gates(gates)
print(report.summary())

# Step 4: Make decision
if report.status == "HALT":
    print("\n⛔ HALT: Investigate before deploying!")
elif report.status == "WARN":
    print("\n⚠️  WARN: Proceed with caution")
else:
    print("\n✅ PASS: Safe to continue")
