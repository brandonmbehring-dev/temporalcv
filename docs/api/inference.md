# Inference

Statistical inference tools for cross-validation results.

## Overview

Provides bootstrap-based inference for test statistics computed across CV folds,
particularly useful when standard asymptotic inference is unreliable due to few folds.

**Knowledge Tier**: [T2] - Wild bootstrap is established, but CV fold independence
assumption requires domain-specific validation.

## Data Classes

### `WildBootstrapResult`

```python
@dataclass
class WildBootstrapResult:
    statistic: float           # Original test statistic
    p_value: float            # Bootstrap p-value
    ci_lower: float           # Lower confidence bound
    ci_upper: float           # Upper confidence bound
    n_bootstrap: int          # Number of bootstrap samples
    bootstrap_dist: np.ndarray  # Bootstrap distribution
```

## Functions

### `wild_cluster_bootstrap`

Wild cluster bootstrap for dependent data:

```python
from temporalcv.inference import wild_cluster_bootstrap

# Bootstrap inference on fold statistics
result = wild_cluster_bootstrap(
    fold_statistics=fold_maes,
    n_bootstrap=1000,
    confidence_level=0.95,
)

print(f"Statistic: {result.statistic:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"p-value: {result.p_value:.4f}")
```

## Usage Example

```python
from temporalcv import WalkForwardCV
from temporalcv.inference import wild_cluster_bootstrap
import numpy as np

# Collect fold statistics
cv = WalkForwardCV(n_splits=10)
fold_maes = []

for train_idx, test_idx in cv.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    fold_maes.append(np.mean(np.abs(y[test_idx] - preds)))

fold_maes = np.array(fold_maes)

# Bootstrap inference
result = wild_cluster_bootstrap(fold_maes, n_bootstrap=1000)

print(f"Mean MAE: {result.statistic:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

## Block Bootstrap Confidence Intervals

For time series data, use block bootstrap to preserve temporal dependence:

```python
from temporalcv.inference import block_bootstrap_ci
import numpy as np

# Time series data with autocorrelation
np.random.seed(42)
n = 200
errors = np.zeros(n)
for t in range(1, n):
    errors[t] = 0.7 * errors[t-1] + np.random.randn()

# Compute block bootstrap CI for the mean
result = block_bootstrap_ci(
    data=errors,
    statistic_func=np.mean,
    n_bootstrap=1000,
    block_length='auto',  # Uses n^(1/3) rule
    confidence_level=0.95,
)

print(f"Point estimate: {result.statistic:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Block length used: {result.block_length}")
```

## Comparing Two Models with Bootstrap

```python
from temporalcv.inference import wild_cluster_bootstrap
import numpy as np

# MAE from two models across 10 CV folds
model_a_maes = np.array([0.45, 0.52, 0.48, 0.51, 0.47, 0.49, 0.53, 0.46, 0.50, 0.48])
model_b_maes = np.array([0.42, 0.48, 0.44, 0.46, 0.43, 0.45, 0.49, 0.42, 0.46, 0.44])

# Test if Model B is significantly better
differences = model_a_maes - model_b_maes  # Positive = B is better

result = wild_cluster_bootstrap(differences, n_bootstrap=1000)

print(f"Mean improvement: {result.statistic:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"p-value (one-sided): {result.p_value:.4f}")

if result.ci_lower > 0:
    print("Model B is significantly better (CI excludes zero)")
```

## When to Use

- **Few CV folds** (< 20): Asymptotic inference unreliable
- **Dependent folds**: Standard errors underestimate uncertainty
- **Confidence intervals**: When point estimates alone are insufficient

## See Also

- [Statistical Tests](statistical_tests.md) - DM test for model comparison
- [Wild Cluster Bootstrap](#wild_cluster_bootstrap) - Cluster-robust inference

## References

- Cameron, Gelbach & Miller (2008). "Bootstrap-Based Improvements for
  Inference with Clustered Errors." Review of Economics and Statistics.
- MacKinnon & Webb (2017). "Wild Bootstrap Inference for Wildly
  Different Cluster Sizes." JASA.
