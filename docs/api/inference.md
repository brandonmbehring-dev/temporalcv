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

## When to Use

- **Few CV folds** (< 20): Asymptotic inference unreliable
- **Dependent folds**: Standard errors underestimate uncertainty
- **Confidence intervals**: When point estimates alone are insufficient

## References

- Cameron, Gelbach & Miller (2008). "Bootstrap-Based Improvements for
  Inference with Clustered Errors." Review of Economics and Statistics.
- MacKinnon & Webb (2017). "Wild Bootstrap Inference for Wildly
  Different Cluster Sizes." JASA.
