# API Reference: Event Metrics

Novel metrics for direction prediction with proper calibration.

---

## Data Classes

### `BrierScoreResult`

Result from Brier score computation.

```python
@dataclass
class BrierScoreResult:
    brier_score: float    # Mean squared error (0 = perfect, 1 = worst)
    reliability: float    # Calibration component (lower = better)
    resolution: float     # Refinement component (higher = better)
    uncertainty: float    # Base rate uncertainty
    n_samples: int        # Number of samples
    n_classes: int        # Number of classes (2 or 3)
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `skill_score` | `float` | BSS = 1 - (BS / uncertainty) |

**Decomposition** (Murphy 1973):
```
BS = Reliability - Resolution + Uncertainty
```

---

### `PRAUCResult`

Result from PR-AUC computation.

```python
@dataclass
class PRAUCResult:
    pr_auc: float                  # Area under PR curve
    baseline: float                # Random classifier PR-AUC
    precision_at_50_recall: float  # Precision at 50% recall
    n_positive: int                # Positive samples
    n_negative: int                # Negative samples
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `lift_over_baseline` | `float` | PR-AUC / baseline |
| `n_total` | `int` | Total samples |
| `imbalance_ratio` | `float` | Majority / minority class ratio |

---

## Functions

### `compute_direction_brier`

Compute Brier score for direction prediction.

```python
def compute_direction_brier(
    pred_probs: np.ndarray,
    actual_directions: np.ndarray,
    n_classes: Literal[2, 3] = 2,
) -> BrierScoreResult
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_probs` | `np.ndarray` | required | Predicted probabilities |
| `actual_directions` | `np.ndarray` | required | Actual directions as integers |
| `n_classes` | `int` | `2` | Number of classes (2 or 3) |

**For 2-class**:
- `pred_probs`: 1D array, P(positive)
- `actual_directions`: 0 = negative, 1 = positive

**For 3-class**:
- `pred_probs`: (n_samples, 3), probabilities for [DOWN, FLAT, UP]
- `actual_directions`: 0 = DOWN, 1 = FLAT, 2 = UP

**Example**:

```python
from temporalcv.metrics.event import compute_direction_brier

# 2-class
probs = np.array([0.7, 0.3, 0.8, 0.2])
actuals = np.array([1, 0, 1, 0])
result = compute_direction_brier(probs, actuals, n_classes=2)
print(f"Brier: {result.brier_score:.4f}")
print(f"Skill: {result.skill_score:.3f}")
```

---

### `compute_pr_auc`

Compute Area Under Precision-Recall Curve.

```python
def compute_pr_auc(
    pred_probs: np.ndarray,
    actual_binary: np.ndarray,
) -> PRAUCResult
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pred_probs` | `np.ndarray` | Predicted probabilities of positive class |
| `actual_binary` | `np.ndarray` | Binary labels (0 or 1) |

**Returns**: `PRAUCResult`

**Notes**:
- Preferred over ROC-AUC for imbalanced classification
- Baseline equals positive class rate (random classifier)

**Example**:

```python
from temporalcv.metrics.event import compute_pr_auc

probs = np.array([0.9, 0.8, 0.3, 0.1, 0.7])
actuals = np.array([1, 1, 0, 0, 1])

result = compute_pr_auc(probs, actuals)
print(f"PR-AUC: {result.pr_auc:.3f}")
print(f"Baseline: {result.baseline:.3f}")
print(f"Lift: {result.lift_over_baseline:.2f}x")
```

---

### `compute_calibrated_direction_brier`

Compute Brier score with reliability diagram data.

```python
def compute_calibrated_direction_brier(
    pred_probs: np.ndarray,
    actual_directions: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pred_probs` | `np.ndarray` | required | Predicted probabilities (1D) |
| `actual_directions` | `np.ndarray` | required | Binary outcomes |
| `n_bins` | `int` | `10` | Number of calibration bins |

**Returns**: `(brier_score, bin_means, bin_true_fractions)`

**Example** (plotting reliability diagram):

```python
brier, bin_means, bin_fracs = compute_calibrated_direction_brier(probs, actuals)

import matplotlib.pyplot as plt
plt.plot(bin_means, bin_fracs, 'o-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.xlabel('Predicted probability')
plt.ylabel('Observed frequency')
plt.legend()
```

---

### `convert_predictions_to_direction_probs`

Convert point predictions with uncertainty to direction probabilities.

```python
def convert_predictions_to_direction_probs(
    point_predictions: np.ndarray,
    prediction_std: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `point_predictions` | `np.ndarray` | required | Point predictions |
| `prediction_std` | `np.ndarray` | required | Prediction standard deviation |
| `threshold` | `float` | `0.0` | UP/DOWN threshold |

**Returns**: P(UP) = P(X > threshold)

**Assumes**: Gaussian prediction distribution

**Example**:

```python
from temporalcv.bagging import create_block_bagger
from temporalcv.metrics.event import (
    convert_predictions_to_direction_probs,
    compute_direction_brier,
)

# Get predictions with uncertainty
mean, std = bagger.predict_with_uncertainty(X_test)

# Convert to direction probabilities
p_up = convert_predictions_to_direction_probs(mean, std, threshold=0.01)

# Compute Brier score
actual_up = (actuals > 0.01).astype(int)
result = compute_direction_brier(p_up, actual_up, n_classes=2)
```

---

## Metric Interpretation

### Brier Score

| Score | Interpretation |
|-------|----------------|
| 0.0 | Perfect |
| 0.25 | Random guessing (50% base rate) |
| 1.0 | Worst possible |

### Brier Skill Score (BSS)

| BSS | Interpretation |
|-----|----------------|
| < 0 | Worse than climatology |
| 0 | Same as climatology |
| > 0 | Skill over climatology |
| 1.0 | Perfect |

### PR-AUC

| Context | Interpretation |
|---------|----------------|
| = baseline | Random classifier |
| > baseline | Some skill |
| = 1.0 | Perfect separation |
