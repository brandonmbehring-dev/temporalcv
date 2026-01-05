# Financial Cross-Validation

Cross-validation with purging and embargo for financial data.

## Overview

Implements CV techniques for financial ML where labels often overlap
(e.g., 5-day forward returns share 4 days of data). Standard CV leaks
information through this overlap.

## Key Concepts

- **Purging**: Remove training samples within `purge_gap` of any test sample
- **Embargo**: Additional percentage of samples removed after test set
- **Label overlap**: When labels use future data (e.g., forward returns)

## Classes

### `PurgedKFold`

K-fold with purging and embargo:

```python
from temporalcv.cv_financial import PurgedKFold

cv = PurgedKFold(
    n_splits=5,
    purge_gap=5,    # Days to purge around test set
    embargo_pct=0.01,  # 1% embargo after test
)

for train_idx, test_idx in cv.split(X, y, times=timestamps):
    # train_idx has samples purged that overlap with test_idx
    pass
```

### `CombinatorialPurgedCV`

All (n choose k) combinations with purging:

```python
from temporalcv.cv_financial import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(
    n_splits=5,
    purge_gap=5,
)
```

### `PurgedWalkForward`

Walk-forward with purging:

```python
from temporalcv.cv_financial import PurgedWalkForward

cv = PurgedWalkForward(
    n_splits=5,
    purge_gap=5,
    embargo_pct=0.01,
)
```

## Data Classes

### `PurgedSplit`

```python
@dataclass(frozen=True)
class PurgedSplit:
    train_indices: np.ndarray   # Indices after purging
    test_indices: np.ndarray    # Test indices
    n_purged: int              # Samples removed by purging
    n_embargoed: int           # Samples removed by embargo
```

## Usage Example

```python
from temporalcv.cv_financial import PurgedKFold
import numpy as np

# Financial data with timestamps
X, y = ...
timestamps = pd.date_range(...)

# Label uses 5-day forward returns → purge_gap=5
cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)

scores = []
for train_idx, test_idx in cv.split(X, y, times=timestamps):
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    scores.append(compute_score(y[test_idx], pred))
```

## References

- De Prado (2018). "Advances in Financial Machine Learning." Wiley. Chapter 7.
- Lopez de Prado & Lewis (2019). "Detection of False Investment Strategies."
