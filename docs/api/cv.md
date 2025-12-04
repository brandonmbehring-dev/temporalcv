# API Reference: Walk-Forward Cross-Validation

sklearn-compatible temporal cross-validation with gap enforcement for h-step forecasting.

---

## Data Classes

### `SplitInfo`

Metadata for a single CV split.

```python
@dataclass
class SplitInfo:
    split_idx: int     # Zero-based split index
    train_start: int   # First training index (inclusive)
    train_end: int     # Last training index (inclusive)
    test_start: int    # First test index (inclusive)
    test_end: int      # Last test index (inclusive)
```

**Properties**:

| Property | Type | Description |
|----------|------|-------------|
| `train_size` | `int` | Number of training samples |
| `test_size` | `int` | Number of test samples |
| `gap` | `int` | Actual gap between train end and test start |

---

## Classes

### `WalkForwardCV`

Walk-forward cross-validation with gap enforcement.

Inherits from `sklearn.model_selection.BaseCrossValidator`.

```python
class WalkForwardCV(BaseCrossValidator):
    def __init__(
        self,
        n_splits: int = 5,
        window_type: Literal["expanding", "sliding"] = "expanding",
        window_size: Optional[int] = None,
        gap: int = 0,
        test_size: int = 1,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | `int` | `5` | Number of CV folds |
| `window_type` | `str` | `"expanding"` | `"expanding"` or `"sliding"` |
| `window_size` | `int` | `None` | Window size (required for sliding) |
| `gap` | `int` | `0` | Gap between train and test (set ≥ h for h-step forecasts) |
| `test_size` | `int` | `1` | Samples in each test fold |

**Window Types**:

| Type | Description | When to Use |
|------|-------------|-------------|
| `expanding` | Training window grows from start | More data always helps |
| `sliding` | Fixed-size window slides forward | Recent data more relevant |

---

#### Methods

##### `split(X, y=None, groups=None)`

Generate train/test indices.

```python
def split(
    self,
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    groups: Optional[ArrayLike] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]
```

**Yields**: `(train_indices, test_indices)` tuples

**Example**:

```python
cv = WalkForwardCV(n_splits=5, window_type="sliding", window_size=100, gap=2)

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
```

---

##### `get_n_splits(X=None, y=None, groups=None)`

Return number of splits.

```python
def get_n_splits(
    self,
    X: Optional[ArrayLike] = None,
    y: Optional[ArrayLike] = None,
    groups: Optional[ArrayLike] = None,
) -> int
```

**Parameters**:
- If `X` provided: returns actual number of valid splits
- If `X` is None: returns configured `n_splits`

---

##### `get_split_info(X)`

Return detailed metadata for all splits.

```python
def get_split_info(self, X: ArrayLike) -> List[SplitInfo]
```

**Returns**: List of `SplitInfo` objects with split boundaries

**Example**:

```python
cv = WalkForwardCV(n_splits=3, gap=2)

for info in cv.get_split_info(X):
    print(f"Split {info.split_idx}:")
    print(f"  Train: [{info.train_start}, {info.train_end}]")
    print(f"  Test:  [{info.test_start}, {info.test_end}]")
    print(f"  Gap: {info.gap}")
```

---

## sklearn Integration

Works with `cross_val_score` and `GridSearchCV`:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from temporalcv import WalkForwardCV

cv = WalkForwardCV(n_splits=5, window_type="expanding", gap=2)

# cross_val_score
scores = cross_val_score(
    Ridge(), X, y, cv=cv,
    scoring="neg_mean_absolute_error"
)

# GridSearchCV
param_grid = {"alpha": [0.1, 1.0, 10.0]}
search = GridSearchCV(Ridge(), param_grid, cv=cv)
search.fit(X, y)
```

---

## Gap Enforcement

For h-step ahead forecasts, set `gap >= h - 1`:

```python
# For 2-step forecasts
cv = WalkForwardCV(
    n_splits=5,
    window_type="sliding",
    window_size=100,
    gap=2,  # Ensures no leakage
    test_size=1
)

for train_idx, test_idx in cv.split(X):
    # Guaranteed: train_idx[-1] + gap < test_idx[0]
    assert train_idx[-1] + cv.gap < test_idx[0]
```

**Why gap matters**:

```
h=2 forecast: y[t+2] = f(y[t], y[t-1], ...)

Without gap:
  - Train ends at t=99, test starts at t=100
  - Test prediction uses y[99] (last training observation)
  - For h=1: OK
  - For h=2: LEAKAGE (y[99] is within horizon window)
```
