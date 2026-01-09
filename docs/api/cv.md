# API Reference: Walk-Forward Cross-Validation

sklearn-compatible temporal cross-validation with gap enforcement for h-step forecasting.

---

## When to Use

```{mermaid}
graph TD
    A[Time Series CV?] --> B{Data characteristics}

    B -->|Recent data more relevant| C[sliding window]
    B -->|More data always helps| D[expanding window]
    B -->|Financial with overlap| E[PurgedKFold]

    C --> F{Multi-step forecast?}
    D --> F
    F -->|Yes, h > 1| G[Set horizon=h]
    F -->|No, h=1| H[horizon=1 or None]

    G --> I{Extra safety?}
    I -->|Yes| J[extra_gap > 0]
    I -->|No| K[extra_gap=0]
```

### Common Mistakes

- **No gap for h-step forecasting**
  - For h=5 forecast: `WalkForwardCV(horizon=5)` required
  - Without gap, target leaks into training features

- **Using KFold on time series**
  - Random splits destroy temporal order → 47%+ fake improvement
  - Always use `WalkForwardCV` or `TimeSeriesSplit`

- **Sliding window too small**
  - Window must be larger than model's memory
  - Rule of thumb: `window_size >= 5 * n_features`

**See Also**: [Walk-Forward Tutorial](../tutorials/walk_forward_cv.md), [Example 07](../tutorials/examples_index.md#07-nested-cv-tuning), [Example 20](../tutorials/examples_index.md#20-kfold-trap-failure)

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
        horizon: Optional[int] = None,
        extra_gap: int = 0,
        test_size: int = 1,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_splits` | `int` | `5` | Number of CV folds |
| `window_type` | `str` | `"expanding"` | `"expanding"` or `"sliding"` |
| `window_size` | `int` | `None` | Window size (required for sliding) |
| `horizon` | `int` | `None` | Forecast horizon (minimum required separation for h-step forecasts) |
| `extra_gap` | `int` | `0` | Additional separation beyond horizon (total = horizon + extra_gap) |
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
cv = WalkForwardCV(n_splits=5, window_type="sliding", window_size=100, horizon=2, extra_gap=0)

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
cv = WalkForwardCV(n_splits=3, horizon=2, extra_gap=0)

for info in cv.get_split_info(X):
    print(f"Split {info.split_idx}:")
    print(f"  Train: [{info.train_start}, {info.train_end}]")
    print(f"  Test:  [{info.test_start}, {info.test_end}]")
    print(f"  Gap: {info.gap}")  # Total gap = horizon + extra_gap
```

---

## sklearn Integration

Works with `cross_val_score` and `GridSearchCV`:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from temporalcv import WalkForwardCV

cv = WalkForwardCV(n_splits=5, window_type="expanding", horizon=2, extra_gap=0)

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

For h-step ahead forecasts, set `horizon=h`:

```python
# For 2-step forecasts: horizon=2
cv = WalkForwardCV(
    n_splits=5,
    window_type="sliding",
    window_size=100,
    horizon=2,      # Minimum required separation for 2-step forecasts
    extra_gap=0,    # Optional: additional safety margin (default: 0)
    test_size=1
)

for train_idx, test_idx in cv.split(X):
    # Guaranteed: train_idx[-1] + total_gap < test_idx[0]
    # where total_gap = horizon + extra_gap
    total_gap = (cv.horizon or 0) + cv.extra_gap
    assert train_idx[-1] + total_gap < test_idx[0]
```

**Gap rule** [T1]: For h-step ahead forecasting, `total_separation = horizon + extra_gap` must be at least `h`.

Per Bergmeir & Benitez (2012): temporal separation must equal or exceed forecast horizon.

**Why gap matters**:

```
h=2 forecast: y[t+2] = f(y[t], y[t-1], ...)

Without gap (horizon=None, extra_gap=0):
  - Train ends at t=99, test starts at t=100 (gap=0)
  - Test prediction y[100] uses features from y[99], y[98], ...
  - For h=1: OK (predicting one step ahead)
  - For h=2: LEAKAGE (y[100] target overlaps training features)

With horizon=2, extra_gap=0:
  - Train ends at t=99, test starts at t=102 (total_gap=2)
  - Test prediction y[102] uses features from y[101], y[100], ...
  - No overlap with training targets (safe for h=2 forecasts)
```

---

## Nested Cross-Validation

### `NestedWalkForwardCV`

Nested walk-forward CV for hyperparameter tuning with temporal integrity.

```python
class NestedWalkForwardCV:
    def __init__(
        self,
        estimator,
        param_grid: Dict[str, List] = None,           # For grid search
        param_distributions: Dict[str, Any] = None,   # For random search
        *,
        n_iter: int = None,           # Required for random search
        n_outer_splits: int = 3,
        n_inner_splits: int = 5,
        horizon: int = 1,
        extra_gap: int = None,        # Defaults to 0
        window_type: str = "expanding",
        scoring: str = "neg_mean_squared_error",
        refit: bool = True,
        random_state: int = None,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `estimator` | sklearn estimator | - | Model with fit/predict methods |
| `param_grid` | `Dict[str, List]` | `None` | Parameters for grid search |
| `param_distributions` | `Dict[str, Any]` | `None` | Distributions for random search |
| `n_iter` | `int` | `None` | Iterations for random search |
| `n_outer_splits` | `int` | `3` | Outer folds (performance estimation) |
| `n_inner_splits` | `int` | `5` | Inner folds (hyperparameter selection) |
| `horizon` | `int` | `1` | Forecast horizon (minimum required separation) |
| `extra_gap` | `int` | `0` | Additional separation beyond horizon (total = horizon + extra_gap) |
| `refit` | `bool` | `True` | Refit on all data with best params |

**Attributes** (after `fit()`):

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_params_` | `Dict` | Best hyperparameters found |
| `best_estimator_` | estimator | Model refitted with best_params_ |
| `outer_scores_` | `np.ndarray` | Unbiased scores per outer fold |
| `mean_outer_score_` | `float` | Mean of outer scores |
| `std_outer_score_` | `float` | Std of outer scores |
| `params_stability_` | `float` | Fraction agreeing with best_params_ |

---

### Example: Grid Search

```python
from temporalcv import NestedWalkForwardCV
from sklearn.linear_model import Ridge

nested_cv = NestedWalkForwardCV(
    estimator=Ridge(),
    param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]},
    n_outer_splits=3,
    n_inner_splits=5,
    horizon=4,
    scoring="neg_mean_squared_error",
)

nested_cv.fit(X, y)

print(f"Best alpha: {nested_cv.best_params_['alpha']}")
print(f"Score: {nested_cv.mean_outer_score_:.4f} ± {nested_cv.std_outer_score_:.4f}")
print(f"Stability: {nested_cv.params_stability_:.1%}")

# Predict with refitted model
predictions = nested_cv.predict(X_new)
```

---

### Example: Randomized Search

```python
from scipy.stats import loguniform

nested_cv = NestedWalkForwardCV(
    estimator=Ridge(),
    param_distributions={"alpha": loguniform(1e-3, 1e2)},
    n_iter=20,
    n_outer_splits=3,
    random_state=42,
)

nested_cv.fit(X, y)
```

---

### When to Use Nested CV vs Single CV

| Use Case | Recommendation |
|----------|----------------|
| Fixed hyperparameters | Single `WalkForwardCV` |
| Few hyperparameters, low impact | Single CV with default params |
| Many hyperparameters, high impact | Nested CV (avoids optimistic bias) |
| Publication/rigorous validation | Nested CV (unbiased estimates) |

---

### `NestedCVResult`

Structured result from nested CV.

```python
@dataclass
class NestedCVResult:
    best_params: Dict[str, Any]        # Optimal parameters
    outer_scores: np.ndarray           # Per-fold scores
    mean_outer_score: float            # Mean score
    std_outer_score: float             # Std of scores
    n_outer_splits: int
    n_inner_splits: int
    scoring: str
    best_params_per_fold: List[Dict]   # Per-fold selections
    params_stability: float            # Consistency metric
```

Access via `nested_cv.get_result()`.

---

## References

**[T1] Time Series Cross-Validation**:

- Bergmeir, C. & Benítez, J.M. (2012). "On the use of cross-validation for time series predictor evaluation." *Information Sciences*, 191, 192-213. [DOI: 10.1016/j.ins.2011.12.028](https://doi.org/10.1016/j.ins.2011.12.028)

- Tashman, L.J. (2000). "Out-of-sample tests of forecasting accuracy: An analysis and review." *International Journal of Forecasting*, 16(4), 437-450. [DOI: 10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)

**[T1] Nested Cross-Validation**:

- Varma, S. & Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection." *BMC Bioinformatics*, 7(1), 91. [DOI: 10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91)
