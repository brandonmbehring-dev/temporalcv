# API Reference: Time Series Bagging

Model-agnostic bagging with time-series-aware bootstrap strategies.

---

## Classes

### `TimeSeriesBagger`

Generic time series bagger.

```python
class TimeSeriesBagger:
    def __init__(
        self,
        base_model: SupportsPredict,
        strategy: BootstrapStrategy,
        n_estimators: int = 20,
        aggregation: Literal["mean", "median"] = "mean",
        random_state: int = 42,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | `SupportsPredict` | required | Model to bag |
| `strategy` | `BootstrapStrategy` | required | Bootstrap strategy |
| `n_estimators` | `int` | `20` | Number of bootstrap estimators |
| `aggregation` | `str` | `"mean"` | How to combine predictions |
| `random_state` | `int` | `42` | Random seed |

#### Methods

##### `fit(X, y)`

Fit all estimators on bootstrap samples.

##### `predict(X)`

Return aggregated predictions.

##### `predict_with_uncertainty(X)`

Return predictions with uncertainty estimates.

```python
def predict_with_uncertainty(
    self,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]
```

**Returns**: `(mean_prediction, std_prediction)`

##### `predict_interval(X, alpha=0.10)`

Return prediction intervals.

```python
def predict_interval(
    self,
    X: np.ndarray,
    alpha: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Returns**: `(mean, lower, upper)`

---

## Bootstrap Strategies

### `MovingBlockBootstrap`

Block bootstrap preserving local autocorrelation (Kunsch 1989).

```python
class MovingBlockBootstrap(BootstrapStrategy):
    def __init__(
        self,
        block_length: Optional[int] = None,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_length` | `int` | `None` | Block length. If None, auto-compute as n^(1/3) |

---

### `StationaryBootstrap`

Geometric block lengths for stationarity (Politis & Romano 1994).

```python
class StationaryBootstrap(BootstrapStrategy):
    def __init__(
        self,
        expected_block_length: Optional[float] = None,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_block_length` | `float` | `None` | Expected block length. If None, auto-compute. |

---

### `FeatureBagging`

Random subspace method (Ho 1998).

```python
class FeatureBagging(BootstrapStrategy):
    def __init__(
        self,
        max_features: float = 0.7,
    )
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_features` | `float` | `0.7` | Fraction of features per estimator |

---

## Factory Functions

### `create_block_bagger`

Create bagger with Moving Block Bootstrap.

```python
def create_block_bagger(
    base_model: SupportsPredict,
    n_estimators: int = 20,
    block_length: Optional[int] = None,
    aggregation: str = "mean",
    random_state: int = 42,
) -> TimeSeriesBagger
```

**Example**:

```python
from sklearn.linear_model import Ridge
from temporalcv.bagging import create_block_bagger

bagger = create_block_bagger(
    Ridge(alpha=1.0),
    n_estimators=50,
    block_length=10,
    random_state=42
)

bagger.fit(X_train, y_train)
mean, lower, upper = bagger.predict_interval(X_test, alpha=0.10)
```

---

### `create_stationary_bagger`

Create bagger with Stationary Bootstrap.

```python
def create_stationary_bagger(
    base_model: SupportsPredict,
    n_estimators: int = 20,
    expected_block_length: Optional[float] = None,
    aggregation: str = "mean",
    random_state: int = 42,
) -> TimeSeriesBagger
```

---

### `create_feature_bagger`

Create bagger with Feature Bagging (Random Subspace).

```python
def create_feature_bagger(
    base_model: SupportsPredict,
    n_estimators: int = 20,
    max_features: float = 0.7,
    aggregation: str = "mean",
    random_state: int = 42,
) -> TimeSeriesBagger
```

---

## Strategy Selection

| Strategy | Use When |
|----------|----------|
| `MovingBlockBootstrap` | Standard time series with known autocorrelation |
| `StationaryBootstrap` | Varying block lengths needed, uncertain structure |
| `FeatureBagging` | High-dimensional features, reduce overfitting |

---

## Complete Example

```python
import numpy as np
from sklearn.linear_model import Ridge
from temporalcv.bagging import create_block_bagger

# Generate data
np.random.seed(42)
n = 300
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.9 * y[t-1] + np.random.randn() * 0.1

X = np.column_stack([np.roll(y, i) for i in range(1, 6)])[5:]
y = y[5:]

# Split
train_end = 200

# Create and fit bagger
bagger = create_block_bagger(
    Ridge(alpha=1.0),
    n_estimators=50,
    block_length=15,
    random_state=42
)
bagger.fit(X[:train_end], y[:train_end])

# Get predictions with uncertainty
mean, std = bagger.predict_with_uncertainty(X[train_end:])
print(f"Mean std: {std.mean():.4f}")

# Get intervals
mean, lower, upper = bagger.predict_interval(X[train_end:], alpha=0.10)
coverage = np.mean((y[train_end:] >= lower) & (y[train_end:] <= upper))
print(f"Coverage: {coverage:.1%}")
print(f"Mean width: {np.mean(upper - lower):.4f}")
```
