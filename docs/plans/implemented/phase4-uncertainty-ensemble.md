# Phase 4: Uncertainty + Ensemble

**Created**: 2025-12-04
**Status**: In Progress

---

## Overview

Implement conformal prediction intervals and time-series-aware bagging for uncertainty quantification.

**Package**: `temporalcv` @ `~/Claude/temporalcv/`
**Prior**: Phase 3 ✓ (162 tests, 95% coverage)

---

## Phase 4 Goals

1. **Conformal prediction** - Distribution-free prediction intervals
2. **Bagging framework** - Model-agnostic bootstrap aggregation for time series

---

## Module 1: conformal.py

### Classes & Functions

```python
@dataclass
class PredictionInterval:
    """Container for prediction intervals."""
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    method: str

    @property
    def width(self) -> np.ndarray
    @property
    def mean_width(self) -> float
    def coverage(self, actuals: np.ndarray) -> float

class SplitConformalPredictor:
    """
    Split Conformal Prediction (Romano et al. 2019).

    Distribution-free coverage guarantee.
    """
    def __init__(self, alpha: float = 0.05)
    def calibrate(self, predictions, actuals) -> self
    def predict_interval(self, predictions) -> PredictionInterval

class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Inference (Gibbs & Candes 2021).

    For non-exchangeable time series data.
    """
    def __init__(self, alpha: float = 0.05, gamma: float = 0.1)
    def initialize(self, predictions, actuals) -> self
    def update(self, prediction: float, actual: float) -> float
    def predict_interval(self, prediction: float) -> tuple[float, float]

def evaluate_interval_quality(intervals, actuals) -> dict:
    """Evaluate coverage, width, interval score."""

def walk_forward_conformal(predictions, actuals, ...) -> tuple[PredictionInterval, dict]:
    """Apply conformal to walk-forward results."""
```

### Key Properties

| Property | Requirement |
|----------|-------------|
| Coverage | ≥ 1 - α (finite sample guarantee) |
| Width | Minimal while maintaining coverage |
| Calibration | MUST be on separate data from test |

### References

- Romano, Sesia, Candes (2019). "Conformalized Quantile Regression"
- Gibbs, Candes (2021). "Adaptive Conformal Inference Under Distribution Shift"

---

## Module 2: bagging/

### Structure

```
src/temporalcv/bagging/
├── __init__.py          # Exports
├── base.py              # BootstrapStrategy ABC, TimeSeriesBagger
└── strategies/
    ├── __init__.py
    ├── block_bootstrap.py    # MovingBlockBootstrap
    ├── stationary_bootstrap.py  # StationaryBootstrap
    └── feature_bagging.py    # FeatureBagging (random subspace)
```

### Core Classes

```python
class BootstrapStrategy(ABC):
    """Abstract base for bootstrap resampling."""

    @abstractmethod
    def generate_samples(
        self, X: np.ndarray, y: np.ndarray, n_samples: int, rng
    ) -> list[tuple[np.ndarray, np.ndarray]]

    def transform_for_predict(
        self, X: np.ndarray, estimator_idx: int
    ) -> np.ndarray

class TimeSeriesBagger:
    """
    Generic bagging wrapper.

    Works with any model implementing fit(X, y) / predict(X).
    """
    def __init__(
        self,
        base_model,
        strategy: BootstrapStrategy,
        n_estimators: int = 20,
        aggregation: str = "mean",
        random_state: int = 42
    )
    def fit(self, X, y) -> self
    def predict(self, X) -> np.ndarray
    def predict_with_uncertainty(self, X) -> tuple[np.ndarray, np.ndarray]
    def predict_interval(self, X, alpha: float = 0.05) -> tuple
```

### Strategies

| Strategy | Block Length | Use Case |
|----------|--------------|----------|
| `MovingBlockBootstrap` | Fixed | Strong local autocorrelation |
| `StationaryBootstrap` | Geometric | Robust to block choice |
| `FeatureBagging` | N/A (features) | High-dimensional features |

### Factory Functions

```python
def create_block_bagger(base_model, n_estimators=20, block_length=None)
def create_stationary_bagger(base_model, n_estimators=20, expected_block_length=None)
def create_feature_bagger(base_model, n_estimators=20, max_features=0.7)
```

### References

- Kunsch (1989). "The Jackknife and Bootstrap for General Stationary"
- Politis & Romano (1994). "The Stationary Bootstrap"
- Ho (1998). "The Random Subspace Method"
- Bergmeir, Hyndman & Benitez (2016). "Bagging Exponential Smoothing"

---

## Test Plan

### test_conformal.py (~40 tests)

| Category | Tests |
|----------|-------|
| `TestPredictionInterval` | Creation, width, coverage |
| `TestSplitConformal` | Calibration, intervals, coverage |
| `TestAdaptiveConformal` | Init, update, adjustment |
| `TestIntervalQuality` | Metrics computation |
| `TestWalkForwardConformal` | Integration, holdout-only |
| `TestCoverageGuarantees` | Finite-sample validity |

### test_bagging.py (~35 tests)

| Category | Tests |
|----------|-------|
| `TestBootstrapStrategy` | ABC interface |
| `TestMovingBlockBootstrap` | Block sampling, length |
| `TestStationaryBootstrap` | Geometric blocks |
| `TestFeatureBagging` | Feature subsets, transform |
| `TestTimeSeriesBagger` | Fit, predict, intervals |
| `TestFactoryFunctions` | Convenience creators |

---

## Design Decisions

### 1. No pandas requirement for bagging

Bagging works on numpy arrays directly. Optional pandas support.

### 2. No joblib parallel by default

Keep dependencies minimal. Users can wrap with joblib if needed.

### 3. Protocol-based model interface

```python
class SupportsPredict(Protocol):
    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...
```

### 4. Skip ResidualBootstrap

Requires STL (statsmodels dependency). Out of scope for Phase 4.

---

## Dependencies

No new required dependencies. Optional:
- `joblib` for parallel fitting (user-provided)
- `pandas` for DataFrame support

---

## Exit Criteria

- [ ] `SplitConformalPredictor` provides valid coverage guarantee
- [ ] `AdaptiveConformalPredictor` adjusts to distribution shift
- [ ] `TimeSeriesBagger` works with any fit/predict model
- [ ] All three bootstrap strategies implemented
- [ ] Factory functions for easy usage
- [ ] Coverage ≥ 90% on new modules
- [ ] All tests pass, mypy strict passes
- [ ] ~220 total tests (162 + ~58 new)

---

## Implementation Order

1. `conformal.py` - PredictionInterval dataclass
2. `conformal.py` - SplitConformalPredictor
3. `conformal.py` - AdaptiveConformalPredictor
4. `conformal.py` - evaluate_interval_quality, walk_forward_conformal
5. `test_conformal.py` - conformal tests
6. `bagging/base.py` - BootstrapStrategy ABC
7. `bagging/base.py` - TimeSeriesBagger
8. `bagging/strategies/` - Three strategies
9. `bagging/__init__.py` - Factory functions
10. `test_bagging.py` - bagging tests
11. Update `__init__.py` exports
12. Final verification
