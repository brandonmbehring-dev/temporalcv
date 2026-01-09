# Phase 1: Core Validation Gates + Statistical Tests

**Timeline**: Weeks 1-4
**Status**: ✓ COMPLETE (2025-12-04)

---

## Goal

Standalone leakage detection + statistical testing with minimal dependencies.

## Modules to Create

| Module | File | Reference |
|--------|------|-----------|
| `temporalcv.gates` | `src/temporalcv/gates.py` | myga: `validation/gates.py` |
| `temporalcv.statistical_tests` | `src/temporalcv/statistical_tests.py` | myga: `evaluation/statistical_tests.py` |
| `temporalcv.metrics` | `src/temporalcv/metrics.py` | myga: `evaluation/metrics.py` |

## Dependencies

- numpy (required)
- scipy (required)
- pandas (optional)
- scikit-learn (optional, API compat)

## Gate Framework Design

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class GateStatus(Enum):
    HALT = "HALT"      # Critical failure, stop pipeline
    WARN = "WARN"      # Warning, continue with caution
    PASS = "PASS"      # Validation passed
    SKIP = "SKIP"      # Insufficient data to run

@dataclass
class GateResult:
    name: str
    status: GateStatus
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

@dataclass
class ValidationReport:
    gates: List[GateResult]

    @property
    def status(self) -> str:
        if any(g.status == GateStatus.HALT for g in self.gates):
            return "HALT"
        if any(g.status == GateStatus.WARN for g in self.gates):
            return "WARN"
        return "PASS"

    @property
    def failures(self) -> List[GateResult]:
        return [g for g in self.gates if g.status == GateStatus.HALT]
```

## Gates to Implement

### 1. Shuffled Target Gate (Priority: Critical)

```python
def gate_signal_verification(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_shuffles: int = 5,
    random_state: int | None = None
) -> GateResult:
    """
    Definitive leakage detection.

    If model performs better than persistence on shuffled target,
    features contain information about target ordering (leakage).
    """
```

### 2. Synthetic AR(1) Gate

```python
def gate_synthetic_ar1(
    model,
    phi: float = 0.95,
    n_samples: int = 500,
    random_state: int | None = None
) -> GateResult:
    """
    Theoretical bound verification.

    Test model on synthetic AR(1) process where optimal
    forecast is phi * y_{t-1}. MAE should match theory.
    """
```

### 3. Suspicious Improvement Gate

```python
def gate_suspicious_improvement(
    model_mae: float,
    persistence_mae: float,
    threshold: float = 0.20
) -> GateResult:
    """
    Heuristic leakage detection.

    >20% improvement over persistence baseline
    triggers investigation.
    """
```

## Statistical Tests to Implement

### 1. Diebold-Mariano Test

```python
def dm_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    h: int = 1,
    loss: str = "squared",
    harvey_correction: bool = True
) -> DMTestResult:
    """
    Compare forecast accuracy with HAC variance estimation.

    Parameters
    ----------
    errors_1, errors_2 : Forecast errors from two models
    h : Forecast horizon (for HAC bandwidth)
    loss : "squared" or "absolute"
    harvey_correction : Apply small-sample correction

    Returns
    -------
    DMTestResult with statistic, pvalue, conclusion
    """
```

### 2. Pesaran-Timmermann Test

```python
def pt_test(
    actual: np.ndarray,
    predicted: np.ndarray,
    move_threshold: float = 0.0
) -> PTTestResult:
    """
    Direction accuracy test (3-class: UP/DOWN/FLAT).

    Parameters
    ----------
    actual : Actual values
    predicted : Predicted values
    move_threshold : Minimum move to count as UP/DOWN
    """
```

## Exit Criteria

- [x] `gate_signal_verification()` implemented with tests
- [x] `gate_synthetic_ar1()` implemented with tests
- [x] `gate_suspicious_improvement()` implemented with tests
- [x] `gate_temporal_boundary()` implemented with tests (bonus)
- [x] `dm_test()` with HAC variance, Harvey correction
- [x] `pt_test()` with 3-class support
- [x] `run_gates()` orchestrator function
- [x] All tests pass (64 tests), mypy strict passes
- [x] 97% coverage (exceeds 80% target)
