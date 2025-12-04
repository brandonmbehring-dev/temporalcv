"""
temporalcv: Temporal cross-validation with leakage protection for time-series ML.

This package provides rigorous validation tools for time-series forecasting,
including:

- Validation gates for detecting data leakage
- Walk-forward cross-validation with gap enforcement
- Statistical tests (Diebold-Mariano, Pesaran-Timmermann)
- High-persistence series handling (MC-SS, move thresholds)

Example
-------
>>> from temporalcv import run_gates, WalkForwardCV
>>> from temporalcv.gates import gate_shuffled_target
>>>
>>> report = run_gates(
...     model=my_model,
...     X=X, y=y,
...     gates=[gate_shuffled_target(n_shuffles=5)]
... )
>>> if report.status == "HALT":
...     raise ValueError(f"Leakage detected: {report.failures}")
"""

from __future__ import annotations

__version__ = "0.1.0"

# Gates module exports
from temporalcv.gates import (
    GateStatus,
    GateResult,
    ValidationReport,
    gate_shuffled_target,
    gate_synthetic_ar1,
    gate_suspicious_improvement,
    gate_temporal_boundary,
    run_gates,
)

# Statistical tests exports
from temporalcv.statistical_tests import (
    DMTestResult,
    PTTestResult,
    dm_test,
    pt_test,
    compute_hac_variance,
)

__all__ = [
    "__version__",
    # Gates
    "GateStatus",
    "GateResult",
    "ValidationReport",
    "gate_shuffled_target",
    "gate_synthetic_ar1",
    "gate_suspicious_improvement",
    "gate_temporal_boundary",
    "run_gates",
    # Statistical tests
    "DMTestResult",
    "PTTestResult",
    "dm_test",
    "pt_test",
    "compute_hac_variance",
]
