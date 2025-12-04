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
__all__ = [
    "__version__",
    # Core exports will be added as modules are implemented:
    # "run_gates",
    # "ValidationReport",
    # "WalkForwardCV",
]
