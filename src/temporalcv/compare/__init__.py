"""
Compare Module - Cross-Package Model Comparison.

Provides unified interface for comparing forecasting models across
different packages (statsforecast, sktime, etc.).

Example
-------
>>> from temporalcv.compare import run_comparison, NaiveAdapter
>>> from temporalcv.benchmarks import create_synthetic_dataset
>>>
>>> dataset = create_synthetic_dataset()
>>> result = run_comparison(dataset, [NaiveAdapter()])
>>> print(f"Best model: {result.best_model}")
>>> print(f"MAE: {result.models[0].get_metric('mae'):.4f}")

.. rubric:: Available Classes

ModelResult
    Result from a single model run
ComparisonResult
    Result from comparing models on a single dataset
ComparisonReport
    Report from comparing models across multiple datasets
ForecastAdapter
    Abstract base class for model adapters
NaiveAdapter
    Naive persistence baseline
SeasonalNaiveAdapter
    Seasonal naive baseline

.. rubric:: Available Functions

run_comparison
    Compare models on a single dataset
run_benchmark_suite
    Compare models across multiple datasets
compare_to_baseline
    Compare a model to a baseline
compute_comparison_metrics
    Compute MAE, RMSE, MAPE, direction accuracy

.. rubric:: Optional Dependencies

statsforecast : For StatsforecastAdapter
    pip install temporalcv[compare]
"""

from __future__ import annotations

# Core classes (always available)
from temporalcv.compare.base import (
    ComparisonReport,
    ComparisonResult,
    ForecastAdapter,
    ModelResult,
    NaiveAdapter,
    SeasonalNaiveAdapter,
    compute_comparison_metrics,
)

# Runner functions (always available)
from temporalcv.compare.runner import (
    compare_to_baseline,
    run_benchmark_suite,
    run_comparison,
)

# Result serialization (always available)
from temporalcv.compare.results import (
    create_run_metadata,
    load_benchmark_results,
    load_checkpoint,
    save_benchmark_results,
    save_checkpoint,
)

# Documentation generation (always available)
from temporalcv.compare.docs import (
    generate_benchmark_docs,
    generate_ranking_table,
    generate_summary_table,
)

__all__ = [
    # Dataclasses
    "ModelResult",
    "ComparisonResult",
    "ComparisonReport",
    # Adapters
    "ForecastAdapter",
    "NaiveAdapter",
    "SeasonalNaiveAdapter",
    # Functions
    "run_comparison",
    "run_benchmark_suite",
    "compare_to_baseline",
    "compute_comparison_metrics",
    # Result serialization
    "create_run_metadata",
    "save_benchmark_results",
    "load_benchmark_results",
    "save_checkpoint",
    "load_checkpoint",
    # Documentation generation
    "generate_benchmark_docs",
    "generate_summary_table",
    "generate_ranking_table",
]

# Conditional imports for optional adapters
try:
    from temporalcv.compare.adapters import StatsforecastAdapter

    __all__.append("StatsforecastAdapter")
except ImportError:
    pass
