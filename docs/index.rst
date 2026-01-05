temporalcv Documentation
========================

**Temporal cross-validation with leakage protection for time-series ML.**

temporalcv provides rigorous validation tools for time-series forecasting, including:

- **Validation gates** for detecting data leakage
- **Walk-forward cross-validation** with gap enforcement
- **Statistical tests** (Diebold-Mariano, Pesaran-Timmermann)
- **High-persistence handling** (MC-SS, move thresholds)
- **Regime classification** (volatility, direction)
- **Conformal prediction** intervals with coverage guarantees
- **Time-series-aware bagging** with bootstrap strategies

Quick Example
-------------

.. code-block:: python

   from temporalcv import run_gates, WalkForwardCV
   from temporalcv.gates import gate_shuffled_target

   # Run leakage detection gates
   # Note: n_shuffles>=100 required for statistical power in permutation mode
   gate_result = gate_shuffled_target(my_model, X, y, n_shuffles=100)
   report = run_gates([gate_result])
   if report.status == "HALT":
       raise ValueError(f"Leakage detected: {report.failures}")

   # Move-conditional metrics for high-persistence series
   from temporalcv import compute_move_threshold, compute_move_conditional_metrics
   threshold = compute_move_threshold(train_actuals)  # From training only!
   mc = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)
   print(f"MC-SS: {mc.skill_score:.3f}")

Installation
------------

.. code-block:: bash

   pip install temporalcv

   # With all optional dependencies
   pip install temporalcv[all]

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Guides

   api/gates
   api/cv
   api/statistical_tests
   api/conformal
   api/persistence
   api/regimes
   api/bagging
   api/diagnostics

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/gates
   api_reference/cv
   api_reference/statistical_tests
   api_reference/conformal
   api_reference/persistence
   api_reference/regimes
   api_reference/bagging
   api_reference/diagnostics
   api_reference/inference
   api_reference/metrics
   api_reference/benchmarks
   api_reference/compare

.. toctree::
   :maxdepth: 1
   :caption: Reference

   glossary
   knowledge/mathematical_foundations
   knowledge/assumptions
   knowledge/notation

.. toctree::
   :maxdepth: 1
   :caption: Help

   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
