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
   from temporalcv.gates import gate_signal_verification

   # Run signal verification gate
   # Note: n_shuffles>=100 required for statistical power in permutation mode
   gate_result = gate_signal_verification(my_model, X, y, n_shuffles=100)
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
   :caption: Getting Started

   quickstart
   guide/why_time_series_is_different
   guide/common_pitfalls
   guide/algorithm_decision_tree

.. toctree::
   :maxdepth: 2
   :caption: Examples Gallery

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Guides

   api/gates
   api/guardrails
   api/cv
   api/cv_financial
   api/statistical_tests
   api/stationarity
   api/lag_selection
   api/conformal
   api/persistence
   api/regimes
   api/bagging
   api/diagnostics
   api/changepoint
   api/inference
   api/metrics
   api/benchmarks
   api/compare

.. toctree:
   :maxdepth: 1
   :caption: Reference

   glossary
   knowledge/mathematical_foundations
   knowledge/assumptions
   knowledge/notation
   testing_strategy

.. toctree::
   :maxdepth: 1
   :caption: Model Cards

   model_cards/README
   model_cards/walk_forward_cv
   model_cards/gate_shuffled_target

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks
   benchmarks/methodology
   benchmarks/reproduce

.. toctree::
   :maxdepth: 1
   :caption: Help

   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
