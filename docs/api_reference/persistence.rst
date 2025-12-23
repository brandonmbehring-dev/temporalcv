Persistence Metrics Module
==========================

Move-conditional metrics for high-persistence time series.

Standard metrics like MAE are misleading for high-persistence series where
simple baselines (persistence, drift) perform well. These metrics focus
evaluation on periods when the target actually moves.

.. seealso::

   :doc:`/api/persistence`
      User guide with detailed usage examples.

   :term:`MC-SS`
      Glossary entry for Move-Conditional Skill Score.

Module Contents
---------------

.. automodule:: temporalcv.persistence
   :members:
   :undoc-members:
   :show-inheritance:
