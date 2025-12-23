Gates Module
============

Validation gates for detecting data leakage and suspicious model behavior.

Gates are the first line of defense against data leakage in time-series ML.
Each gate returns a :class:`GateResult` with status PASS, WARN, HALT, or SKIP.

.. seealso::

   :doc:`/api/gates`
      User guide with detailed usage examples.

   :doc:`/glossary`
      Definitions of gate statuses and leakage terms.

Module Contents
---------------

.. automodule:: temporalcv.gates
   :members:
   :undoc-members:
   :show-inheritance:
