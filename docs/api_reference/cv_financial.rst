Financial Cross-Validation Module
===================================

Cross-validation with purging and embargo for financial data.

Implements CV techniques for financial ML where labels often overlap
(e.g., 5-day forward returns share 4 days of data). Standard CV leaks
information through this overlap.

.. seealso::

   :doc:`/api/cv_financial`
      User guide with detailed usage examples.

   :doc:`/glossary`
      Definitions of purging, embargo, and label overlap.

Module Contents
---------------

.. automodule:: temporalcv.cv_financial
   :members:
   :undoc-members:
   :show-inheritance:
