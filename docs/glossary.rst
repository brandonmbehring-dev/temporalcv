Glossary
========

This glossary defines key terms used throughout the temporalcv documentation,
including the Knowledge Tier system used to tag confidence levels in claims.

Knowledge Tiers
---------------

.. glossary::

   T1
      **Academically Validated** |---| Claims with full academic citations and
      peer-reviewed support. Apply directly; these are established statistical methods.

      *Examples*:

      - Diebold-Mariano test (Diebold & Mariano, 1995)
      - Harvey small-sample correction (Harvey et al., 1997)
      - HAC variance estimation (Newey & West, 1987)
      - Cross-fitting debiasing (Chernozhukov et al., 2018)

      *Action*: Trust and apply. These methods have strong theoretical foundations.

   T2
      **Empirical Finding** |---| Results validated through prior project work
      but lacking formal publication. May need adjustment for different domains.

      *Examples*:

      - 70th percentile threshold from prior forecasting work
      - 20% improvement suspicion threshold
      - Webb weights for < 13 clusters

      *Action*: Apply but monitor. Verify in your specific context.

   T3
      **Assumption Needing Justification** |---| Domain-specific choices that
      require explicit justification for your data. Document why the assumption
      holds.

      *Examples*:

      - 13-week volatility window
      - AR(1) process assumption for residuals
      - Minimum 30 samples for DM test

      *Action*: Question first. Run sensitivity analysis when possible.

.. |---| unicode:: U+2014
   :trim:

Statistical Terms
-----------------

.. glossary::

   HAC
      **Heteroskedasticity and Autocorrelation Consistent** variance estimation.
      Used in the DM test to account for serial correlation in forecast errors.
      Implementation uses Bartlett kernel with Andrews (1991) automatic bandwidth.
      See :func:`temporalcv.compute_hac_variance`.

   DM test
      **Diebold-Mariano test** for comparing predictive accuracy of two forecasts.
      Tests null hypothesis of equal predictive accuracy.
      See :func:`temporalcv.dm_test`. :term:`T1`

   PT test
      **Pesaran-Timmermann test** for directional forecast accuracy.
      Tests whether directional predictions are better than random.
      See :func:`temporalcv.pt_test`. :term:`T1`

   MC-SS
      **Move-Conditional Skill Score**. Measures forecast skill only during
      periods when the target variable moves beyond a threshold. Essential for
      high-persistence series where persistence baselines inflate apparent skill.
      See :func:`temporalcv.compute_move_conditional_metrics`. :term:`T2`

Cross-Validation Terms
----------------------

.. glossary::

   walk-forward CV
      Time-series cross-validation where training data always precedes test data.
      Can use expanding (all prior data) or sliding (fixed window) approach.
      See :class:`temporalcv.WalkForwardCV`.

   cross-fitting
      Double ML-style K-fold where each observation receives out-of-sample
      predictions. Eliminates regularization bias in metric estimates.
      See :class:`temporalcv.CrossFitCV`. :term:`T1`

   gap
      Number of samples skipped between training end and test start.
      Should be ``>= horizon`` to prevent information leakage in forecasting.

   leakage
      Information from the test period improperly entering the training process.
      Examples: future values in features, full-series normalization,
      threshold computation using test data.

Validation Terms
----------------

.. glossary::

   gate
      A validation check that returns PASS, WARN, HALT, or SKIP status.
      Gates form the first line of defense against data leakage.
      See :mod:`temporalcv.gates`.

   HALT
      Gate status indicating critical failure. Pipeline should stop and
      investigate before proceeding. Typically indicates data leakage.

   WARN
      Gate status indicating potential issue. Pipeline may continue but
      results should be interpreted with caution.

   PASS
      Gate status indicating check passed. No evidence of problems found.

   SKIP
      Gate status indicating check was skipped due to insufficient data
      or inapplicable conditions.

Uncertainty Quantification
--------------------------

.. glossary::

   conformal prediction
      Distribution-free prediction intervals with finite-sample coverage
      guarantees. Uses calibration residuals to quantify uncertainty.
      See :class:`temporalcv.SplitConformalPredictor`. :term:`T1`

   wild bootstrap
      Bootstrap method for clustered data that preserves within-cluster
      correlation structure. Uses random weight multipliers.
      See :func:`temporalcv.wild_cluster_bootstrap`. :term:`T1`

   coverage
      Proportion of test observations falling within prediction intervals.
      Target is typically 1 - alpha (e.g., 95% for alpha=0.05).

Regime Terms
------------

.. glossary::

   volatility regime
      Classification of periods as HIGH or LOW volatility based on
      rolling standard deviation relative to median. :term:`T3`
      See :func:`temporalcv.classify_volatility_regime`.

   direction regime
      Classification of periods as UP, DOWN, or FLAT based on
      returns exceeding threshold. :term:`T3`
      See :func:`temporalcv.classify_direction_regime`.
