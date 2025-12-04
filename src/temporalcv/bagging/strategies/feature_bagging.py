"""
Feature Bagging Strategy (Random Subspace Method).

Implements the Random Subspace Method from Ho (1998).
Bootstraps features instead of observations, preserving temporal structure.

Example
-------
>>> from temporalcv.bagging import FeatureBagging
>>>
>>> strategy = FeatureBagging(max_features=0.7)
>>> samples = strategy.generate_samples(X, y, n_samples=20, rng=rng)

References
----------
- Ho, T. K. (1998). "The Random Subspace Method for Constructing
  Decision Forests." IEEE TPAMI, 20(8), 832-844.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from temporalcv.bagging.base import BootstrapStrategy


class FeatureBagging(BootstrapStrategy):
    """
    Feature Bagging / Random Subspace Method (Ho, 1998).

    Bootstraps features instead of observations. Uses ALL temporal
    observations, avoiding temporal violation. Reduces multicollinearity
    effects by using different feature subsets per estimator.

    CRITICAL: Stores feature indices for use in transform_for_predict().
    Each estimator sees only its subset during both fit AND predict.

    Parameters
    ----------
    max_features : float, default=0.7
        Fraction of features to use per estimator (0.0 < max_features <= 1.0).
        Default: 0.7 (70% of features)

    Attributes
    ----------
    max_features : float
        Configured feature fraction
    feature_indices_ : list[np.ndarray]
        Feature indices used by each estimator (populated after generate_samples)

    Notes
    -----
    - Unlike block bootstrap, this preserves ALL temporal observations
    - Feature subsets are stored for use during prediction
    - transform_for_predict() is CRITICAL: ensures estimators see same
      features during predict as during fit

    Examples
    --------
    >>> strategy = FeatureBagging(max_features=0.7)
    >>> # 70% of features per estimator

    References
    ----------
    Ho (1998), IEEE TPAMI: Random Subspace Method
    """

    def __init__(self, max_features: float = 0.7):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(f"max_features must be in (0.0, 1.0], got {max_features}")
        self.max_features = max_features
        self.feature_indices_: List[np.ndarray] = []

    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap samples using Feature Bagging.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_obs, n_features)
        y : np.ndarray
            Target series (n_obs,)
        n_samples : int
            Number of bootstrap samples
        rng : np.random.Generator
            Random number generator

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            Bootstrap samples (each with different feature subset)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D array")

        n_features = X.shape[1]
        n_select = max(1, int(self.max_features * n_features))

        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.feature_indices_ = []

        for _ in range(n_samples):
            # Select random feature subset (without replacement)
            feature_idx = np.sort(rng.choice(n_features, size=n_select, replace=False))
            self.feature_indices_.append(feature_idx)

            # Use numpy column indexing
            X_boot = X[:, feature_idx]
            # y unchanged - all observations preserved
            samples.append((X_boot, y.copy()))

        return samples

    def transform_for_predict(
        self,
        X: np.ndarray,
        estimator_idx: int,
    ) -> np.ndarray:
        """
        CRITICAL: Slice X to the feature subset used by this estimator.

        Without this, predict() would pass full X to estimators trained
        on subsets, causing dimension mismatch errors.

        Parameters
        ----------
        X : np.ndarray
            Full feature matrix
        estimator_idx : int
            Index of the estimator

        Returns
        -------
        np.ndarray
            Feature matrix with only the columns this estimator was trained on

        Raises
        ------
        RuntimeError
            If called before generate_samples()
        """
        if not self.feature_indices_:
            raise RuntimeError(
                "FeatureBagging must generate samples before "
                "transform_for_predict. Call generate_samples() first."
            )
        X = np.asarray(X)
        feature_idx = self.feature_indices_[estimator_idx]
        return X[:, feature_idx]

    def __repr__(self) -> str:
        return f"FeatureBagging(max_features={self.max_features})"


# =============================================================================
# Public API
# =============================================================================

__all__ = ["FeatureBagging"]
