"""
Time Series Bagging Framework - Base Classes.

Generic, model-agnostic bagging framework for time series that wraps ANY
model with fit/predict interface using time-series-aware resampling strategies.

Example
-------
>>> from temporalcv.bagging import TimeSeriesBagger, MovingBlockBootstrap
>>> from sklearn.linear_model import Ridge
>>>
>>> strategy = MovingBlockBootstrap(block_length=10)
>>> bagger = TimeSeriesBagger(Ridge(), strategy, n_estimators=20)
>>> bagger.fit(X_train, y_train)
>>> predictions = bagger.predict(X_test)
>>> mean, std = bagger.predict_with_uncertainty(X_test)

References
----------
- Bergmeir, Hyndman & Benitez (2016). "Bagging Exponential Smoothing Methods"
- Kunsch (1989). "The Jackknife and Bootstrap for General Stationary Observations"
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsPredict(Protocol):
    """Protocol for models with fit/predict interface."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SupportsPredict":
        """Fit model to data."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...


class BootstrapStrategy(ABC):
    """
    Abstract base class for bootstrap resampling strategies.

    All strategies must implement generate_samples() to produce bootstrap
    samples while preserving time series properties.

    The transform_for_predict() method allows strategies to transform X
    during prediction (critical for feature bagging).
    """

    @abstractmethod
    def generate_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate bootstrap samples preserving time series properties.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target series (n_samples,)
        n_samples : int
            Number of bootstrap samples to generate
        rng : np.random.Generator
            Random number generator for reproducibility

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (X_boot, y_boot) bootstrap samples
        """
        pass

    def transform_for_predict(
        self,
        X: np.ndarray,
        estimator_idx: int,
    ) -> np.ndarray:
        """
        Transform X for prediction (default: identity).

        Override this method for strategies like feature bagging where
        each estimator sees a different feature subset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction
        estimator_idx : int
            Index of the estimator (0 to n_estimators-1)

        Returns
        -------
        np.ndarray
            Transformed feature matrix
        """
        return X


def _clone_model(model: SupportsPredict) -> SupportsPredict:
    """
    Clone a model, handling both sklearn and custom model instances.

    Uses sklearn.base.clone for sklearn estimators, deepcopy for custom models.

    Parameters
    ----------
    model : SupportsPredict
        Model to clone

    Returns
    -------
    SupportsPredict
        Fresh, unfitted clone of the model
    """
    # Check if model has sklearn's get_params (sklearn-compatible)
    if hasattr(model, "get_params") and callable(model.get_params):
        try:
            from sklearn.base import clone

            cloned: SupportsPredict = clone(model)
            return cloned
        except (ImportError, Exception):
            # Fallback to deepcopy if clone fails
            pass

    # Use deepcopy for custom models
    cloned_model: SupportsPredict = copy.deepcopy(model)
    return cloned_model


class TimeSeriesBagger:
    """
    Generic bagging wrapper for any model with fit/predict interface.

    Implements Bootstrap AGGregatING with time-series-aware resampling
    strategies.

    Parameters
    ----------
    base_model : SupportsPredict
        Any model implementing fit(X, y) and predict(X)
    strategy : BootstrapStrategy
        Resampling strategy (MBB, Stationary, Feature)
    n_estimators : int, default=20
        Number of bootstrap samples/estimators
    aggregation : {"mean", "median"}, default="mean"
        How to combine predictions
    random_state : int or None, default=None
        For reproducibility. None for non-deterministic behavior.

    Attributes
    ----------
    estimators_ : list[SupportsPredict]
        Fitted estimators (after fit)
    is_fitted : bool
        Whether the bagger has been fitted

    Examples
    --------
    >>> from temporalcv.bagging import TimeSeriesBagger, MovingBlockBootstrap
    >>> from sklearn.linear_model import Ridge
    >>>
    >>> bagger = TimeSeriesBagger(
    ...     Ridge(alpha=1.0),
    ...     MovingBlockBootstrap(block_length=10),
    ...     n_estimators=50
    ... )
    >>> bagger.fit(X_train, y_train)
    >>> predictions = bagger.predict(X_test)

    References
    ----------
    - Bergmeir, Hyndman & Benitez (2016). "Bagging exponential smoothing"
    - Kunsch (1989). "The Jackknife and Bootstrap for General Stationary"

    See Also
    --------
    create_block_bagger : Factory function for block bootstrap.
    create_stationary_bagger : Factory function for stationary bootstrap.
    BootstrapUncertainty : Alternative uncertainty quantification.
    """

    def __init__(
        self,
        base_model: SupportsPredict,
        strategy: BootstrapStrategy,
        n_estimators: int = 20,
        aggregation: Literal["mean", "median"] = "mean",
        random_state: Optional[int] = None,
    ):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if aggregation not in ("mean", "median"):
            raise ValueError(f"aggregation must be 'mean' or 'median', got {aggregation}")

        self.base_model = base_model
        self.strategy = strategy
        self.n_estimators = n_estimators
        self.aggregation = aggregation
        self.random_state = random_state
        self.estimators_: List[SupportsPredict] = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the bagger has been fitted."""
        return self._fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TimeSeriesBagger":
        """
        Fit n_estimators on bootstrap samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Target series (n_samples,)

        Returns
        -------
        TimeSeriesBagger
            Fitted bagger (self)

        Raises
        ------
        ValueError
            If X and y have mismatched lengths or are empty
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("Cannot fit on empty data")

        # Use SeedSequence for reproducible per-estimator seeds
        ss = np.random.SeedSequence(self.random_state)
        child_seeds = ss.spawn(self.n_estimators + 1)

        # First child seed for bootstrap sample generation
        rng = np.random.default_rng(child_seeds[0])
        samples = self.strategy.generate_samples(X, y, self.n_estimators, rng)

        # Fit estimators
        self.estimators_ = []
        for i, (X_boot, y_boot) in enumerate(samples):
            estimator = _clone_model(self.base_model)
            estimator.fit(X_boot, y_boot)
            self.estimators_.append(estimator)

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        """Raise RuntimeError if not fitted."""
        if not self._fitted:
            raise RuntimeError(
                "TimeSeriesBagger must be fitted before predict. Call fit() first."
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Aggregate predictions across all estimators.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Aggregated predictions
        """
        self._check_fitted()
        X = np.asarray(X)

        # Use strategy's transform_for_predict for each estimator
        # (critical for feature bagging!)
        predictions = np.array(
            [
                est.predict(self.strategy.transform_for_predict(X, i))
                for i, est in enumerate(self.estimators_)
            ]
        )

        if self.aggregation == "mean":
            result: np.ndarray = np.mean(predictions, axis=0)
            return result
        result_median: np.ndarray = np.median(predictions, axis=0)
        return result_median

    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all estimators.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        np.ndarray
            Shape (n_estimators, n_samples) array of predictions
        """
        self._check_fitted()
        X = np.asarray(X)
        all_preds: np.ndarray = np.array(
            [
                est.predict(self.strategy.transform_for_predict(X, i))
                for i, est in enumerate(self.estimators_)
            ]
        )
        return all_preds

    def predict_with_uncertainty(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean, std) across estimators for uncertainty quantification.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean_predictions, std_predictions)
        """
        predictions = self._get_all_predictions(X)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (mean, lower, upper) prediction interval.

        Uses empirical quantiles across bootstrap estimates.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        alpha : float, default=0.05
            Significance level (default: 0.05 for 95% CI)

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            (mean, lower_bound, upper_bound)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        predictions = self._get_all_predictions(X)
        mean = np.mean(predictions, axis=0)
        lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
        upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)
        return mean, lower, upper

    def __repr__(self) -> str:
        return (
            f"TimeSeriesBagger("
            f"base_model={self.base_model!r}, "
            f"strategy={self.strategy.__class__.__name__}, "
            f"n_estimators={self.n_estimators})"
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "SupportsPredict",
    "BootstrapStrategy",
    "TimeSeriesBagger",
]
