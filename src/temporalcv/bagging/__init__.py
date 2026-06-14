"""
Time Series Bagging Framework.

Generic, model-agnostic bagging for time series with methodologically
correct bootstrap strategies from the literature.

Available Strategies
--------------------
- MovingBlockBootstrap: Block bootstrap preserving local autocorrelation
- StationaryBootstrap: Geometric block lengths for stationarity
- FeatureBagging: Random subspace method (feature bootstrap)

Factory Functions
-----------------
- create_block_bagger: Create bagger with Moving Block Bootstrap
- create_stationary_bagger: Create bagger with Stationary Bootstrap
- create_feature_bagger: Create bagger with Feature Bagging

Example
-------
>>> import numpy as np
>>> from sklearn.linear_model import Ridge
>>> from temporalcv.bagging import create_block_bagger
>>>
>>> # Synthetic train/test split (fixed seed for a reproducible doctest)
>>> rng = np.random.default_rng(0)
>>> X_train = rng.standard_normal((100, 3))
>>> y_train = X_train @ np.array([1.0, -0.5, 0.25]) + 0.1 * rng.standard_normal(100)
>>> X_test = rng.standard_normal((10, 3))
>>>
>>> # Create and fit a bagged Ridge model
>>> bagged_ridge = create_block_bagger(Ridge(), n_estimators=20, random_state=42)
>>> _ = bagged_ridge.fit(X_train, y_train)
>>> predictions = bagged_ridge.predict(X_test)
>>> predictions.shape
(10,)
>>> mean, lower, upper = bagged_ridge.predict_interval(X_test)
>>> bool(np.all(lower <= upper))
True

References
----------
- Kunsch (1989). "The Jackknife and Bootstrap for General Stationary"
- Politis & Romano (1994). "The Stationary Bootstrap"
- Ho (1998). "The Random Subspace Method"
- Bergmeir, Hyndman & Benitez (2016). "Bagging Exponential Smoothing"
"""

from temporalcv.bagging.base import (
    BootstrapStrategy,
    SupportsFitPredict,
    TimeSeriesBagger,
)
from temporalcv.bagging.strategies import (
    FeatureBagging,
    MovingBlockBootstrap,
    ResidualBootstrap,
    StationaryBootstrap,
    create_residual_bagger,
)


def create_block_bagger(
    base_model: SupportsFitPredict,
    n_estimators: int = 20,
    block_length: int | None = None,
    aggregation: str = "mean",
    random_state: int | None = None,
) -> TimeSeriesBagger:
    """
    Create bagged model with Moving Block Bootstrap.

    Parameters
    ----------
    base_model : SupportsFitPredict
        Model to bag (e.g., Ridge, ElasticNet)
    n_estimators : int, default=20
        Number of bootstrap estimators
    block_length : int or None, default=None
        Block length. If None, auto-compute as n^(1/3)
    aggregation : {"mean", "median"}, default="mean"
        How to combine predictions
    random_state : int or None, default=None
        Random seed for reproducibility. None for non-deterministic.

    Returns
    -------
    TimeSeriesBagger
        Bagged model ready for fit/predict

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> rng = np.random.default_rng(0)
    >>> X_train = rng.standard_normal((80, 3))
    >>> y_train = X_train[:, 0] + 0.1 * rng.standard_normal(80)
    >>> bagger = create_block_bagger(Ridge(), n_estimators=50, random_state=42)
    >>> _ = bagger.fit(X_train, y_train)
    >>> bagger.is_fitted
    True

    See Also
    --------
    create_stationary_bagger : Alternative with geometric block lengths.
    create_feature_bagger : Feature subspace method instead of row bootstrap.
    MovingBlockBootstrap : The underlying bootstrap strategy.
    """
    strategy = MovingBlockBootstrap(block_length=block_length)
    return TimeSeriesBagger(
        base_model,
        strategy,
        n_estimators=n_estimators,
        aggregation=aggregation,  # type: ignore[arg-type]
        random_state=random_state,
    )


def create_stationary_bagger(
    base_model: SupportsFitPredict,
    n_estimators: int = 20,
    expected_block_length: float | None = None,
    aggregation: str = "mean",
    random_state: int | None = None,
) -> TimeSeriesBagger:
    """
    Create bagged model with Stationary Bootstrap.

    Parameters
    ----------
    base_model : SupportsFitPredict
        Model to bag (e.g., Ridge, ElasticNet)
    n_estimators : int, default=20
        Number of bootstrap estimators
    expected_block_length : float or None, default=None
        Expected block length. If None, auto-compute as n^(1/3)
    aggregation : {"mean", "median"}, default="mean"
        How to combine predictions
    random_state : int or None, default=None
        Random seed for reproducibility. None for non-deterministic.

    Returns
    -------
    TimeSeriesBagger
        Bagged model ready for fit/predict

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> bagger = create_stationary_bagger(ElasticNet(), n_estimators=50)
    >>> bagger.n_estimators
    50

    See Also
    --------
    create_block_bagger : Fixed block lengths (simpler).
    StationaryBootstrap : The underlying bootstrap strategy.
    """
    strategy = StationaryBootstrap(expected_block_length=expected_block_length)
    return TimeSeriesBagger(
        base_model,
        strategy,
        n_estimators=n_estimators,
        aggregation=aggregation,  # type: ignore[arg-type]
        random_state=random_state,
    )


def create_feature_bagger(
    base_model: SupportsFitPredict,
    n_estimators: int = 20,
    max_features: float = 0.7,
    aggregation: str = "mean",
    random_state: int | None = None,
) -> TimeSeriesBagger:
    """
    Create bagged model with Feature Bagging (Random Subspace).

    Parameters
    ----------
    base_model : SupportsFitPredict
        Model to bag (e.g., Ridge, ElasticNet)
    n_estimators : int, default=20
        Number of bootstrap estimators
    max_features : float, default=0.7
        Fraction of features per estimator (0.0-1.0)
    aggregation : {"mean", "median"}, default="mean"
        How to combine predictions
    random_state : int or None, default=None
        Random seed for reproducibility. None for non-deterministic.

    Returns
    -------
    TimeSeriesBagger
        Bagged model ready for fit/predict

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> bagger = create_feature_bagger(Ridge(), max_features=0.6)
    >>> bagger.strategy.max_features
    0.6

    See Also
    --------
    create_block_bagger : Bootstrap rows instead of features.
    FeatureBagging : The underlying bootstrap strategy.
    """
    strategy = FeatureBagging(max_features=max_features)
    return TimeSeriesBagger(
        base_model,
        strategy,
        n_estimators=n_estimators,
        aggregation=aggregation,  # type: ignore[arg-type]
        random_state=random_state,
    )


__all__ = [
    # Core classes
    "SupportsFitPredict",
    "BootstrapStrategy",
    "TimeSeriesBagger",
    # Strategies
    "MovingBlockBootstrap",
    "StationaryBootstrap",
    "FeatureBagging",
    "ResidualBootstrap",
    # Factory functions
    "create_block_bagger",
    "create_stationary_bagger",
    "create_feature_bagger",
    "create_residual_bagger",
]
