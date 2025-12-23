"""
Conformal Prediction Module.

Distribution-free prediction intervals with finite-sample coverage guarantees.

Key concepts:
- **Split Conformal**: Calibrate on holdout, apply to test
- **Adaptive Conformal**: Dynamic adjustment for distribution shift
- **Coverage guarantee**: P(Y ∈ interval) ≥ 1 - α

Knowledge Tiers
---------------
[T1] Split conformal prediction (Romano, Patterson & Candès 2019)
[T1] Finite-sample coverage guarantee: P(Y ∈ Ĉ) ≥ 1 - α (Vovk et al. 2005)
[T1] Adaptive conformal inference for distribution shift (Gibbs & Candès 2021)
[T1] Quantile formula: q = ceil((n+1)(1-α))/n (standard conformal result)
[T2] Bootstrap uncertainty as complementary approach (empirical)
[T3] Default gamma=0.1 for adaptive conformal (recommended in paper, may need tuning)
[T3] Calibration fraction=0.3 as default split (implementation choice)

Example
-------
>>> from temporalcv.conformal import (
...     SplitConformalPredictor,
...     walk_forward_conformal,
... )
>>>
>>> # Calibrate on held-out predictions
>>> conformal = SplitConformalPredictor(alpha=0.05)
>>> conformal.calibrate(cal_predictions, cal_actuals)
>>>
>>> # Generate intervals for new predictions
>>> intervals = conformal.predict_interval(test_predictions)
>>> print(f"Coverage: {intervals.coverage(test_actuals):.1%}")

References
----------
[T1] Romano, Y., Patterson, E. & Candès, E.J. (2019). Conformalized quantile
     regression. NeurIPS.
[T1] Gibbs, I. & Candès, E.J. (2021). Adaptive conformal inference under
     distribution shift. NeurIPS.
[T1] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in
     a Random World. Springer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PredictionInterval:
    """
    Container for prediction intervals.

    Attributes
    ----------
    point : np.ndarray
        Point predictions
    lower : np.ndarray
        Lower bound of interval
    upper : np.ndarray
        Upper bound of interval
    confidence : float
        Nominal confidence level (1 - alpha)
    method : str
        Method used for interval construction

    Examples
    --------
    >>> interval = PredictionInterval(
    ...     point=np.array([1.0, 2.0]),
    ...     lower=np.array([0.5, 1.5]),
    ...     upper=np.array([1.5, 2.5]),
    ...     confidence=0.95,
    ...     method="split_conformal"
    ... )
    >>> print(f"Mean width: {interval.mean_width:.3f}")
    """

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    confidence: float
    method: str

    @property
    def width(self) -> np.ndarray:
        """Interval width at each point."""
        result: np.ndarray = self.upper - self.lower
        return result

    @property
    def mean_width(self) -> float:
        """Mean interval width."""
        return float(np.mean(self.width))

    def coverage(self, actuals: np.ndarray) -> float:
        """
        Compute empirical coverage.

        Parameters
        ----------
        actuals : np.ndarray
            Actual values

        Returns
        -------
        float
            Fraction of actuals within intervals
        """
        actuals = np.asarray(actuals)
        within = (actuals >= self.lower) & (actuals <= self.upper)
        return float(np.mean(within))

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "point": self.point.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "confidence": self.confidence,
            "method": self.method,
            "mean_width": self.mean_width,
        }


class SplitConformalPredictor:
    """
    Split Conformal Prediction for regression.

    Uses a calibration set to compute nonconformity scores,
    then applies to new predictions for valid prediction intervals.

    [T1] Distribution-free finite-sample coverage guarantee (Romano et al. 2019)

    Parameters
    ----------
    alpha : float, default=0.05
        Miscoverage rate (1 - confidence). For 95% intervals, use alpha=0.05.

    Attributes
    ----------
    alpha : float
        Miscoverage rate
    quantile_ : float or None
        Calibrated quantile of residuals (set after calibrate())

    Examples
    --------
    >>> scp = SplitConformalPredictor(alpha=0.10)  # 90% intervals
    >>> scp.calibrate(cal_preds, cal_actuals)
    >>> intervals = scp.predict_interval(test_preds)
    >>> print(f"Quantile: {scp.quantile_:.4f}")

    References
    ----------
    Romano, Sesia, Candes (2019). "Conformalized Quantile Regression"
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize conformal predictor.

        Parameters
        ----------
        alpha : float
            Miscoverage rate (default: 0.05 for 95% intervals)

        Raises
        ------
        ValueError
            If alpha not in (0, 1)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.quantile_: Optional[float] = None

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> "SplitConformalPredictor":
        """
        Calibrate conformal predictor on held-out data.

        CRITICAL: Calibration data MUST be separate from test data
        to maintain the coverage guarantee.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions on calibration set
        actuals : np.ndarray
            Actual values on calibration set

        Returns
        -------
        SplitConformalPredictor
            Calibrated predictor (self)

        Raises
        ------
        ValueError
            If fewer than 10 calibration samples

        Notes
        -----
        Uses the quantile formula: ceil((n+1)(1-alpha))/n
        which provides finite-sample coverage guarantee.

        Warning
        -------
        SplitConformalPredictor assumes exchangeability (i.i.d. data).
        For time series with autocorrelation, this assumption is violated
        and coverage guarantees may not hold. Consider AdaptiveConformalPredictor
        or walk_forward_conformal for temporal data.
        """
        warnings.warn(
            "SplitConformalPredictor assumes exchangeability (i.i.d. data). "
            "For time series, consider AdaptiveConformalPredictor instead.",
            UserWarning,
            stacklevel=2,
        )

        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
                "must have same length"
            )

        if len(predictions) < 10:
            raise ValueError(
                f"Need at least 10 calibration samples, got {len(predictions)}"
            )

        # Nonconformity scores: absolute residuals
        scores = np.abs(actuals - predictions)

        # Quantile for coverage guarantee
        # Use ceiling((n+1)(1-alpha))/n quantile for finite-sample validity
        # method="higher" ensures conservative coverage (ceiling interpolation)
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)  # Cap at 1.0

        self.quantile_ = float(np.quantile(scores, q, method="higher"))

        return self

    def predict_interval(
        self,
        predictions: np.ndarray,
    ) -> PredictionInterval:
        """
        Construct prediction intervals.

        Parameters
        ----------
        predictions : np.ndarray
            Point predictions

        Returns
        -------
        PredictionInterval
            Prediction intervals with coverage guarantee

        Raises
        ------
        RuntimeError
            If predictor not calibrated
        """
        if self.quantile_ is None:
            raise RuntimeError("Predictor not calibrated. Call calibrate() first.")

        predictions = np.asarray(predictions)

        lower = predictions - self.quantile_
        upper = predictions + self.quantile_

        return PredictionInterval(
            point=predictions,
            lower=lower,
            upper=upper,
            confidence=1 - self.alpha,
            method="split_conformal",
        )


class AdaptiveConformalPredictor:
    """
    Adaptive Conformal Inference for time series.

    Adjusts quantile dynamically based on recent coverage,
    addressing the challenge of non-exchangeable data in time series.

    [T1] Gibbs & Candes (2021) ACI for distribution shift

    Parameters
    ----------
    alpha : float, default=0.05
        Target miscoverage rate
    gamma : float, default=0.1
        Adaptation rate (higher = faster adaptation)

    Attributes
    ----------
    alpha : float
        Target miscoverage rate
    gamma : float
        Adaptation rate
    quantile_history : list[float]
        History of adaptive quantiles

    Examples
    --------
    >>> acp = AdaptiveConformalPredictor(alpha=0.10, gamma=0.1)
    >>> acp.initialize(initial_preds, initial_actuals)
    >>>
    >>> # Online updates
    >>> for pred, actual in stream:
    ...     lower, upper = acp.predict_interval(pred)
    ...     acp.update(pred, actual)

    References
    ----------
    Gibbs, Candes (2021). "Adaptive Conformal Inference Under Distribution Shift"
    """

    def __init__(
        self,
        alpha: float = 0.05,
        gamma: float = 0.1,
    ):
        """
        Initialize adaptive conformal predictor.

        Parameters
        ----------
        alpha : float
            Target miscoverage rate
        gamma : float
            Adaptation rate (higher = faster adaptation)

        Raises
        ------
        ValueError
            If alpha or gamma not in (0, 1)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0 < gamma < 1:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")

        self.alpha = alpha
        self.gamma = gamma
        self.quantile_history: List[float] = []
        self._current_quantile: Optional[float] = None

    def initialize(
        self,
        initial_predictions: np.ndarray,
        initial_actuals: np.ndarray,
    ) -> "AdaptiveConformalPredictor":
        """
        Initialize with calibration data.

        Parameters
        ----------
        initial_predictions : np.ndarray
            Initial predictions
        initial_actuals : np.ndarray
            Initial actuals

        Returns
        -------
        AdaptiveConformalPredictor
            Initialized predictor (self)
        """
        initial_predictions = np.asarray(initial_predictions)
        initial_actuals = np.asarray(initial_actuals)

        scores = np.abs(initial_actuals - initial_predictions)
        n = len(scores)

        if n == 0:
            raise ValueError("Cannot initialize with empty data")

        # method="higher" ensures conservative coverage (ceiling interpolation)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)

        self._current_quantile = float(np.quantile(scores, q, method="higher"))
        self.quantile_history = [self._current_quantile]

        return self

    def update(
        self,
        prediction: float,
        actual: float,
    ) -> float:
        """
        Update quantile based on coverage feedback.

        Parameters
        ----------
        prediction : float
            Latest prediction
        actual : float
            Actual value

        Returns
        -------
        float
            Updated quantile

        Raises
        ------
        RuntimeError
            If predictor not initialized
        """
        if self._current_quantile is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        # Check if covered
        error = abs(actual - prediction)
        covered = error <= self._current_quantile

        # Update quantile: increase if not covered, decrease if covered
        if covered:
            # Covered: could tighten interval
            update = -self.gamma * self.alpha
        else:
            # Not covered: need wider interval
            update = self.gamma * (1 - self.alpha)

        self._current_quantile = max(0.0, self._current_quantile + update)
        self.quantile_history.append(self._current_quantile)

        return self._current_quantile

    def predict_interval(
        self,
        prediction: float,
    ) -> Tuple[float, float]:
        """
        Construct prediction interval for single prediction.

        Parameters
        ----------
        prediction : float
            Point prediction

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds

        Raises
        ------
        RuntimeError
            If predictor not initialized
        """
        if self._current_quantile is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        lower = prediction - self._current_quantile
        upper = prediction + self._current_quantile

        return lower, upper

    @property
    def current_quantile(self) -> Optional[float]:
        """Return current adaptive quantile."""
        return self._current_quantile


class BootstrapUncertainty:
    """
    Bootstrap-based prediction intervals.

    Uses residual bootstrap to estimate prediction uncertainty.
    Useful for comparison with conformal methods.

    [T1] Efron & Tibshirani (1993) bootstrap theory

    Parameters
    ----------
    n_bootstrap : int, default=100
        Number of bootstrap samples
    alpha : float, default=0.05
        Miscoverage rate
    random_state : int, default=42
        Random seed for reproducibility

    Examples
    --------
    >>> boot = BootstrapUncertainty(n_bootstrap=100, alpha=0.10)
    >>> boot.fit(cal_preds, cal_actuals)
    >>> intervals = boot.predict_interval(test_preds)
    """

    def __init__(
        self,
        n_bootstrap: int = 100,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        """
        Initialize bootstrap uncertainty estimator.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap samples
        alpha : float
            Miscoverage rate
        random_state : int
            Random seed
        """
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.residuals_: Optional[np.ndarray] = None

    def fit(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> "BootstrapUncertainty":
        """
        Fit bootstrap estimator.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions
        actuals : np.ndarray
            Actuals

        Returns
        -------
        BootstrapUncertainty
            Fitted estimator (self)
        """
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
                "must have same length"
            )

        self.residuals_ = actuals - predictions
        return self

    def predict_interval(
        self,
        predictions: np.ndarray,
    ) -> PredictionInterval:
        """
        Construct bootstrap prediction intervals.

        Parameters
        ----------
        predictions : np.ndarray
            Point predictions

        Returns
        -------
        PredictionInterval
            Bootstrap prediction intervals
        """
        if self.residuals_ is None:
            raise RuntimeError("Estimator not fitted. Call fit() first.")

        predictions = np.asarray(predictions)
        rng = np.random.RandomState(self.random_state)

        # Bootstrap resampling of residuals
        n_pred = len(predictions)
        bootstrap_samples = np.zeros((self.n_bootstrap, n_pred))

        for i in range(self.n_bootstrap):
            # Sample residuals with replacement
            sampled_residuals = rng.choice(self.residuals_, size=n_pred, replace=True)
            bootstrap_samples[i] = predictions + sampled_residuals

        # Compute quantiles
        lower_q = self.alpha / 2
        upper_q = 1 - self.alpha / 2

        lower = np.percentile(bootstrap_samples, lower_q * 100, axis=0)
        upper = np.percentile(bootstrap_samples, upper_q * 100, axis=0)

        return PredictionInterval(
            point=predictions,
            lower=lower,
            upper=upper,
            confidence=1 - self.alpha,
            method="bootstrap",
        )


def evaluate_interval_quality(
    intervals: PredictionInterval,
    actuals: np.ndarray,
) -> dict[str, object]:
    """
    Evaluate prediction interval quality.

    Parameters
    ----------
    intervals : PredictionInterval
        Prediction intervals
    actuals : np.ndarray
        Actual values

    Returns
    -------
    dict
        Quality metrics:
        - coverage: empirical coverage
        - target_coverage: nominal coverage (1 - alpha)
        - coverage_gap: coverage - target
        - mean_width: average interval width
        - interval_score: proper scoring rule (lower is better)
        - conditional_gap: difference in coverage by prediction magnitude

    Examples
    --------
    >>> quality = evaluate_interval_quality(intervals, actuals)
    >>> print(f"Coverage: {quality['coverage']:.1%}")
    >>> print(f"Gap: {quality['coverage_gap']:+.1%}")
    """
    actuals = np.asarray(actuals)

    coverage = intervals.coverage(actuals)
    mean_width = intervals.mean_width
    target_coverage = intervals.confidence

    # Coverage deviation
    coverage_gap = coverage - target_coverage

    # Conditional coverage: check if coverage varies with prediction magnitude
    n = len(actuals)
    if n >= 20:
        # Split into low/high prediction magnitude
        median_pred = np.median(np.abs(intervals.point))
        low_mask = np.abs(intervals.point) < median_pred
        high_mask = ~low_mask

        def _cond_coverage(mask: np.ndarray) -> float:
            if mask.sum() == 0:
                return float("nan")
            masked_actuals = actuals[mask]
            masked_lower = intervals.lower[mask]
            masked_upper = intervals.upper[mask]
            within = (masked_actuals >= masked_lower) & (masked_actuals <= masked_upper)
            return float(np.mean(within))

        low_coverage = _cond_coverage(low_mask)
        high_coverage = _cond_coverage(high_mask)
        if not np.isnan(low_coverage) and not np.isnan(high_coverage):
            conditional_gap = abs(low_coverage - high_coverage)
        else:
            conditional_gap = float("nan")
    else:
        low_coverage = float("nan")
        high_coverage = float("nan")
        conditional_gap = float("nan")

    # Interval score (proper scoring rule for intervals)
    # Lower is better
    alpha = 1 - intervals.confidence
    width = intervals.upper - intervals.lower
    below = (actuals < intervals.lower).astype(float)
    above = (actuals > intervals.upper).astype(float)
    interval_score = float(
        np.mean(
            width
            + (2 / alpha) * (intervals.lower - actuals) * below
            + (2 / alpha) * (actuals - intervals.upper) * above
        )
    )

    return {
        "coverage": coverage,
        "target_coverage": target_coverage,
        "coverage_gap": coverage_gap,
        "mean_width": mean_width,
        "interval_score": interval_score,
        "low_coverage": low_coverage,
        "high_coverage": high_coverage,
        "conditional_gap": conditional_gap,
        "method": intervals.method,
    }


def walk_forward_conformal(
    predictions: np.ndarray,
    actuals: np.ndarray,
    calibration_fraction: float = 0.3,
    alpha: float = 0.05,
) -> Tuple[PredictionInterval, dict[str, object]]:
    """
    Apply conformal prediction to walk-forward results.

    CRITICAL: Coverage is computed ONLY on post-calibration holdout
    to avoid inflated coverage from calibration points.

    Parameters
    ----------
    predictions : np.ndarray
        Walk-forward predictions (all splits)
    actuals : np.ndarray
        Corresponding actuals
    calibration_fraction : float, default=0.3
        Fraction of data for calibration (default: 30%)
    alpha : float, default=0.05
        Miscoverage rate (default: 0.05 for 95% intervals)

    Returns
    -------
    tuple[PredictionInterval, dict]
        (intervals_on_holdout, quality_metrics)

    Raises
    ------
    ValueError
        If insufficient calibration or holdout points

    Notes
    -----
    [T1] Romano, Sesia, Candes (2019). "Conformalized Quantile Regression"

    The key insight is that coverage must be evaluated on data NOT used
    for calibration. Using calibration data in coverage computation
    inflates the reported coverage.

    Examples
    --------
    >>> intervals, quality = walk_forward_conformal(predictions, actuals)
    >>> print(f"Coverage (holdout only): {quality['coverage']:.1%}")
    >>> print(f"Calibration size: {quality['calibration_size']}")
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    n = len(predictions)

    if len(actuals) != n:
        raise ValueError(
            f"predictions ({n}) and actuals ({len(actuals)}) must have same length"
        )

    cal_size = int(n * calibration_fraction)

    if cal_size < 10:
        raise ValueError(
            f"Need >= 10 calibration points, got {cal_size}. "
            f"Either increase data size or reduce calibration_fraction."
        )

    holdout_size = n - cal_size
    if holdout_size < 10:
        raise ValueError(
            f"Need >= 10 holdout points, got {holdout_size}. "
            f"Either increase data size or reduce calibration_fraction."
        )

    # Calibrate on first portion
    conformal = SplitConformalPredictor(alpha=alpha)
    conformal.calibrate(predictions[:cal_size], actuals[:cal_size])

    # Intervals on holdout ONLY
    holdout_preds = predictions[cal_size:]
    holdout_actuals = actuals[cal_size:]

    intervals = conformal.predict_interval(holdout_preds)
    quality = evaluate_interval_quality(intervals, holdout_actuals)

    # Add metadata for transparency
    quality["calibration_size"] = cal_size
    quality["holdout_size"] = holdout_size
    quality["calibration_fraction"] = calibration_fraction
    quality["quantile"] = conformal.quantile_

    return intervals, quality


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Dataclasses
    "PredictionInterval",
    # Predictors
    "SplitConformalPredictor",
    "AdaptiveConformalPredictor",
    "BootstrapUncertainty",
    # Functions
    "evaluate_interval_quality",
    "walk_forward_conformal",
]
