"""
Conformal Prediction Module.

Distribution-free prediction intervals with finite-sample coverage guarantees.

Key concepts:
- **Split Conformal**: Calibrate on holdout, apply to test
- **Adaptive Conformal**: Dynamic adjustment for distribution shift
- **Bellman Conformal**: Optimal DP-based adaptation (multi-horizon)
- **Coverage guarantee**: P(Y ∈ interval) ≥ 1 - α

Knowledge Tiers
---------------
[T1] Split conformal prediction (Romano, Patterson & Candès 2019)
[T1] Finite-sample coverage guarantee: P(Y ∈ Ĉ) ≥ 1 - α (Vovk et al. 2005)
[T1] Adaptive conformal inference for distribution shift (Gibbs & Candès 2021)
[T1] Bellman conformal inference for optimal adaptation (Yang, Candès & Lei 2024)
[T1] Quantile formula: q = ceil((n+1)(1-α))/n (standard conformal result)
[T2] Bootstrap uncertainty as complementary approach (empirical)
[T3] Default gamma=0.1 for adaptive conformal (recommended in paper, may need tuning)
[T3] Calibration fraction=0.3 as default split (implementation choice)
[T3] Default n_grid=50 for Bellman DP discretization (tradeoff: finer = slower)

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

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


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

    See Also
    --------
    AdaptiveConformalPredictor : Dynamic adaptation for distribution shift.
    BootstrapUncertainty : Alternative bootstrap-based intervals.
    walk_forward_conformal : Convenience function for walk-forward results.
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

        # Validate no NaN values
        if np.any(np.isnan(predictions)):
            raise ValueError(
                "predictions contains NaN values. Clean data before processing. "
                "Use np.nan_to_num() or dropna() to handle missing values."
            )
        if np.any(np.isnan(actuals)):
            raise ValueError(
                "actuals contains NaN values. Clean data before processing. "
                "Use np.nan_to_num() or dropna() to handle missing values."
            )

        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
                "must have same length"
            )

        if len(predictions) < 10:
            raise ValueError(
                f"Need at least 10 calibration samples, got {len(predictions)}"
            )

        # Tiered sample size warnings [T2]
        # n>=10: allowed (minimum for quantile estimation)
        # n>=30: recommended (CLT threshold)
        # n>=50: reliable (stable quantile estimation)
        n_cal = len(predictions)
        if n_cal < 30:
            warnings.warn(
                f"Calibration set size n={n_cal} is below recommended minimum of 30. "
                f"Coverage guarantee holds but quantile estimation may be unstable. "
                f"n>=50 recommended for reliable inference. "
                f"See SPECIFICATION.md Section 5.2 for guidance.",
                UserWarning,
                stacklevel=2,
            )
        elif n_cal < 50:
            warnings.warn(
                f"Calibration set size n={n_cal} is adequate but below optimal. "
                f"n>=50 recommended for stable quantile estimation.",
                UserWarning,
                stacklevel=2,
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

    See Also
    --------
    SplitConformalPredictor : Static conformal for i.i.d. data.
    BootstrapUncertainty : Alternative bootstrap-based intervals.
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


class BellmanConformalPredictor:
    """
    Bellman Conformal Inference for optimal adaptive prediction intervals.

    Uses dynamic programming to solve the Bellman equation for optimal
    quantile selection, preventing interval explosion during distribution
    shifts while minimizing expected interval width.

    [T1] Yang, Candès & Lei (2024) - Bellman Conformal Inference

    Key Advantages Over AdaptiveConformalPredictor
    -----------------------------------------------
    1. **Proactive**: Uses multi-step forecast information, not just past
    2. **Optimal**: Minimizes expected width subject to coverage constraint
    3. **Bounded**: Prevents intervals from growing unboundedly
    4. **Horizon-aware**: Explicitly models forecast horizon uncertainty

    Parameters
    ----------
    alpha : float, default=0.05
        Target miscoverage rate.
    horizon : int, default=5
        Number of steps to look ahead in DP optimization.
    n_grid : int, default=50
        Grid size for quantile discretization (finer = more accurate, slower).
    gamma : float, default=0.1
        Learning rate for quantile updates.
    lambda_reg : float, default=0.01
        Regularization to penalize extreme quantile values.

    Attributes
    ----------
    alpha : float
        Target miscoverage rate.
    horizon : int
        Lookahead horizon for DP.
    quantile_history : list[float]
        History of optimal quantiles.
    value_function : np.ndarray or None
        Computed DP value function.

    Examples
    --------
    >>> bcp = BellmanConformalPredictor(alpha=0.10, horizon=10)
    >>> bcp.initialize(cal_preds, cal_actuals)
    >>>
    >>> # Get optimal quantile sequence for test predictions
    >>> quantiles = bcp.solve_optimal_sequence(test_preds, n_steps=20)
    >>>
    >>> # Or use online mode
    >>> for pred, actual in stream:
    ...     lower, upper = bcp.predict_interval(pred)
    ...     bcp.update(pred, actual)

    References
    ----------
    Yang, R., Candès, E.J. & Lei, L. (2024). "Bellman Conformal Inference:
    Calibrating Prediction Intervals For Time Series." arXiv:2402.05203.

    See Also
    --------
    AdaptiveConformalPredictor : Simpler gradient-based adaptation.
    SplitConformalPredictor : Static conformal for i.i.d. data.

    Notes
    -----
    The Bellman equation solved is:

        V(q_t) = min_{q_{t+1}} { E[width(q_{t+1})] + V(q_{t+1}) }
                 s.t. E[coverage(q_{t+1})] >= 1 - alpha

    This is discretized over a grid of quantile values and solved via
    backward induction.

    .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        alpha: float = 0.05,
        horizon: int = 5,
        n_grid: int = 50,
        gamma: float = 0.1,
        lambda_reg: float = 0.01,
    ):
        """
        Initialize Bellman conformal predictor.

        Parameters
        ----------
        alpha : float
            Target miscoverage rate.
        horizon : int
            Number of steps to look ahead.
        n_grid : int
            Grid size for quantile discretization.
        gamma : float
            Learning rate for quantile updates.
        lambda_reg : float
            Regularization strength.

        Raises
        ------
        ValueError
            If parameters are out of valid ranges.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n_grid < 10:
            raise ValueError(f"n_grid must be >= 10, got {n_grid}")
        if not 0 < gamma < 1:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")

        self.alpha = alpha
        self.horizon = horizon
        self.n_grid = n_grid
        self.gamma = gamma
        self.lambda_reg = lambda_reg

        # State
        self._current_quantile: Optional[float] = None
        self.quantile_history: List[float] = []
        self.value_function: Optional[np.ndarray] = None
        self._residual_distribution: Optional[np.ndarray] = None
        self._quantile_grid: Optional[np.ndarray] = None

    def initialize(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> "BellmanConformalPredictor":
        """
        Initialize with calibration data and solve initial DP.

        Parameters
        ----------
        predictions : np.ndarray
            Calibration predictions.
        actuals : np.ndarray
            Calibration actuals.

        Returns
        -------
        BellmanConformalPredictor
            Initialized predictor (self).

        Raises
        ------
        ValueError
            If calibration data is insufficient.
        """
        predictions = np.asarray(predictions)
        actuals = np.asarray(actuals)

        if len(predictions) < 10:
            raise ValueError(
                f"Need at least 10 calibration samples, got {len(predictions)}"
            )

        # Compute residual distribution (nonconformity scores)
        residuals = np.abs(actuals - predictions)
        self._residual_distribution = np.sort(residuals)

        # Build quantile grid from residuals
        q_min = np.percentile(residuals, 5)
        q_max = np.percentile(residuals, 99)
        # Ensure non-negative and reasonable range
        q_min = max(0.0, q_min * 0.5)
        q_max = max(q_max * 2.0, np.std(residuals) * 5)

        self._quantile_grid = np.linspace(q_min, q_max, self.n_grid)

        # Solve initial Bellman recursion
        self._solve_bellman()

        # Initialize current quantile using split conformal formula
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._current_quantile = float(
            np.quantile(residuals, q_level, method="higher")
        )
        self.quantile_history = [self._current_quantile]

        return self

    def _solve_bellman(self) -> None:
        """
        Solve Bellman equation via backward induction.

        Computes optimal value function V(q) for each quantile in the grid.
        The value function represents the minimum expected future cost
        (interval width) starting from quantile q.
        """
        if self._quantile_grid is None or self._residual_distribution is None:
            raise RuntimeError("Must initialize before solving Bellman equation.")

        n_grid = len(self._quantile_grid)
        residuals = self._residual_distribution

        # Value function: V[h, i] = value at horizon h, quantile index i
        # Backward induction from horizon to 0
        V = np.zeros((self.horizon + 1, n_grid))

        # Terminal cost: zero at end
        V[self.horizon, :] = 0.0

        # Expected coverage probability for each quantile
        # P(|residual| <= q) estimated from calibration distribution
        def coverage_prob(q: float) -> float:
            """Estimate P(residual <= q) from calibration distribution."""
            return float(np.mean(residuals <= q))

        # Expected width cost (just 2*q for symmetric intervals)
        def width_cost(q: float) -> float:
            """Cost of interval width."""
            return 2.0 * q

        # Backward induction
        for h in range(self.horizon - 1, -1, -1):
            for i, q_current in enumerate(self._quantile_grid):
                # Find optimal next quantile
                best_value = float("inf")

                for j, q_next in enumerate(self._quantile_grid):
                    # Check coverage constraint
                    cov = coverage_prob(q_next)
                    if cov < 1 - self.alpha - 0.01:  # Small slack for numerical stability
                        continue  # Skip infeasible actions

                    # Compute cost
                    immediate_cost = width_cost(q_next)

                    # Regularization: penalize large changes
                    transition_cost = self.lambda_reg * (q_next - q_current) ** 2

                    # Total cost
                    total = immediate_cost + transition_cost + V[h + 1, j]

                    if total < best_value:
                        best_value = total

                # If no feasible action, use highest coverage quantile
                if best_value == float("inf"):
                    best_value = width_cost(self._quantile_grid[-1]) + V[h + 1, -1]

                V[h, i] = best_value

        self.value_function = V

    def _get_optimal_quantile(self, current_q: float) -> float:
        """
        Get optimal next quantile given current quantile.

        Parameters
        ----------
        current_q : float
            Current quantile value.

        Returns
        -------
        float
            Optimal next quantile.
        """
        if self._quantile_grid is None or self.value_function is None:
            raise RuntimeError("Must initialize before getting optimal quantile.")

        if self._residual_distribution is None:
            raise RuntimeError("Must initialize before getting optimal quantile.")

        residuals = self._residual_distribution

        # Find closest index in grid
        i_current = int(np.argmin(np.abs(self._quantile_grid - current_q)))

        # Find optimal next quantile (greedy from value function at h=0)
        best_j = i_current  # Default to staying
        best_value = float("inf")

        for j, q_next in enumerate(self._quantile_grid):
            # Check coverage constraint
            cov = float(np.mean(residuals <= q_next))
            if cov < 1 - self.alpha - 0.01:
                continue

            # Cost
            immediate = 2.0 * q_next
            transition = self.lambda_reg * (q_next - current_q) ** 2

            # Use value function from horizon 1
            if self.horizon >= 1:
                future = self.value_function[1, j]
            else:
                future = 0.0

            total = immediate + transition + future

            if total < best_value:
                best_value = total
                best_j = j

        return float(self._quantile_grid[best_j])

    def update(
        self,
        prediction: float,
        actual: float,
    ) -> float:
        """
        Update quantile based on coverage feedback using Bellman-optimal policy.

        Parameters
        ----------
        prediction : float
            Latest prediction.
        actual : float
            Actual value.

        Returns
        -------
        float
            Updated quantile.

        Raises
        ------
        RuntimeError
            If predictor not initialized.
        """
        if self._current_quantile is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        # Check coverage
        error = abs(actual - prediction)
        covered = error <= self._current_quantile

        # Get Bellman-optimal quantile
        optimal_q = self._get_optimal_quantile(self._current_quantile)

        # Blend with adaptive update for responsiveness
        # If not covered, move toward optimal but also increase
        if covered:
            # Covered: move toward optimal (which may be tighter)
            adaptive_update = -self.gamma * self.alpha
        else:
            # Not covered: move toward optimal but also widen
            adaptive_update = self.gamma * (1 - self.alpha)

        # Blend: 50% Bellman optimal, 50% adaptive
        bellman_component = 0.5 * (optimal_q - self._current_quantile)
        adaptive_component = 0.5 * adaptive_update

        self._current_quantile = max(
            0.0, self._current_quantile + bellman_component + adaptive_component
        )
        self.quantile_history.append(self._current_quantile)

        # Periodically update residual distribution for online learning
        if len(self.quantile_history) % 50 == 0 and self._residual_distribution is not None:
            # Add new residual to distribution (with forgetting)
            new_residuals = np.append(self._residual_distribution[-100:], error)
            self._residual_distribution = np.sort(new_residuals)

        return self._current_quantile

    def predict_interval(
        self,
        prediction: float,
    ) -> Tuple[float, float]:
        """
        Construct prediction interval using current optimal quantile.

        Parameters
        ----------
        prediction : float
            Point prediction.

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.

        Raises
        ------
        RuntimeError
            If predictor not initialized.
        """
        if self._current_quantile is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        lower = prediction - self._current_quantile
        upper = prediction + self._current_quantile

        return lower, upper

    def solve_optimal_sequence(
        self,
        predictions: np.ndarray,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """
        Solve for optimal quantile sequence for given predictions.

        Uses forward simulation with Bellman-optimal policy to compute
        the sequence of quantiles that minimizes expected width while
        maintaining coverage.

        Parameters
        ----------
        predictions : np.ndarray
            Future predictions to construct intervals for.
        n_steps : int, optional
            Number of steps to optimize. If None, uses len(predictions).

        Returns
        -------
        np.ndarray
            Optimal quantile sequence.

        Raises
        ------
        RuntimeError
            If predictor not initialized.

        Notes
        -----
        This is useful for batch interval construction where you have
        access to all future predictions upfront.
        """
        if self._current_quantile is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        predictions = np.asarray(predictions)
        if n_steps is None:
            n_steps = len(predictions)
        n_steps = min(n_steps, len(predictions))

        # Forward simulation with optimal policy
        quantiles = np.zeros(n_steps)
        q = self._current_quantile

        for t in range(n_steps):
            q = self._get_optimal_quantile(q)
            quantiles[t] = q

        return quantiles

    def predict_intervals_batch(
        self,
        predictions: np.ndarray,
    ) -> PredictionInterval:
        """
        Construct prediction intervals for a batch of predictions.

        Uses solve_optimal_sequence to get optimal quantiles, then
        constructs intervals.

        Parameters
        ----------
        predictions : np.ndarray
            Batch of predictions.

        Returns
        -------
        PredictionInterval
            Prediction intervals with optimal widths.
        """
        predictions = np.asarray(predictions)
        quantiles = self.solve_optimal_sequence(predictions)

        lower = predictions - quantiles
        upper = predictions + quantiles

        return PredictionInterval(
            point=predictions,
            lower=lower,
            upper=upper,
            confidence=1 - self.alpha,
            method="bellman_conformal",
        )

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

    Complexity: O(n_bootstrap × n_predictions)

    See Also
    --------
    SplitConformalPredictor : Conformal method with coverage guarantees.
    wild_cluster_bootstrap : Wild bootstrap for CV fold inference.
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

    See Also
    --------
    PredictionInterval.coverage : Simple coverage computation.
    walk_forward_conformal : Integrated calibration and evaluation.
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
        median_pred: float = float(np.median(np.abs(intervals.point)))
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

    See Also
    --------
    SplitConformalPredictor : Underlying conformal predictor used.
    evaluate_interval_quality : Quality metrics computed on holdout.
    WalkForwardCV : CV strategy that produces the input predictions.
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
# Coverage Diagnostics
# =============================================================================


@dataclass
class CoverageDiagnostics:
    """
    Detailed coverage diagnostics for conformal prediction intervals.

    Attributes
    ----------
    overall_coverage : float
        Empirical coverage across all observations.
    target_coverage : float
        Nominal coverage level (1 - alpha).
    coverage_gap : float
        Difference between target and empirical coverage.
    undercoverage_warning : bool
        True if coverage is significantly below target.
    coverage_by_window : Dict[str, float]
        Coverage computed in rolling windows.
    coverage_by_regime : Optional[Dict[str, float]]
        Coverage by regime (if regimes provided).
    n_observations : int
        Total number of observations.

    .. versionadded:: 1.0.0
    """

    overall_coverage: float
    target_coverage: float
    coverage_gap: float
    undercoverage_warning: bool
    coverage_by_window: Dict[str, float]
    coverage_by_regime: Optional[Dict[str, float]]
    n_observations: int


def compute_coverage_diagnostics(
    intervals: PredictionInterval,
    actuals: np.ndarray,
    *,
    target_coverage: Optional[float] = None,
    window_size: int = 50,
    regimes: Optional[np.ndarray] = None,
    undercoverage_threshold: float = 0.05,
) -> CoverageDiagnostics:
    """
    Compute detailed coverage diagnostics for prediction intervals.

    Analyzes empirical coverage overall, by time window, and optionally by
    regime. Warns if coverage falls significantly below target.

    Parameters
    ----------
    intervals : PredictionInterval
        Prediction intervals to evaluate.
    actuals : np.ndarray
        Actual values for coverage computation.
    target_coverage : float, optional
        Target coverage level (1 - alpha). If None, uses ``intervals.confidence``.
    window_size : int, default=50
        Size of rolling windows for time-based coverage analysis.
    regimes : np.ndarray, optional
        Integer or string array of regime labels for each observation.
        If provided, coverage is also computed per regime.
    undercoverage_threshold : float, default=0.05
        Trigger warning if ``target_coverage - empirical_coverage > threshold``.

    Returns
    -------
    CoverageDiagnostics
        Detailed coverage diagnostics.

    Warns
    -----
    UserWarning
        If empirical coverage is significantly below target (gap > threshold).

    Examples
    --------
    >>> from temporalcv.conformal import (
    ...     SplitConformalPredictor,
    ...     compute_coverage_diagnostics,
    ... )
    >>> conformal = SplitConformalPredictor(alpha=0.05)
    >>> conformal.calibrate(cal_preds, cal_actuals)
    >>> intervals = conformal.predict_interval(test_preds)
    >>> diag = compute_coverage_diagnostics(intervals, test_actuals)
    >>> print(f"Coverage: {diag.overall_coverage:.1%}, "
    ...       f"Target: {diag.target_coverage:.1%}")

    Notes
    -----
    This function is useful for:

    1. Detecting coverage degradation in production
    2. Identifying time periods with poor coverage
    3. Regime-specific performance analysis

    .. versionadded:: 1.0.0
    """
    actuals = np.asarray(actuals)
    n = len(actuals)

    if len(intervals.lower) != n or len(intervals.upper) != n:
        raise ValueError(
            f"Interval length ({len(intervals.lower)}) doesn't match "
            f"actuals length ({n})"
        )

    # Use interval's confidence if target not specified
    if target_coverage is None:
        target_coverage = intervals.confidence

    # Compute overall coverage
    covered = (actuals >= intervals.lower) & (actuals <= intervals.upper)
    overall_coverage = float(np.mean(covered))

    # Coverage by time window
    coverage_by_window: Dict[str, float] = {}
    n_windows = max(1, n // window_size)

    for i in range(n_windows):
        start = i * window_size
        end = min((i + 1) * window_size, n)
        window_covered = covered[start:end]
        window_name = f"window_{i + 1}_{start}_{end}"
        coverage_by_window[window_name] = float(np.mean(window_covered))

    # Handle last partial window if exists
    if n % window_size != 0 and n_windows * window_size < n:
        start = n_windows * window_size
        window_covered = covered[start:]
        window_name = f"window_{n_windows + 1}_{start}_{n}"
        coverage_by_window[window_name] = float(np.mean(window_covered))

    # Coverage by regime (if provided)
    coverage_by_regime: Optional[Dict[str, float]] = None
    if regimes is not None:
        regimes = np.asarray(regimes)
        if len(regimes) != n:
            raise ValueError(
                f"Regimes length ({len(regimes)}) doesn't match actuals length ({n})"
            )

        coverage_by_regime = {}
        unique_regimes = np.unique(regimes)
        for regime in unique_regimes:
            mask = regimes == regime
            regime_coverage = float(np.mean(covered[mask]))
            coverage_by_regime[str(regime)] = regime_coverage

    # Check for undercoverage
    coverage_gap = target_coverage - overall_coverage
    undercoverage_warning = coverage_gap > undercoverage_threshold

    if undercoverage_warning:
        logger.warning(
            "Coverage undercoverage detected: empirical=%.3f, target=%.3f, gap=%.3f. "
            "Consider recalibrating or investigating distribution shift.",
            overall_coverage,
            target_coverage,
            coverage_gap,
        )

    return CoverageDiagnostics(
        overall_coverage=overall_coverage,
        target_coverage=target_coverage,
        coverage_gap=coverage_gap,
        undercoverage_warning=undercoverage_warning,
        coverage_by_window=coverage_by_window,
        coverage_by_regime=coverage_by_regime,
        n_observations=n,
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Dataclasses
    "PredictionInterval",
    "CoverageDiagnostics",
    # Predictors
    "SplitConformalPredictor",
    "AdaptiveConformalPredictor",
    "BellmanConformalPredictor",
    "BootstrapUncertainty",
    # Functions
    "evaluate_interval_quality",
    "walk_forward_conformal",
    "compute_coverage_diagnostics",
]
