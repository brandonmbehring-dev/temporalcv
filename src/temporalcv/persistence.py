"""
High-Persistence Series Metrics Module.

Specialized tools for evaluating forecasts on high-persistence time series
where the persistence baseline (predict no change) is trivially good.

Key concepts:
- **Move threshold**: Separates "significant" moves from noise (FLAT)
- **MC-SS**: Move-Conditional Skill Score relative to persistence
- **3-class direction**: UP/DOWN/FLAT makes persistence a fair baseline

Knowledge Tiers
---------------
[T1] Persistence baseline = predict no change (standard in forecasting literature)
[T1] Skill score formula: SS = 1 - (model_error / baseline_error) (Murphy 1988)
[T1] Directional accuracy testing framework (Pesaran & Timmermann 1992)
[T2] MC-SS = skill score computed on moves only (myga-forecasting-v2 Phase 11)
[T2] 70th percentile threshold defines "significant" moves (v2 empirical finding)
[T2] Threshold MUST come from training data only (BUG-003 fix in v2)
[T3] 10 samples per direction for reliability (rule of thumb, not validated)
[T3] Scale-aware epsilon for numerical stability (implementation choice)

Example
-------
>>> from temporalcv.persistence import (
...     compute_move_threshold,
...     compute_move_conditional_metrics,
... )
>>>
>>> # Compute threshold from training data (CRITICAL: training only!)
>>> threshold = compute_move_threshold(train_actuals, percentile=70.0)
>>>
>>> # Evaluate on test data
>>> mc = compute_move_conditional_metrics(predictions, actuals, threshold=threshold)
>>> print(f"MC-SS: {mc.skill_score:.3f}")
>>> print(f"Reliable: {mc.is_reliable}")

References
----------
[T1] Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy:
     An analysis and review. International Journal of Forecasting, 16(4), 437-450.
[T1] Pesaran, M.H. & Timmermann, A. (1992). A simple nonparametric test of
     predictive performance. Journal of Business & Economic Statistics, 10(4), 461-465.
[T1] Murphy, A.H. (1988). Skill scores based on the mean square error and their
     relationships to the correlation coefficient. Monthly Weather Review, 116, 2417-2424.
[T2] Move-conditional evaluation: myga-forecasting-v2 Phase 11 analysis.
     Rationale: Overall MAE is dominated by FLAT periods where persistence trivially wins.
     Conditioning on moves isolates genuine forecasting skill.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Tuple

import numpy as np

# Type alias for target mode
TargetMode = Literal["change", "level"]


class MoveDirection(Enum):
    """Direction of value change."""

    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass
class MoveConditionalResult:
    """
    Move-conditional evaluation results.

    Evaluates performance conditional on actual movement:
    - UP: actual > threshold
    - DOWN: actual < -threshold
    - FLAT: |actual| <= threshold

    Attributes
    ----------
    mae_up : float
        MAE for upward moves
    mae_down : float
        MAE for downward moves
    mae_flat : float
        MAE for flat periods
    n_up : int
        Count of upward moves
    n_down : int
        Count of downward moves
    n_flat : int
        Count of flat periods
    skill_score : float
        MC-SS = 1 - (model_mae_moves / persistence_mae_moves)
    move_threshold : float
        Threshold used for classification
    """

    mae_up: float
    mae_down: float
    mae_flat: float
    n_up: int
    n_down: int
    n_flat: int
    skill_score: float
    move_threshold: float

    @property
    def n_total(self) -> int:
        """Total sample count."""
        return self.n_up + self.n_down + self.n_flat

    @property
    def n_moves(self) -> int:
        """Count of significant moves (UP + DOWN)."""
        return self.n_up + self.n_down

    @property
    def is_reliable(self) -> bool:
        """
        Check if results are statistically reliable.

        Requires at least 10 samples per move direction.
        """
        return self.n_up >= 10 and self.n_down >= 10

    @property
    def move_fraction(self) -> float:
        """Fraction of samples that are moves (not FLAT)."""
        if self.n_total == 0:
            return 0.0
        return self.n_moves / self.n_total

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "mae_up": self.mae_up,
            "mae_down": self.mae_down,
            "mae_flat": self.mae_flat,
            "n_up": self.n_up,
            "n_down": self.n_down,
            "n_flat": self.n_flat,
            "n_moves": self.n_moves,
            "n_total": self.n_total,
            "skill_score": self.skill_score,
            "move_threshold": self.move_threshold,
            "is_reliable": self.is_reliable,
            "move_fraction": self.move_fraction,
        }


def compute_move_threshold(
    actuals: np.ndarray,
    percentile: float = 70.0,
    target_mode: TargetMode = "change",
) -> float:
    """
    Compute move threshold from historical changes.

    Default: 70th percentile of |actuals|.

    Parameters
    ----------
    actuals : np.ndarray
        Historical actual values (from training data).
        Should be *changes* (returns/differences), not raw levels.
    percentile : float, default=70.0
        Percentile of |actuals| to use as threshold
    target_mode : {"change", "level"}, default="change"
        Whether actuals are changes/returns ("change") or raw levels ("level").
        - "change": Data already represents differences (recommended)
        - "level": Will raise ValueError; convert to changes first

    Returns
    -------
    float
        Move threshold

    Raises
    ------
    ValueError
        If actuals is empty, percentile invalid, or target_mode="level"

    Notes
    -----
    CRITICAL: The threshold MUST be computed from training data only
    to prevent regime threshold leakage (BUG-003 in myga-forecasting).

    Using 70th percentile means ~30% of historical changes are "moves"
    and ~70% are "flat". This provides a meaningful signal-to-noise ratio.

    **Why target_mode="level" raises an error:**
    Persistence metrics assume the baseline predicts "no change" (zero).
    This only makes sense when data represents changes/returns, not raw levels.
    If you have level data, convert to changes first:
    ``changes = np.diff(levels)``

    Examples
    --------
    >>> train_actuals = np.array([-0.1, -0.05, 0.0, 0.02, 0.05, 0.1])
    >>> threshold = compute_move_threshold(train_actuals, percentile=70)
    >>> print(f"Threshold: {threshold:.4f}")

    See Also
    --------
    compute_move_conditional_metrics : Main MC-SS computation using threshold.
    classify_moves : Classify values into UP/DOWN/FLAT using threshold.
    """
    if target_mode == "level":
        raise ValueError(
            "target_mode='level' not supported for persistence metrics. "
            "Persistence baseline assumes data represents changes/returns. "
            "Convert levels to changes first: changes = np.diff(levels)"
        )

    actuals = np.asarray(actuals)

    if len(actuals) == 0:
        raise ValueError("Cannot compute threshold from empty array")

    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")

    return float(np.percentile(np.abs(actuals), percentile))


def classify_moves(
    values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Classify values into UP, DOWN, FLAT categories.

    Parameters
    ----------
    values : np.ndarray
        Values to classify (typically actuals or predictions)
    threshold : float
        Threshold for flat classification

    Returns
    -------
    np.ndarray
        Array of MoveDirection enums

    Raises
    ------
    ValueError
        If threshold is negative

    Examples
    --------
    >>> values = np.array([0.1, -0.1, 0.02, -0.02, 0.0])
    >>> moves = classify_moves(values, threshold=0.05)
    >>> print([m.value for m in moves])  # ['up', 'down', 'flat', 'flat', 'flat']
    """
    values = np.asarray(values)

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    classifications: list[MoveDirection] = []
    for v in values:
        if v > threshold:
            classifications.append(MoveDirection.UP)
        elif v < -threshold:
            classifications.append(MoveDirection.DOWN)
        else:
            classifications.append(MoveDirection.FLAT)

    result: np.ndarray = np.array(classifications)
    return result


def _compute_mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute MAE (internal helper)."""
    if len(predictions) == 0:
        return float("nan")
    return float(np.mean(np.abs(actuals - predictions)))


def _get_scale_aware_epsilon(values: np.ndarray) -> float:
    """
    Compute scale-aware epsilon for division safety.

    Uses median absolute value to determine appropriate epsilon,
    avoiding issues with very small magnitude data where fixed
    thresholds like 1e-10 may be inappropriate.

    Parameters
    ----------
    values : np.ndarray
        Array of values to determine scale from

    Returns
    -------
    float
        Scale-appropriate epsilon (min 1e-10)
    """
    nonzero = values[values != 0]
    if len(nonzero) > 0:
        scale = float(np.median(np.abs(nonzero)))
        return max(1e-10, scale * 1e-8)
    return 1e-10


def compute_move_conditional_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    threshold: Optional[float] = None,
    threshold_percentile: float = 70.0,
    target_mode: TargetMode = "change",
) -> MoveConditionalResult:
    """
    Compute move-conditional evaluation metrics.

    Evaluates model performance separately for:
    - UP moves: actual > threshold
    - DOWN moves: actual < -threshold
    - FLAT periods: |actual| <= threshold

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions (should be predicted changes, not levels)
    actuals : np.ndarray
        Actual values (should be actual changes, not levels)
    threshold : float, optional
        Move threshold. If None, computed from actuals (NOT recommended
        for walk-forward; use training data threshold instead).
    threshold_percentile : float, default=70.0
        Percentile for threshold if computed from data
    target_mode : {"change", "level"}, default="change"
        Whether data represents changes/returns ("change") or raw levels ("level").
        - "change": Data already represents differences (recommended)
        - "level": Will raise ValueError; convert to changes first

    Returns
    -------
    MoveConditionalResult
        Move-conditional metrics including MC-SS

    Raises
    ------
    ValueError
        If arrays have different lengths, contain NaN, or target_mode="level"

    Notes
    -----
    MC-SS (Move-Conditional Skill Score) formula:
        MC-SS = 1 - (model_mae_on_moves / persistence_mae_on_moves)

    Where:
    - model_mae_on_moves = MAE of predictions on UP and DOWN only
    - persistence_mae_on_moves = mean(|actual|) on UP and DOWN only
      (because persistence predicts 0, its error equals |actual|)

    CRITICAL: For walk-forward evaluation, `threshold` should be computed
    from training data only to prevent leakage.

    **Why target_mode="level" raises an error:**
    The persistence baseline assumes predicting "no change" (zero).
    This only makes sense when data represents changes/returns.
    If you have level data, convert first:
    ``changes = np.diff(levels)``

    Examples
    --------
    >>> # Compute threshold from training
    >>> threshold = compute_move_threshold(train_actuals)
    >>>
    >>> # Evaluate on test
    >>> mc = compute_move_conditional_metrics(preds, actuals, threshold=threshold)
    >>> print(f"MC-SS: {mc.skill_score:.3f}")
    """
    if target_mode == "level":
        raise ValueError(
            "target_mode='level' not supported for persistence metrics. "
            "Persistence baseline assumes data represents changes/returns. "
            "Convert levels to changes first: changes = np.diff(levels)"
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
            f"Arrays must have same length. "
            f"predictions: {len(predictions)}, actuals: {len(actuals)}"
        )

    if len(predictions) == 0:
        return MoveConditionalResult(
            mae_up=float("nan"),
            mae_down=float("nan"),
            mae_flat=float("nan"),
            n_up=0,
            n_down=0,
            n_flat=0,
            skill_score=float("nan"),
            move_threshold=0.0,
        )

    # Compute or use provided threshold
    if threshold is None:
        threshold = compute_move_threshold(actuals, threshold_percentile)

    # Classify moves based on ACTUALS
    classifications = classify_moves(actuals, threshold)

    # Create masks
    up_mask = np.array([c == MoveDirection.UP for c in classifications])
    down_mask = np.array([c == MoveDirection.DOWN for c in classifications])
    flat_mask = np.array([c == MoveDirection.FLAT for c in classifications])

    # Counts
    n_up = int(np.sum(up_mask))
    n_down = int(np.sum(down_mask))
    n_flat = int(np.sum(flat_mask))

    # Conditional MAEs with warnings for empty subsets
    if n_up > 0:
        mae_up = _compute_mae(predictions[up_mask], actuals[up_mask])
    else:
        warnings.warn(
            "No UP moves in sample (all values at or below threshold). "
            f"mae_up will be NaN. Total samples: {len(actuals)}, "
            f"threshold: {threshold:.4g}. Consider lowering the move threshold.",
            UserWarning,
            stacklevel=2,
        )
        mae_up = float("nan")

    if n_down > 0:
        mae_down = _compute_mae(predictions[down_mask], actuals[down_mask])
    else:
        warnings.warn(
            "No DOWN moves in sample (all values at or above negative threshold). "
            f"mae_down will be NaN. Total samples: {len(actuals)}, "
            f"threshold: {threshold:.4g}. Consider lowering the move threshold.",
            UserWarning,
            stacklevel=2,
        )
        mae_down = float("nan")

    if n_flat > 0:
        mae_flat = _compute_mae(predictions[flat_mask], actuals[flat_mask])
    else:
        warnings.warn(
            "No FLAT periods in sample (all values outside threshold). "
            f"mae_flat will be NaN. Total samples: {len(actuals)}, "
            f"threshold: {threshold:.4g}. This is unusual - check threshold.",
            UserWarning,
            stacklevel=2,
        )
        mae_flat = float("nan")

    # Compute MC-SS on moves only (UP + DOWN)
    move_mask = up_mask | down_mask
    n_moves = n_up + n_down

    if n_moves > 0:
        # Model MAE on moves
        model_mae_moves = _compute_mae(predictions[move_mask], actuals[move_mask])

        # Persistence MAE on moves
        # Persistence predicts 0, so error = |actual|
        persistence_mae_moves = float(np.mean(np.abs(actuals[move_mask])))

        # Guard against division by zero (scale-aware epsilon)
        epsilon = _get_scale_aware_epsilon(actuals[move_mask])
        if persistence_mae_moves > epsilon:
            skill_score = 1.0 - (model_mae_moves / persistence_mae_moves)
        else:
            warnings.warn(
                "Persistence MAE on moves is near zero (all moves are negligibly small). "
                f"skill_score will be NaN. persistence_mae_moves={persistence_mae_moves:.4e}, "
                f"epsilon={epsilon:.4e}. Consider raising the move threshold.",
                UserWarning,
                stacklevel=2,
            )
            skill_score = float("nan")
    else:
        warnings.warn(
            "No moves (UP or DOWN) in sample - all observations are FLAT. "
            f"skill_score will be NaN. Total samples: {len(actuals)}, n_flat: {n_flat}. "
            "Consider lowering the move threshold or checking data scale.",
            UserWarning,
            stacklevel=2,
        )
        skill_score = float("nan")

    return MoveConditionalResult(
        mae_up=mae_up,
        mae_down=mae_down,
        mae_flat=mae_flat,
        n_up=n_up,
        n_down=n_down,
        n_flat=n_flat,
        skill_score=skill_score,
        move_threshold=threshold,
    )


def compute_direction_accuracy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    move_threshold: Optional[float] = None,
) -> float:
    """
    Compute directional accuracy.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    actuals : np.ndarray
        Actual values
    move_threshold : float, optional
        If provided, uses 3-class (UP/DOWN/FLAT) comparison.
        If None, uses 2-class (positive/negative sign) comparison.

    Returns
    -------
    float
        Direction accuracy as fraction (0-1)

    Notes
    -----
    **Without threshold (2-class)**:
    - Compares signs: both positive OR both negative = correct
    - Zero actuals are excluded
    - Persistence (predicts 0) gets 0% accuracy

    **With threshold (3-class)**:
    - UP: value > threshold
    - DOWN: value < -threshold
    - FLAT: |value| <= threshold
    - Correct if both have same class (including both FLAT)
    - Persistence (predicts 0 = FLAT) gets credit when actual is also FLAT

    The 3-class version provides a meaningful baseline for persistence model
    comparison. Without it, persistence trivially gets 0% making all
    comparisons "significant".

    Examples
    --------
    >>> # 2-class (sign-based)
    >>> acc = compute_direction_accuracy(preds, actuals)
    >>>
    >>> # 3-class (with threshold)
    >>> threshold = compute_move_threshold(train_actuals)
    >>> acc = compute_direction_accuracy(preds, actuals, move_threshold=threshold)

    See Also
    --------
    pt_test : Statistical test for directional accuracy significance.
    compute_move_conditional_metrics : Full move-conditional evaluation.
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Arrays must have same length. "
            f"predictions: {len(predictions)}, actuals: {len(actuals)}"
        )

    if len(predictions) == 0:
        return 0.0

    if move_threshold is not None:
        # 3-class comparison
        pred_dirs = classify_moves(predictions, move_threshold)
        actual_dirs = classify_moves(actuals, move_threshold)

        correct = np.array([p == a for p, a in zip(pred_dirs, actual_dirs)])
        return float(np.mean(correct))

    # 2-class (sign) comparison
    # Exclude near-zero actuals to avoid floating-point issues
    epsilon = 1e-10
    nonzero_mask = np.abs(actuals) > epsilon

    if np.sum(nonzero_mask) == 0:
        return 0.0

    pred_signs = np.sign(predictions[nonzero_mask])
    actual_signs = np.sign(actuals[nonzero_mask])

    return float(np.mean(pred_signs == actual_signs))


def compute_move_only_mae(
    predictions: np.ndarray,
    actuals: np.ndarray,
    threshold: float,
) -> Tuple[float, int]:
    """
    Compute MAE only on moves (excluding FLAT).

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions
    actuals : np.ndarray
        Actual values
    threshold : float
        Move threshold

    Returns
    -------
    tuple
        (mae, n_moves) - MAE on moves and count of moves

    Notes
    -----
    This isolates model performance on "significant" moves,
    excluding periods where nothing happened (FLAT).

    Examples
    --------
    >>> mae, n = compute_move_only_mae(preds, actuals, threshold=0.05)
    >>> if n >= 20:
    ...     print(f"Move-only MAE: {mae:.4f} (n={n})")
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    if len(predictions) != len(actuals):
        raise ValueError(
            f"Arrays must have same length. "
            f"predictions: {len(predictions)}, actuals: {len(actuals)}"
        )

    # Identify moves (UP or DOWN, not FLAT)
    move_mask = np.abs(actuals) > threshold
    n_moves = int(np.sum(move_mask))

    if n_moves == 0:
        return float("nan"), 0

    mae = _compute_mae(predictions[move_mask], actuals[move_mask])
    return mae, n_moves


def compute_persistence_mae(
    actuals: np.ndarray,
    threshold: Optional[float] = None,
) -> float:
    """
    Compute MAE of persistence baseline.

    Persistence predicts 0 (no change), so MAE = mean(|actual|).

    Parameters
    ----------
    actuals : np.ndarray
        Actual values
    threshold : float, optional
        If provided, computes MAE only on moves

    Returns
    -------
    float
        Persistence baseline MAE

    Examples
    --------
    >>> persistence_mae = compute_persistence_mae(actuals)
    >>> persistence_mae_moves = compute_persistence_mae(actuals, threshold=0.05)
    """
    actuals = np.asarray(actuals)

    if len(actuals) == 0:
        return float("nan")

    if threshold is not None:
        move_mask = np.abs(actuals) > threshold
        if np.sum(move_mask) == 0:
            return float("nan")
        return float(np.mean(np.abs(actuals[move_mask])))

    return float(np.mean(np.abs(actuals)))


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Type aliases
    "TargetMode",
    # Enums and dataclasses
    "MoveDirection",
    "MoveConditionalResult",
    # Functions
    "compute_move_threshold",
    "classify_moves",
    "compute_move_conditional_metrics",
    "compute_direction_accuracy",
    "compute_move_only_mae",
    "compute_persistence_mae",
]
