"""
Event-Aware Metrics Module.

Novel metrics for direction prediction with proper calibration.

Key concepts:
- **Brier score**: Probabilistic direction accuracy (2 and 3-class)
- **PR-AUC**: Area under precision-recall curve (for imbalanced classes)
- **Direction Brier**: Brier with confidence calibration

Note: MC-SS and move-only MAE already exist in persistence.py.

Example
-------
>>> from temporalcv.metrics.event import (
...     compute_direction_brier,
...     compute_pr_auc,
...     compute_calibrated_direction_brier,
... )
>>>
>>> # Basic Brier score for direction
>>> brier = compute_direction_brier(pred_probs, actual_directions)
>>>
>>> # PR-AUC for imbalanced UP/DOWN classification
>>> prauc = compute_pr_auc(pred_probs_up, actual_up)

References
----------
- Brier (1950). Verification of forecasts expressed in terms of probability.
- Murphy (1973). A new vector partition of the probability score.
- Davis & Goadrich (2006). The relationship between PR and ROC curves.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


class UndefinedMetricWarning(UserWarning):
    """
    Warning for mathematically undefined metrics.

    Raised when a metric cannot be computed meaningfully due to
    degenerate input (e.g., all samples belong to one class).
    Similar to sklearn.exceptions.UndefinedMetricWarning.
    """

    pass


@dataclass
class BrierScoreResult:
    """
    Result from Brier score computation.

    Attributes
    ----------
    brier_score : float
        Mean squared error of probability forecasts (0 = perfect, 1 = worst)
    reliability : float
        Calibration component (lower is better)
    resolution : float
        Refinement component (higher is better)
    uncertainty : float
        Base rate uncertainty
    n_samples : int
        Number of samples
    n_classes : int
        Number of classes (2 or 3)
    """

    brier_score: float
    reliability: float
    resolution: float
    uncertainty: float
    n_samples: int
    n_classes: int

    @property
    def skill_score(self) -> float:
        """
        Brier skill score relative to climatology.

        Returns
        -------
        float
            BSS = 1 - (BS / uncertainty)
            Positive values indicate skill over climatology.
        """
        if self.uncertainty == 0 or np.isnan(self.uncertainty):
            return 0.0
        return 1.0 - (self.brier_score / self.uncertainty)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "brier_score": self.brier_score,
            "reliability": self.reliability,
            "resolution": self.resolution,
            "uncertainty": self.uncertainty,
            "skill_score": self.skill_score,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
        }


@dataclass
class PRAUCResult:
    """
    Result from PR-AUC computation.

    Attributes
    ----------
    pr_auc : float
        Area under precision-recall curve
    baseline : float
        Baseline PR-AUC (positive class rate)
    precision_at_50_recall : float
        Precision at 50% recall
    n_positive : int
        Number of positive samples
    n_negative : int
        Number of negative samples
    """

    pr_auc: float
    baseline: float
    precision_at_50_recall: float
    n_positive: int
    n_negative: int

    @property
    def lift_over_baseline(self) -> float:
        """
        PR-AUC lift over baseline.

        Returns
        -------
        float
            Ratio of PR-AUC to baseline (random classifier).
        """
        if self.baseline == 0 or np.isnan(self.baseline):
            return float("nan")
        return self.pr_auc / self.baseline

    @property
    def n_total(self) -> int:
        """Total number of samples."""
        return self.n_positive + self.n_negative

    @property
    def imbalance_ratio(self) -> float:
        """
        Class imbalance ratio.

        Returns
        -------
        float
            Ratio of majority to minority class.
        """
        if self.n_positive == 0 or self.n_negative == 0:
            return float("inf")
        return max(self.n_positive, self.n_negative) / min(self.n_positive, self.n_negative)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "pr_auc": self.pr_auc,
            "baseline": self.baseline,
            "lift_over_baseline": self.lift_over_baseline,
            "precision_at_50_recall": self.precision_at_50_recall,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_total": self.n_total,
            "imbalance_ratio": self.imbalance_ratio,
        }


def _compute_brier_decomposition(
    pred_probs: np.ndarray,
    actual_binary: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float, float]:
    """
    Compute Brier score decomposition (Murphy 1973).

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities for positive class (1D)
    actual_binary : np.ndarray
        Actual binary outcomes (0 or 1)
    n_bins : int, default=10
        Number of probability bins

    Returns
    -------
    tuple[float, float, float]
        (reliability, resolution, uncertainty)

    Notes
    -----
    Murphy (1973) decomposition:
        BS = Reliability - Resolution + Uncertainty

    Where:
        Reliability = (1/N) * sum_k n_k * (f_k - o_k)^2  (calibration error)
        Resolution = (1/N) * sum_k n_k * (o_k - o_bar)^2  (refinement)
        Uncertainty = o_bar * (1 - o_bar)  (inherent uncertainty)

    f_k = mean forecast probability in bin k
    o_k = observed frequency (positive rate) in bin k
    o_bar = overall positive rate
    n_k = number of samples in bin k
    """
    n = len(pred_probs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    # Overall positive rate
    o_bar = float(np.mean(actual_binary))

    # Uncertainty is base rate variance
    uncertainty = o_bar * (1 - o_bar)

    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pred_probs, bins) - 1
    # Clip to valid range [0, n_bins-1]
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        n_b = mask.sum()

        if n_b == 0:
            continue

        # Mean forecast probability in bin
        f_b = float(np.mean(pred_probs[mask]))
        # Observed positive rate in bin
        o_b = float(np.mean(actual_binary[mask]))

        # Reliability contribution: weighted squared calibration error
        reliability += n_b * (f_b - o_b) ** 2
        # Resolution contribution: weighted squared deviation from climatology
        resolution += n_b * (o_b - o_bar) ** 2

    # Normalize by total samples
    reliability /= n
    resolution /= n

    return reliability, resolution, uncertainty


def compute_direction_brier(
    pred_probs: np.ndarray,
    actual_directions: np.ndarray,
    n_classes: Literal[2, 3] = 2,
) -> BrierScoreResult:
    """
    Compute Brier score for direction prediction.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities. Shape depends on n_classes:
        - n_classes=2: (n_samples,) probability of positive direction
        - n_classes=3: (n_samples, 3) probabilities for [DOWN, FLAT, UP]
    actual_directions : np.ndarray
        Actual directions as integers:
        - n_classes=2: 0 = negative, 1 = positive
        - n_classes=3: 0 = DOWN, 1 = FLAT, 2 = UP

    Returns
    -------
    BrierScoreResult
        Brier score with decomposition

    Raises
    ------
    ValueError
        If arrays have mismatched shapes or invalid values

    Notes
    -----
    Brier score = (1/N) * sum((p_i - o_i)^2)

    where p_i is predicted probability and o_i is actual outcome (0 or 1).

    For 3-class, we compute multiclass Brier:
    BS = (1/N) * sum_i sum_k (p_ik - o_ik)^2

    Examples
    --------
    >>> # 2-class: predict probability of UP
    >>> probs = np.array([0.7, 0.3, 0.8, 0.2])  # P(UP)
    >>> actuals = np.array([1, 0, 1, 0])  # 1=UP, 0=DOWN
    >>> result = compute_direction_brier(probs, actuals, n_classes=2)
    >>> print(f"Brier: {result.brier_score:.4f}")

    >>> # 3-class: predict probability vector [P(DOWN), P(FLAT), P(UP)]
    >>> probs_3 = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    >>> actuals_3 = np.array([2, 0])  # 2=UP, 0=DOWN
    >>> result_3 = compute_direction_brier(probs_3, actuals_3, n_classes=3)
    """
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    actual_directions = np.asarray(actual_directions)

    n = len(actual_directions)
    if n == 0:
        return BrierScoreResult(
            brier_score=float("nan"),
            reliability=float("nan"),
            resolution=float("nan"),
            uncertainty=float("nan"),
            n_samples=0,
            n_classes=n_classes,
        )

    if n_classes == 2:
        # Binary Brier score
        if pred_probs.ndim > 1:
            raise ValueError(
                f"For n_classes=2, pred_probs should be 1D (P(positive)). "
                f"Got shape {pred_probs.shape}"
            )

        if len(pred_probs) != n:
            raise ValueError(
                f"Length mismatch: pred_probs has {len(pred_probs)}, "
                f"actual_directions has {n}"
            )

        # Validate probability range
        if np.any(pred_probs < 0) or np.any(pred_probs > 1):
            raise ValueError("Probabilities must be in [0, 1]")

        # Convert to float for Brier calculation
        actual_onehot: np.ndarray = actual_directions.astype(np.float64)
        brier = float(np.mean((pred_probs - actual_onehot) ** 2))

        # Decomposition (Murphy 1973)
        base_rate = float(np.mean(actual_onehot))
        uncertainty = base_rate * (1 - base_rate)

        # Murphy (1973) binned decomposition
        reliability, resolution, _ = _compute_brier_decomposition(
            pred_probs, actual_onehot, n_bins=10
        )

    else:  # n_classes == 3
        if pred_probs.ndim != 2 or pred_probs.shape[1] != 3:
            raise ValueError(
                f"For n_classes=3, pred_probs should be (n_samples, 3). "
                f"Got shape {pred_probs.shape}"
            )

        if pred_probs.shape[0] != n:
            raise ValueError(
                f"Length mismatch: pred_probs has {pred_probs.shape[0]} samples, "
                f"actual_directions has {n}"
            )

        # Validate probability sums
        prob_sums = np.sum(pred_probs, axis=1)
        if not np.allclose(prob_sums, 1.0, atol=1e-6):
            raise ValueError(
                f"Probability vectors must sum to 1.0. "
                f"Got sums ranging from {prob_sums.min():.4f} to {prob_sums.max():.4f}"
            )

        # One-hot encode actuals
        actual_onehot = np.zeros((n, 3), dtype=np.float64)
        for i, a in enumerate(actual_directions):
            if not 0 <= a <= 2:
                raise ValueError(
                    f"For n_classes=3, actual_directions must be 0, 1, or 2. "
                    f"Got {a} at index {i}"
                )
            actual_onehot[i, int(a)] = 1.0

        # Multiclass Brier score
        brier = float(np.mean(np.sum((pred_probs - actual_onehot) ** 2, axis=1)))

        # Base rates per class
        class_rates = np.mean(actual_onehot, axis=0)
        uncertainty = float(np.sum(class_rates * (1 - class_rates)))

        # Murphy (1973) decomposition per class (one-vs-rest), then aggregate
        reliability = 0.0
        resolution = 0.0
        for k in range(3):
            rel_k, res_k, _ = _compute_brier_decomposition(
                pred_probs[:, k], actual_onehot[:, k], n_bins=10
            )
            reliability += rel_k
            resolution += res_k

    return BrierScoreResult(
        brier_score=brier,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        n_samples=n,
        n_classes=n_classes,
    )


def compute_pr_auc(
    pred_probs: np.ndarray,
    actual_binary: np.ndarray,
) -> PRAUCResult:
    """
    Compute Area Under Precision-Recall Curve.

    PR-AUC is preferred over ROC-AUC for imbalanced classification,
    which is common in direction prediction (moves are rare).

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities of positive class
    actual_binary : np.ndarray
        Actual binary labels (0 or 1)

    Returns
    -------
    PRAUCResult
        PR-AUC with baseline comparison

    Raises
    ------
    ValueError
        If arrays have different lengths or invalid values

    Notes
    -----
    For direction classification where moves (UP/DOWN) are rare
    compared to FLAT periods, PR-AUC provides a more informative
    metric than accuracy or ROC-AUC.

    Baseline PR-AUC equals the positive class rate (random classifier).

    **Difference from sklearn.metrics.average_precision_score**:
    This implementation uses trapezoidal integration, which can give
    slightly different values from sklearn's average_precision_score
    (which uses step function integration). For jagged PR curves,
    differences can be up to a few percentage points. For
    sklearn-compatible behavior, use:

        from sklearn.metrics import average_precision_score
        ap = average_precision_score(actual_binary, pred_probs)

    Examples
    --------
    >>> probs = np.array([0.9, 0.8, 0.3, 0.1, 0.7])
    >>> actuals = np.array([1, 1, 0, 0, 1])
    >>> result = compute_pr_auc(probs, actuals)
    >>> print(f"PR-AUC: {result.pr_auc:.3f} (baseline: {result.baseline:.3f})")

    See Also
    --------
    compute_direction_brier : Brier score for probabilistic forecasts.
    compute_move_conditional_metrics : MC-SS for high-persistence series.
    """
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    actual_binary = np.asarray(actual_binary)

    if len(pred_probs) != len(actual_binary):
        raise ValueError(
            f"Length mismatch: pred_probs has {len(pred_probs)}, "
            f"actual_binary has {len(actual_binary)}"
        )

    n_positive = int(np.sum(actual_binary))
    n_negative = len(actual_binary) - n_positive

    # Handle degenerate cases with explicit warnings (NEVER FAIL SILENTLY)
    if n_positive == 0:
        warnings.warn(
            "PR-AUC undefined: no positive samples. "
            "Returning 0.0 as skill-equivalent fallback.",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return PRAUCResult(
            pr_auc=0.0,
            baseline=0.0,
            precision_at_50_recall=float("nan"),
            n_positive=0,
            n_negative=n_negative,
        )

    if n_negative == 0:
        warnings.warn(
            "PR-AUC degenerate: all positive samples. "
            "Returning 1.0 (trivially correct).",
            UndefinedMetricWarning,
            stacklevel=2,
        )
        return PRAUCResult(
            pr_auc=1.0,
            baseline=1.0,
            precision_at_50_recall=1.0,
            n_positive=n_positive,
            n_negative=0,
        )

    # Sort by predicted probability descending
    sorted_idx = np.argsort(-pred_probs)
    sorted_actuals = actual_binary[sorted_idx]

    # Compute precision-recall curve
    tp_cumsum = np.cumsum(sorted_actuals)
    n_pred_positive = np.arange(1, len(sorted_actuals) + 1)

    precision = tp_cumsum / n_pred_positive
    recall = tp_cumsum / n_positive

    # Baseline is positive class rate (random classifier precision)
    baseline = n_positive / (n_positive + n_negative)

    # Compute AUC using trapezoidal rule
    # Add (0, baseline) point - at 0% recall, precision is the baseline rate
    # NOT (0, 1) which would overestimate PR-AUC
    recall_extended = np.concatenate([[0], recall])
    precision_extended = np.concatenate([[baseline], precision])

    # Use trapezoidal integration (recall is ascending, so positive result)
    pr_auc = float(np.trapezoid(precision_extended, recall_extended))

    # Precision at 50% recall
    recall_50_idx = np.argmin(np.abs(recall - 0.5))
    precision_at_50 = float(precision[recall_50_idx])

    return PRAUCResult(
        pr_auc=pr_auc,
        baseline=baseline,
        precision_at_50_recall=precision_at_50,
        n_positive=n_positive,
        n_negative=n_negative,
    )


def compute_calibrated_direction_brier(
    pred_probs: np.ndarray,
    actual_directions: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Brier score with reliability diagram data.

    Returns Brier score along with binned calibration data
    for plotting reliability diagrams.

    Parameters
    ----------
    pred_probs : np.ndarray
        Predicted probabilities (1D, for positive class)
    actual_directions : np.ndarray
        Actual binary outcomes (0 or 1)
    n_bins : int, default=10
        Number of bins for calibration

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (brier_score, bin_means, bin_true_fractions)
        NaN values indicate empty bins.

    Raises
    ------
    ValueError
        If arrays have different lengths or invalid values

    Notes
    -----
    Reliability diagrams show calibration:
    - X-axis: Mean predicted probability in each bin
    - Y-axis: Fraction of positives in each bin
    - Perfect calibration: diagonal line

    Examples
    --------
    >>> brier, bin_means, bin_fracs = compute_calibrated_direction_brier(
    ...     pred_probs, actuals, n_bins=10
    ... )
    >>> # Plot reliability diagram
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(bin_means, bin_fracs, 'o-')
    >>> plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration
    """
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    actual_directions = np.asarray(actual_directions, dtype=np.float64)

    if len(pred_probs) != len(actual_directions):
        raise ValueError(
            f"Length mismatch: pred_probs has {len(pred_probs)}, "
            f"actual_directions has {len(actual_directions)}"
        )

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    if len(pred_probs) == 0:
        return float("nan"), np.array([]), np.array([])

    # Compute Brier score
    brier = float(np.mean((pred_probs - actual_directions) ** 2))

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = np.full(n_bins, np.nan)
    bin_true_fractions = np.full(n_bins, np.nan)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
        else:
            # Include right edge for last bin
            mask = (pred_probs >= bin_edges[i]) & (pred_probs <= bin_edges[i + 1])

        if np.sum(mask) > 0:
            bin_means[i] = float(np.mean(pred_probs[mask]))
            bin_true_fractions[i] = float(np.mean(actual_directions[mask]))

    return brier, bin_means, bin_true_fractions


def convert_predictions_to_direction_probs(
    point_predictions: np.ndarray,
    prediction_std: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Convert point predictions with uncertainty to direction probabilities.

    Useful for converting regression predictions to probabilistic
    direction forecasts for Brier score computation.

    Parameters
    ----------
    point_predictions : np.ndarray
        Point predictions (typically changes)
    prediction_std : np.ndarray
        Prediction standard deviation (from ensemble or conformal)
    threshold : float, default=0.0
        Threshold for UP/DOWN classification

    Returns
    -------
    np.ndarray
        Probability of UP direction (value > threshold)

    Raises
    ------
    ValueError
        If arrays have different lengths or negative std values

    Notes
    -----
    Assumes Gaussian prediction distribution:
    P(UP) = P(X > threshold) = 1 - Phi((threshold - mean) / std)

    Examples
    --------
    >>> from temporalcv import TimeSeriesBagger
    >>> mean, std = bagger.predict_with_uncertainty(X_test)
    >>> p_up = convert_predictions_to_direction_probs(mean, std, threshold=0.01)
    """
    from scipy import stats

    point_predictions = np.asarray(point_predictions)
    prediction_std = np.asarray(prediction_std)

    if len(point_predictions) != len(prediction_std):
        raise ValueError(
            f"Length mismatch: point_predictions has {len(point_predictions)}, "
            f"prediction_std has {len(prediction_std)}"
        )

    if np.any(prediction_std < 0):
        raise ValueError("prediction_std must be non-negative")

    # Handle zero std (deterministic predictions)
    std_safe = np.maximum(prediction_std, 1e-10)

    # P(X > threshold) using normal CDF
    z_scores = (threshold - point_predictions) / std_safe
    p_up: np.ndarray = 1.0 - stats.norm.cdf(z_scores)

    return p_up


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Warnings
    "UndefinedMetricWarning",
    # Dataclasses
    "BrierScoreResult",
    "PRAUCResult",
    # Functions
    "compute_direction_brier",
    "compute_pr_auc",
    "compute_calibrated_direction_brier",
    "convert_predictions_to_direction_probs",
]
