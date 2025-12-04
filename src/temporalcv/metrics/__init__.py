"""
Event-Aware Metrics Subpackage.

Novel metrics for direction prediction with proper calibration,
complementing the move-conditional metrics in persistence.py.

Key components:
- **Brier score**: Probabilistic direction accuracy (2 and 3-class)
- **PR-AUC**: Area under precision-recall curve (for imbalanced classes)
- **Direction Brier**: Brier with confidence calibration

Example
-------
>>> from temporalcv.metrics import (
...     compute_direction_brier,
...     compute_pr_auc,
...     BrierScoreResult,
...     PRAUCResult,
... )
>>>
>>> # Brier score for direction probabilities
>>> result = compute_direction_brier(pred_probs, actual_directions)
>>> print(f"Brier: {result.brier_score:.4f}")
>>>
>>> # PR-AUC for imbalanced UP/DOWN classification
>>> prauc = compute_pr_auc(pred_probs_up, actual_up)
>>> print(f"PR-AUC: {prauc.pr_auc:.3f} (baseline: {prauc.baseline:.3f})")

References
----------
- Brier (1950). Verification of forecasts expressed in terms of probability.
- Murphy (1973). A new vector partition of the probability score.
- Davis & Goadrich (2006). The relationship between PR and ROC curves.
"""

from temporalcv.metrics.event import (
    BrierScoreResult,
    PRAUCResult,
    compute_calibrated_direction_brier,
    compute_direction_brier,
    compute_pr_auc,
    convert_predictions_to_direction_probs,
)

__all__ = [
    # Dataclasses
    "BrierScoreResult",
    "PRAUCResult",
    # Functions
    "compute_direction_brier",
    "compute_pr_auc",
    "compute_calibrated_direction_brier",
    "convert_predictions_to_direction_probs",
]
