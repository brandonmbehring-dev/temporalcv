"""
Inference Module.

Statistical inference tools for cross-validation results.

This module provides bootstrap-based inference for test statistics
computed across CV folds, particularly useful when standard asymptotic
inference is unreliable due to few folds.

Knowledge Tier: [T2] - Wild bootstrap is established, but CV fold
independence assumption requires domain-specific validation.
"""

from __future__ import annotations

from temporalcv.inference.wild_bootstrap import (
    WildBootstrapResult,
    wild_cluster_bootstrap,
)

__all__ = [
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
]
