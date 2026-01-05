"""
Inference Module.

Statistical inference tools for cross-validation results.

This module provides bootstrap-based inference for test statistics
computed across CV folds, particularly useful when standard asymptotic
inference is unreliable due to few folds.

Knowledge Tiers
---------------
[T1] Moving Block Bootstrap (Kunsch 1989, Politis & Romano 1994)
[T1] Wild bootstrap theory (Wu 1986, Liu 1988, Mammen 1993)
[T2] Application to CV folds assumes approximate independence
"""

from __future__ import annotations

from temporalcv.inference.block_bootstrap_ci import (
    BlockBootstrapResult,
    bootstrap_ci_mae,
    bootstrap_ci_mean,
    compute_block_length,
    moving_block_bootstrap,
)
from temporalcv.inference.wild_bootstrap import (
    WildBootstrapResult,
    wild_cluster_bootstrap,
)

__all__ = [
    # Block bootstrap
    "BlockBootstrapResult",
    "compute_block_length",
    "moving_block_bootstrap",
    "bootstrap_ci_mean",
    "bootstrap_ci_mae",
    # Wild bootstrap
    "WildBootstrapResult",
    "wild_cluster_bootstrap",
]
