"""Library-wide eq/hash invariant: every array-bearing result object uses identity eq/hash.

A ``@dataclass(frozen=True)`` that carries an ``np.ndarray`` field must declare ``eq=False``.
Otherwise the generated ``__eq__`` raises ``ValueError`` (ambiguous truth value of an array)
and the generated ``__hash__`` raises ``TypeError`` (ndarray is unhashable). ``eq=False``
restores safe object-identity semantics, so ``==`` and ``hash()`` never raise.

This is the single canonical home for that invariant. When a new array-bearing result object
is added to the library, add a factory for it to ``RESULT_FACTORIES`` below.

History: the diagnostics/inference/financial five fixed/tested here in M1. (The cv-result
trio was first fixed in R1, 5f35882.) Result objects with no array fields — e.g.
``SplitInfo`` — keep value eq/hash and are covered in ``test_cv_result_objects.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from temporalcv.cv_financial import PurgedSplit
from temporalcv.diagnostics.influence import InfluenceDiagnostic
from temporalcv.diagnostics.sensitivity import GapSensitivityResult
from temporalcv.inference.block_bootstrap_ci import BlockBootstrapResult
from temporalcv.inference.wild_bootstrap import WildBootstrapResult


def _purged_split() -> PurgedSplit:
    return PurgedSplit(
        train_indices=np.array([0, 1, 2]),
        test_indices=np.array([5, 6]),
        n_purged=2,
        n_embargoed=1,
    )


def _influence_diagnostic() -> InfluenceDiagnostic:
    return InfluenceDiagnostic(
        observation_influence=np.array([0.1, 0.2, 0.3]),
        observation_high_mask=np.array([False, False, True]),
        block_influence=np.array([0.15, 0.25]),
        block_high_mask=np.array([False, True]),
        block_indices=[(0, 2), (2, 3)],
        n_high_influence_obs=1,
        n_high_influence_blocks=1,
        influence_threshold=2.0,
    )


def _gap_sensitivity_result() -> GapSensitivityResult:
    return GapSensitivityResult(
        gap_values=np.array([0, 1, 2]),
        metrics=np.array([0.5, 0.55, 0.6]),
        metric_name="mae",
        break_even_gap=1,
        sensitivity_score=0.08,
        degradation_threshold=0.1,
        baseline_metric=0.5,
        baseline_gap=0,
    )


def _block_bootstrap_result() -> BlockBootstrapResult:
    return BlockBootstrapResult(
        estimate=1.0,
        ci_lower=0.8,
        ci_upper=1.2,
        alpha=0.05,
        std_error=0.1,
        n_bootstrap=100,
        block_length=5,
        bootstrap_distribution=np.array([0.9, 1.0, 1.1]),
    )


def _wild_bootstrap_result() -> WildBootstrapResult:
    return WildBootstrapResult(
        estimate=1.0,
        se=0.1,
        ci_lower=0.8,
        ci_upper=1.2,
        p_value=0.04,
        n_bootstrap=100,
        n_clusters=5,
        weight_type="rademacher",
        bootstrap_distribution=np.array([0.9, 1.0, 1.1]),
    )


# Every array-bearing result object in the library. Add new ones here.
RESULT_FACTORIES: list[tuple[str, Callable[[], object]]] = [
    ("PurgedSplit", _purged_split),
    ("InfluenceDiagnostic", _influence_diagnostic),
    ("GapSensitivityResult", _gap_sensitivity_result),
    ("BlockBootstrapResult", _block_bootstrap_result),
    ("WildBootstrapResult", _wild_bootstrap_result),
]


@pytest.mark.parametrize(
    "factory",
    [factory for _, factory in RESULT_FACTORIES],
    ids=[name for name, _ in RESULT_FACTORIES],
)
def test_identity_eq_hash_never_raises(factory: Callable[[], object]) -> None:
    """== and hash() resolve by identity and never raise on the np.ndarray fields."""
    a = factory()
    b = factory()
    assert a == a  # reflexive
    assert (a == b) is False  # distinct instances are unequal under identity eq
    assert isinstance(hash(a), int)
    assert len({a, b}) == 2  # hashable and distinct
