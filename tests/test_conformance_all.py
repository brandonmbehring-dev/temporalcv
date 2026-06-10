"""Canonical library-wide splitter conformance gate (A4 closeout).

EVERY splitter the library ships is classified here, exactly once:

- **Forward-only splitters** must pass ``check_temporal_splitter`` (the
  executable no-lookahead/determinism/disjointness contract) — including the
  purged forward-only ``PurgedWalkForward``, which previously had no
  conformance coverage at all.
- **Bidirectional purged K-folds** (``PurgedKFold``, ``CombinatorialPurgedCV``)
  are excluded BY DESIGN: they train on both sides of the test block (the de
  Prado evaluation setting), so they must FAIL the forward-only contract —
  and that failure is asserted, not assumed.
- **``NestedWalkForwardCV``** is a tuning meta-estimator, not a splitter
  (A1 decision); it does not satisfy the ``Splitter`` Protocol — asserted.

An exhaustiveness guard walks ``cv.py``/``cv_financial.py`` and fails the
moment a new splitter class ships without being classified here — the same
fail-loud pattern as ``test_result_objects.py::test_registry_is_exhaustive``.

(The strategy/adapter conformance counterpart is exhaustive in
``test_seam_vocab.py``: ``ALL_STRATEGIES`` / ``ALL_ADAPTERS``.)
"""

from __future__ import annotations

import inspect

import pytest
from sklearn.linear_model import LinearRegression

import temporalcv.cv
import temporalcv.cv_financial
from temporalcv import (
    BlockedTimeSeriesCV,
    CombinatorialPurgedCV,
    CrossFitCV,
    NestedWalkForwardCV,
    PurgedKFold,
    PurgedWalkForward,
    TimeSeriesCrossValidator,
    WalkForwardCV,
    check_temporal_splitter,
)
from temporalcv.protocols import Splitter

# ---------------------------------------------------------------------------
# The classified inventory (one entry per shipped splitter class)
# ---------------------------------------------------------------------------

FORWARD_ONLY_SPLITTERS = [
    WalkForwardCV(n_splits=3),
    WalkForwardCV(n_splits=4, window_type="sliding", window_size=20),
    TimeSeriesCrossValidator(n_splits=3),
    TimeSeriesCrossValidator(n_splits=3, gap=2, purge_length=1),
    BlockedTimeSeriesCV(n_splits=3),
    CrossFitCV(n_splits=3),
    CrossFitCV(n_splits=4, extra_gap=2),
    PurgedWalkForward(n_splits=3),
    PurgedWalkForward(n_splits=3, purge_gap=2, embargo_pct=0.02),
]

BIDIRECTIONAL_BY_DESIGN = [
    PurgedKFold(n_splits=4),
    CombinatorialPurgedCV(n_splits=5, n_test_splits=2),
]

# Classes deliberately NOT in either list above, with the reason.
NON_SPLITTER_CLASSES = {
    "NestedWalkForwardCV": "tuning meta-estimator, not a splitter (A1 decision)",
}


class TestForwardOnlyConformance:
    @pytest.mark.parametrize(
        "splitter",
        FORWARD_ONLY_SPLITTERS,
        ids=lambda s: f"{type(s).__name__}-{id(s) % 1000}",
    )
    def test_passes_temporal_splitter_contract(self, splitter: Splitter) -> None:
        check_temporal_splitter(splitter)  # must not raise

    def test_purged_walk_forward_satisfies_protocol(self) -> None:
        # The forward-only purged splitter is a first-class Splitter — the
        # Track-B migration target for dml_ts's purged_cv strategy (#23).
        assert isinstance(PurgedWalkForward(n_splits=3), Splitter)


class TestBidirectionalExclusions:
    @pytest.mark.parametrize("splitter", BIDIRECTIONAL_BY_DESIGN, ids=lambda s: type(s).__name__)
    def test_fails_forward_only_contract_as_designed(self, splitter: Splitter) -> None:
        # These purged K-folds train on BOTH sides of the test block (the
        # de Prado evaluation setting). The forward-only contract must
        # reject them on the no-lookahead invariant — loudly, so nobody
        # mistakes them for walk-forward splitters.
        with pytest.raises(AssertionError, match="lookahead"):
            check_temporal_splitter(splitter)

    def test_they_still_satisfy_the_structural_protocol(self) -> None:
        # Structurally they ARE Splitters (split/get_n_splits) — the
        # exclusion is behavioral (forward-only), not structural.
        for splitter in BIDIRECTIONAL_BY_DESIGN:
            assert isinstance(splitter, Splitter)


class TestNonSplitterExclusions:
    def test_nested_walk_forward_is_not_a_splitter(self) -> None:
        nested = NestedWalkForwardCV(
            estimator=LinearRegression(), param_grid={"fit_intercept": [True, False]}
        )
        assert not isinstance(nested, Splitter)


class TestInventoryIsExhaustive:
    """Fail the moment a new splitter ships without being classified here."""

    def test_every_shipped_splitter_class_is_classified(self) -> None:
        classified = {type(s).__name__ for s in FORWARD_ONLY_SPLITTERS}
        classified |= {type(s).__name__ for s in BIDIRECTIONAL_BY_DESIGN}

        # Detector: classes defined in the two CV modules exposing the
        # splitter surface (split + get_n_splits). NestedWalkForwardCV has
        # NEITHER (pure tuning meta-estimator), so it is not collected —
        # its exclusion is documented in NON_SPLITTER_CLASSES and verified
        # separately below.
        shipped: set[str] = set()
        for module in (temporalcv.cv, temporalcv.cv_financial):
            for name, obj in vars(module).items():
                if (
                    inspect.isclass(obj)
                    and obj.__module__ == module.__name__
                    and callable(getattr(obj, "split", None))
                    and callable(getattr(obj, "get_n_splits", None))
                ):
                    shipped.add(name)

        unclassified = shipped - classified
        assert not unclassified, (
            f"New splitter class(es) shipped without conformance classification: "
            f"{sorted(unclassified)}. Add each to FORWARD_ONLY_SPLITTERS (must pass "
            f"check_temporal_splitter), BIDIRECTIONAL_BY_DESIGN (must fail it, "
            f"asserted), or NON_SPLITTER_CLASSES (with a reason) in "
            f"tests/test_conformance_all.py."
        )
        # And nothing classified that no longer ships (stale inventory).
        stale = classified - shipped
        assert not stale, f"Classified but no longer shipped: {sorted(stale)}"

    def test_non_splitter_exclusions_exist_and_lack_the_surface(self) -> None:
        # Each documented non-splitter must still exist (no stale entries)
        # and must NOT expose the splitter surface (else it belongs in the
        # main inventory).
        for name in NON_SPLITTER_CLASSES:
            obj = getattr(temporalcv.cv, name, None) or getattr(temporalcv.cv_financial, name, None)
            assert obj is not None, f"NON_SPLITTER_CLASSES entry '{name}' no longer exists"
            assert not (
                callable(getattr(obj, "split", None))
                and callable(getattr(obj, "get_n_splits", None))
            ), f"'{name}' now exposes the splitter surface — classify it in the inventory"
