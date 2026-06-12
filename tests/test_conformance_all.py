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

An exhaustiveness guard walks EVERY module in the package (pkgutil, not a
hardcoded module list — a splitter defined in a future ``cv_panel.py`` is
caught too) and fails the moment a new splitter class ships without being
classified here — the same fail-loud pattern as
``test_result_objects.py::test_registry_is_exhaustive``.

(The strategy/adapter conformance counterpart is exhaustive in
``test_seam_vocab.py``: ``ALL_STRATEGIES`` / ``ALL_ADAPTERS``.)
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

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
    # Note: embargo_pct is structurally a no-op for this forward-only
    # splitter (it only embargoes train indices AFTER the test block, which
    # never exist here) — real embargo behavior is exercised by the
    # bidirectional PurgedKFold tests. The config stays as a smoke config.
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
        # Deterministic ids (an id(s)-based scheme broke --last-failed and
        # node-id reruns — review finding).
        ids=[f"{type(s).__name__}-{i}" for i, s in enumerate(FORWARD_ONLY_SPLITTERS)],
    )
    def test_passes_temporal_splitter_contract(self, splitter: Splitter) -> None:
        check_temporal_splitter(splitter)  # must not raise

    def test_purged_walk_forward_satisfies_protocol(self) -> None:
        # The forward-only purged splitter is a first-class Splitter — the
        # Track-B migration target for dml_ts's purged_cv strategy (#23).
        assert isinstance(PurgedWalkForward(n_splits=3), Splitter)

    def test_under_provisioned_purged_walk_forward_raises(self) -> None:
        # #32 hardening: PurgedWalkForward raises on under-provisioned
        # configs like its v2.0 siblings (BlockedTimeSeriesCV/
        # TimeSeriesCrossValidator) instead of silently dropping folds.
        # This was the KNOWN-LIMITATION pin (the conformance checker's
        # count-consistency invariant exposed the silent drop); the raise
        # now fires from the splitter itself before conformance can even
        # collect folds.
        with pytest.raises(ValueError, match="empty train window"):
            check_temporal_splitter(
                PurgedWalkForward(n_splits=5, train_size=50, test_size=20), n_samples=30
            )


class TestBidirectionalExclusions:
    @pytest.mark.parametrize("splitter", BIDIRECTIONAL_BY_DESIGN, ids=lambda s: type(s).__name__)
    def test_fails_forward_only_contract_as_designed(self, splitter: Splitter) -> None:
        # These purged K-folds train on BOTH sides of the test block (the
        # de Prado evaluation setting). The forward-only contract must
        # reject them on the FOLD-LEVEL no-lookahead invariant — the match
        # is anchored to "fold N: lookahead" because a bare "lookahead"
        # would also match the tags-consistency message that fires in the
        # OPPOSITE scenario (review finding).
        with pytest.raises(AssertionError, match=r"fold \d+: lookahead"):
            check_temporal_splitter(splitter)

    @pytest.mark.parametrize("splitter", BIDIRECTIONAL_BY_DESIGN, ids=lambda s: type(s).__name__)
    def test_they_still_satisfy_the_structural_protocol(self, splitter: Splitter) -> None:
        # Structurally they ARE Splitters (split/get_n_splits) — the
        # exclusion is behavioral (forward-only), not structural.
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

        # Detector: walk EVERY module in the package for classes exposing
        # the splitter surface (split + get_n_splits). A two-module
        # hardcoded walk was bypassable by a splitter defined in a new
        # module (review finding) — pkgutil closes that. Protocol
        # DEFINITIONS (Splitter/CrossFitter) are excluded; sklearn-imported
        # classes are excluded via the temporalcv __module__ prefix.
        # NestedWalkForwardCV exposes NEITHER method (pure tuning
        # meta-estimator), so it is not collected — its exclusion is
        # documented in NON_SPLITTER_CLASSES and verified separately below.
        shipped: set[str] = set()
        for info in pkgutil.walk_packages(temporalcv.__path__, "temporalcv."):
            module = importlib.import_module(info.name)
            for name, obj in vars(module).items():
                if (
                    inspect.isclass(obj)
                    and obj.__module__.startswith("temporalcv.")
                    and obj.__module__ == module.__name__
                    and callable(getattr(obj, "split", None))
                    and callable(getattr(obj, "get_n_splits", None))
                ):
                    shipped.add(name)
        shipped -= {"Splitter", "CrossFitter"}  # Protocol definitions, not splitters

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
