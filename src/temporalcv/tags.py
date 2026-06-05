"""Capability tags for temporalcv splitters (issue #14).

Capabilities are declared as *data* — a frozen :class:`TemporalTags` descriptor — rather than
encoded in a class hierarchy. Declaring capabilities as data is the lever for non-breaking
evolution (hub ``library-design-playbook.md``): a new capability is a new field, not a new
subclass level. A splitter exposes its tags via a ``temporal_tags()`` method, and the conformance
suite (:func:`~temporalcv.check_temporal_splitter`) cross-validates the *declared* tags against the
splitter's *observed* behavior, so a tag cannot silently drift from reality.

:class:`TemporalTags` is a capability **descriptor**, deliberately **not** a versioned result
object: it carries no ``SCHEMA_VERSION`` and is intentionally excluded from the result-object
registry enforced by ``tests/test_result_objects.py``. Tags evolve by *adding* fields with
defaults (non-breaking), never by renaming or removing existing ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["TemporalTags"]


@dataclass(frozen=True, slots=True)
class TemporalTags:
    """Declared capabilities of a temporal cross-validation splitter.

    Every field is conformance-verifiable: :func:`~temporalcv.check_temporal_splitter` checks each
    declaration against the splitter's observed behavior.

    Attributes
    ----------
    forward_only : bool
        Every fold satisfies ``max(train_idx) < min(test_idx)`` — the splitter never trains on the
        future of its own test block. All splitters in temporalcv's ``cv.py`` are forward-only; the
        de Prado purged K-folds in ``cv_financial.py`` are *bidirectional* and are not.
    deterministic : bool
        ``split(X)`` yields identical folds across repeated calls — no dependence on global RNG
        state.
    produces_oof : bool
        The splitter is also a :class:`~temporalcv.CrossFitter`: it produces out-of-fold
        predictions / residuals (rows with no valid out-of-fold prediction are ``NaN``).
    requires_groups : bool
        ``split`` requires a non-``None`` ``groups`` argument to produce its folds.
    """

    forward_only: bool
    deterministic: bool
    produces_oof: bool
    requires_groups: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``{field: value}`` mapping (JSON-friendly)."""
        return asdict(self)
