"""Static typed seams for temporalcv (v2.0).

These Protocols are a **static typing aid only**. ``@runtime_checkable`` verifies member
*presence*, not signatures, and is comparatively slow — use it for a one-shot boundary
check, never as a validator or on a hot path. Concrete splitters additionally subclass
sklearn ``BaseCrossValidator`` for ``cross_val_score``/``Pipeline`` interop; this Protocol
describes the shape typed consumers (e.g. ``dml_ts``) annotate against, without forcing an
import of temporalcv or sklearn.

See ``STYLE.md`` and ``docs/adr/0001-v2-seams-and-layout.md`` for the seam strategy, and
``~/Claude/lever_of_archimedes/patterns/library-design-playbook.md`` for the rationale.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["Splitter", "CrossFitter"]


@runtime_checkable
class Splitter(Protocol):
    """A temporal cross-validation splitter.

    Yields ``(train_idx, test_idx)`` integer arrays in temporal order. Concrete
    implementations also subclass sklearn ``BaseCrossValidator`` so they work with
    ``cross_val_score``/``Pipeline``; this Protocol is the typed seam only.
    """

    def split(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` integer index arrays."""
        ...

    def get_n_splits(
        self,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        groups: ArrayLike | None = None,
    ) -> int:
        """Number of splits the splitter will produce."""
        ...


@runtime_checkable
class CrossFitter(Splitter, Protocol):
    """A splitter that also produces out-of-fold predictions / residuals.

    Forward-only semantics: rows with no valid out-of-fold prediction (e.g. fold 0,
    which has no history to train on) are returned as ``NaN``. This is the seam a
    debiased/orthogonalized consumer (Double ML, partialling-out) builds on.
    """

    def fit_predict(self, model: object, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Out-of-fold predictions aligned to ``y`` (uncovered rows are ``NaN``)."""
        ...

    def fit_predict_residuals(self, model: object, X: ArrayLike, y: ArrayLike) -> np.ndarray:
        """Out-of-fold residuals ``y - y_hat`` (uncovered rows are ``NaN``)."""
        ...
