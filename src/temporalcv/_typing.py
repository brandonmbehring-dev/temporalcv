"""Backend-agnostic typing seam for temporalcv (v2.0, issue #15).

Public *input* parameters are typed :data:`~numpy.typing.ArrayLike` (they accept arrays, lists,
tuples — anything ``np.asarray`` materializes), while implementations and **return** types stay
concrete ``np.ndarray``. The annotation is the contract: *widening* ``np.ndarray`` -> ``ArrayLike``
later would be a breaking change, so the public contract is widened now even though the
implementation only ever runs on numpy today (hub ``library-design-playbook.md`` §4, ADR 0001 §3).

This module is the single reserved seam for future backends: an array-API (``xp``) namespace and a
narwhals dataframe boundary would be introduced HERE, without touching call sites. Until then,
:func:`as_array` is the one normalization boundary — a public function widens its signature to
``ArrayLike`` and calls :func:`as_array` (or an existing ``np.asarray``) to recover a concrete array
for the numpy implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

__all__ = ["ArrayLike", "as_array"]


def as_array(x: ArrayLike, dtype: DTypeLike | None = None) -> np.ndarray:
    """Normalize an ``ArrayLike`` input to a concrete ``np.ndarray`` at a public boundary.

    The single reserved seam where a future array-API / narwhals backend would dispatch; today it is
    a thin :func:`numpy.asarray`. ``dtype=None`` infers the dtype (numpy's default).

    Parameters
    ----------
    x : ArrayLike
        Any array-like input (ndarray, list, tuple, scalar sequence, ...).
    dtype : DTypeLike, optional
        Target dtype; ``None`` (default) infers it.

    Returns
    -------
    np.ndarray
        A concrete numpy array (a view when ``x`` is already a matching ndarray).
    """
    return np.asarray(x, dtype=dtype)
