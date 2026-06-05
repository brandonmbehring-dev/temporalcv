"""Backend-agnostic typing seam for temporalcv (v2.0, issue #15).

Public *input* parameters are typed :data:`~numpy.typing.ArrayLike` (they accept arrays, lists,
tuples — anything ``np.asarray`` materializes), while implementations and **return** types stay
concrete ``np.ndarray``. The annotation is the contract: *widening* ``np.ndarray`` -> ``ArrayLike``
later would be a breaking change, so the public contract is widened now even though the
implementation only ever runs on numpy today (hub ``library-design-playbook.md`` §4, ADR 0001 §3).

This module is the **reserved** home for a future backend seam: an array-API (``xp``) namespace and
a narwhals dataframe boundary would be introduced HERE. :func:`as_array` is the normalization
boundary for code written against it. Note (truth-in-advertising): most existing public functions
recover their concrete array via a direct ``np.asarray`` call rather than :func:`as_array`, so a
future backend swap would migrate those sites to :func:`as_array` too — it is the *reserved* seam,
not yet a single universal chokepoint.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

__all__ = ["ArrayLike", "as_array"]


def as_array(x: ArrayLike, dtype: DTypeLike | None = None) -> np.ndarray:
    """Normalize an ``ArrayLike`` input to a concrete ``np.ndarray`` at a public boundary.

    The reserved seam where a future array-API / narwhals backend would dispatch (most existing call
    sites currently use ``np.asarray`` directly); today a thin :func:`numpy.asarray`. ``dtype=None``
    infers the dtype (numpy's default).

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
