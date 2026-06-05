"""Internal serialization helpers for result-object ``to_dict()`` methods.

The v2.0 result-object pattern (see ``STYLE.md``): every result is a frozen, slotted value
object with an explicit JSON-serializable ``to_dict()`` carrying a ``SCHEMA_VERSION``. These
helpers centralize the recurring conversions so every module serializes the same way —
datetime → ISO string, ndarray → nested list, and non-string mapping keys → string (JSON
object keys must be strings).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def date_to_json(value: Any) -> Any:
    """Serialize an optional datetime-like field to an ISO string (else pass through)."""
    return value.isoformat() if hasattr(value, "isoformat") else value


def array_to_list(value: ArrayLike) -> Any:
    """Convert an array-like field to a JSON-serializable (possibly nested) list."""
    return np.asarray(value).tolist()


def jsonify_key(key: Any) -> str:
    """Render a mapping key as a JSON-safe string.

    JSON object keys must be strings. Tuples become a comma-joined string
    (e.g. ``("AR", "RW") -> "AR,RW"``); everything else falls back to ``str()``.
    """
    if isinstance(key, tuple):
        return ",".join(str(part) for part in key)
    return str(key)
