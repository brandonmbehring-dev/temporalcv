"""Internal serialization helpers for result-object ``to_dict()`` methods.

The v2.0 result-object pattern (see ``STYLE.md``): every result is a frozen, slotted value
object with an explicit JSON-serializable ``to_dict()`` carrying a ``SCHEMA_VERSION``. These
helpers centralize the recurring conversions so every module serializes the same way —
datetime → ISO string, ndarray → nested list, and non-string mapping keys → string (JSON
object keys must be strings).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
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


def _jsonify(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable form.

    Order matters: numpy scalars are checked *before* the python-scalar shortcut because
    ``np.float64`` is a subclass of ``float`` (and would otherwise pass through unconverted,
    breaking ``json.dumps``).
    """
    if isinstance(value, np.generic):  # numpy scalar (np.float64 is-a float!) -> python scalar
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):  # date / datetime
        return value.isoformat()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):  # nested result object
        return to_dict()
    if isinstance(value, Mapping):
        return {jsonify_key(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(item) for item in value]
    return value


def result_to_dict(obj: Any) -> dict[str, Any]:
    """JSON-serializable mapping for a frozen result dataclass.

    Emits ``schema_version`` (from the class ``SCHEMA_VERSION``) followed by every dataclass
    field, each recursively normalized by :func:`_jsonify` (numpy scalar/array → python,
    datetime → ISO string, nested result → its ``to_dict()``, mapping keys → string, sequences
    → list). Result objects that need renamed keys or derived fields hand-write ``to_dict()``
    instead (e.g. the ``cv.py`` results). ``SCHEMA_VERSION`` is a ``ClassVar`` and so is not a
    dataclass field — it is surfaced explicitly here.
    """
    out: dict[str, Any] = {"schema_version": obj.SCHEMA_VERSION}
    for field in dataclasses.fields(obj):
        out[field.name] = _jsonify(getattr(obj, field.name))
    return out
