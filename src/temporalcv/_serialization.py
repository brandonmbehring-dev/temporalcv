"""Internal serialization helpers for result-object ``to_dict()`` methods.

The v2.0 result-object pattern (see ``STYLE.md``): every result is a frozen, slotted value
object with an explicit JSON-serializable ``to_dict()`` carrying a ``SCHEMA_VERSION``. These
helpers centralize the recurring conversions so every module serializes the same way —
datetime → ISO string, ndarray → nested list, and non-string scalar mapping keys → string
(JSON object keys must be strings; tuple keys are refused — see ``jsonify_key``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def date_to_json(value: Any) -> Any:
    """Serialize an optional datetime-like field to an ISO string (else pass through).

    Handles python ``date``/``datetime`` (``isoformat``) and numpy ``datetime64`` (which has no
    ``isoformat``); ``None`` and anything else pass through unchanged.
    """
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(value)
    return value


def array_to_list(value: ArrayLike) -> Any:
    """Convert an array-like field to a JSON-serializable (possibly nested) list."""
    return np.asarray(value).tolist()


def jsonify_key(key: Any) -> str:
    """Render a mapping key as a JSON-safe string.

    JSON object keys must be strings; scalars fall back to ``str()``.

    Tuple keys are REFUSED: the former comma-join (``("AR", "RW") ->
    "AR,RW"``) collided with ``("AR,RW",)`` and could not round-trip model
    names that legitimately contain commas (``ARIMA(1,1,0)``), silently
    overwriting entries on collision (#21). A result object with a
    tuple-keyed dict must hand-write ``to_dict()`` and emit a list of
    records instead — see ``MultiModelComparisonResult.to_dict``.
    """
    if isinstance(key, tuple):
        raise TypeError(
            f"tuple mapping key {key!r} cannot serialize losslessly as a JSON "
            f"object key — emit a list of records instead (see "
            f"MultiModelComparisonResult.to_dict, #21)"
        )
    return str(key)


def _jsonify(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable form.

    Order matters: numpy scalars are checked *before* the python-scalar shortcut because
    ``np.float64`` is a subclass of ``float`` (and would otherwise pass through unconverted,
    breaking ``json.dumps``). Raises ``TypeError`` on any value it cannot convert — failing loud
    at the serializer rather than silently emitting a value that explodes later at ``json.dumps``.
    """
    if isinstance(value, np.generic):
        # numpy scalar -> python scalar; recurse so datetime64 -> date/datetime -> ISO below.
        return _jsonify(value.item())
    if isinstance(value, Enum):  # serialize enums by value (e.g. GateStatus -> "PASS")
        return _jsonify(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):  # date / datetime
        return value.isoformat()
    # Nested result object: gate on the SCHEMA_VERSION marker so a stray third-party to_dict
    # (e.g. pandas Series/DataFrame, which have a to_dict) is not silently mis-serialized.
    if hasattr(type(value), "SCHEMA_VERSION"):
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
    if isinstance(value, Mapping):
        return {jsonify_key(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonify(item) for item in value]
    raise TypeError(f"_jsonify: cannot serialize value of type {type(value).__name__!r} to JSON")


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
