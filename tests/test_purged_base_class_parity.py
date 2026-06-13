"""Golden-parity gate for the #25 BaseCrossValidator realign.

The three purged splitters gained ``sklearn.model_selection.BaseCrossValidator``
as a base class (#25) — a declaration-only change. This test pins SHA-256
digests of every ``split()`` output across a parameter grid, generated on
pre-realign main (post-#35/#36, commit 3ae04fb): the realign must not move a
single index. Same proof pattern as the A2 splitter ports.

If a future change legitimately alters split geometry, regenerate the digests
in the same change and say so in the commit — this file failing means "the
indices moved", which must never happen silently.

Regenerated at v2.2.0 for the #38 one-sided per-run embargo fix. The grid has
20 entries: the 18 #25-realign baseline configs (3ae04fb) plus 2
``PurgedKFold(purge_gap=0, embargo_pct=0.05)`` configs added here so a PurgedKFold
embargo regression moves a digest (its only other PKF-with-embargo config is
purge-masked). Among the 18 baseline digests, 9 moved at this fix — all 9 are
``embargo_pct>0`` (14 of the 18 carry ``embargo_pct>0``); the other 9 are
byte-identical (4 ``embargo_pct=0.0``, 5 ``embargo_pct>0`` but purge-masked or
forward-only no-ops), so they still pin the realign baseline.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from temporalcv.cv_financial import (
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedWalkForward,
)

# (constructor expression, n_samples, sha256[:16] of concatenated index bytes)
# Baseline main @ 3ae04fb (#25 realign); embargo_pct>0 digests regenerated at
# the #38 one-sided per-run embargo fix (v2.2.0) — see module docstring.
PARITY_GRID: list[tuple[str, int, str]] = [
    ("PurgedKFold(n_splits=5)", 120, "45cf8260bde9d87f"),
    ("PurgedKFold(n_splits=5)", 257, "8410c26c2f38f0d0"),
    ("PurgedKFold(n_splits=4, purge_gap=5, embargo_pct=0.02)", 120, "af09f50f394ac38c"),
    ("PurgedKFold(n_splits=4, purge_gap=5, embargo_pct=0.02)", 257, "d51a243b03e803f7"),
    ("PurgedKFold(n_splits=3, purge_gap=10, embargo_pct=0.0)", 120, "bf19711573cdf00f"),
    ("PurgedKFold(n_splits=3, purge_gap=10, embargo_pct=0.0)", 257, "3887856d80cb8590"),
    # purge_gap=0: embargo is unmasked, so this digest moves with PKF embargo (#38).
    ("PurgedKFold(n_splits=4, purge_gap=0, embargo_pct=0.05)", 120, "a50f5df50e619284"),
    ("PurgedKFold(n_splits=4, purge_gap=0, embargo_pct=0.05)", 257, "99612963040a7819"),
    ("CombinatorialPurgedCV(n_splits=5, n_test_splits=2)", 120, "3c3e1cbfd0566fbc"),
    ("CombinatorialPurgedCV(n_splits=5, n_test_splits=2)", 257, "842ff6bd009a3699"),
    (
        "CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_gap=3)",
        120,
        "438e602540f8da22",
    ),
    (
        "CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_gap=3)",
        257,
        "4c9dd91f40d09bf6",
    ),
    (
        "CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo_pct=0.05)",
        120,
        "a50f5df50e619284",
    ),
    (
        "CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo_pct=0.05)",
        257,
        "99612963040a7819",
    ),
    ("PurgedWalkForward(n_splits=5)", 120, "6b0dd043f04cd5fc"),
    ("PurgedWalkForward(n_splits=5)", 257, "2bd4e387f7d6e14c"),
    (
        "PurgedWalkForward(n_splits=3, train_size=50, test_size=20, purge_gap=5)",
        120,
        "708d8ebd190b6ecd",
    ),
    (
        "PurgedWalkForward(n_splits=3, train_size=50, test_size=20, purge_gap=5)",
        257,
        "303f7f22d29cf4e6",
    ),
    (
        "PurgedWalkForward(n_splits=4, test_size=15, extra_gap=3, embargo_pct=0.0)",
        120,
        "54b7700b64f08c6b",
    ),
    (
        "PurgedWalkForward(n_splits=4, test_size=15, extra_gap=3, embargo_pct=0.0)",
        257,
        "5a5c0232e1fda511",
    ),
]


def _split_digest(cv: object, n_samples: int) -> str:
    """SHA-256 (truncated) over every fold's train/test index bytes, in order."""
    h = hashlib.sha256()
    for train, test in cv.split(np.zeros((n_samples, 1))):  # type: ignore[attr-defined]
        h.update(np.asarray(train, dtype=np.int64).tobytes())
        h.update(b"|")
        h.update(np.asarray(test, dtype=np.int64).tobytes())
        h.update(b";")
    return h.hexdigest()[:16]


@pytest.mark.parametrize(
    ("expr", "n_samples", "expected"),
    PARITY_GRID,
    ids=[f"{expr}-n{n}" for expr, n, _ in PARITY_GRID],
)
def test_split_indices_bit_identical_across_realign(
    expr: str, n_samples: int, expected: str
) -> None:
    cv = eval(expr)  # noqa: S307 — grid of literal constructor expressions above
    assert _split_digest(cv, n_samples) == expected


class TestSklearnInterop:
    """The realign's payoff: first-class sklearn ecosystem citizenship (#25)."""

    @pytest.mark.parametrize(
        "cv",
        [
            PurgedKFold(n_splits=5),
            CombinatorialPurgedCV(n_splits=5, n_test_splits=2),
            PurgedWalkForward(n_splits=3),
        ],
        ids=lambda cv: type(cv).__name__,
    )
    def test_isinstance_base_cross_validator(self, cv: object) -> None:
        from sklearn.model_selection import BaseCrossValidator

        assert isinstance(cv, BaseCrossValidator)

    @pytest.mark.parametrize(
        "cv",
        [
            PurgedKFold(n_splits=5),
            CombinatorialPurgedCV(n_splits=5, n_test_splits=2),
            PurgedWalkForward(n_splits=3),
        ],
        ids=lambda cv: type(cv).__name__,
    )
    def test_sklearn_repr_reflects_params(self, cv: object) -> None:
        # BaseCrossValidator.__repr__ (_build_repr) introspects __init__
        # params via same-named attributes — all three store them that way.
        assert type(cv).__name__ in repr(cv)
        assert "n_splits" in repr(cv)

    def test_cross_val_score_runs(self) -> None:
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(7)
        X = rng.normal(size=(120, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=120) * 0.1

        for cv in (
            PurgedKFold(n_splits=5, purge_gap=3),
            PurgedWalkForward(n_splits=3),
        ):
            scores = cross_val_score(LinearRegression(), X, y, cv=cv)
            assert len(scores) == cv.get_n_splits()
            assert np.all(np.isfinite(scores))
