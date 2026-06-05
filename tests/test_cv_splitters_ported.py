"""Conformance + dml_ts golden-parity for the ported forward-only splitters (A2).

Covers the v2.0 A2 port of two forward-only CV splitters from
``dml_ts/dml/cross_fitting.py`` into :mod:`temporalcv.cv`:

- :class:`~temporalcv.TimeSeriesCrossValidator` — expanding/sliding, test-from-end, gap/purge.
- :class:`~temporalcv.BlockedTimeSeriesCV` — whole-block-preserving (issue #7), now fail-loud
  on an under-provisioned config instead of silently dropping a fold.

The **golden-parity** tests pin each ported splitter's fold indices to a *verbatim,
dependency-free* reference reproducing dml_ts's exact index math (the same pattern as
``_dmlts_reference_residualize`` in ``test_pilot_crossfitter_seam.py``). A one-off live
cross-check against the actual ``dml_ts`` package confirmed these references — and the ported
classes — match the live source bit-exactly across the parameter grid below (48 TSCV + 18
skip-free Blocked configs, 0 mismatches). dml_ts is intentionally **not** imported here so the
gate runs in temporalcv CI without that dependency.

``PurgedGroupTimeSeriesCV`` is deliberately *not* ported — it is a bidirectional purged K-fold
redundant with ``cv_financial.PurgedKFold`` and is reconciled in issue #23 (defer to Track B).

See ``docs/adr/0001-v2-seams-and-layout.md``, ``STYLE.md``, ``docs/plans/v2_roadmap.md``.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from temporalcv import (
    BlockedTimeSeriesCV,
    SplitInfo,
    TimeSeriesCrossValidator,
    check_temporal_splitter,
)

# =============================================================================
# Golden reference: verbatim dml_ts index math (dependency-free)
# =============================================================================


def _dmlts_tscv_reference(
    n: int,
    n_splits: int,
    gap: int,
    purge_length: int,
    expanding: bool,
    test_size: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reproduce dml_ts ``TimeSeriesCrossValidator`` fold indices exactly.

    Mirrors ``_get_fold_indices`` with ``min_train_size`` defaulting to
    ``n // (n_splits + 1)`` (the dml_ts default when unset).
    """
    ts = test_size if test_size is not None else n // (n_splits + 1)
    min_train = n // (n_splits + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        test_end = n - (n_splits - 1 - fold_idx) * ts
        test_start = test_end - ts
        train_end = test_start - gap - purge_length
        if expanding:
            train_start = 0
        else:
            window = min_train + fold_idx * ts
            train_start = max(0, train_end - window)
        if train_end - train_start < min_train:
            train_start = max(0, train_end - min_train)
        folds.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))
    return folds


def _dmlts_blocked_reference(
    n: int,
    n_splits: int,
    block_size: int,
    gap_blocks: int,
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Reproduce dml_ts ``BlockedTimeSeriesCV`` fold indices exactly.

    Returns ``None`` when dml_ts would *silently skip* a fold (``train_block_end <= 0``) —
    exactly the configs where the ported class fails loud with ``ValueError`` instead.
    """
    n_blocks = n // block_size
    test_blocks = max(1, n_blocks // (n_splits + 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        test_block_end = n_blocks - (n_splits - 1 - fold_idx) * test_blocks
        test_block_start = test_block_end - test_blocks
        train_block_end = test_block_start - gap_blocks
        if train_block_end <= 0:
            return None
        train_end_sample = train_block_end * block_size
        test_start_sample = test_block_start * block_size
        test_end_sample = min(test_block_end * block_size, n)
        folds.append(
            (np.arange(0, train_end_sample), np.arange(test_start_sample, test_end_sample))
        )
    return folds


def _as_lists(folds: list[tuple[np.ndarray, np.ndarray]]) -> list[list[list[int]]]:
    return [[tr.tolist(), te.tolist()] for tr, te in folds]


# =============================================================================
# Conformance — the ported splitters satisfy the forward-only contract
# =============================================================================


class TestConformance:
    @pytest.mark.parametrize(
        "splitter",
        [
            TimeSeriesCrossValidator(n_splits=5),
            TimeSeriesCrossValidator(n_splits=3, gap=3, purge_length=2),
            TimeSeriesCrossValidator(n_splits=4, expanding=False),
            BlockedTimeSeriesCV(n_splits=5),
            BlockedTimeSeriesCV(n_splits=3, gap_blocks=1),
        ],
    )
    def test_passes_check_temporal_splitter(self, splitter: object) -> None:
        check_temporal_splitter(splitter)


# =============================================================================
# Golden parity — fold indices match the dml_ts originals exactly
# =============================================================================

_N = 300

_TSCV_GRID = list(
    product((3, 5), (0, 5, 10), (0, 3), (True, False), (None, 20))
)  # n_splits, gap, purge_length, expanding, test_size

_BLOCKED_GRID = list(product((2, 3), (10, 20, 25), (0, 1, 2)))  # n_splits, block_size, gap_blocks


class TestGoldenParity:
    @pytest.mark.parametrize("n_splits,gap,purge,expanding,test_size", _TSCV_GRID)
    def test_tscv_matches_dmlts(
        self, n_splits: int, gap: int, purge: int, expanding: bool, test_size: int | None
    ) -> None:
        X = np.arange(_N).reshape(-1, 1)
        cv = TimeSeriesCrossValidator(
            n_splits=n_splits, gap=gap, purge_length=purge, expanding=expanding, test_size=test_size
        )
        ref = _dmlts_tscv_reference(_N, n_splits, gap, purge, expanding, test_size)
        assert _as_lists(list(cv.split(X))) == _as_lists(ref)

    @pytest.mark.parametrize("n_splits,block_size,gap_blocks", _BLOCKED_GRID)
    def test_blocked_matches_dmlts_or_fails_loud(
        self, n_splits: int, block_size: int, gap_blocks: int
    ) -> None:
        X = np.arange(_N).reshape(-1, 1)
        cv = BlockedTimeSeriesCV(n_splits=n_splits, block_size=block_size, gap_blocks=gap_blocks)
        ref = _dmlts_blocked_reference(_N, n_splits, block_size, gap_blocks)
        if ref is None:
            # dml_ts would silently drop a fold here; the port fails loud instead.
            with pytest.raises(ValueError, match="no training blocks"):
                list(cv.split(X))
        else:
            assert _as_lists(list(cv.split(X))) == _as_lists(ref)

    def test_tscv_indices_are_integer_dtype(self) -> None:
        X = np.arange(_N).reshape(-1, 1)
        for tr, te in TimeSeriesCrossValidator(n_splits=3).split(X):
            assert np.issubdtype(tr.dtype, np.integer) and np.issubdtype(te.dtype, np.integer)


# =============================================================================
# get_split_info reuses SplitInfo (CVFold reconciled away)
# =============================================================================


class TestGetSplitInfo:
    def test_returns_splitinfo_with_correct_boundaries(self) -> None:
        X = np.arange(100).reshape(-1, 1)
        cv = TimeSeriesCrossValidator(n_splits=3, gap=5, purge_length=2)
        infos = cv.get_split_info(X)
        folds = list(cv.split(X))
        assert len(infos) == 3
        for idx, (info, (tr, te)) in enumerate(zip(infos, folds, strict=True)):
            assert isinstance(info, SplitInfo)
            assert info.split_idx == idx
            # inclusive-end convention; raw arrays still available via split()
            assert info.train_start == int(tr[0])
            assert info.train_end == int(tr[-1])
            assert info.test_start == int(te[0])
            assert info.test_end == int(te[-1])

    def test_gap_property_reflects_configured_separation(self) -> None:
        X = np.arange(200).reshape(-1, 1)
        cv = TimeSeriesCrossValidator(n_splits=3, gap=6, purge_length=4)
        # SplitInfo.gap = test_start - train_end - 1 (inclusive train_end). With exclusive
        # arange the gap+purge buffer is gap + purge_length.
        for info in cv.get_split_info(X):
            assert info.gap == 6 + 4


# =============================================================================
# Fail-loud — no silent failures
# =============================================================================


class TestFailLoud:
    def test_blocked_raises_on_degenerate_config(self) -> None:
        X = np.arange(200).reshape(-1, 1)
        cv = BlockedTimeSeriesCV(n_splits=5, block_size=10, gap_blocks=8)
        with pytest.raises(ValueError, match="no training blocks"):
            list(cv.split(X))

    def test_blocked_raises_on_too_few_blocks(self) -> None:
        X = np.arange(30).reshape(-1, 1)
        cv = BlockedTimeSeriesCV(n_splits=3, block_size=25, gap_blocks=1)
        with pytest.raises(ValueError, match="Not enough blocks"):
            list(cv.split(X))

    def test_tscv_raises_on_too_small(self) -> None:
        X = np.arange(20).reshape(-1, 1)
        cv = TimeSeriesCrossValidator(n_splits=5, gap=50)
        with pytest.raises(ValueError, match="Not enough samples"):
            list(cv.split(X))

    @pytest.mark.parametrize("bad", [{"n_splits": 0}, {"gap": -1}, {"purge_length": -1}])
    def test_tscv_init_validation(self, bad: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            TimeSeriesCrossValidator(**bad)

    @pytest.mark.parametrize("bad", [{"n_splits": 0}, {"gap_blocks": -1}])
    def test_blocked_init_validation(self, bad: dict[str, int]) -> None:
        with pytest.raises(ValueError):
            BlockedTimeSeriesCV(**bad)


# =============================================================================
# Behavioral — the meaningful dml_ts invariants, adapted
# =============================================================================


class TestBehavioral:
    def test_forward_only_and_disjoint(self) -> None:
        X = np.arange(200).reshape(-1, 1)
        for cv in (
            TimeSeriesCrossValidator(n_splits=5),
            BlockedTimeSeriesCV(n_splits=3, block_size=20, gap_blocks=1),
        ):
            for tr, te in cv.split(X):
                assert tr.max() < te.min()
                assert np.intersect1d(tr, te).size == 0

    def test_gap_creates_separation(self) -> None:
        gap = 10
        X = np.arange(150).reshape(-1, 1)
        for tr, te in TimeSeriesCrossValidator(n_splits=3, gap=gap).split(X):
            assert te[0] - tr[-1] - 1 >= gap

    def test_purge_removes_from_train_end(self) -> None:
        purge = 5
        X = np.arange(150).reshape(-1, 1)
        no = TimeSeriesCrossValidator(n_splits=3, purge_length=0).split(X)
        yes = TimeSeriesCrossValidator(n_splits=3, purge_length=purge).split(X)
        for (tr0, _), (tr1, _) in zip(no, yes, strict=True):
            assert len(tr0) - len(tr1) == purge

    def test_expanding_grows_sliding_does_not_grow_faster(self) -> None:
        X = np.arange(150).reshape(-1, 1)
        exp = [len(tr) for tr, _ in TimeSeriesCrossValidator(n_splits=3, expanding=True).split(X)]
        slid = [len(tr) for tr, _ in TimeSeriesCrossValidator(n_splits=3, expanding=False).split(X)]
        assert exp[-1] > exp[0]  # expanding grows
        assert (slid[-1] - slid[0]) <= (exp[-1] - exp[0])  # sliding grows no faster

    def test_test_sets_sequential(self) -> None:
        X = np.arange(200).reshape(-1, 1)
        starts = [int(te[0]) for _, te in TimeSeriesCrossValidator(n_splits=4).split(X)]
        assert all(starts[i] > starts[i - 1] for i in range(1, len(starts)))

    def test_blocked_blocks_not_split(self) -> None:
        bs = 10
        X = np.arange(100).reshape(-1, 1)
        for _tr, te in BlockedTimeSeriesCV(n_splits=2, block_size=bs, gap_blocks=0).split(X):
            assert int(te[0]) % bs == 0  # test starts on a block boundary

    def test_custom_test_size(self) -> None:
        X = np.arange(200).reshape(-1, 1)
        for _, te in TimeSeriesCrossValidator(n_splits=3, test_size=20).split(X):
            assert len(te) == 20

    def test_1d_array_input(self) -> None:
        assert len(list(TimeSeriesCrossValidator(n_splits=3).split(np.arange(100)))) == 3

    def test_get_n_splits(self) -> None:
        assert TimeSeriesCrossValidator(n_splits=7).get_n_splits() == 7
        assert BlockedTimeSeriesCV(n_splits=4).get_n_splits(np.zeros(100)) == 4

    @pytest.mark.parametrize(
        "cv",
        [TimeSeriesCrossValidator(n_splits=3), BlockedTimeSeriesCV(n_splits=3, block_size=20)],
    )
    def test_sklearn_cross_val_score_interop(self, cv: object) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((120, 4))
        y = X.sum(axis=1) + rng.standard_normal(120) * 0.1
        scores = cross_val_score(Ridge(), X, y, cv=cv)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_repr_roundtrips_key_params(self) -> None:
        assert "n_splits=3" in repr(TimeSeriesCrossValidator(n_splits=3, gap=5))
        assert "gap_blocks=2" in repr(BlockedTimeSeriesCV(n_splits=2, gap_blocks=2))
