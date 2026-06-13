"""
Tests for financial cross-validation with purging and embargo.

Test categories:
1. PurgedKFold tests
2. CombinatorialPurgedCV tests
3. PurgedWalkForward tests
4. Utility function tests
5. Edge cases
"""

import warnings

import numpy as np
import pytest

from temporalcv.cv_financial import (
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedSplit,
    PurgedWalkForward,
    _apply_purge_and_embargo,
    compute_label_overlap,
    estimate_purge_gap,
)


class TestComputeLabelOverlap:
    """Tests for compute_label_overlap function."""

    def test_no_overlap_large_horizon(self) -> None:
        """With horizon >= n_samples, all samples overlap."""
        overlap = compute_label_overlap(n_samples=10, horizon=10)

        assert overlap.shape == (10, 10)
        assert np.all(overlap)  # All True

    def test_no_overlap_horizon_one(self) -> None:
        """With horizon=1, only self overlaps."""
        overlap = compute_label_overlap(n_samples=5, horizon=1)

        # Diagonal should be True (self-overlap)
        assert np.all(np.diag(overlap))
        # Off-diagonal should be False
        assert not overlap[0, 1]
        assert not overlap[0, 4]

    def test_partial_overlap(self) -> None:
        """With horizon=3, samples within 3 of each other overlap."""
        overlap = compute_label_overlap(n_samples=10, horizon=3)

        assert overlap[0, 0]  # Self
        assert overlap[0, 1]  # Within 3
        assert overlap[0, 2]  # Within 3
        assert not overlap[0, 3]  # Exactly 3, not < 3
        assert not overlap[0, 5]  # Beyond 3

    def test_symmetric(self) -> None:
        """Overlap matrix should be symmetric."""
        overlap = compute_label_overlap(n_samples=20, horizon=5)

        np.testing.assert_array_equal(overlap, overlap.T)


class TestEstimatePurgeGap:
    """Tests for estimate_purge_gap function."""

    def test_default_decay(self) -> None:
        """Default decay factor of 1.0 should return horizon."""
        assert estimate_purge_gap(horizon=5) == 5
        assert estimate_purge_gap(horizon=10) == 10

    def test_custom_decay(self) -> None:
        """Custom decay factor should scale horizon."""
        assert estimate_purge_gap(horizon=5, decay_factor=1.5) == 7  # floor(5 * 1.5) = 7
        assert estimate_purge_gap(horizon=10, decay_factor=0.5) == 5

    def test_minimum_one(self) -> None:
        """Result should be at least 1."""
        assert estimate_purge_gap(horizon=1, decay_factor=0.1) >= 1


class TestPurgedKFold:
    """Tests for PurgedKFold cross-validator."""

    def test_initialization(self) -> None:
        """Should initialize with valid parameters."""
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)

        assert cv.n_splits == 5
        assert cv.purge_gap == 5
        assert cv.embargo_pct == 0.01
        assert cv.shuffle is False

    def test_invalid_n_splits(self) -> None:
        """Should raise for invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFold(n_splits=1)

    def test_invalid_purge_gap(self) -> None:
        """Should raise for negative purge_gap."""
        with pytest.raises(ValueError, match="purge_gap must be >= 0"):
            PurgedKFold(purge_gap=-1)

    def test_invalid_embargo_pct(self) -> None:
        """Should raise for invalid embargo_pct."""
        with pytest.raises(ValueError, match="embargo_pct must be in"):
            PurgedKFold(embargo_pct=1.5)

    def test_shuffle_deprecated_at_construction(self) -> None:
        """shuffle=True warns once, at construction (#39).

        Construction-time (not split-time) so the warning fires before any
        CV loop and exactly once per object.
        """
        with pytest.warns(DeprecationWarning, match="shuffle=True"):
            PurgedKFold(n_splits=4, shuffle=True)

    def test_shuffle_no_warning_when_false(self) -> None:
        """The default shuffle=False path is warning-free (#39)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes an error
            PurgedKFold(n_splits=4, shuffle=False)

    def test_shuffle_seeded_reproducible_and_consistent(self) -> None:
        """The deprecated shuffle is seeded: reproducible, split==split_detailed (#39).

        The pre-#39 ``np.random.shuffle`` consumed the unseedable global RNG,
        so two ``split()`` calls disagreed and ``split`` != ``split_detailed``.
        """
        X = np.arange(40).reshape(-1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            a = [(tr.tolist(), te.tolist()) for tr, te in PurgedKFold(4, shuffle=True).split(X)]
            b = [(tr.tolist(), te.tolist()) for tr, te in PurgedKFold(4, shuffle=True).split(X)]
            cv = PurgedKFold(4, shuffle=True)
            via_split = [(tr.tolist(), te.tolist()) for tr, te in cv.split(X)]
            via_detailed = [
                (d.train_indices.tolist(), d.test_indices.tolist()) for d in cv.split_detailed(X)
            ]
        assert a == b  # reproducible across instances (seeded, not global RNG)
        assert via_split == via_detailed  # the two methods describe the same folds
        # shuffle actually scrambles order: at least one test fold is non-contiguous
        assert any(te != list(range(min(te), max(te) + 1)) for _, te in a)

    def test_splits_count(self) -> None:
        """Should generate correct number of splits."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5)

        splits = list(cv.split(X))

        assert len(splits) == 5

    def test_get_n_splits(self) -> None:
        """get_n_splits should return n_splits."""
        cv = PurgedKFold(n_splits=5)

        assert cv.get_n_splits() == 5

    def test_no_overlap_train_test(self) -> None:
        """Train and test should not overlap."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, purge_gap=0)

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    def test_purging_removes_nearby_samples(self) -> None:
        """Purging should remove training samples near test samples."""
        X = np.arange(100).reshape(-1, 1)
        cv_no_purge = PurgedKFold(n_splits=5, purge_gap=0)
        cv_with_purge = PurgedKFold(n_splits=5, purge_gap=5)

        for (train_no, test_no), (train_with, test_with) in zip(
            cv_no_purge.split(X), cv_with_purge.split(X)
        ):
            # With purging, training set should be smaller
            assert len(train_with) < len(train_no)
            # Test sets should be the same
            np.testing.assert_array_equal(test_no, test_with)

    def test_split_detailed(self) -> None:
        """split_detailed should return PurgedSplit objects."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)

        for split in cv.split_detailed(X):
            assert isinstance(split, PurgedSplit)
            assert split.n_purged >= 0
            assert split.n_embargoed >= 0

    def test_purge_empties_train_raises(self) -> None:
        """An over-aggressive purge_gap raises instead of yielding empty trains (#36).

        Pre-fix this config silently yielded (train len=0, test len=10) twice.
        """
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedKFold(n_splits=2, purge_gap=100)

        with pytest.raises(ValueError, match="purge/embargo removal emptied"):
            list(cv.split(X))

    def test_raises_at_call_time(self) -> None:
        """The empty-train raise fires at split() call time, before iteration (#36)."""
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedKFold(n_splits=2, purge_gap=100)

        with pytest.raises(ValueError, match="purge/embargo removal emptied"):
            cv.split(X)  # no iteration

    def test_split_detailed_raises_at_call_time(self) -> None:
        """split_detailed's own documented call-time contract holds (#36).

        Pins it independently of split(): a `yield from` refactor of
        split_detailed would defer the raise to first next() and split()
        would still mask it via its eager list comprehension.
        """
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedKFold(n_splits=2, purge_gap=100)

        with pytest.raises(ValueError, match="purge/embargo removal emptied"):
            cv.split_detailed(X)  # no iteration

    def test_more_splits_than_samples_raises(self) -> None:
        """n_samples < n_splits raises a clear message, not a numpy crash (#36)."""
        X = np.arange(3).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5)

        with pytest.raises(ValueError, match="test folds would be empty"):
            list(cv.split(X))

    def test_n_samples_equals_n_splits_is_legal(self) -> None:
        """The exact n_samples == n_splits boundary yields all folds (#36).

        Guards against an off-by-one (`<=`) regression in the empty-test
        guard: five samples into five folds is minimal but legal.
        """
        X = np.arange(5).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)

        splits = list(cv.split(X))
        assert len(splits) == 5
        for train, test in splits:
            assert len(train) == 4
            assert len(test) == 1

    def test_split_matches_split_detailed(self) -> None:
        """split() yields exactly split_detailed()'s index pairs (dedup, #36)."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.02)

        for (train, test), detailed in zip(cv.split(X), cv.split_detailed(X), strict=True):
            np.testing.assert_array_equal(train, detailed.train_indices)
            np.testing.assert_array_equal(test, detailed.test_indices)

    def test_train_indices_integer_dtype(self) -> None:
        """Train indices keep an integer dtype usable for array indexing (#36)."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, purge_gap=5)

        for train, _ in cv.split(X):
            assert np.issubdtype(train.dtype, np.integer)


class TestCombinatorialPurgedCV:
    """Tests for CombinatorialPurgedCV cross-validator."""

    def test_initialization(self) -> None:
        """Should initialize with valid parameters."""
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        assert cv.n_splits == 5
        assert cv.n_test_splits == 2

    def test_invalid_n_test_splits(self) -> None:
        """Should raise for invalid n_test_splits."""
        with pytest.raises(ValueError, match="n_test_splits must be in"):
            CombinatorialPurgedCV(n_splits=5, n_test_splits=5)

        with pytest.raises(ValueError, match="n_test_splits must be in"):
            CombinatorialPurgedCV(n_splits=5, n_test_splits=0)

    def test_correct_number_of_paths(self) -> None:
        """Should generate C(n_splits, n_test_splits) paths."""
        from math import comb

        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        assert cv.get_n_splits() == comb(5, 2)  # 10

        X = np.arange(100).reshape(-1, 1)
        splits = list(cv.split(X))

        assert len(splits) == 10

    def test_no_overlap_train_test(self) -> None:
        """Train and test should not overlap."""
        X = np.arange(100).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    def test_purging_applied(self) -> None:
        """Purging should reduce training set size."""
        X = np.arange(100).reshape(-1, 1)
        cv_no_purge = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=0)
        cv_with_purge = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)

        for (train_no, _), (train_with, _) in zip(cv_no_purge.split(X), cv_with_purge.split(X)):
            # With purging, training set should be smaller (or equal if no overlap)
            assert len(train_with) <= len(train_no)

    def test_purge_empties_train_raises(self) -> None:
        """An over-aggressive purge_gap raises instead of yielding empty trains (#36)."""
        X = np.arange(20).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=2, n_test_splits=1, purge_gap=100)

        with pytest.raises(ValueError, match="purge/embargo removal emptied"):
            list(cv.split(X))

    def test_raises_at_call_time(self) -> None:
        """The empty-train raise fires at split() call time, before iteration (#36)."""
        X = np.arange(20).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=2, n_test_splits=1, purge_gap=100)

        with pytest.raises(ValueError, match="purge/embargo removal emptied"):
            cv.split(X)  # no iteration

    def test_more_splits_than_samples_raises(self) -> None:
        """n_samples < n_splits raises a clear message, not a numpy crash (#36)."""
        X = np.arange(3).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        with pytest.raises(ValueError, match="groups would be empty"):
            list(cv.split(X))

    def test_n_samples_equals_n_splits_is_legal(self) -> None:
        """The exact n_samples == n_splits boundary yields all paths (#36).

        np.array_split makes five singleton groups — minimal but legal;
        guards against an off-by-one (`<=`) regression in the group guard.
        """
        X = np.arange(5).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, embargo_pct=0.0)

        splits = list(cv.split(X))
        assert len(splits) == cv.get_n_splits() == 10
        for train, test in splits:
            assert len(train) == 3
            assert len(test) == 2

    def test_train_indices_integer_dtype(self) -> None:
        """Train indices keep an integer dtype usable for array indexing (#36)."""
        X = np.arange(100).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=5)

        for train, _ in cv.split(X):
            assert np.issubdtype(train.dtype, np.integer)

    def test_interior_block_embargo_fires(self) -> None:
        """Every contiguous test run is embargoed, not just the global max (#38).

        n=100, n_splits=5, n_test_splits=2, embargo_pct=0.10, purge_gap=0:
        path (g0, g2) tests [0..19] u [40..59]. Pre-#38 the embargo used the
        global test_max only, leaving rows 20-29 (right after the FIRST test
        block) in train — the headline leak. The per-run fix embargoes
        ceil(0.10*100)=10 rows after EACH run: 20-29 AND 60-69. This is the
        byte-exact corrected-output pin for the bug.
        """
        X = np.arange(100).reshape(-1, 1)
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=0, embargo_pct=0.10)
        target = set(range(0, 20)) | set(range(40, 60))
        train = next(tr for tr, te in cv.split(X) if set(te.tolist()) == target)
        train_set = set(train.tolist())

        assert set(range(20, 30)).isdisjoint(train_set)  # interior boundary (was the leak)
        assert set(range(60, 70)).isdisjoint(train_set)  # trailing boundary
        assert set(range(30, 40)).issubset(train_set)  # beyond embargo span: kept
        assert set(range(70, 100)).issubset(train_set)
        # exact corrected train set for this path
        assert sorted(train_set) == list(range(30, 40)) + list(range(70, 100))


class TestPurgedWalkForward:
    """Tests for PurgedWalkForward cross-validator."""

    def test_initialization(self) -> None:
        """Should initialize with valid parameters."""
        cv = PurgedWalkForward(n_splits=5, train_size=100, test_size=20, purge_gap=5)

        assert cv.n_splits == 5
        assert cv.train_size == 100
        assert cv.test_size == 20
        assert cv.purge_gap == 5

    def test_invalid_parameters(self) -> None:
        """Should raise for invalid parameters."""
        with pytest.raises(ValueError):
            PurgedWalkForward(n_splits=0)

        with pytest.raises(ValueError):
            PurgedWalkForward(train_size=0)

        with pytest.raises(ValueError):
            PurgedWalkForward(test_size=0)

        with pytest.raises(ValueError):
            PurgedWalkForward(purge_gap=-1)

    def test_temporal_order(self) -> None:
        """Test indices should always come after train indices."""
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=5, train_size=50, test_size=20)

        for train_idx, test_idx in cv.split(X):
            assert np.max(train_idx) < np.min(test_idx)

    def test_expanding_window(self) -> None:
        """With train_size=None, should use expanding window."""
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=3, train_size=None, test_size=20)

        splits = list(cv.split(X))
        train_sizes = [len(train) for train, _ in splits]

        # Expanding window: each split should have more training data
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_fixed_window(self) -> None:
        """With train_size specified, should use fixed window."""
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=3, train_size=50, test_size=20, purge_gap=0)

        train_sizes = [len(train) for train, _ in cv.split(X)]

        # Fixed window: all training sets should have same size
        assert all(size == train_sizes[0] for size in train_sizes)

    def test_purging_creates_gap(self) -> None:
        """Purging should create gap between train and test."""
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=3, train_size=50, test_size=20, purge_gap=10)

        for train_idx, test_idx in cv.split(X):
            gap = np.min(test_idx) - np.max(train_idx)
            assert gap >= 10  # At least purge_gap

    def test_split_detailed(self) -> None:
        """split_detailed should return PurgedSplit objects."""
        X = np.arange(200).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=3, train_size=50, test_size=20, purge_gap=5)

        for split in cv.split_detailed(X):
            assert isinstance(split, PurgedSplit)
            assert split.n_purged >= 0

    def test_gap_parameter(self) -> None:
        """Additional gap should create larger separation."""
        X = np.arange(200).reshape(-1, 1)
        cv_no_gap = PurgedWalkForward(n_splits=3, test_size=20, extra_gap=0, purge_gap=0)
        cv_with_gap = PurgedWalkForward(n_splits=3, test_size=20, extra_gap=10, purge_gap=0)

        for (train_no, test_no), (train_with, test_with) in zip(
            cv_no_gap.split(X), cv_with_gap.split(X)
        ):
            gap_no = np.min(test_no) - np.max(train_no)
            gap_with = np.min(test_with) - np.max(train_with)
            assert gap_with > gap_no

    def test_fixed_window_truncation_raises(self) -> None:
        """A fixed train_size that cannot fit raises instead of truncating (#35).

        Fold 0 has 50 samples of history but train_size=60; pre-fix the
        window was silently clipped to [0, 50).
        """
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=5, train_size=60, test_size=10, purge_gap=0)

        with pytest.raises(ValueError, match="does not fit"):
            list(cv.split(X))

    def test_auto_test_size_insufficient_raises(self) -> None:
        """The auto test_size branch raises when nothing is available (#35 repro).

        Pre-fix `max(1, available // n_splits)` clamped available=-100 to a
        1-sample test window and the truncated trains went unnoticed.
        """
        X = np.zeros(100)
        cv = PurgedWalkForward(n_splits=5, train_size=200)

        with pytest.raises(ValueError, match="cannot auto-size test windows"):
            cv.split(X)  # no iteration: raise fires at call time

    def test_fixed_window_exact_fit_passes(self) -> None:
        """A fixed window that exactly fits yields folds of exactly train_size (#35).

        Fold 0 is binding: train [0, 50) with zero slack. With purge_gap and
        embargo zeroed, every fold's train is the untouched geometric window.
        """
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedWalkForward(
            n_splits=5, train_size=50, test_size=10, purge_gap=0, embargo_pct=0.0
        )

        splits = list(cv.split(X))
        assert len(splits) == 5
        for train, _ in splits:
            assert len(train) == 50


class TestEdgeCases:
    """Edge case tests."""

    def test_small_dataset_purged_kfold(self) -> None:
        """Should handle small datasets."""
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedKFold(n_splits=2, purge_gap=2)

        splits = list(cv.split(X))
        assert len(splits) == 2

    def test_large_purge_gap(self) -> None:
        """Large purge_gap should heavily reduce training set."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedKFold(n_splits=5, purge_gap=20)

        for train_idx, _ in cv.split(X):
            # Training set should be significantly reduced
            assert len(train_idx) < 80  # Less than 80% of data

    def test_embargo_removes_samples(self) -> None:
        """Embargo should remove additional samples."""
        X = np.arange(100).reshape(-1, 1)
        cv_no_embargo = PurgedKFold(n_splits=5, embargo_pct=0.0)
        cv_with_embargo = PurgedKFold(n_splits=5, embargo_pct=0.1)

        for (train_no, _), (train_with, _) in zip(cv_no_embargo.split(X), cv_with_embargo.split(X)):
            assert len(train_with) <= len(train_no)

    def test_list_input(self) -> None:
        """Should accept list input."""
        X_list = [[i, i + 1] for i in range(100)]
        cv = PurgedKFold(n_splits=5)

        splits = list(cv.split(X_list))
        assert len(splits) == 5

    def test_combinatorial_single_test_split(self) -> None:
        """CombinatorialPurgedCV with n_test_splits=1 should equal K-fold paths."""
        from math import comb

        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1)

        assert cv.get_n_splits() == comb(5, 1)  # 5 paths

    def test_walk_forward_insufficient_data_raises(self) -> None:
        """An under-provisioned config raises instead of dropping folds (#32).

        Since #35 the fixed-window branch reports this as a window that
        does not fit (rather than the post-truncation "empty train window").
        """
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=10, train_size=15, test_size=5)

        with pytest.raises(ValueError, match="does not fit"):
            list(cv.split(X))

    def test_walk_forward_raises_at_call_time(self) -> None:
        """The under-provisioned raise fires at split() call time (#32).

        Pins the call-time contract: a consumer that stores the result
        without iterating still sees the error immediately.
        """
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=10, train_size=15, test_size=5)

        with pytest.raises(ValueError, match="does not fit"):
            cv.split(X)  # no iteration

    def test_walk_forward_split_detailed_raises_at_call_time(self) -> None:
        """split_detailed's own documented call-time contract holds (#32).

        Pins it independently of split(): a `yield from` refactor of
        split_detailed would defer the raise to first next() and split()
        would still mask it via its eager list comprehension.
        """
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=10, train_size=15, test_size=5)

        with pytest.raises(ValueError, match="does not fit"):
            cv.split_detailed(X)  # no iteration

    def test_apply_purge_and_embargo_empty_result_integer_dtype(self) -> None:
        """An emptied purged train keeps an integer dtype (#36).

        Pins the dtype=np.intp pin directly: the public splitters now raise
        before an empty train can surface, so only this unit-level test
        catches a revert to bare np.array([]) (which infers float64).
        """
        purged_train, n_purged, _ = _apply_purge_and_embargo(
            train_indices=np.arange(0, 5),
            test_indices=np.arange(5, 10),
            n_samples=10,
            purge_gap=100,
            embargo_pct=0.0,
        )

        assert len(purged_train) == 0
        assert np.issubdtype(purged_train.dtype, np.integer)
        assert n_purged == 5

    def test_apply_purge_and_embargo_one_sided_per_run(self) -> None:
        """Embargo is one-sided (after) and per contiguous run (#38).

        A two-block test set [10..14] u [30..34] with embargo_pct=0.10, n=50
        (n_embargo = ceil(5.0) = 5) embargoes only the 5 rows AFTER each run —
        [15..19] and [35..39] — and nothing before either run (purge_gap=0),
        directly pinning the per-run one-sided semantics at the helper level.
        """
        test = np.concatenate([np.arange(10, 15), np.arange(30, 35)])
        train = np.array([i for i in range(50) if i not in set(test.tolist())])
        purged, n_purged, n_emb = _apply_purge_and_embargo(
            train, test, n_samples=50, purge_gap=0, embargo_pct=0.10
        )
        removed = set(train.tolist()) - set(purged.tolist())

        assert removed == set(range(15, 20)) | set(range(35, 40))
        assert n_purged == 0  # purge_gap=0, test indices are not in train
        assert n_emb == 10
        assert purged.dtype == np.intp

    def test_walk_forward_expanding_window_insufficient_data_raises(self) -> None:
        """The expanding-window branch raises on a squeezed train window (#32)."""
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=5, train_size=None, test_size=20)

        with pytest.raises(ValueError, match="empty train window"):
            list(cv.split(X))

    def test_walk_forward_high_embargo_is_noop(self) -> None:
        """In forward-only geometry the one-sided embargo is a no-op (#38).

        The train window [70, 80) precedes the test window [80, 100), so the
        after-test embargo removes nothing even at embargo_pct=0.5 — the window
        is delivered intact. (Pre-#38 a non-canonical *pre-test* embargo
        swallowed all 10 rows and raised; the embargo is now one-sided/after
        each test run, matching De Prado.)
        """
        X = np.arange(100).reshape(-1, 1)
        cv = PurgedWalkForward(
            n_splits=1, train_size=10, test_size=20, purge_gap=0, embargo_pct=0.5
        )

        (detailed,) = list(cv.split_detailed(X))
        assert detailed.n_embargoed == 0
        np.testing.assert_array_equal(detailed.train_indices, np.arange(70, 80))

    def test_walk_forward_provisioned_yields_all_splits(self) -> None:
        """A provisioned config yields exactly get_n_splits folds (#32)."""
        X = np.arange(300).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=5, train_size=50, test_size=20, purge_gap=5)

        splits = list(cv.split(X))
        assert len(splits) == cv.get_n_splits() == 5


class TestIntegration:
    """Integration tests with mock models."""

    def test_purged_kfold_with_model(self) -> None:
        """Test PurgedKFold with a simple model."""
        from sklearn.linear_model import Ridge

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

        cv = PurgedKFold(n_splits=5, purge_gap=3)
        scores = []

        for train_idx, test_idx in cv.split(X, y):
            model = Ridge()
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
            scores.append(score)

        assert len(scores) == 5
        assert all(s > 0.9 for s in scores)  # Ridge should perform well

    def test_walk_forward_with_model(self) -> None:
        """Test PurgedWalkForward with a simple model."""
        from sklearn.linear_model import Ridge

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(n) * 0.1

        cv = PurgedWalkForward(n_splits=3, train_size=50, test_size=20, purge_gap=5)
        scores = []

        for train_idx, test_idx in cv.split(X, y):
            model = Ridge()
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
            scores.append(score)

        assert len(scores) >= 1  # At least one split should work
