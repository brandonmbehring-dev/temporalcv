"""
Tests for financial cross-validation with purging and embargo.

Test categories:
1. PurgedKFold tests
2. CombinatorialPurgedCV tests
3. PurgedWalkForward tests
4. Utility function tests
5. Edge cases
"""

import numpy as np
import pytest

from temporalcv.cv_financial import (
    CombinatorialPurgedCV,
    PurgedKFold,
    PurgedSplit,
    PurgedWalkForward,
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

        for (train_no, _), (train_with, _) in zip(
            cv_no_purge.split(X), cv_with_purge.split(X)
        ):
            # With purging, training set should be smaller (or equal if no overlap)
            assert len(train_with) <= len(train_no)


class TestPurgedWalkForward:
    """Tests for PurgedWalkForward cross-validator."""

    def test_initialization(self) -> None:
        """Should initialize with valid parameters."""
        cv = PurgedWalkForward(
            n_splits=5, train_size=100, test_size=20, purge_gap=5
        )

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

        for (train_no, _), (train_with, _) in zip(
            cv_no_embargo.split(X), cv_with_embargo.split(X)
        ):
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

    def test_walk_forward_insufficient_data(self) -> None:
        """Walk-forward should handle insufficient data gracefully."""
        X = np.arange(20).reshape(-1, 1)
        cv = PurgedWalkForward(n_splits=10, train_size=15, test_size=5)

        # Should still produce some splits (maybe fewer than requested)
        splits = list(cv.split(X))
        assert len(splits) <= 10


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
