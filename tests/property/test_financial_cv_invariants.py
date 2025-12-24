"""Property-based tests for financial CV module.

Tests invariants of PurgedKFold, CombinatorialPurgedCV, and PurgedWalkForward.
"""

import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from math import comb

from temporalcv.cv_financial import (
    PurgedKFold,
    CombinatorialPurgedCV,
    PurgedWalkForward,
    compute_label_overlap,
    estimate_purge_gap,
)


# Custom strategies
@st.composite
def valid_purged_kfold_params(draw: st.DrawFn) -> dict:
    """Generate valid parameters for PurgedKFold."""
    n_samples = draw(st.integers(min_value=50, max_value=500))
    n_splits = draw(st.integers(min_value=2, max_value=min(10, n_samples // 5)))
    purge_gap = draw(st.integers(min_value=0, max_value=10))
    embargo_pct = draw(st.floats(min_value=0.0, max_value=0.1))

    # Ensure there's enough data for splits
    min_fold_size = n_samples // n_splits
    assume(min_fold_size > purge_gap)

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "purge_gap": purge_gap,
        "embargo_pct": embargo_pct,
    }


@st.composite
def valid_cpcv_params(draw: st.DrawFn) -> dict:
    """Generate valid parameters for CombinatorialPurgedCV."""
    n_samples = draw(st.integers(min_value=100, max_value=500))
    n_splits = draw(st.integers(min_value=3, max_value=6))
    n_test_splits = draw(st.integers(min_value=1, max_value=n_splits - 1))
    purge_gap = draw(st.integers(min_value=0, max_value=5))

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "purge_gap": purge_gap,
    }


@st.composite
def valid_walk_forward_params(draw: st.DrawFn) -> dict:
    """Generate valid parameters for PurgedWalkForward."""
    n_samples = draw(st.integers(min_value=200, max_value=1000))
    n_splits = draw(st.integers(min_value=2, max_value=10))
    test_size = draw(st.integers(min_value=10, max_value=50))
    train_size = draw(st.integers(min_value=50, max_value=200))
    purge_gap = draw(st.integers(min_value=0, max_value=10))

    # Ensure enough data
    min_required = train_size + test_size + purge_gap
    assume(n_samples >= min_required * 2)

    return {
        "n_samples": n_samples,
        "n_splits": n_splits,
        "train_size": train_size,
        "test_size": test_size,
        "purge_gap": purge_gap,
    }


class TestPurgedKFoldInvariants:
    """Property tests for PurgedKFold."""

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_train_test_never_overlap(self, params: dict) -> None:
        """Train and test indices must never overlap."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
            embargo_pct=params["embargo_pct"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0, "Train and test overlap!"

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_purge_gap_respected(self, params: dict) -> None:
        """Training samples must respect purge gap from test samples."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
            embargo_pct=0.0,  # Test purge without embargo
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            # Find minimum distance from any train sample to any test sample
            train_arr = np.array(list(train_idx))
            test_arr = np.array(list(test_idx))

            # For each test index, check distance to nearest train index
            for t_idx in test_arr:
                if len(train_arr) > 0:
                    distances = np.abs(train_arr - t_idx)
                    min_dist = np.min(distances)
                    # Must be at least purge_gap away (or 1 if purge_gap=0)
                    assert min_dist >= max(1, params["purge_gap"]), \
                        f"Purge gap violated: min_dist={min_dist}, purge_gap={params['purge_gap']}"

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_correct_number_of_splits(self, params: dict) -> None:
        """Should produce exactly n_splits folds."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        splits = list(cv.split(X))
        assert len(splits) == params["n_splits"]

    @given(params=valid_purged_kfold_params())
    @settings(max_examples=100)
    def test_all_indices_valid(self, params: dict) -> None:
        """All indices must be in valid range [0, n_samples)."""
        cv = PurgedKFold(
            n_splits=params["n_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            assert all(0 <= i < params["n_samples"] for i in train_idx)
            assert all(0 <= i < params["n_samples"] for i in test_idx)


class TestCombinatorialPurgedCVInvariants:
    """Property tests for CombinatorialPurgedCV."""

    @given(params=valid_cpcv_params())
    @settings(max_examples=50)
    def test_correct_number_of_paths(self, params: dict) -> None:
        """Should produce C(n_splits, n_test_splits) paths."""
        cv = CombinatorialPurgedCV(
            n_splits=params["n_splits"],
            n_test_splits=params["n_test_splits"],
            purge_gap=params["purge_gap"],
        )

        expected_paths = comb(params["n_splits"], params["n_test_splits"])
        assert cv.get_n_splits() == expected_paths

    @given(params=valid_cpcv_params())
    @settings(max_examples=50)
    def test_train_test_never_overlap(self, params: dict) -> None:
        """Train and test indices must never overlap."""
        cv = CombinatorialPurgedCV(
            n_splits=params["n_splits"],
            n_test_splits=params["n_test_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0

    @given(params=valid_cpcv_params())
    @settings(max_examples=50)
    def test_all_indices_valid(self, params: dict) -> None:
        """All indices must be in valid range."""
        cv = CombinatorialPurgedCV(
            n_splits=params["n_splits"],
            n_test_splits=params["n_test_splits"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            assert all(0 <= i < params["n_samples"] for i in train_idx)
            assert all(0 <= i < params["n_samples"] for i in test_idx)


class TestPurgedWalkForwardInvariants:
    """Property tests for PurgedWalkForward."""

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_train_precedes_test(self, params: dict) -> None:
        """All train indices must precede test indices."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert np.max(train_idx) < np.min(test_idx), \
                    "Train indices must precede test indices!"

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_purge_gap_creates_separation(self, params: dict) -> None:
        """Purge gap should create separation between train and test."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            if len(train_idx) > 0 and len(test_idx) > 0:
                gap = np.min(test_idx) - np.max(train_idx) - 1
                assert gap >= params["purge_gap"], \
                    f"Gap {gap} less than purge_gap {params['purge_gap']}"

    @given(params=valid_walk_forward_params())
    @settings(max_examples=100)
    def test_train_test_never_overlap(self, params: dict) -> None:
        """Train and test must never overlap."""
        cv = PurgedWalkForward(
            n_splits=params["n_splits"],
            train_size=params["train_size"],
            test_size=params["test_size"],
            purge_gap=params["purge_gap"],
        )
        X = np.zeros((params["n_samples"], 1))

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx)
            test_set = set(test_idx)
            assert len(train_set & test_set) == 0


class TestLabelOverlapInvariants:
    """Property tests for compute_label_overlap."""

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_overlap_matrix_symmetric(self, n_samples: int, horizon: int) -> None:
        """Overlap matrix must be symmetric."""
        overlap = compute_label_overlap(n_samples=n_samples, horizon=horizon)
        np.testing.assert_array_equal(overlap, overlap.T)

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_overlap_matrix_diagonal_true(self, n_samples: int, horizon: int) -> None:
        """Diagonal must always be True (self-overlap)."""
        overlap = compute_label_overlap(n_samples=n_samples, horizon=horizon)
        assert np.all(np.diag(overlap))

    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        horizon=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=100)
    def test_overlap_matrix_shape(self, n_samples: int, horizon: int) -> None:
        """Overlap matrix must be (n_samples, n_samples)."""
        overlap = compute_label_overlap(n_samples=n_samples, horizon=horizon)
        assert overlap.shape == (n_samples, n_samples)


class TestEstimatePurgeGapInvariants:
    """Property tests for estimate_purge_gap."""

    @given(
        horizon=st.integers(min_value=1, max_value=100),
        decay_factor=st.floats(min_value=0.1, max_value=3.0),
    )
    @settings(max_examples=100)
    def test_purge_gap_non_negative(self, horizon: int, decay_factor: float) -> None:
        """Purge gap must always be non-negative."""
        gap = estimate_purge_gap(horizon=horizon, decay_factor=decay_factor)
        assert gap >= 0

    @given(
        horizon=st.integers(min_value=1, max_value=100),
        decay_factor=st.floats(min_value=0.1, max_value=3.0),
    )
    @settings(max_examples=100)
    def test_purge_gap_integer(self, horizon: int, decay_factor: float) -> None:
        """Purge gap must be an integer."""
        gap = estimate_purge_gap(horizon=horizon, decay_factor=decay_factor)
        assert isinstance(gap, int)

    @given(horizon=st.integers(min_value=1, max_value=100))
    @settings(max_examples=50)
    def test_default_decay_returns_horizon(self, horizon: int) -> None:
        """Default decay factor of 1.0 should return horizon."""
        gap = estimate_purge_gap(horizon=horizon, decay_factor=1.0)
        assert gap == horizon
