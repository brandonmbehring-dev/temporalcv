"""Property-based tests for cross-validation invariants.

These tests verify invariants that should ALWAYS hold:
1. Train and test indices never overlap
2. Gap is always respected: train_end + gap < test_start
3. All indices are within valid range
4. Window size is respected
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from temporalcv.cv import WalkForwardCV, SplitInfo


# === Strategies ===

@st.composite
def valid_cv_params(draw):
    """Generate valid parameters for WalkForwardCV."""
    n_samples = draw(st.integers(min_value=50, max_value=500))
    window_size = draw(st.integers(min_value=10, max_value=n_samples // 2))
    gap = draw(st.integers(min_value=0, max_value=10))
    test_size = draw(st.integers(min_value=1, max_value=5))
    window_type = draw(st.sampled_from(["sliding", "expanding"]))

    # Ensure we have enough data for at least one split
    min_required = window_size + gap + test_size
    assume(n_samples >= min_required + 10)  # At least 10 extra for multiple splits

    return {
        "n_samples": n_samples,
        "window_size": window_size,
        "gap": gap,
        "test_size": test_size,
        "window_type": window_type,
    }


# === Core CV Invariants ===

class TestCVCoreInvariants:
    """Test core invariants that must always hold."""

    @given(valid_cv_params())
    @settings(max_examples=100)
    def test_train_test_never_overlap(self, params):
        """Train and test indices must never overlap."""
        n_samples = params["n_samples"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type=params["window_type"],
            window_size=params["window_size"],
            extra_gap=params["gap"],
            test_size=params["test_size"],
        )

        for train_idx, test_idx in cv.split(X, y):
            train_set = set(train_idx)
            test_set = set(test_idx)

            # No overlap
            assert len(train_set & test_set) == 0, (
                f"Train and test overlap: {train_set & test_set}"
            )

    @given(valid_cv_params())
    @settings(max_examples=100)
    def test_gap_always_respected(self, params):
        """Gap between train end and test start must be respected."""
        n_samples = params["n_samples"]
        gap = params["gap"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type=params["window_type"],
            window_size=params["window_size"],
            extra_gap=gap,
            test_size=params["test_size"],
        )

        for train_idx, test_idx in cv.split(X, y):
            train_end = train_idx[-1]
            test_start = test_idx[0]

            # Gap must be respected
            actual_gap = test_start - train_end - 1
            assert actual_gap >= gap, (
                f"Gap violated: expected >= {gap}, got {actual_gap}"
            )

    @given(valid_cv_params())
    @settings(max_examples=100)
    def test_indices_within_bounds(self, params):
        """All indices must be within valid range [0, n_samples)."""
        n_samples = params["n_samples"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type=params["window_type"],
            window_size=params["window_size"],
            extra_gap=params["gap"],
            test_size=params["test_size"],
        )

        for train_idx, test_idx in cv.split(X, y):
            # All train indices in range
            assert all(0 <= i < n_samples for i in train_idx)

            # All test indices in range
            assert all(0 <= i < n_samples for i in test_idx)

    @given(valid_cv_params())
    @settings(max_examples=100)
    def test_train_precedes_test(self, params):
        """All train indices must precede all test indices."""
        n_samples = params["n_samples"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type=params["window_type"],
            window_size=params["window_size"],
            extra_gap=params["gap"],
            test_size=params["test_size"],
        )

        for train_idx, test_idx in cv.split(X, y):
            assert max(train_idx) < min(test_idx), (
                f"Train should precede test: max(train)={max(train_idx)}, "
                f"min(test)={min(test_idx)}"
            )


# === Window Type Invariants ===

class TestWindowTypeInvariants:
    """Test invariants specific to window types."""

    @given(valid_cv_params())
    @settings(max_examples=50)
    def test_sliding_window_size_constant(self, params):
        """Sliding window should maintain constant training size."""
        assume(params["window_type"] == "sliding")

        n_samples = params["n_samples"]
        window_size = params["window_size"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type="sliding",
            window_size=window_size,
            extra_gap=params["gap"],
            test_size=params["test_size"],
        )

        train_sizes = [len(train_idx) for train_idx, _ in cv.split(X, y)]

        if train_sizes:  # Only check if we have splits
            # All training sets should be same size
            assert all(s == train_sizes[0] for s in train_sizes), (
                f"Sliding window should have constant size: {set(train_sizes)}"
            )

    @given(valid_cv_params())
    @settings(max_examples=50)
    def test_expanding_window_grows(self, params):
        """Expanding window should grow or stay same over splits."""
        assume(params["window_type"] == "expanding")

        n_samples = params["n_samples"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type="expanding",
            window_size=params["window_size"],
            extra_gap=params["gap"],
            test_size=params["test_size"],
        )

        train_sizes = [len(train_idx) for train_idx, _ in cv.split(X, y)]

        if len(train_sizes) > 1:
            # Each training set should be >= previous
            for i in range(1, len(train_sizes)):
                assert train_sizes[i] >= train_sizes[i-1], (
                    f"Expanding window should grow: {train_sizes}"
                )


# === Test Size Invariants ===

class TestTestSizeInvariants:
    """Test invariants for test set size."""

    @given(valid_cv_params())
    @settings(max_examples=100)
    def test_test_size_respected(self, params):
        """Test size should match requested size (except possibly last split)."""
        n_samples = params["n_samples"]
        test_size = params["test_size"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type=params["window_type"],
            window_size=params["window_size"],
            extra_gap=params["gap"],
            test_size=test_size,
        )

        splits = list(cv.split(X, y))

        if splits:
            # All but possibly last should have exact test_size
            for train_idx, test_idx in splits[:-1]:
                assert len(test_idx) == test_size

            # Last can be <= test_size (may have fewer remaining samples)
            _, last_test_idx = splits[-1]
            assert len(last_test_idx) <= test_size


# === SplitInfo Invariants ===

class TestSplitInfoInvariants:
    """Test invariants for SplitInfo metadata."""

    @given(valid_cv_params())
    @settings(max_examples=50)
    def test_split_indices_monotonic(self, params):
        """Split indices should be monotonically increasing."""
        n_samples = params["n_samples"]
        X = np.random.randn(n_samples, 3)
        y = np.random.randn(n_samples)

        cv = WalkForwardCV(
            window_type=params["window_type"],
            window_size=params["window_size"],
            extra_gap=params["gap"],
            test_size=params["test_size"],
        )

        prev_test_end = -1
        for train_idx, test_idx in cv.split(X, y):
            test_start = min(test_idx)
            test_end = max(test_idx)

            # This split's test should start after previous split's test
            assert test_start > prev_test_end, (
                f"Splits should progress: test_start={test_start}, "
                f"prev_test_end={prev_test_end}"
            )

            prev_test_end = test_end
