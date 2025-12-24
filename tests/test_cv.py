"""
Tests for temporalcv.cv module.

Tests walk-forward cross-validation including:
- SplitInfo dataclass
- WalkForwardCV splitter
- Gap enforcement (leakage prevention)
- Window types (expanding, sliding)
- sklearn compatibility
- Edge cases and error handling
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from temporalcv.cv import SplitInfo, WalkForwardCV


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data for testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 5))
    y = rng.standard_normal(n)
    return X, y


@pytest.fixture
def small_data() -> tuple[np.ndarray, np.ndarray]:
    """Small dataset for edge case testing."""
    rng = np.random.default_rng(42)
    n = 20
    X = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    return X, y


# =============================================================================
# SplitInfo Tests
# =============================================================================


class TestSplitInfo:
    """Tests for SplitInfo dataclass."""

    def test_basic_creation(self) -> None:
        """SplitInfo should be creatable with valid indices."""
        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=102,
        )
        assert info.split_idx == 0
        assert info.train_start == 0
        assert info.train_end == 99
        assert info.test_start == 102
        assert info.test_end == 102

    def test_train_size_property(self) -> None:
        """train_size should be computed correctly."""
        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=102,
        )
        assert info.train_size == 100

    def test_test_size_property(self) -> None:
        """test_size should be computed correctly."""
        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
        )
        assert info.test_size == 3

    def test_gap_property(self) -> None:
        """gap should be computed correctly."""
        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=102,
        )
        assert info.gap == 2  # 102 - 99 - 1 = 2

    def test_invalid_temporal_ordering(self) -> None:
        """Should raise error if train_end >= test_start."""
        with pytest.raises(ValueError, match="Temporal leakage"):
            SplitInfo(
                split_idx=0,
                train_start=0,
                train_end=100,
                test_start=100,  # Same as train_end
                test_end=101,
            )

    def test_overlapping_raises_error(self) -> None:
        """Should raise error if train and test overlap."""
        with pytest.raises(ValueError, match="Temporal leakage"):
            SplitInfo(
                split_idx=0,
                train_start=0,
                train_end=105,
                test_start=100,  # Before train_end
                test_end=110,
            )

    def test_frozen_immutability(self) -> None:
        """SplitInfo should be immutable (frozen dataclass)."""
        from dataclasses import FrozenInstanceError

        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=100,
            test_end=109,
        )
        # Attempting to modify any field should raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            info.split_idx = 1  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            info.train_end = 50  # type: ignore[misc]


# =============================================================================
# WalkForwardCV Basic Tests
# =============================================================================


class TestWalkForwardCV:
    """Tests for WalkForwardCV core functionality."""

    def test_default_initialization(self) -> None:
        """Default parameters should be valid."""
        cv = WalkForwardCV()
        assert cv.n_splits == 5
        assert cv.window_type == "expanding"
        assert cv.window_size is None
        assert cv.gap == 0
        assert cv.test_size == 1

    def test_custom_initialization(self) -> None:
        """Custom parameters should be stored."""
        cv = WalkForwardCV(
            n_splits=10,
            window_type="sliding",
            window_size=50,
            gap=2,
            test_size=3,
        )
        assert cv.n_splits == 10
        assert cv.window_type == "sliding"
        assert cv.window_size == 50
        assert cv.gap == 2
        assert cv.test_size == 3

    def test_repr(self) -> None:
        """__repr__ should be informative."""
        cv = WalkForwardCV(n_splits=3, gap=2)
        repr_str = repr(cv)
        assert "WalkForwardCV" in repr_str
        assert "n_splits=3" in repr_str
        assert "gap=2" in repr_str

    def test_split_yields_correct_count(self, sample_data: tuple) -> None:
        """split() should yield n_splits tuples."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5)

        splits = list(cv.split(X, y))
        assert len(splits) == 5

    def test_split_yields_numpy_arrays(self, sample_data: tuple) -> None:
        """split() should yield numpy arrays."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3)

        for train, test in cv.split(X, y):
            assert isinstance(train, np.ndarray)
            assert isinstance(test, np.ndarray)
            assert train.dtype == np.intp
            assert test.dtype == np.intp

    def test_get_n_splits_without_data(self) -> None:
        """get_n_splits() without X returns configured value."""
        cv = WalkForwardCV(n_splits=7)
        assert cv.get_n_splits() == 7

    def test_get_n_splits_with_data(self, sample_data: tuple) -> None:
        """get_n_splits() with X returns actual splits."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5)
        assert cv.get_n_splits(X) == 5

    def test_get_split_info(self, sample_data: tuple) -> None:
        """get_split_info() should return SplitInfo objects."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, gap=2)

        infos = cv.get_split_info(X)
        assert len(infos) == 3
        for i, info in enumerate(infos):
            assert isinstance(info, SplitInfo)
            assert info.split_idx == i
            assert info.gap >= 2


# =============================================================================
# Gap Enforcement Tests
# =============================================================================


class TestGapEnforcement:
    """Tests for gap parameter enforcement."""

    def test_gap_enforced_between_splits(self, sample_data: tuple) -> None:
        """Gap should be maintained between train and test."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, gap=3)

        for train, test in cv.split(X):
            # train[-1] + gap + 1 <= test[0]
            actual_gap = test[0] - train[-1] - 1
            assert actual_gap >= 3, f"Gap {actual_gap} < required 3"

    def test_gap_zero_allowed(self, sample_data: tuple) -> None:
        """gap=0 should work (adjacent train/test)."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, gap=0)

        for train, test in cv.split(X):
            # With gap=0, test should start right after train
            assert test[0] == train[-1] + 1

    def test_gap_prevents_leakage(self, sample_data: tuple) -> None:
        """Train indices should never include test indices."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, gap=2)

        for train, test in cv.split(X):
            train_set = set(train)
            test_set = set(test)
            overlap = train_set & test_set
            assert len(overlap) == 0, f"Overlap detected: {overlap}"

    def test_large_gap(self, sample_data: tuple) -> None:
        """Large gap should still work."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, gap=10)

        splits = list(cv.split(X))
        assert len(splits) == 3

        for train, test in splits:
            actual_gap = test[0] - train[-1] - 1
            assert actual_gap >= 10

    def test_no_overlap_between_consecutive_tests(self, sample_data: tuple) -> None:
        """Consecutive test sets should not overlap."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, test_size=1)

        prev_test: set[int] = set()
        for train, test in cv.split(X):
            test_set = set(test)
            if prev_test:
                overlap = prev_test & test_set
                assert len(overlap) == 0, f"Test overlap: {overlap}"
            prev_test = test_set


# =============================================================================
# Window Type Tests
# =============================================================================


class TestWindowTypes:
    """Tests for expanding and sliding window types."""

    def test_expanding_window_grows(self, sample_data: tuple) -> None:
        """Expanding window should grow with each split."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, window_type="expanding")

        train_sizes = [len(train) for train, test in cv.split(X)]

        # Each subsequent training set should be larger or equal
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1], (
                f"Expanding window shrunk: {train_sizes[i-1]} -> {train_sizes[i]}"
            )

    def test_sliding_window_fixed_size(self, sample_data: tuple) -> None:
        """Sliding window should maintain fixed size."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, window_type="sliding", window_size=50)

        for train, test in cv.split(X):
            assert len(train) == 50, f"Train size {len(train)} != 50"

    def test_sliding_requires_window_size(self) -> None:
        """Sliding window should require window_size."""
        with pytest.raises(ValueError, match="window_size is required"):
            WalkForwardCV(window_type="sliding")

    def test_expanding_with_min_size(self, sample_data: tuple) -> None:
        """Expanding window should respect minimum size."""
        X, y = sample_data
        cv = WalkForwardCV(
            n_splits=3,
            window_type="expanding",
            window_size=30,  # Minimum size
        )

        train_sizes = [len(train) for train, test in cv.split(X)]

        # First split should have at least 30 samples
        assert train_sizes[0] >= 30

    def test_sliding_vs_expanding_difference(self, sample_data: tuple) -> None:
        """Sliding and expanding should produce different splits."""
        X, y = sample_data

        cv_expanding = WalkForwardCV(
            n_splits=3,
            window_type="expanding",
            window_size=50,
        )
        cv_sliding = WalkForwardCV(
            n_splits=3,
            window_type="sliding",
            window_size=50,
        )

        expanding_sizes = [len(train) for train, _ in cv_expanding.split(X)]
        sliding_sizes = [len(train) for train, _ in cv_sliding.split(X)]

        # Sliding should have constant size
        assert all(s == 50 for s in sliding_sizes)

        # Expanding should grow
        assert expanding_sizes[-1] > expanding_sizes[0] or len(expanding_sizes) == 1


# =============================================================================
# sklearn Compatibility Tests
# =============================================================================


class TestSklearnCompatibility:
    """Tests for sklearn integration."""

    def test_cross_val_score_works(self, sample_data: tuple) -> None:
        """Should work with sklearn's cross_val_score."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, gap=0)

        scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=cv, scoring="r2")

        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)

    def test_cross_val_score_with_gap(self, sample_data: tuple) -> None:
        """cross_val_score should work with gap parameter."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, gap=5)

        scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=cv, scoring="r2")

        assert len(scores) == 3

    def test_cross_val_score_sliding(self, sample_data: tuple) -> None:
        """cross_val_score should work with sliding window."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, window_type="sliding", window_size=50)

        scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=cv, scoring="r2")

        assert len(scores) == 3

    def test_compatible_with_base_cv_interface(self) -> None:
        """Should implement BaseCrossValidator interface."""
        from sklearn.model_selection import BaseCrossValidator

        cv = WalkForwardCV()
        assert isinstance(cv, BaseCrossValidator)
        assert hasattr(cv, "split")
        assert hasattr(cv, "get_n_splits")


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_n_splits(self) -> None:
        """Should raise error for invalid n_splits."""
        with pytest.raises(ValueError, match="n_splits must be >= 1"):
            WalkForwardCV(n_splits=0)

    def test_invalid_window_type(self) -> None:
        """Should raise error for invalid window_type."""
        with pytest.raises(ValueError, match="window_type must be"):
            WalkForwardCV(window_type="invalid")  # type: ignore[arg-type]

    def test_invalid_window_size(self) -> None:
        """Should raise error for invalid window_size."""
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            WalkForwardCV(window_type="sliding", window_size=0)

    def test_invalid_gap(self) -> None:
        """Should raise error for negative gap."""
        with pytest.raises(ValueError, match="gap must be >= 0"):
            WalkForwardCV(gap=-1)

    def test_invalid_test_size(self) -> None:
        """Should raise error for invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be >= 1"):
            WalkForwardCV(test_size=0)

    def test_insufficient_data(self, small_data: tuple) -> None:
        """Should raise error if not enough data."""
        X, y = small_data  # 20 samples

        cv = WalkForwardCV(
            n_splits=10,
            window_type="sliding",
            window_size=50,  # More than available
        )

        with pytest.raises(ValueError, match="Not enough samples"):
            list(cv.split(X))

    def test_single_split(self, sample_data: tuple) -> None:
        """n_splits=1 should work."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=1)

        splits = list(cv.split(X))
        assert len(splits) == 1

    def test_large_test_size(self, sample_data: tuple) -> None:
        """Large test_size should work."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, test_size=10)

        for train, test in cv.split(X):
            assert len(test) == 10

    def test_test_indices_are_contiguous(self, sample_data: tuple) -> None:
        """Test indices should be contiguous."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, test_size=3)

        for train, test in cv.split(X):
            expected = np.arange(test[0], test[-1] + 1)
            np.testing.assert_array_equal(test, expected)

    def test_train_indices_are_contiguous(self, sample_data: tuple) -> None:
        """Train indices should be contiguous."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5)

        for train, test in cv.split(X):
            expected = np.arange(train[0], train[-1] + 1)
            np.testing.assert_array_equal(train, expected)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with temporalcv gates."""

    def test_splits_pass_temporal_boundary_gate(self, sample_data: tuple) -> None:
        """All splits should pass temporal boundary gate."""
        from temporalcv.gates import GateStatus, gate_temporal_boundary

        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, gap=2)

        for train, test in cv.split(X):
            result = gate_temporal_boundary(
                train_end_idx=int(train[-1]),
                test_start_idx=int(test[0]),
                horizon=2,
                gap=0,
            )
            assert result.status != GateStatus.HALT, (
                f"Split failed temporal boundary: {result.message}"
            )

    def test_splits_with_suspicious_improvement_gate(self, sample_data: tuple) -> None:
        """Splits can be validated with suspicious improvement gate."""
        from temporalcv.gates import GateStatus, gate_suspicious_improvement

        X, y = sample_data
        cv = WalkForwardCV(n_splits=3)

        for train, test in cv.split(X):
            # Train a simple model
            model = Ridge(alpha=1.0)
            model.fit(X[train], y[train])
            preds = model.predict(X[test])

            # Compute MAE
            model_mae = float(np.mean(np.abs(y[test] - preds)))
            # Baseline: mean predictor
            baseline_mae = float(np.mean(np.abs(y[test] - np.mean(y[train]))))

            # Check improvement isn't suspicious
            result = gate_suspicious_improvement(
                model_metric=model_mae,
                baseline_metric=baseline_mae,
                threshold=0.50,  # Relaxed for random data
            )
            # Just verify it runs (random data may have any result)
            assert result.status in (GateStatus.PASS, GateStatus.WARN, GateStatus.HALT)


# =============================================================================
# Horizon Validation Tests (Phase 2 Feature)
# =============================================================================


class TestHorizonValidation:
    """
    Tests for horizon parameter validation in WalkForwardCV.

    [T1] Per Bergmeir & Benitez (2012): gap must equal or exceed forecast horizon
    to prevent target leakage in multi-step forecasting.
    """

    def test_horizon_with_sufficient_gap_passes(self) -> None:
        """Horizon with gap >= horizon should work fine."""
        # gap == horizon: valid
        cv = WalkForwardCV(n_splits=3, horizon=3, gap=3)
        assert cv.horizon == 3
        assert cv.gap == 3

        # gap > horizon: also valid
        cv = WalkForwardCV(n_splits=3, horizon=2, gap=5)
        assert cv.horizon == 2
        assert cv.gap == 5

    def test_horizon_with_insufficient_gap_raises(self) -> None:
        """Horizon with gap < horizon should raise ValueError."""
        with pytest.raises(ValueError, match="gap.*must be >= horizon"):
            WalkForwardCV(n_splits=3, horizon=3, gap=2)

        with pytest.raises(ValueError, match="gap.*must be >= horizon"):
            WalkForwardCV(n_splits=3, horizon=5, gap=0)

    def test_horizon_none_allows_any_gap(self) -> None:
        """When horizon is None, any gap value is allowed."""
        # No horizon means no validation
        cv = WalkForwardCV(n_splits=3, gap=0)
        assert cv.horizon is None
        assert cv.gap == 0

        cv = WalkForwardCV(n_splits=3, gap=10)
        assert cv.horizon is None
        assert cv.gap == 10

    def test_horizon_validation_error_message_is_helpful(self) -> None:
        """Error message should explain how to fix the issue."""
        with pytest.raises(ValueError) as exc_info:
            WalkForwardCV(n_splits=3, horizon=4, gap=2)

        error_msg = str(exc_info.value)
        assert "gap (2)" in error_msg
        assert "horizon (4)" in error_msg
        assert "target leakage" in error_msg
        assert "4-step forecasting" in error_msg
        assert "gap >= 4" in error_msg

    def test_horizon_must_be_positive(self) -> None:
        """Horizon must be >= 1 if provided."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            WalkForwardCV(n_splits=3, horizon=0, gap=0)

        with pytest.raises(ValueError, match="horizon must be >= 1"):
            WalkForwardCV(n_splits=3, horizon=-1, gap=0)

    def test_horizon_is_stored_as_attribute(self) -> None:
        """Horizon should be accessible as instance attribute."""
        cv = WalkForwardCV(n_splits=5, horizon=3, gap=3)
        assert hasattr(cv, "horizon")
        assert cv.horizon == 3

    def test_splits_work_with_horizon(self, sample_data: tuple) -> None:
        """CV should generate valid splits when horizon is set."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, horizon=2, gap=2)

        splits = list(cv.split(X))
        assert len(splits) == 3

        for train, test in splits:
            # Gap is enforced
            assert train[-1] + cv.gap < test[0]
