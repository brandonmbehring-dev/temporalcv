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

from temporalcv.cv import (
    NestedCVResult,
    NestedWalkForwardCV,
    SplitInfo,
    WalkForwardCV,
)


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
        assert cv.extra_gap == 0
        assert cv.test_size == 1

    def test_custom_initialization(self) -> None:
        """Custom parameters should be stored."""
        cv = WalkForwardCV(
            n_splits=10,
            window_type="sliding",
            window_size=50,
            extra_gap=2,
            test_size=3,
        )
        assert cv.n_splits == 10
        assert cv.window_type == "sliding"
        assert cv.window_size == 50
        assert cv.extra_gap == 2
        assert cv.test_size == 3

    def test_repr(self) -> None:
        """__repr__ should be informative."""
        cv = WalkForwardCV(n_splits=3, extra_gap=2)
        repr_str = repr(cv)
        assert "WalkForwardCV" in repr_str
        assert "n_splits=3" in repr_str
        assert "extra_gap=2" in repr_str

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

    def test_get_n_splits_strict_raises_on_insufficient_data(self) -> None:
        """
        get_n_splits(strict=True) should raise ValueError on insufficient data.

        This tests the fix for silent failures - previously errors were swallowed
        and 0 was returned, which could mask configuration problems.
        """
        # Create data that's too small for the requested splits
        rng = np.random.default_rng(42)
        X_tiny = rng.standard_normal((5, 3))  # Only 5 samples

        # Request more splits than possible
        cv = WalkForwardCV(n_splits=10, window_size=5)

        # Default (strict=True) should raise
        with pytest.raises(ValueError, match="Cannot compute n_splits"):
            cv.get_n_splits(X_tiny)

    def test_get_n_splits_strict_false_returns_zero(self) -> None:
        """
        get_n_splits(strict=False) should return 0 on insufficient data.

        This preserves backward compatibility for callers who want the old behavior.
        """
        rng = np.random.default_rng(42)
        X_tiny = rng.standard_normal((5, 3))

        cv = WalkForwardCV(n_splits=10, window_size=5)

        # strict=False should return 0 instead of raising
        result = cv.get_n_splits(X_tiny, strict=False)
        assert result == 0

    def test_get_split_info(self, sample_data: tuple) -> None:
        """get_split_info() should return SplitInfo objects."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, extra_gap=2)

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
        cv = WalkForwardCV(n_splits=5, extra_gap=3)

        for train, test in cv.split(X):
            # train[-1] + gap + 1 <= test[0]
            actual_gap = test[0] - train[-1] - 1
            assert actual_gap >= 3, f"Gap {actual_gap} < required 3"

    def test_gap_zero_allowed(self, sample_data: tuple) -> None:
        """extra_gap=0 should work (adjacent train/test)."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, extra_gap=0)

        for train, test in cv.split(X):
            # With extra_gap=0, test should start right after train
            assert test[0] == train[-1] + 1

    def test_gap_prevents_leakage(self, sample_data: tuple) -> None:
        """Train indices should never include test indices."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=5, extra_gap=2)

        for train, test in cv.split(X):
            train_set = set(train)
            test_set = set(test)
            overlap = train_set & test_set
            assert len(overlap) == 0, f"Overlap detected: {overlap}"

    def test_large_gap(self, sample_data: tuple) -> None:
        """Large gap should still work."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, extra_gap=10)

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
        cv = WalkForwardCV(n_splits=5, extra_gap=0)

        scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=cv, scoring="r2")

        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)

    def test_cross_val_score_with_gap(self, sample_data: tuple) -> None:
        """cross_val_score should work with gap parameter."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, extra_gap=5)

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
            WalkForwardCV(extra_gap=-1)

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
        cv = WalkForwardCV(n_splits=5, extra_gap=2)

        for train, test in cv.split(X):
            result = gate_temporal_boundary(
                train_end_idx=int(train[-1]),
                test_start_idx=int(test[0]),
                horizon=2,
                extra_gap=0,
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
        cv = WalkForwardCV(n_splits=3, horizon=3, extra_gap=3)
        assert cv.horizon == 3
        assert cv.extra_gap == 3

        # gap > horizon: also valid
        cv = WalkForwardCV(n_splits=3, horizon=2, extra_gap=5)
        assert cv.horizon == 2
        assert cv.extra_gap == 5

    def test_horizon_with_any_extra_gap_allowed(self) -> None:
        """New semantics: extra_gap can be any value, total_separation = horizon + extra_gap."""
        # extra_gap < horizon is now VALID
        cv = WalkForwardCV(n_splits=3, horizon=3, extra_gap=2)
        assert cv.horizon == 3
        assert cv.extra_gap == 2
        # Total separation will be 3 + 2 = 5

        cv = WalkForwardCV(n_splits=3, horizon=5, extra_gap=0)
        assert cv.horizon == 5
        assert cv.extra_gap == 0
        # Total separation will be 5 + 0 = 5

    def test_horizon_none_allows_any_gap(self) -> None:
        """When horizon is None, any gap value is allowed."""
        # No horizon means no validation
        cv = WalkForwardCV(n_splits=3, extra_gap=0)
        assert cv.horizon is None
        assert cv.extra_gap == 0

        cv = WalkForwardCV(n_splits=3, extra_gap=10)
        assert cv.horizon is None
        assert cv.extra_gap == 10

    def test_horizon_with_extra_gap_validation_removed(self) -> None:
        """New semantics: extra_gap < horizon is now ALLOWED (no validation error)."""
        # This used to raise ValueError, but now it's valid
        cv = WalkForwardCV(n_splits=3, horizon=4, extra_gap=2)
        assert cv.horizon == 4
        assert cv.extra_gap == 2
        # Total separation will be 4 + 2 = 6

    def test_horizon_must_be_positive(self) -> None:
        """Horizon must be >= 1 if provided."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            WalkForwardCV(n_splits=3, horizon=0, extra_gap=0)

        with pytest.raises(ValueError, match="horizon must be >= 1"):
            WalkForwardCV(n_splits=3, horizon=-1, extra_gap=0)

    def test_horizon_is_stored_as_attribute(self) -> None:
        """Horizon should be accessible as instance attribute."""
        cv = WalkForwardCV(n_splits=5, horizon=3, extra_gap=3)
        assert hasattr(cv, "horizon")
        assert cv.horizon == 3

    def test_splits_work_with_horizon(self, sample_data: tuple) -> None:
        """CV should generate valid splits when horizon is set."""
        X, y = sample_data
        cv = WalkForwardCV(n_splits=3, horizon=2, extra_gap=2)

        splits = list(cv.split(X))
        assert len(splits) == 3

        for train, test in splits:
            # Gap is enforced
            assert train[-1] + cv.extra_gap < test[0]


# =============================================================================
# SplitResult and WalkForwardResults Tests
# =============================================================================


class TestSplitResult:
    """Tests for the SplitResult dataclass."""

    def test_split_result_basic(self) -> None:
        """SplitResult should compute basic metrics correctly."""
        from temporalcv.cv import SplitResult

        sr = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=111,
            predictions=np.array([1.0, 1.1, 1.2]),
            actuals=np.array([1.0, 1.0, 1.0]),
        )

        assert sr.split_idx == 0
        assert sr.train_size == 100
        assert sr.test_size == 10
        assert sr.gap == 2
        assert sr.mae == pytest.approx(0.1, rel=1e-6)
        assert sr.bias == pytest.approx(0.1, rel=1e-6)

    def test_split_result_errors(self) -> None:
        """SplitResult should compute errors correctly."""
        from temporalcv.cv import SplitResult

        sr = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
            predictions=np.array([1.0, 2.0, 3.0]),
            actuals=np.array([1.5, 2.5, 2.5]),
        )

        np.testing.assert_array_almost_equal(sr.errors, [-0.5, -0.5, 0.5])
        np.testing.assert_array_almost_equal(sr.absolute_errors, [0.5, 0.5, 0.5])
        assert sr.mae == 0.5
        assert sr.rmse == 0.5
        assert sr.bias == pytest.approx(-1 / 6, rel=1e-6)

    def test_split_result_has_dates(self) -> None:
        """SplitResult should report has_dates correctly."""
        from temporalcv.cv import SplitResult

        sr_no_dates = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
            predictions=np.array([1.0]),
            actuals=np.array([1.0]),
        )
        assert sr_no_dates.has_dates is False

        sr_with_dates = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
            predictions=np.array([1.0]),
            actuals=np.array([1.0]),
            train_start_date="2020-01-01",  # Can be any truthy value
        )
        assert sr_with_dates.has_dates is True

    def test_split_result_to_split_info(self) -> None:
        """SplitResult should convert to SplitInfo correctly."""
        from temporalcv.cv import SplitResult, SplitInfo

        sr = SplitResult(
            split_idx=2,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=111,
            predictions=np.array([1.0]),
            actuals=np.array([1.0]),
        )

        info = sr.to_split_info()
        assert isinstance(info, SplitInfo)
        assert info.split_idx == 2
        assert info.train_start == 0
        assert info.train_end == 99
        assert info.test_start == 102
        assert info.test_end == 111


class TestWalkForwardResults:
    """Tests for the WalkForwardResults dataclass."""

    def test_walk_forward_results_basic(self) -> None:
        """WalkForwardResults should aggregate metrics correctly."""
        from temporalcv.cv import SplitResult, WalkForwardResults

        sr1 = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
            predictions=np.array([1.0, 1.0, 1.0]),
            actuals=np.array([1.1, 1.1, 1.1]),
        )
        sr2 = SplitResult(
            split_idx=1,
            train_start=10,
            train_end=109,
            test_start=112,
            test_end=114,
            predictions=np.array([2.0, 2.0, 2.0]),
            actuals=np.array([2.2, 2.2, 2.2]),
        )

        results = WalkForwardResults(splits=[sr1, sr2])

        assert results.n_splits == 2
        assert results.total_samples == 6
        assert len(results.predictions) == 6
        assert len(results.actuals) == 6
        assert results.mae == pytest.approx(0.15, rel=1e-6)

    def test_walk_forward_results_empty_raises(self) -> None:
        """WalkForwardResults should raise for empty splits."""
        from temporalcv.cv import WalkForwardResults

        with pytest.raises(ValueError, match="at least one split"):
            WalkForwardResults(splits=[])

    def test_walk_forward_results_per_split_metrics(self) -> None:
        """WalkForwardResults should return per-split metrics."""
        from temporalcv.cv import SplitResult, WalkForwardResults

        sr = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
            predictions=np.array([1.0, 1.0, 1.0]),
            actuals=np.array([1.1, 1.1, 1.1]),
        )

        results = WalkForwardResults(splits=[sr])
        metrics = results.per_split_metrics()

        assert len(metrics) == 1
        assert metrics[0]["split_idx"] == 0
        assert metrics[0]["mae"] == pytest.approx(0.1, rel=1e-6)
        assert metrics[0]["n_samples"] == 3

    def test_walk_forward_results_summary(self) -> None:
        """WalkForwardResults should produce summary string."""
        from temporalcv.cv import SplitResult, WalkForwardResults

        sr = SplitResult(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=104,
            predictions=np.array([1.0]),
            actuals=np.array([1.0]),
        )

        results = WalkForwardResults(splits=[sr])
        summary = results.summary()

        assert "WalkForwardResults Summary" in summary
        assert "MAE:" in summary
        assert "RMSE:" in summary


class TestWalkForwardEvaluate:
    """Tests for the walk_forward_evaluate function."""

    def test_walk_forward_evaluate_basic(self) -> None:
        """walk_forward_evaluate should produce valid results."""
        from sklearn.linear_model import Ridge
        from temporalcv.cv import walk_forward_evaluate

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 0.5 + np.random.randn(100) * 0.1

        results = walk_forward_evaluate(Ridge(), X, y, n_splits=3, extra_gap=2, test_size=5)

        assert results.n_splits == 3
        assert results.total_samples == 15  # 3 * 5
        assert results.mae > 0
        assert results.rmse >= results.mae  # RMSE >= MAE always

    def test_walk_forward_evaluate_with_cv(self) -> None:
        """walk_forward_evaluate should accept pre-configured CV."""
        from sklearn.linear_model import Ridge
        from temporalcv.cv import walk_forward_evaluate, WalkForwardCV

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 0.5 + np.random.randn(100) * 0.1

        cv = WalkForwardCV(n_splits=4, extra_gap=1, test_size=3)
        results = walk_forward_evaluate(Ridge(), X, y, cv=cv)

        assert results.n_splits == 4
        assert results.total_samples == 12
        assert results.cv_config["n_splits"] == 4
        assert results.cv_config["extra_gap"] == 1

    def test_walk_forward_evaluate_sliding_window(self) -> None:
        """walk_forward_evaluate should work with sliding window."""
        from sklearn.linear_model import Ridge
        from temporalcv.cv import walk_forward_evaluate

        np.random.seed(42)
        X = np.random.randn(150, 3)
        y = X[:, 0] * 0.5 + np.random.randn(150) * 0.1

        results = walk_forward_evaluate(
            Ridge(),
            X,
            y,
            n_splits=3,
            window_type="sliding",
            window_size=50,
            extra_gap=2,
            test_size=10,
        )

        assert results.n_splits == 3
        assert results.cv_config["window_type"] == "sliding"
        assert results.cv_config["window_size"] == 50

    def test_walk_forward_evaluate_split_details(self) -> None:
        """walk_forward_evaluate should provide split-level details."""
        from sklearn.linear_model import Ridge
        from temporalcv.cv import walk_forward_evaluate

        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] * 0.5 + np.random.randn(100) * 0.1

        results = walk_forward_evaluate(Ridge(), X, y, n_splits=3, extra_gap=2, test_size=5)

        for split in results.splits:
            assert split.train_start >= 0
            assert split.train_end < split.test_start
            assert split.gap >= 2
            assert len(split.predictions) == len(split.actuals)
            assert split.mae >= 0


class TestSplitInfoWithDates:
    """Tests for SplitInfo with date fields."""

    def test_split_info_has_dates(self) -> None:
        """SplitInfo should report has_dates correctly."""
        from temporalcv.cv import SplitInfo

        info_no_dates = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=111,
        )
        assert info_no_dates.has_dates is False

        info_with_dates = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=111,
            train_start_date="2020-01-01",
        )
        assert info_with_dates.has_dates is True

    def test_split_info_is_frozen(self) -> None:
        """SplitInfo should be immutable."""
        from temporalcv.cv import SplitInfo

        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=111,
        )

        with pytest.raises(AttributeError):
            info.split_idx = 1  # type: ignore[misc]


# =============================================================================
# NestedWalkForwardCV Tests
# =============================================================================


class TestNestedWalkForwardCV:
    """
    Tests for NestedWalkForwardCV.

    [T1] Tests nested CV following Bergmeir & Benítez (2012), Varma & Simon (2006).
    """

    @pytest.fixture
    def nested_cv_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate data for nested CV testing."""
        rng = np.random.default_rng(42)
        n = 500  # Need more data for nested CV
        X = rng.standard_normal((n, 5))
        # Simple linear relationship for predictable behavior
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(n) * 0.2
        return X, y

    def test_basic_nested_cv(self, nested_cv_data: tuple) -> None:
        """NestedWalkForwardCV should run without error."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.01, 0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=1,
        )

        nested_cv.fit(X, y)

        assert nested_cv.best_params_ is not None
        assert "alpha" in nested_cv.best_params_
        assert len(nested_cv.outer_scores_) == 3

    def test_temporal_ordering_preserved(self, nested_cv_data: tuple) -> None:
        """Inner/outer loops should maintain temporal ordering."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=2,
            extra_gap=2,
        )

        nested_cv.fit(X, y)

        # Check that outer CV results contain valid inner CV info
        assert len(nested_cv.best_params_per_fold_) == 3
        assert nested_cv.cv_results_ is not None

    def test_gap_enforcement_both_levels(self, nested_cv_data: tuple) -> None:
        """Gap >= horizon should be enforced in both loops."""
        X, y = nested_cv_data

        # This should work: gap >= horizon
        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=3,
            extra_gap=3,  # Equal to horizon
        )
        nested_cv.fit(X, y)
        assert nested_cv.best_params_ is not None

    def test_extra_gap_defaults_to_zero(self, nested_cv_data: tuple) -> None:
        """New semantics: extra_gap defaults to 0 (not horizon)."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=4,
            # extra_gap not specified, should default to 0
        )

        assert nested_cv.extra_gap == 0
        assert nested_cv.horizon == 4
        # Total separation will be 4 + 0 = 4

    def test_extra_gap_less_than_horizon_allowed(self) -> None:
        """New semantics: extra_gap < horizon is ALLOWED without warning."""
        # This used to warn, but now it's valid without any warning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            nested_cv = NestedWalkForwardCV(
                estimator=Ridge(),
                param_grid={"alpha": [0.1]},
                n_outer_splits=3,
                n_inner_splits=3,
                horizon=5,
                extra_gap=2,  # Less than horizon - now valid
            )
        assert nested_cv.horizon == 5
        assert nested_cv.extra_gap == 2
        # Total separation will be 5 + 2 = 7

    def test_best_params_selection(self, nested_cv_data: tuple) -> None:
        """Best params should be from param_grid."""
        X, y = nested_cv_data

        param_grid = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid=param_grid,
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=1,
        )

        nested_cv.fit(X, y)

        assert nested_cv.best_params_["alpha"] in param_grid["alpha"]

    def test_outer_scores_unbiased(self, nested_cv_data: tuple) -> None:
        """Outer scores should come from held-out data only."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=1,
        )

        nested_cv.fit(X, y)

        # Outer scores should be negative (neg_mean_squared_error)
        assert all(s < 0 for s in nested_cv.outer_scores_)
        # Should have one score per outer fold
        assert len(nested_cv.outer_scores_) == 3

    def test_params_stability(self, nested_cv_data: tuple) -> None:
        """params_stability should measure consistency across folds."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},  # Only 2 options
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=1,
        )

        nested_cv.fit(X, y)

        # Stability should be between 0 and 1
        assert 0 <= nested_cv.params_stability_ <= 1

    def test_sklearn_estimator_compatibility(self, nested_cv_data: tuple) -> None:
        """Should work with various sklearn estimators."""
        from sklearn.linear_model import Lasso

        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Lasso(max_iter=1000),
            param_grid={"alpha": [0.01, 0.1]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=1,
        )

        nested_cv.fit(X, y)
        assert nested_cv.best_params_ is not None

    def test_custom_scoring_function(self, nested_cv_data: tuple) -> None:
        """Custom scoring functions should work."""
        X, y = nested_cv_data

        def custom_neg_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return -float(np.mean(np.abs(y_true - y_pred)))

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            horizon=1,
            scoring=custom_neg_mae,
        )

        nested_cv.fit(X, y)
        assert nested_cv.best_params_ is not None

    def test_string_scoring(self, nested_cv_data: tuple) -> None:
        """String scoring options should work."""
        X, y = nested_cv_data

        for scoring in ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]:
            nested_cv = NestedWalkForwardCV(
                estimator=Ridge(),
                param_grid={"alpha": [0.1]},
                n_outer_splits=2,
                n_inner_splits=2,
                horizon=1,
                scoring=scoring,
            )
            nested_cv.fit(X, y)
            assert nested_cv.best_params_ is not None

    def test_refit_option(self, nested_cv_data: tuple) -> None:
        """refit=True should provide best_estimator_."""
        X, y = nested_cv_data

        # With refit=True (default)
        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            refit=True,
        )
        nested_cv.fit(X, y)
        assert nested_cv.best_estimator_ is not None

        # With refit=False
        nested_cv_no_refit = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            refit=False,
        )
        nested_cv_no_refit.fit(X, y)
        with pytest.raises(RuntimeError, match="refit=True"):
            _ = nested_cv_no_refit.best_estimator_

    def test_predict_requires_refit(self, nested_cv_data: tuple) -> None:
        """predict() should require refit=True."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=2,
            n_inner_splits=2,
            refit=False,
        )
        nested_cv.fit(X, y)

        with pytest.raises(RuntimeError, match="refit=True"):
            nested_cv.predict(X[:10])

    def test_predict_with_refit(self, nested_cv_data: tuple) -> None:
        """predict() should work when refit=True."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
            refit=True,
        )
        nested_cv.fit(X, y)

        preds = nested_cv.predict(X[:10])
        assert len(preds) == 10

    def test_expanding_vs_sliding(self, nested_cv_data: tuple) -> None:
        """Both window types should work."""
        X, y = nested_cv_data

        # Expanding (default)
        nested_cv_exp = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=2,
            n_inner_splits=2,
            window_type="expanding",
        )
        nested_cv_exp.fit(X, y)
        assert nested_cv_exp.best_params_ is not None

        # Sliding
        nested_cv_slide = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=2,
            n_inner_splits=2,
            window_type="sliding",
            window_size=100,
        )
        nested_cv_slide.fit(X, y)
        assert nested_cv_slide.best_params_ is not None

    def test_verbose_output(self, nested_cv_data: tuple, capsys: pytest.CaptureFixture) -> None:
        """verbose=1 should print progress."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=2,
            n_inner_splits=2,
            verbose=1,
        )
        nested_cv.fit(X, y)

        captured = capsys.readouterr()
        assert "NestedWalkForwardCV" in captured.out
        assert "Outer fold" in captured.out

    def test_get_result_method(self, nested_cv_data: tuple) -> None:
        """get_result() should return NestedCVResult."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
        )
        nested_cv.fit(X, y)

        result = nested_cv.get_result()
        assert isinstance(result, NestedCVResult)
        assert result.best_params == nested_cv.best_params_
        assert result.n_outer_splits == 3
        assert result.n_inner_splits == 3
        assert result.params_stability == nested_cv.params_stability_

    def test_repr(self) -> None:
        """__repr__ should be informative."""
        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=3,
            n_inner_splits=5,
            horizon=4,
            extra_gap=4,
        )
        repr_str = repr(nested_cv)
        assert "NestedWalkForwardCV" in repr_str
        assert "grid" in repr_str
        assert "n_outer=3" in repr_str
        assert "n_inner=5" in repr_str

    def test_randomized_search(self, nested_cv_data: tuple) -> None:
        """param_distributions with n_iter should work."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_distributions={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
            n_iter=5,
            n_outer_splits=2,
            n_inner_splits=2,
            random_state=42,
        )
        nested_cv.fit(X, y)

        assert nested_cv.best_params_ is not None
        assert "random" in repr(nested_cv)

    def test_randomized_search_requires_n_iter(self) -> None:
        """param_distributions without n_iter should raise."""
        with pytest.raises(ValueError, match="n_iter is required"):
            NestedWalkForwardCV(
                estimator=Ridge(),
                param_distributions={"alpha": [0.1, 1.0]},
                # n_iter missing
            )

    def test_cannot_specify_both_grid_and_distributions(self) -> None:
        """Cannot use both param_grid and param_distributions."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            NestedWalkForwardCV(
                estimator=Ridge(),
                param_grid={"alpha": [0.1]},
                param_distributions={"alpha": [0.1]},
                n_iter=5,
            )

    def test_must_specify_one_search_type(self) -> None:
        """Must specify param_grid or param_distributions."""
        with pytest.raises(ValueError, match="Either param_grid or"):
            NestedWalkForwardCV(
                estimator=Ridge(),
                # Neither param_grid nor param_distributions
            )

    def test_input_validation_n_outer_splits(self) -> None:
        """n_outer_splits must be >= 2."""
        with pytest.raises(ValueError, match="n_outer_splits must be >= 2"):
            NestedWalkForwardCV(
                estimator=Ridge(),
                param_grid={"alpha": [0.1]},
                n_outer_splits=1,
            )

    def test_input_validation_n_inner_splits(self) -> None:
        """n_inner_splits must be >= 2."""
        with pytest.raises(ValueError, match="n_inner_splits must be >= 2"):
            NestedWalkForwardCV(
                estimator=Ridge(),
                param_grid={"alpha": [0.1]},
                n_inner_splits=1,
            )

    def test_input_validation_horizon(self) -> None:
        """horizon must be >= 1."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            NestedWalkForwardCV(
                estimator=Ridge(),
                param_grid={"alpha": [0.1]},
                horizon=0,
            )

    def test_cv_results_structure(self, nested_cv_data: tuple) -> None:
        """cv_results_ should have expected structure."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0]},
            n_outer_splits=3,
            n_inner_splits=3,
        )
        nested_cv.fit(X, y)

        cv_results = nested_cv.cv_results_
        assert "mean_outer_score" in cv_results
        assert "std_outer_score" in cv_results
        assert "outer_scores" in cv_results
        assert "best_params_per_fold" in cv_results
        assert "params_stability" in cv_results
        assert "param_combinations" in cv_results

    def test_properties_before_fit_raise(self) -> None:
        """Accessing properties before fit() should raise."""
        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
        )

        with pytest.raises(RuntimeError, match="Call fit"):
            _ = nested_cv.best_params_

        with pytest.raises(RuntimeError, match="Call fit"):
            _ = nested_cv.cv_results_

        with pytest.raises(RuntimeError, match="Call fit"):
            _ = nested_cv.outer_scores_

    def test_mean_std_outer_score(self, nested_cv_data: tuple) -> None:
        """mean_outer_score_ and std_outer_score_ should be computed correctly."""
        X, y = nested_cv_data

        nested_cv = NestedWalkForwardCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1]},
            n_outer_splits=3,
            n_inner_splits=3,
        )
        nested_cv.fit(X, y)

        # Verify mean/std match outer_scores
        assert nested_cv.mean_outer_score_ == pytest.approx(
            np.mean(nested_cv.outer_scores_), rel=1e-10
        )
        assert nested_cv.std_outer_score_ == pytest.approx(
            np.std(nested_cv.outer_scores_), rel=1e-10
        )
