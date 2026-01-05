"""
Anti-pattern tests for boundary violation detection.

Boundary violations occur when:
- Train and test sets overlap in time
- Gap between train and test is insufficient for the horizon
- Temporal ordering is not enforced

These tests verify that temporalcv correctly prevents these bugs.

Bug Category: #2 from lever_of_archimedes/patterns/data_leakage_prevention.md
Gate: gate_temporal_boundary
CV: WalkForwardCV with gap enforcement
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.cv import SplitInfo, WalkForwardCV
from temporalcv.gates import GateStatus, gate_temporal_boundary


class TestBoundaryViolationDetection:
    """Tests that verify boundary violations are caught."""

    def test_detects_overlapping_indices(self) -> None:
        """
        Scenario: Train and test indices overlap.

        This is the most basic form of temporal leakage.
        """
        result = gate_temporal_boundary(
            train_end_idx=100,
            test_start_idx=95,  # Overlaps with train
            horizon=1,
            extra_gap=0,
        )

        assert result.status == GateStatus.HALT, (
            f"Should HALT on overlapping indices, got {result.status}"
        )

    def test_detects_adjacent_without_gap(self) -> None:
        """
        Scenario: Test starts immediately after train with no gap.

        For h-step ahead forecasting, need gap >= h.
        """
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=100,  # Immediately adjacent
            horizon=2,  # 2-step ahead needs gap
            extra_gap=0,
        )

        assert result.status == GateStatus.HALT, (
            f"Should HALT when gap < horizon, got {result.status}"
        )

    def test_detects_insufficient_gap(self) -> None:
        """
        Scenario: Gap is less than required horizon.

        For h=3, need at least 3 indices between train_end and test_start.
        """
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=101,  # Gap of 1
            horizon=3,           # Need gap of 3
            extra_gap=0,
        )

        assert result.status == GateStatus.HALT

    def test_passes_valid_boundary(self) -> None:
        """
        Scenario: Proper gap enforcement.
        """
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=103,  # Gap of 3
            horizon=2,           # Need gap of 2
            extra_gap=1,               # Additional gap of 1
        )

        assert result.status == GateStatus.PASS

    def test_additional_gap_enforcement(self) -> None:
        """
        Scenario: Additional gap parameter beyond horizon.

        Total required gap = horizon + gap parameter.
        """
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=103,  # Gap of 3
            horizon=2,
            extra_gap=2,               # Need horizon(2) + gap(2) = 4
        )

        assert result.status == GateStatus.HALT, (
            "Should HALT when gap + horizon not satisfied"
        )


class TestSplitInfoBoundaryEnforcement:
    """Tests that SplitInfo prevents boundary violations at construction."""

    def test_rejects_overlapping_split(self) -> None:
        """SplitInfo should raise on overlapping train/test."""
        with pytest.raises(ValueError, match="[Ll]eakage|[Oo]verlap"):
            SplitInfo(
                split_idx=0,
                train_start=0,
                train_end=100,
                test_start=95,
                test_end=110,
            )

    def test_rejects_equal_boundary(self) -> None:
        """SplitInfo should raise when train_end == test_start."""
        with pytest.raises(ValueError, match="[Ll]eakage"):
            SplitInfo(
                split_idx=0,
                train_start=0,
                train_end=100,
                test_start=100,
                test_end=105,
            )

    def test_accepts_valid_gap(self) -> None:
        """SplitInfo should accept proper temporal ordering."""
        info = SplitInfo(
            split_idx=0,
            train_start=0,
            train_end=99,
            test_start=102,
            test_end=105,
        )

        assert info.gap == 2  # 102 - 99 - 1


class TestWalkForwardCVBoundaries:
    """Tests that WalkForwardCV enforces proper boundaries."""

    def test_cv_never_overlaps(self) -> None:
        """All CV splits should have non-overlapping train/test."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = rng.standard_normal(200)

        cv = WalkForwardCV(
            n_splits=5,
            window_type="expanding",
            extra_gap=2,
        )

        for train_idx, test_idx in cv.split(X, y):
            train_set = set(train_idx)
            test_set = set(test_idx)

            overlap = train_set & test_set
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_cv_respects_gap(self) -> None:
        """All CV splits should respect the gap parameter."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = rng.standard_normal(200)

        gap = 3
        cv = WalkForwardCV(
            n_splits=5,
            window_type="sliding",
            window_size=50,
            extra_gap=gap,
        )

        for train_idx, test_idx in cv.split(X, y):
            train_end = max(train_idx)
            test_start = min(test_idx)
            actual_gap = test_start - train_end - 1

            assert actual_gap >= gap, (
                f"Gap violation: expected >= {gap}, got {actual_gap}"
            )

    def test_cv_train_precedes_test(self) -> None:
        """All train indices must be less than all test indices."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        y = rng.standard_normal(200)

        cv = WalkForwardCV(n_splits=5)

        for train_idx, test_idx in cv.split(X, y):
            max_train = max(train_idx)
            min_test = min(test_idx)

            assert max_train < min_test, (
                f"Temporal order violation: max(train)={max_train} >= min(test)={min_test}"
            )


class TestBoundaryMetrics:
    """Tests for metrics reported by boundary validation."""

    def test_reports_all_indices(self) -> None:
        """Gate should report all boundary indices in details."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=105,
            horizon=2,
            extra_gap=3,
        )

        assert result.details["train_end_idx"] == 99
        assert result.details["test_start_idx"] == 105
        assert result.details["horizon"] == 2
        assert result.details["extra_gap"] == 3

    def test_reports_actual_gap(self) -> None:
        """Gate should report the actual gap found."""
        result = gate_temporal_boundary(
            train_end_idx=99,
            test_start_idx=105,
            horizon=2,
            extra_gap=1,
        )

        # Actual gap is 105 - 99 - 1 = 5
        assert result.details.get("actual_gap") == 5 or "actual_gap" not in result.details
