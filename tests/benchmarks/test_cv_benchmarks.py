"""Benchmark tests for cross-validation operations.

Measures performance of CV splitting across various configurations.
"""

import numpy as np
import pytest

from temporalcv.cv import WalkForwardCV, CrossFitCV
from temporalcv.cv_financial import PurgedKFold, CombinatorialPurgedCV, PurgedWalkForward


class TestWalkForwardCVBenchmarks:
    """Benchmarks for WalkForwardCV splitting."""

    @pytest.fixture
    def small_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Small dataset: n=500."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 10))
        y = rng.standard_normal(500)
        return X, y

    @pytest.fixture
    def medium_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: n=2000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2000, 10))
        y = rng.standard_normal(2000)
        return X, y

    @pytest.fixture
    def large_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Large dataset: n=10000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10000, 10))
        y = rng.standard_normal(10000)
        return X, y

    def test_walk_forward_small(
        self, benchmark, small_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """WalkForwardCV with n=500, 5 splits."""
        X, y = small_data
        cv = WalkForwardCV(n_splits=5)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 5

    def test_walk_forward_medium(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """WalkForwardCV with n=2000, 10 splits."""
        X, y = medium_data
        cv = WalkForwardCV(n_splits=10)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 10

    def test_walk_forward_large(
        self, benchmark, large_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """WalkForwardCV with n=10000, 20 splits."""
        X, y = large_data
        cv = WalkForwardCV(n_splits=20)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 20

    def test_walk_forward_many_splits(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """WalkForwardCV with n=2000, 50 splits."""
        X, y = medium_data
        cv = WalkForwardCV(n_splits=50)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 50


class TestCrossFitCVBenchmarks:
    """Benchmarks for CrossFitCV splitting."""

    @pytest.fixture
    def medium_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: n=2000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2000, 10))
        y = rng.standard_normal(2000)
        return X, y

    def test_crossfit_5_folds(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CrossFitCV with 5 folds."""
        X, y = medium_data
        cv = CrossFitCV(n_splits=5, extra_gap=5)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        # CrossFitCV yields n_splits - 1 pairs (fold 0 has no training data)
        assert len(result) == 4

    def test_crossfit_10_folds(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CrossFitCV with 10 folds."""
        X, y = medium_data
        cv = CrossFitCV(n_splits=10, extra_gap=5)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        # CrossFitCV yields n_splits - 1 pairs (fold 0 has no training data)
        assert len(result) == 9


class TestPurgedKFoldBenchmarks:
    """Benchmarks for PurgedKFold splitting."""

    @pytest.fixture
    def medium_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: n=2000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2000, 10))
        y = rng.standard_normal(2000)
        return X, y

    @pytest.fixture
    def large_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Large dataset: n=10000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10000, 10))
        y = rng.standard_normal(10000)
        return X, y

    def test_purged_kfold_no_purge(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedKFold with purge_gap=0 (baseline)."""
        X, y = medium_data
        cv = PurgedKFold(n_splits=5, purge_gap=0)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 5

    def test_purged_kfold_with_purge(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedKFold with purge_gap=10."""
        X, y = medium_data
        cv = PurgedKFold(n_splits=5, purge_gap=10)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 5

    def test_purged_kfold_with_embargo(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedKFold with purge_gap=10 and embargo_pct=0.05."""
        X, y = medium_data
        cv = PurgedKFold(n_splits=5, purge_gap=10, embargo_pct=0.05)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 5

    def test_purged_kfold_large(
        self, benchmark, large_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedKFold with n=10000, purge_gap=20."""
        X, y = large_data
        cv = PurgedKFold(n_splits=5, purge_gap=20)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 5


class TestCombinatorialPurgedCVBenchmarks:
    """Benchmarks for CombinatorialPurgedCV splitting."""

    @pytest.fixture
    def medium_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: n=1000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 10))
        y = rng.standard_normal(1000)
        return X, y

    def test_cpcv_5_choose_2(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CombinatorialPurgedCV with C(5,2)=10 paths."""
        X, y = medium_data
        cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 10  # C(5,2) = 10

    def test_cpcv_6_choose_2(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CombinatorialPurgedCV with C(6,2)=15 paths."""
        X, y = medium_data
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) == 15  # C(6,2) = 15


class TestPurgedWalkForwardBenchmarks:
    """Benchmarks for PurgedWalkForward splitting."""

    @pytest.fixture
    def medium_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Medium dataset: n=2000."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((2000, 10))
        y = rng.standard_normal(2000)
        return X, y

    def test_purged_walk_forward_fixed_window(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedWalkForward with fixed train window."""
        X, y = medium_data
        cv = PurgedWalkForward(
            n_splits=10, train_size=200, test_size=50, purge_gap=10
        )

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) >= 1

    def test_purged_walk_forward_expanding_window(
        self, benchmark, medium_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """PurgedWalkForward with expanding train window."""
        X, y = medium_data
        cv = PurgedWalkForward(
            n_splits=10, train_size=None, test_size=50, purge_gap=10
        )

        def split_all() -> list:
            return list(cv.split(X, y))

        result = benchmark(split_all)
        assert len(result) >= 1
