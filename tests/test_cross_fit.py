"""
Tests for CrossFitCV temporal cross-fitting.

Tests the forward-only cross-fitting implementation for debiased metrics.

Knowledge Tier: [T1] - Cross-fitting debiasing, [T2] - Temporal adaptation
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from temporalcv.cv import CrossFitCV


class TestCrossFitCVBasic:
    """Basic functionality tests for CrossFitCV."""

    def test_basic_split_runs(self) -> None:
        """Basic split should run without errors."""
        cv = CrossFitCV(n_splits=5)
        X = np.random.default_rng(42).standard_normal((100, 3))

        splits = list(cv.split(X))

        # n_splits - 1 because fold 0 is skipped (no training data)
        assert len(splits) == 4

    def test_forward_only_semantics(self) -> None:
        """Training indices should always be before test indices."""
        cv = CrossFitCV(n_splits=5)
        X = np.random.default_rng(42).standard_normal((100, 3))

        for train_idx, test_idx in cv.split(X):
            # All training indices should be before all test indices
            assert train_idx.max() < test_idx.min()

    def test_non_overlapping_test_folds(self) -> None:
        """Each observation should appear in exactly one test fold."""
        cv = CrossFitCV(n_splits=5)
        X = np.random.default_rng(42).standard_normal((100, 3))

        all_test_indices = []
        for _, test_idx in cv.split(X):
            all_test_indices.extend(test_idx)

        # Check no duplicates
        assert len(all_test_indices) == len(set(all_test_indices))

    def test_fold_0_not_included(self) -> None:
        """Fold 0 should not appear in test indices (no training data)."""
        cv = CrossFitCV(n_splits=5)
        n_samples = 100
        X = np.random.default_rng(42).standard_normal((n_samples, 3))
        fold_size = n_samples // 5  # 20

        all_test_indices = []
        for _, test_idx in cv.split(X):
            all_test_indices.extend(test_idx)

        # Fold 0 is indices 0-19, should NOT be in test indices
        for i in range(fold_size):
            assert i not in all_test_indices


class TestFitPredict:
    """Tests for fit_predict method."""

    def test_fit_predict_returns_correct_shape(self) -> None:
        """fit_predict should return array of same length as y."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = X @ rng.standard_normal(3) + rng.standard_normal(n) * 0.1

        cv = CrossFitCV(n_splits=5)
        predictions = cv.fit_predict(Ridge(), X, y)

        assert len(predictions) == n

    def test_first_fold_is_nan(self) -> None:
        """First fold predictions should be NaN (no training data)."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = X @ rng.standard_normal(3) + rng.standard_normal(n) * 0.1

        cv = CrossFitCV(n_splits=5)
        predictions = cv.fit_predict(Ridge(), X, y)

        fold_size = n // 5  # 20
        # First fold (indices 0-19) should be NaN
        assert np.all(np.isnan(predictions[:fold_size]))
        # Other folds should NOT be NaN
        assert np.all(~np.isnan(predictions[fold_size:]))

    def test_predictions_are_out_of_sample(self) -> None:
        """Predictions should be truly out-of-sample."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        # Create relationship that would overfit in-sample
        y = X @ np.array([1.0, 2.0, 3.0]) + rng.standard_normal(n) * 0.5

        cv = CrossFitCV(n_splits=5)

        # Get out-of-sample predictions
        oos_predictions = cv.fit_predict(Ridge(alpha=0.0), X, y)

        # Compare to in-sample predictions
        model = Ridge(alpha=0.0)
        model.fit(X, y)
        in_sample_predictions = model.predict(X)

        # Out-of-sample predictions should differ from in-sample
        valid_mask = ~np.isnan(oos_predictions)
        assert not np.allclose(
            oos_predictions[valid_mask], in_sample_predictions[valid_mask]
        )


class TestFitPredictResiduals:
    """Tests for fit_predict_residuals method."""

    def test_residuals_are_y_minus_predictions(self) -> None:
        """Residuals should equal y - predictions."""
        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = X @ rng.standard_normal(3) + rng.standard_normal(n) * 0.1

        cv = CrossFitCV(n_splits=5)
        predictions = cv.fit_predict(Ridge(), X, y)
        residuals = cv.fit_predict_residuals(Ridge(), X, y)

        valid_mask = ~np.isnan(predictions)
        np.testing.assert_allclose(
            residuals[valid_mask], y[valid_mask] - predictions[valid_mask]
        )


class TestGapEnforcement:
    """Tests for gap parameter."""

    def test_gap_enforced_in_splits(self) -> None:
        """Gap should be enforced between train and test."""
        cv = CrossFitCV(n_splits=5, extra_gap=5)
        X = np.random.default_rng(42).standard_normal((100, 3))

        for train_idx, test_idx in cv.split(X):
            actual_gap = test_idx.min() - train_idx.max() - 1
            assert actual_gap >= 5

    def test_large_gap_reduces_splits(self) -> None:
        """Large gap may reduce number of valid splits."""
        cv_no_gap = CrossFitCV(n_splits=5, extra_gap=0)
        cv_large_gap = CrossFitCV(n_splits=5, extra_gap=10)
        X = np.random.default_rng(42).standard_normal((50, 3))

        splits_no_gap = list(cv_no_gap.split(X))
        splits_large_gap = list(cv_large_gap.split(X))

        # Large gap should have fewer or equal splits
        assert len(splits_large_gap) <= len(splits_no_gap)


class TestGetNSplits:
    """Tests for get_n_splits method."""

    def test_get_n_splits_without_x(self) -> None:
        """Without X, should return n_splits - 1."""
        cv = CrossFitCV(n_splits=5)

        assert cv.get_n_splits() == 4

    def test_get_n_splits_with_x(self) -> None:
        """With X, should return actual number of valid splits."""
        cv = CrossFitCV(n_splits=5)
        X = np.random.default_rng(42).standard_normal((100, 3))

        actual = cv.get_n_splits(X)
        expected = len(list(cv.split(X)))

        assert actual == expected


class TestFoldIndices:
    """Tests for get_fold_indices method."""

    def test_fold_indices_correct(self) -> None:
        """Fold indices should divide data correctly."""
        cv = CrossFitCV(n_splits=5)
        X = np.random.default_rng(42).standard_normal((100, 3))

        folds = cv.get_fold_indices(X)

        assert len(folds) == 5
        # Check consecutive
        for i in range(len(folds) - 1):
            assert folds[i][1] == folds[i + 1][0]

        # Check covers all data
        assert folds[0][0] == 0
        assert folds[-1][1] == 100


class TestEdgeCases:
    """Edge case tests for CrossFitCV."""

    def test_minimum_splits(self) -> None:
        """Should work with minimum 2 splits."""
        cv = CrossFitCV(n_splits=2)
        X = np.random.default_rng(42).standard_normal((100, 3))

        splits = list(cv.split(X))

        assert len(splits) == 1  # Only fold 1, fold 0 skipped

    def test_one_split_raises(self) -> None:
        """Should raise error with n_splits=1."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            CrossFitCV(n_splits=1)

    def test_small_data_with_many_splits(self) -> None:
        """Should handle small data gracefully."""
        cv = CrossFitCV(n_splits=5)
        X = np.random.default_rng(42).standard_normal((10, 3))

        # Should work but have few samples per fold
        splits = list(cv.split(X))
        assert len(splits) <= 4

    def test_negative_gap_raises(self) -> None:
        """Should raise error for negative gap."""
        with pytest.raises(ValueError, match="gap must be >= 0"):
            CrossFitCV(n_splits=5, extra_gap=-1)

    def test_test_size_parameter(self) -> None:
        """Custom test_size should be respected."""
        cv = CrossFitCV(n_splits=5, test_size=10)
        X = np.random.default_rng(42).standard_normal((100, 3))

        for _, test_idx in cv.split(X):
            # All but last fold should have test_size samples
            if test_idx[-1] < 99:  # Not last fold
                assert len(test_idx) == 10


class TestSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_with_cross_val_score(self) -> None:
        """Should work with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score

        rng = np.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 3))
        y = X @ rng.standard_normal(3) + rng.standard_normal(n) * 0.1

        cv = CrossFitCV(n_splits=5)
        scores = cross_val_score(Ridge(), X, y, cv=cv)

        assert len(scores) == 4  # n_splits - 1

    def test_repr(self) -> None:
        """Repr should be informative."""
        cv = CrossFitCV(n_splits=5, extra_gap=2, test_size=10)

        repr_str = repr(cv)

        assert "CrossFitCV" in repr_str
        assert "n_splits=5" in repr_str
        assert "extra_gap=2" in repr_str
        assert "test_size=10" in repr_str
