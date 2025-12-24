"""
Tests for compare module.

Tests ModelResult, ComparisonResult, adapters, and runner functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.compare import (
    ComparisonReport,
    ComparisonResult,
    ForecastAdapter,
    ModelResult,
    NaiveAdapter,
    SeasonalNaiveAdapter,
    compare_to_baseline,
    compute_comparison_metrics,
    run_benchmark_suite,
    run_comparison,
)


# =============================================================================
# Test ModelResult
# =============================================================================


class TestModelResult:
    """Test ModelResult dataclass."""

    def test_basic_creation(self) -> None:
        """Should create result with required fields."""
        result = ModelResult(
            model_name="TestModel",
            package="test_package",
            metrics={"mae": 0.5, "rmse": 0.7},
            predictions=np.array([1.0, 2.0, 3.0]),
            runtime_seconds=1.5,
        )

        assert result.model_name == "TestModel"
        assert result.package == "test_package"
        assert result.metrics["mae"] == 0.5
        assert result.runtime_seconds == 1.5

    def test_empty_model_name_raises(self) -> None:
        """Should reject empty model name."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            ModelResult(
                model_name="",
                package="test",
                metrics={"mae": 0.5},
                predictions=np.array([1.0]),
                runtime_seconds=1.0,
            )

    def test_negative_runtime_raises(self) -> None:
        """Should reject negative runtime."""
        with pytest.raises(ValueError, match="runtime_seconds cannot be negative"):
            ModelResult(
                model_name="test",
                package="test",
                metrics={"mae": 0.5},
                predictions=np.array([1.0]),
                runtime_seconds=-1.0,
            )

    def test_get_metric(self) -> None:
        """Should get metric by name."""
        result = ModelResult(
            model_name="test",
            package="test",
            metrics={"mae": 0.5, "RMSE": 0.7},
            predictions=np.array([1.0]),
            runtime_seconds=1.0,
        )

        assert result.get_metric("mae") == 0.5
        # Case-insensitive
        assert result.get_metric("MAE") == 0.5
        assert result.get_metric("rmse") == 0.7

    def test_get_metric_not_found(self) -> None:
        """Should raise KeyError for missing metric."""
        result = ModelResult(
            model_name="test",
            package="test",
            metrics={"mae": 0.5},
            predictions=np.array([1.0]),
            runtime_seconds=1.0,
        )

        with pytest.raises(KeyError, match="mape"):
            result.get_metric("mape")

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        result = ModelResult(
            model_name="test",
            package="test",
            metrics={"mae": 0.5},
            predictions=np.array([1.0]),
            runtime_seconds=1.0,
            model_params={"alpha": 0.1},
        )

        d = result.to_dict()
        assert d["model_name"] == "test"
        assert d["metrics"]["mae"] == 0.5
        assert d["model_params"]["alpha"] == 0.1


# =============================================================================
# Test ComparisonResult
# =============================================================================


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    @pytest.fixture
    def sample_results(self) -> list[ModelResult]:
        """Create sample model results."""
        return [
            ModelResult(
                model_name="ModelA",
                package="pkg",
                metrics={"mae": 0.5, "rmse": 0.7},
                predictions=np.array([1.0, 2.0]),
                runtime_seconds=1.0,
            ),
            ModelResult(
                model_name="ModelB",
                package="pkg",
                metrics={"mae": 0.3, "rmse": 0.4},
                predictions=np.array([1.1, 2.1]),
                runtime_seconds=2.0,
            ),
        ]

    def test_basic_creation(self, sample_results: list[ModelResult]) -> None:
        """Should create comparison with best model computed."""
        result = ComparisonResult(
            dataset_name="test_dataset",
            models=sample_results,
            primary_metric="mae",
        )

        assert result.dataset_name == "test_dataset"
        assert result.best_model == "ModelB"  # Lower MAE

    def test_empty_models_raises(self) -> None:
        """Should reject empty models list."""
        with pytest.raises(ValueError, match="models list cannot be empty"):
            ComparisonResult(
                dataset_name="test",
                models=[],
                primary_metric="mae",
            )

    def test_missing_metric_raises(self) -> None:
        """Should raise if no model has primary metric."""
        results = [
            ModelResult(
                model_name="test",
                package="pkg",
                metrics={"rmse": 0.5},
                predictions=np.array([1.0]),
                runtime_seconds=1.0,
            )
        ]

        with pytest.raises(ValueError, match="No model has metric 'mae'"):
            ComparisonResult(
                dataset_name="test",
                models=results,
                primary_metric="mae",
            )

    def test_get_ranking(self, sample_results: list[ModelResult]) -> None:
        """Should return models ranked by metric."""
        result = ComparisonResult(
            dataset_name="test",
            models=sample_results,
            primary_metric="mae",
        )

        ranking = result.get_ranking()
        assert ranking[0][0] == "ModelB"  # Best (lowest MAE)
        assert ranking[1][0] == "ModelA"

    def test_get_ranking_different_metric(
        self, sample_results: list[ModelResult]
    ) -> None:
        """Should rank by specified metric."""
        result = ComparisonResult(
            dataset_name="test",
            models=sample_results,
            primary_metric="mae",
        )

        ranking = result.get_ranking(metric="rmse")
        assert ranking[0][0] == "ModelB"  # Best RMSE

    def test_to_dict(self, sample_results: list[ModelResult]) -> None:
        """Should convert to dictionary."""
        result = ComparisonResult(
            dataset_name="test",
            models=sample_results,
            primary_metric="mae",
        )

        d = result.to_dict()
        assert d["dataset_name"] == "test"
        assert d["best_model"] == "ModelB"
        assert len(d["models"]) == 2


# =============================================================================
# Test ComparisonReport
# =============================================================================


class TestComparisonReport:
    """Test ComparisonReport dataclass."""

    @pytest.fixture
    def sample_comparisons(self) -> list[ComparisonResult]:
        """Create sample comparison results."""
        results_1 = [
            ModelResult(
                model_name="ModelA",
                package="pkg",
                metrics={"mae": 0.5},
                predictions=np.array([1.0]),
                runtime_seconds=1.0,
            ),
            ModelResult(
                model_name="ModelB",
                package="pkg",
                metrics={"mae": 0.3},
                predictions=np.array([1.0]),
                runtime_seconds=1.0,
            ),
        ]
        results_2 = [
            ModelResult(
                model_name="ModelA",
                package="pkg",
                metrics={"mae": 0.4},
                predictions=np.array([1.0]),
                runtime_seconds=1.0,
            ),
            ModelResult(
                model_name="ModelB",
                package="pkg",
                metrics={"mae": 0.6},
                predictions=np.array([1.0]),
                runtime_seconds=1.0,
            ),
        ]

        return [
            ComparisonResult(
                dataset_name="dataset_1", models=results_1, primary_metric="mae"
            ),
            ComparisonResult(
                dataset_name="dataset_2", models=results_2, primary_metric="mae"
            ),
        ]

    def test_basic_creation(
        self, sample_comparisons: list[ComparisonResult]
    ) -> None:
        """Should create report with summary."""
        report = ComparisonReport(results=sample_comparisons)

        assert report.summary["n_datasets"] == 2
        assert "ModelA" in report.summary["wins_by_model"]
        assert "ModelB" in report.summary["wins_by_model"]

    def test_win_counts(self, sample_comparisons: list[ComparisonResult]) -> None:
        """Should count wins correctly."""
        report = ComparisonReport(results=sample_comparisons)

        # ModelB won dataset_1, ModelA won dataset_2
        wins = report.summary["wins_by_model"]
        assert wins["ModelA"] == 1
        assert wins["ModelB"] == 1

    def test_mean_metrics(self, sample_comparisons: list[ComparisonResult]) -> None:
        """Should compute mean metrics."""
        report = ComparisonReport(results=sample_comparisons)

        means = report.summary["mean_metrics_by_model"]
        # ModelA: (0.5 + 0.4) / 2 = 0.45
        assert means["ModelA"]["mae"] == pytest.approx(0.45)
        # ModelB: (0.3 + 0.6) / 2 = 0.45
        assert means["ModelB"]["mae"] == pytest.approx(0.45)

    def test_to_markdown(self, sample_comparisons: list[ComparisonResult]) -> None:
        """Should generate markdown report."""
        report = ComparisonReport(results=sample_comparisons)

        md = report.to_markdown()
        assert "# Model Comparison Report" in md
        assert "dataset_1" in md
        assert "dataset_2" in md
        assert "Model Wins" in md


# =============================================================================
# Test Adapters
# =============================================================================


class TestNaiveAdapter:
    """Test NaiveAdapter."""

    def test_properties(self) -> None:
        """Should have correct properties."""
        adapter = NaiveAdapter()
        assert adapter.model_name == "Naive"
        assert adapter.package_name == "temporalcv"

    def test_single_series_prediction(self) -> None:
        """Should predict last value for single series."""
        adapter = NaiveAdapter()
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        predictions = adapter.fit_predict(train, test_size=3, horizon=1)

        assert len(predictions) == 3
        assert all(p == 5.0 for p in predictions)

    def test_multi_series_prediction(self) -> None:
        """Should predict last value per series."""
        adapter = NaiveAdapter()
        train = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

        predictions = adapter.fit_predict(train, test_size=2, horizon=1)

        assert predictions.shape == (2, 2)
        assert all(predictions[0, :] == 3.0)
        assert all(predictions[1, :] == 30.0)

    def test_satisfies_protocol(self) -> None:
        """Should satisfy ForecastAdapter protocol."""
        adapter = NaiveAdapter()
        assert isinstance(adapter, ForecastAdapter)


class TestSeasonalNaiveAdapter:
    """Test SeasonalNaiveAdapter."""

    def test_properties(self) -> None:
        """Should have correct properties."""
        adapter = SeasonalNaiveAdapter(season_length=4)
        assert adapter.model_name == "SeasonalNaive_4"
        assert adapter.package_name == "temporalcv"

    def test_seasonal_prediction(self) -> None:
        """Should use seasonal lag."""
        adapter = SeasonalNaiveAdapter(season_length=4)
        # Pattern: 1, 2, 3, 4, 1, 2, 3, 4
        train = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])

        predictions = adapter.fit_predict(train, test_size=4, horizon=1)

        # Should repeat the pattern
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(predictions, expected)

    def test_get_params(self) -> None:
        """Should return parameters."""
        adapter = SeasonalNaiveAdapter(season_length=52)
        params = adapter.get_params()
        assert params["season_length"] == 52


# =============================================================================
# Test compute_comparison_metrics
# =============================================================================


class TestComputeComparisonMetrics:
    """Test compute_comparison_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should have zero errors."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        metrics = compute_comparison_metrics(preds, actuals)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_mae_calculation(self) -> None:
        """Should compute MAE correctly."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 2.0, 2.0])

        metrics = compute_comparison_metrics(preds, actuals)

        # |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2; MAE = 2/3
        assert metrics["mae"] == pytest.approx(2.0 / 3.0)

    def test_rmse_calculation(self) -> None:
        """Should compute RMSE correctly."""
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([2.0, 2.0, 2.0])

        metrics = compute_comparison_metrics(preds, actuals)

        # MSE = (1 + 0 + 1) / 3 = 2/3; RMSE = sqrt(2/3)
        assert metrics["rmse"] == pytest.approx(np.sqrt(2.0 / 3.0))

    def test_length_mismatch_raises(self) -> None:
        """Should raise on length mismatch."""
        preds = np.array([1.0, 2.0])
        actuals = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Length mismatch"):
            compute_comparison_metrics(preds, actuals)

    def test_direction_accuracy(self) -> None:
        """Should compute direction accuracy."""
        # Both go up
        preds = np.array([1.0, 2.0, 3.0])
        actuals = np.array([1.0, 2.0, 3.0])

        metrics = compute_comparison_metrics(preds, actuals)

        # Both diffs are positive
        assert metrics["direction_accuracy"] == 1.0


# =============================================================================
# Test Runner Functions
# =============================================================================


class TestRunComparison:
    """Test run_comparison function."""

    @pytest.fixture
    def synthetic_dataset(self) -> "Dataset":
        """Create synthetic dataset for testing."""
        from temporalcv.benchmarks import create_synthetic_dataset

        return create_synthetic_dataset(n_obs=50, train_fraction=0.8, seed=42)

    def test_single_adapter(self, synthetic_dataset: "Dataset") -> None:
        """Should run comparison with single adapter."""
        adapter = NaiveAdapter()

        result = run_comparison(
            synthetic_dataset,
            [adapter],
            primary_metric="mae",
            include_dm_test=False,
        )

        assert result.dataset_name == "synthetic_ar1"
        assert len(result.models) == 1
        assert result.best_model == "Naive"

    def test_multiple_adapters(self, synthetic_dataset: "Dataset") -> None:
        """Should compare multiple adapters."""
        adapters = [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=4),
        ]

        result = run_comparison(
            synthetic_dataset,
            adapters,
            primary_metric="mae",
            include_dm_test=False,
        )

        assert len(result.models) == 2

    def test_empty_adapters_raises(self, synthetic_dataset: "Dataset") -> None:
        """Should raise on empty adapters."""
        with pytest.raises(ValueError, match="adapters list cannot be empty"):
            run_comparison(synthetic_dataset, [], primary_metric="mae")


class TestRunBenchmarkSuite:
    """Test run_benchmark_suite function."""

    @pytest.fixture
    def datasets(self) -> list:
        """Create multiple synthetic datasets."""
        from temporalcv.benchmarks import create_synthetic_dataset

        return [
            create_synthetic_dataset(n_obs=50, seed=i) for i in range(3)
        ]

    def test_multiple_datasets(self, datasets: list) -> None:
        """Should run across multiple datasets."""
        adapters = [NaiveAdapter()]

        report = run_benchmark_suite(
            datasets,
            adapters,
            primary_metric="mae",
            include_dm_test=False,
        )

        assert report.summary["n_datasets"] == 3

    def test_empty_datasets_raises(self) -> None:
        """Should raise on empty datasets."""
        with pytest.raises(ValueError, match="datasets list cannot be empty"):
            run_benchmark_suite([], [NaiveAdapter()])


class TestCompareToBaseline:
    """Test compare_to_baseline function."""

    @pytest.fixture
    def synthetic_dataset(self) -> "Dataset":
        """Create synthetic dataset."""
        from temporalcv.benchmarks import create_synthetic_dataset

        return create_synthetic_dataset(n_obs=50, seed=42)

    def test_compare_to_default_baseline(self, synthetic_dataset: "Dataset") -> None:
        """Should compare to Naive baseline by default."""
        adapter = SeasonalNaiveAdapter(season_length=4)

        result = compare_to_baseline(
            synthetic_dataset,
            adapter,
            primary_metric="mae",
        )

        assert result["model_name"] == "SeasonalNaive_4"
        assert result["baseline_name"] == "Naive"
        assert "improvement_pct" in result
        assert "model_is_better" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestCompareIntegration:
    """Integration tests for compare module."""

    def test_full_workflow(self) -> None:
        """Complete workflow with synthetic data."""
        from temporalcv.benchmarks import create_synthetic_dataset

        # Create dataset
        dataset = create_synthetic_dataset(n_obs=100, seed=42)

        # Compare adapters
        adapters = [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=4),
        ]

        result = run_comparison(
            dataset,
            adapters,
            primary_metric="mae",
            include_dm_test=True,
        )

        # Verify structure
        assert len(result.models) == 2
        assert result.best_model in ["Naive", "SeasonalNaive_4"]

        # Check metrics exist
        for model in result.models:
            assert "mae" in model.metrics
            assert "rmse" in model.metrics
            assert model.runtime_seconds > 0

    def test_report_generation(self) -> None:
        """Should generate markdown report."""
        from temporalcv.benchmarks import create_synthetic_dataset

        datasets = [
            create_synthetic_dataset(n_obs=50, seed=i) for i in range(2)
        ]
        adapters = [NaiveAdapter()]

        report = run_benchmark_suite(
            datasets, adapters, include_dm_test=False
        )

        md = report.to_markdown()
        assert "# Model Comparison Report" in md
        assert "Naive" in md


# =============================================================================
# Regression Tests (Critical Bug Fixes)
# =============================================================================


class TestDMTestRunnerIntegration:
    """
    Regression test for DM test parameter name fix in runner.py.

    Bug: Parameter names errors1/errors2/horizon didn't match dm_test signature
         (errors_1/errors_2/h) causing silent TypeError (2025-12-23)
    Impact: All benchmark DM tests silently failed, caught by broad except Exception.
    Fix: Changed to errors_1, errors_2, h to match dm_test signature.
    """

    def test_dm_test_runs_through_runner(self) -> None:
        """
        DM test should actually run through run_comparison, not silently fail.

        Before fix: statistical_tests would contain {"error": "..."} for every model.
        After fix: statistical_tests should contain actual statistics and p-values.
        """
        from temporalcv.benchmarks import create_synthetic_dataset

        # Use n_obs=200 to ensure sufficient test samples (DM requires n >= 30)
        dataset = create_synthetic_dataset(n_obs=200, seed=42)
        adapters = [NaiveAdapter(), SeasonalNaiveAdapter(season_length=4)]

        result = run_comparison(
            dataset,
            adapters,
            primary_metric="mae",
            include_dm_test=True,
        )

        # Check that statistical_tests exist and contain actual statistics
        assert result.statistical_tests is not None, "statistical_tests should not be None"
        assert len(result.statistical_tests) > 0, "statistical_tests should not be empty"

        # At least one comparison should have real statistics, not error
        has_valid_result = False
        for model_name, dm_data in result.statistical_tests.items():
            if isinstance(dm_data, dict) and "error" not in dm_data:
                has_valid_result = True
                # Verify actual DM test output structure
                assert "statistic" in dm_data, f"Missing 'statistic' for {model_name}"
                assert "p_value" in dm_data, f"Missing 'p_value' for {model_name}"
                assert "significant" in dm_data, f"Missing 'significant' for {model_name}"
                # Verify values are reasonable
                assert isinstance(dm_data["statistic"], float), "statistic should be float"
                assert 0 <= dm_data["p_value"] <= 1, "p_value should be in [0, 1]"

        assert has_valid_result, (
            "All DM tests failed with errors - likely parameter name mismatch. "
            f"Results: {result.statistical_tests}"
        )

    def test_dm_results_not_all_errors(self) -> None:
        """
        DM results should not be 100% error payloads.

        This catches the silent failure pattern where TypeError is caught
        by broad exception handler.
        """
        from temporalcv.benchmarks import create_synthetic_dataset

        # Use n_obs=200 to ensure sufficient test samples (DM requires n >= 30)
        dataset = create_synthetic_dataset(n_obs=200, seed=123)
        adapters = [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=7),
        ]

        result = run_comparison(
            dataset,
            adapters,
            include_dm_test=True,
        )

        if result.statistical_tests:
            error_count = sum(
                1 for dm_data in result.statistical_tests.values()
                if isinstance(dm_data, dict) and "error" in dm_data
            )
            total_count = len(result.statistical_tests)

            # Should not have 100% error rate (which would indicate silent failure)
            assert error_count < total_count, (
                f"All {total_count} DM tests returned errors - "
                "this suggests dm_test is not being called correctly"
            )
