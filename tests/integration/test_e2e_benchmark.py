"""
End-to-end benchmark test.

Runs the full benchmark pipeline with bundled synthetic data.
Verifies integration of all components:
- Dataset loading
- Model comparison
- Statistical tests
- Report generation

Enhanced with:
- M4 real data tests (optional dependency)
- statsforecast adapter tests (optional dependency)
- Multi-frequency synthetic tests
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.benchmarks import create_synthetic_dataset
from temporalcv.compare import (
    ComparisonReport,
    NaiveAdapter,
    SeasonalNaiveAdapter,
    compare_to_baseline,
    run_benchmark_suite,
    run_comparison,
)

# =============================================================================
# Optional Dependency Detection
# =============================================================================

try:
    from datasetsforecast.m4 import M4  # noqa: F401

    HAS_DATASETSFORECAST = True
except ImportError:
    HAS_DATASETSFORECAST = False

try:
    import statsforecast  # noqa: F401

    HAS_STATSFORECAST = True
except ImportError:
    HAS_STATSFORECAST = False


class TestE2EBenchmarkPipeline:
    """End-to-end tests for benchmark pipeline."""

    @pytest.fixture
    def bundled_datasets(self):
        """Create bundled synthetic datasets for offline testing."""
        return [
            create_synthetic_dataset(
                n_obs=150,
                ar_coef=0.9,
                noise_std=0.1,
                seed=seed,
                horizon=2,
            )
            for seed in [42, 123, 456]
        ]

    @pytest.fixture
    def test_adapters(self):
        """Standard test adapters."""
        return [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=4),
        ]

    def test_full_pipeline_synthetic(
        self, bundled_datasets, test_adapters
    ) -> None:
        """
        Full pipeline: datasets -> comparison -> report -> markdown.

        This is the primary E2E test validating all components integrate.
        """
        # Run full benchmark suite
        report = run_benchmark_suite(
            datasets=bundled_datasets,
            adapters=test_adapters,
            primary_metric="mae",
            include_dm_test=True,
        )

        # Validate report structure
        assert isinstance(report, ComparisonReport)
        assert report.summary["n_datasets"] == 3
        assert len(report.results) == 3

        # Validate each result has expected structure
        for result in report.results:
            assert len(result.models) == 2
            assert result.best_model in ["Naive", "SeasonalNaive_4"]

            # Each model should have valid metrics
            for model in result.models:
                assert "mae" in model.metrics
                assert "rmse" in model.metrics
                assert model.metrics["mae"] > 0, "MAE should be positive"
                assert np.isfinite(model.metrics["mae"]), "MAE should be finite"
                assert model.runtime_seconds >= 0

        # Validate markdown generation
        md = report.to_markdown()
        assert "# Model Comparison Report" in md
        assert "Naive" in md
        assert "SeasonalNaive_4" in md

    def test_dm_tests_run_in_pipeline(self, bundled_datasets) -> None:
        """DM tests should actually execute with sufficient data."""
        # Use larger dataset for reliable DM test
        large_dataset = create_synthetic_dataset(n_obs=200, seed=42)

        adapters = [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=7),
        ]

        result = run_comparison(
            dataset=large_dataset,
            adapters=adapters,
            include_dm_test=True,
        )

        # DM tests should have run
        assert result.statistical_tests is not None

        # At least one comparison should have actual statistics
        has_valid_test = False
        for model_name, dm_data in result.statistical_tests.items():
            if isinstance(dm_data, dict) and "error" not in dm_data:
                has_valid_test = True
                assert "statistic" in dm_data
                assert "p_value" in dm_data
                assert 0 <= dm_data["p_value"] <= 1

        assert has_valid_test, "No valid DM tests ran"

    def test_aggregation_modes(self, test_adapters) -> None:
        """Test different aggregation modes for multi-series."""
        # Multi-series dataset
        dataset = create_synthetic_dataset(
            n_obs=100,
            n_series=5,
            seed=42,
        )

        # Test each aggregation mode
        for mode in ["flatten", "per_series_mean", "per_series_median"]:
            result = run_comparison(
                dataset=dataset,
                adapters=test_adapters,
                aggregation_mode=mode,
                include_dm_test=False,
            )

            assert len(result.models) == 2
            for model in result.models:
                assert "mae" in model.metrics
                assert np.isfinite(model.metrics["mae"])

    def test_compare_to_baseline(self) -> None:
        """Test single model vs baseline comparison."""
        dataset = create_synthetic_dataset(n_obs=100, seed=42)

        result = compare_to_baseline(
            dataset=dataset,
            adapter=SeasonalNaiveAdapter(season_length=4),
            primary_metric="mae",
        )

        assert result["model_name"] == "SeasonalNaive_4"
        assert result["baseline_name"] == "Naive"
        assert "improvement_pct" in result
        assert "model_is_better" in result
        assert isinstance(result["model_is_better"], bool)

    def test_benchmark_reproducibility(self) -> None:
        """Same inputs should produce same outputs."""
        datasets = [create_synthetic_dataset(seed=42)]
        adapters = [NaiveAdapter()]

        report1 = run_benchmark_suite(
            datasets=datasets,
            adapters=adapters,
            include_dm_test=False,
        )

        report2 = run_benchmark_suite(
            datasets=datasets,
            adapters=adapters,
            include_dm_test=False,
        )

        # Same MAE values
        mae1 = report1.results[0].models[0].metrics["mae"]
        mae2 = report2.results[0].models[0].metrics["mae"]
        assert mae1 == mae2

    def test_empty_datasets_raises(self, test_adapters) -> None:
        """Empty datasets should raise clear error."""
        with pytest.raises(ValueError, match="datasets list cannot be empty"):
            run_benchmark_suite([], test_adapters)

    def test_empty_adapters_raises(self, bundled_datasets) -> None:
        """Empty adapters should raise clear error."""
        with pytest.raises(ValueError, match="adapters list cannot be empty"):
            run_benchmark_suite(bundled_datasets, [])


class TestBenchmarkDatasetIntegrity:
    """Test bundled dataset integrity."""

    def test_synthetic_dataset_structure(self) -> None:
        """Synthetic dataset should have valid structure."""
        dataset = create_synthetic_dataset(n_obs=100, seed=42)

        assert dataset.metadata.name == "synthetic_ar1"
        assert dataset.metadata.n_series == 1
        assert dataset.metadata.horizon == 2
        assert len(dataset.values) == 100

        # Should be able to split
        train, test = dataset.get_train_test_split()
        assert len(train) + len(test) == 100

    def test_multi_series_synthetic(self) -> None:
        """Multi-series synthetic dataset."""
        dataset = create_synthetic_dataset(n_obs=50, n_series=3, seed=42)

        assert dataset.metadata.n_series == 3
        assert dataset.values.shape == (3, 50)

    def test_ar1_properties(self) -> None:
        """Synthetic AR(1) should have expected properties."""
        # High persistence
        dataset = create_synthetic_dataset(
            n_obs=500,
            ar_coef=0.95,
            noise_std=0.1,
            seed=42,
        )

        # Compute lag-1 autocorrelation
        y = dataset.values
        acf1 = np.corrcoef(y[:-1], y[1:])[0, 1]

        # Should be close to ar_coef
        assert 0.85 < acf1 < 0.99, f"ACF(1) = {acf1} not in expected range"

    def test_frequency_specific_seasonality(self) -> None:
        """Different season_lengths for different frequencies."""
        configs = [
            ("yearly", 1, 50),
            ("quarterly", 4, 80),
            ("monthly", 12, 200),
            ("weekly", 52, 200),
        ]

        adapters_base = [NaiveAdapter()]

        for freq_name, season_len, n_obs in configs:
            # Create synthetic with appropriate length
            dataset = create_synthetic_dataset(
                n_obs=n_obs,
                ar_coef=0.7,
                seed=42,
            )

            # Add seasonal adapter only if season_len > 1
            adapters = adapters_base.copy()
            if season_len > 1:
                adapters.append(SeasonalNaiveAdapter(season_length=season_len))

            result = run_comparison(
                dataset=dataset,
                adapters=adapters,
                include_dm_test=False,
            )

            # Pipeline should complete
            assert len(result.models) == len(adapters)
            for model in result.models:
                assert "mae" in model.metrics
                assert np.isfinite(model.metrics["mae"])


# =============================================================================
# M4 Real Data Tests (Optional Dependency)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not HAS_DATASETSFORECAST,
    reason="datasetsforecast not installed (pip install datasetsforecast)",
)
class TestE2EWithM4Data:
    """
    E2E tests with real M4 competition data.

    These tests validate the pipeline with actual M4 data.
    They're marked slow because M4 loading requires network access.
    """

    def test_m4_monthly_pipeline(self) -> None:
        """Full pipeline on M4 monthly subset (10 series)."""
        from temporalcv.benchmarks import load_m4

        # Small sample for fast test
        dataset = load_m4(subset="monthly", sample_size=10)

        adapters = [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=12),
        ]

        report = run_benchmark_suite(
            datasets=[dataset],
            adapters=adapters,
            primary_metric="mae",
            include_dm_test=True,
        )

        # Validate report
        assert isinstance(report, ComparisonReport)
        assert report.summary["n_datasets"] == 1
        assert len(report.results) == 1

        result = report.results[0]
        assert result.dataset_name == "m4_monthly"
        assert len(result.models) == 2
        assert result.best_model in ["Naive", "SeasonalNaive_12"]

    def test_m4_all_frequencies_small(self) -> None:
        """Test all 6 M4 frequencies with minimal samples (5 series each)."""
        from temporalcv.benchmarks import load_m4

        m4_configs = [
            ("yearly", 1),      # no seasonality for yearly data
            ("quarterly", 4),   # quarterly seasonality
            ("monthly", 12),    # monthly seasonality
            ("weekly", 52),     # weekly → yearly seasonality
            ("daily", 7),       # daily → weekly seasonality
            ("hourly", 24),     # hourly → daily seasonality
        ]

        results_summary = []

        for freq, season_len in m4_configs:
            dataset = load_m4(subset=freq, sample_size=5)

            adapters = [NaiveAdapter()]
            if season_len > 1:
                adapters.append(SeasonalNaiveAdapter(season_length=season_len))

            result = run_comparison(
                dataset=dataset,
                adapters=adapters,
                include_dm_test=False,  # Skip for speed
            )

            assert len(result.models) >= 1
            assert result.dataset_name == f"m4_{freq}"

            results_summary.append({
                "frequency": freq,
                "n_series": dataset.metadata.n_series,
                "horizon": dataset.metadata.horizon,
                "best_model": result.best_model,
            })

        # All 6 frequencies should have completed
        assert len(results_summary) == 6

    def test_m4_reproducibility(self) -> None:
        """Same M4 sample should produce same results."""
        from temporalcv.benchmarks import load_m4

        # Load same data twice
        dataset1 = load_m4(subset="quarterly", sample_size=5)
        dataset2 = load_m4(subset="quarterly", sample_size=5)

        adapters = [NaiveAdapter()]

        result1 = run_comparison(dataset=dataset1, adapters=adapters)
        result2 = run_comparison(dataset=dataset2, adapters=adapters)

        mae1 = result1.models[0].metrics["mae"]
        mae2 = result2.models[0].metrics["mae"]

        assert np.isclose(mae1, mae2), f"MAE mismatch: {mae1} vs {mae2}"


# =============================================================================
# statsforecast Adapter Tests (Optional Dependency)
# =============================================================================


@pytest.mark.skipif(
    not HAS_STATSFORECAST,
    reason="statsforecast not installed (pip install statsforecast)",
)
class TestE2EWithStatsforecast:
    """
    E2E tests with statsforecast models.

    These tests validate the pipeline with statsforecast models
    (AutoARIMA, AutoETS, AutoTheta).
    """

    @pytest.fixture
    def synthetic_dataset(self):
        """Create synthetic dataset for testing."""
        return create_synthetic_dataset(
            n_obs=100,
            n_series=3,
            ar_coef=0.8,
            seed=42,
        )

    def test_autoarima_pipeline(self, synthetic_dataset) -> None:
        """AutoARIMA on synthetic data."""
        from temporalcv.compare.adapters import MultiSeriesAdapter, StatsforecastAdapter

        adapters = [
            NaiveAdapter(),
            MultiSeriesAdapter(StatsforecastAdapter("AutoARIMA", season_length=1)),
        ]

        result = run_comparison(
            dataset=synthetic_dataset,
            adapters=adapters,
            include_dm_test=False,
        )

        assert len(result.models) == 2

        # Find AutoARIMA results
        autoarima_model = next(
            (m for m in result.models if "AutoARIMA" in m.model_name), None
        )
        assert autoarima_model is not None
        assert "mae" in autoarima_model.metrics
        assert np.isfinite(autoarima_model.metrics["mae"])

    def test_autoets_pipeline(self, synthetic_dataset) -> None:
        """AutoETS on synthetic data."""
        from temporalcv.compare.adapters import MultiSeriesAdapter, StatsforecastAdapter

        adapters = [
            NaiveAdapter(),
            MultiSeriesAdapter(StatsforecastAdapter("AutoETS", season_length=1)),
        ]

        result = run_comparison(
            dataset=synthetic_dataset,
            adapters=adapters,
            include_dm_test=False,
        )

        assert len(result.models) == 2

        autoets_model = next(
            (m for m in result.models if "AutoETS" in m.model_name), None
        )
        assert autoets_model is not None
        assert "mae" in autoets_model.metrics

    def test_autotheta_pipeline(self, synthetic_dataset) -> None:
        """AutoTheta on synthetic data."""
        from temporalcv.compare.adapters import MultiSeriesAdapter, StatsforecastAdapter

        adapters = [
            NaiveAdapter(),
            MultiSeriesAdapter(StatsforecastAdapter("AutoTheta", season_length=1)),
        ]

        result = run_comparison(
            dataset=synthetic_dataset,
            adapters=adapters,
            include_dm_test=False,
        )

        assert len(result.models) == 2

        autotheta_model = next(
            (m for m in result.models if "AutoTheta" in m.model_name), None
        )
        assert autotheta_model is not None
        assert "mae" in autotheta_model.metrics

    @pytest.mark.slow
    def test_full_model_suite_small(self, synthetic_dataset) -> None:
        """All 9 models from benchmark script on small synthetic data."""
        from temporalcv.compare.adapters import MultiSeriesAdapter, StatsforecastAdapter

        # Same model suite as run_benchmark.py
        adapters = [
            NaiveAdapter(),
            SeasonalNaiveAdapter(season_length=4),
            MultiSeriesAdapter(StatsforecastAdapter("AutoARIMA", season_length=1)),
            MultiSeriesAdapter(StatsforecastAdapter("AutoETS", season_length=1)),
            MultiSeriesAdapter(StatsforecastAdapter("AutoTheta", season_length=1)),
            MultiSeriesAdapter(StatsforecastAdapter("CrostonClassic")),
            MultiSeriesAdapter(StatsforecastAdapter("ADIDA")),
            MultiSeriesAdapter(StatsforecastAdapter("IMAPA")),
            MultiSeriesAdapter(StatsforecastAdapter("HistoricAverage")),
        ]

        result = run_comparison(
            dataset=synthetic_dataset,
            adapters=adapters,
            include_dm_test=False,  # Skip for speed
        )

        # All 9 models should have results
        assert len(result.models) == 9

        # Each model should have valid metrics
        for model in result.models:
            assert "mae" in model.metrics
            assert np.isfinite(model.metrics["mae"])
            assert model.runtime_seconds >= 0

        # Should identify a best model
        assert result.best_model is not None

    @pytest.mark.slow
    def test_statsforecast_with_dm_test(self, synthetic_dataset) -> None:
        """Statsforecast models with DM test validation."""
        from temporalcv.compare.adapters import MultiSeriesAdapter, StatsforecastAdapter

        adapters = [
            NaiveAdapter(),
            MultiSeriesAdapter(StatsforecastAdapter("AutoETS", season_length=1)),
        ]

        result = run_comparison(
            dataset=synthetic_dataset,
            adapters=adapters,
            include_dm_test=True,
        )

        assert len(result.models) == 2
        assert result.statistical_tests is not None

    def test_statsforecast_model_names(self) -> None:
        """Verify all supported statsforecast model names."""
        from temporalcv.compare.adapters import StatsforecastAdapter

        valid_models = [
            "AutoARIMA",
            "AutoETS",
            "AutoTheta",
            "CrostonClassic",
            "ADIDA",
            "IMAPA",
            "HistoricAverage",
            "Naive",
            "SeasonalNaive",
            "WindowAverage",
        ]

        for model_name in valid_models:
            # Should not raise
            adapter = StatsforecastAdapter(model_name)
            assert adapter.model_name == model_name
            assert adapter.package_name == "statsforecast"

    def test_statsforecast_invalid_model(self) -> None:
        """Invalid model name should raise clear error."""
        from temporalcv.compare.adapters import StatsforecastAdapter

        with pytest.raises(ValueError, match="Unknown model"):
            StatsforecastAdapter("NonExistentModel")
