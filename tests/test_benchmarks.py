"""
Tests for benchmark dataset infrastructure.

Tests Dataset protocol, TimeSeriesDataset, and synthetic data creation.
External loaders (FRED, M5, etc.) are tested with mocks where possible.
"""

from __future__ import annotations

import numpy as np
import pytest

from temporalcv.benchmarks import (
    Dataset,
    DatasetMetadata,
    DatasetNotFoundError,
    TimeSeriesDataset,
    create_synthetic_dataset,
    validate_dataset,
)


# =============================================================================
# Test DatasetMetadata
# =============================================================================


class TestDatasetMetadata:
    """Test DatasetMetadata dataclass."""

    def test_basic_creation(self) -> None:
        """Should create metadata with required fields."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=100,
        )

        assert metadata.name == "test"
        assert metadata.frequency == "W"
        assert metadata.horizon == 2
        assert metadata.n_series == 1
        assert metadata.total_observations == 100

    def test_optional_fields(self) -> None:
        """Should handle optional fields correctly."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=100,
            train_end_idx=80,
            characteristics={"key": "value"},
            license="public_domain",
            source_url="https://example.com",
        )

        assert metadata.train_end_idx == 80
        assert metadata.characteristics == {"key": "value"}
        assert metadata.license == "public_domain"
        assert metadata.source_url == "https://example.com"

    def test_invalid_horizon(self) -> None:
        """Should reject horizon < 1."""
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            DatasetMetadata(
                name="test",
                frequency="W",
                horizon=0,
                n_series=1,
                total_observations=100,
            )

    def test_invalid_n_series(self) -> None:
        """Should reject n_series < 1."""
        with pytest.raises(ValueError, match="n_series must be >= 1"):
            DatasetMetadata(
                name="test",
                frequency="W",
                horizon=2,
                n_series=0,
                total_observations=100,
            )

    def test_invalid_total_observations(self) -> None:
        """Should reject total_observations < 1."""
        with pytest.raises(ValueError, match="total_observations must be >= 1"):
            DatasetMetadata(
                name="test",
                frequency="W",
                horizon=2,
                n_series=1,
                total_observations=0,
            )

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=100,
        )

        d = metadata.to_dict()
        assert d["name"] == "test"
        assert d["frequency"] == "W"
        assert d["horizon"] == 2


# =============================================================================
# Test TimeSeriesDataset
# =============================================================================


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset class."""

    @pytest.fixture
    def simple_metadata(self) -> DatasetMetadata:
        """Create simple metadata for tests."""
        return DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=100,
            train_end_idx=80,
        )

    def test_single_series_creation(self, simple_metadata: DatasetMetadata) -> None:
        """Should create dataset with single series."""
        values = np.random.randn(100)
        dataset = TimeSeriesDataset(metadata=simple_metadata, values=values)

        assert len(dataset.values) == 100
        assert dataset.n_obs == 100

    def test_multi_series_creation(self) -> None:
        """Should create dataset with multiple series."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=5,
            total_observations=500,
            train_end_idx=80,
        )
        values = np.random.randn(5, 100)  # 5 series, 100 obs each
        dataset = TimeSeriesDataset(metadata=metadata, values=values)

        assert dataset.values.shape == (5, 100)
        assert dataset.n_obs == 100

    def test_train_test_split_single(self, simple_metadata: DatasetMetadata) -> None:
        """Should correctly split single series."""
        values = np.arange(100)
        dataset = TimeSeriesDataset(metadata=simple_metadata, values=values)

        train, test = dataset.get_train_test_split()

        assert len(train) == 80
        assert len(test) == 20
        assert train[-1] == 79
        assert test[0] == 80

    def test_train_test_split_multi(self) -> None:
        """Should correctly split multiple series."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=3,
            total_observations=300,
            train_end_idx=80,
        )
        values = np.random.randn(3, 100)
        dataset = TimeSeriesDataset(metadata=metadata, values=values)

        train, test = dataset.get_train_test_split()

        assert train.shape == (3, 80)
        assert test.shape == (3, 20)

    def test_no_split_raises(self) -> None:
        """Should raise if no train_end_idx defined."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=100,
            # No train_end_idx
        )
        values = np.random.randn(100)
        dataset = TimeSeriesDataset(metadata=metadata, values=values)

        with pytest.raises(ValueError, match="no standard train/test split"):
            dataset.get_train_test_split()

    def test_empty_values_raises(self) -> None:
        """Should raise on empty values."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=1,  # Will fail validation
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            TimeSeriesDataset(metadata=metadata, values=np.array([]))

    def test_has_exogenous(self, simple_metadata: DatasetMetadata) -> None:
        """Should track exogenous features."""
        values = np.random.randn(100)
        dataset_no_exog = TimeSeriesDataset(metadata=simple_metadata, values=values)
        assert not dataset_no_exog.has_exogenous

        exog = np.random.randn(100, 3)
        dataset_with_exog = TimeSeriesDataset(
            metadata=simple_metadata, values=values, exogenous=exog
        )
        assert dataset_with_exog.has_exogenous

    def test_satisfies_protocol(self, simple_metadata: DatasetMetadata) -> None:
        """Should satisfy Dataset protocol."""
        values = np.random.randn(100)
        dataset = TimeSeriesDataset(metadata=simple_metadata, values=values)

        assert isinstance(dataset, Dataset)


# =============================================================================
# Test DatasetNotFoundError
# =============================================================================


class TestDatasetNotFoundError:
    """Test DatasetNotFoundError exception."""

    def test_error_message_format(self) -> None:
        """Should format error message correctly."""
        error = DatasetNotFoundError(
            dataset_name="M5",
            download_url="https://kaggle.com/m5",
            instructions="Download from Kaggle",
        )

        assert "M5" in str(error)
        assert "https://kaggle.com/m5" in str(error)
        assert "Download from Kaggle" in str(error)

    def test_attributes_preserved(self) -> None:
        """Should preserve all attributes."""
        error = DatasetNotFoundError(
            dataset_name="test",
            download_url="https://example.com",
            instructions="Instructions here",
        )

        assert error.dataset_name == "test"
        assert error.download_url == "https://example.com"
        assert error.instructions == "Instructions here"

    def test_is_file_not_found_error(self) -> None:
        """Should be a FileNotFoundError subclass."""
        error = DatasetNotFoundError(
            dataset_name="test",
            download_url="https://example.com",
            instructions="Instructions",
        )

        assert isinstance(error, FileNotFoundError)


# =============================================================================
# Test validate_dataset
# =============================================================================


class TestValidateDataset:
    """Test validate_dataset function."""

    def test_valid_dataset_passes(self) -> None:
        """Valid dataset should pass validation."""
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=100,
        )
        values = np.random.randn(100)
        dataset = TimeSeriesDataset(metadata=metadata, values=values)

        # Should not raise
        validate_dataset(dataset)

    def test_empty_values_fails(self) -> None:
        """Empty values should fail validation."""
        # Create a mock dataset with empty values
        metadata = DatasetMetadata(
            name="test",
            frequency="W",
            horizon=2,
            n_series=1,
            total_observations=1,
        )

        # Create dataset and manually set empty values (bypassing __post_init__)
        dataset = TimeSeriesDataset(metadata=metadata, values=np.array([1.0]))
        dataset.values = np.array([])

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_dataset(dataset)


# =============================================================================
# Test create_synthetic_dataset
# =============================================================================


class TestCreateSyntheticDataset:
    """Test synthetic dataset creation."""

    def test_default_parameters(self) -> None:
        """Should create dataset with default parameters."""
        dataset = create_synthetic_dataset()

        assert dataset.metadata.name == "synthetic_ar1"
        assert dataset.metadata.n_series == 1
        assert dataset.n_obs == 200
        assert dataset.metadata.train_end_idx == 160  # 80% of 200

    def test_custom_n_obs(self) -> None:
        """Should respect n_obs parameter."""
        dataset = create_synthetic_dataset(n_obs=500)

        assert dataset.n_obs == 500

    def test_custom_n_series(self) -> None:
        """Should create multiple series."""
        dataset = create_synthetic_dataset(n_obs=100, n_series=10)

        assert dataset.metadata.n_series == 10
        assert dataset.values.shape == (10, 100)

    def test_custom_train_fraction(self) -> None:
        """Should respect train_fraction."""
        dataset = create_synthetic_dataset(n_obs=100, train_fraction=0.5)

        assert dataset.metadata.train_end_idx == 50

    def test_ar_coefficient_affects_persistence(self) -> None:
        """Higher AR coefficient should produce more persistent series."""
        high_ar = create_synthetic_dataset(n_obs=1000, ar_coef=0.99, seed=42)
        low_ar = create_synthetic_dataset(n_obs=1000, ar_coef=0.1, seed=42)

        # High AR should have higher autocorrelation
        high_acf = np.corrcoef(high_ar.values[:-1], high_ar.values[1:])[0, 1]
        low_acf = np.corrcoef(low_ar.values[:-1], low_ar.values[1:])[0, 1]

        assert high_acf > low_acf

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical datasets."""
        ds1 = create_synthetic_dataset(seed=12345)
        ds2 = create_synthetic_dataset(seed=12345)

        np.testing.assert_array_equal(ds1.values, ds2.values)

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds should produce different data."""
        ds1 = create_synthetic_dataset(seed=1)
        ds2 = create_synthetic_dataset(seed=2)

        assert not np.array_equal(ds1.values, ds2.values)

    def test_metadata_characteristics(self) -> None:
        """Should include AR parameters in characteristics."""
        dataset = create_synthetic_dataset(ar_coef=0.95, noise_std=0.2)

        assert dataset.metadata.characteristics["ar_coef"] == 0.95
        assert dataset.metadata.characteristics["noise_std"] == 0.2
        assert dataset.metadata.characteristics["synthetic"] is True

    def test_can_get_train_test_split(self) -> None:
        """Should be able to split synthetic data."""
        dataset = create_synthetic_dataset(n_obs=100, train_fraction=0.8)

        train, test = dataset.get_train_test_split()

        assert len(train) == 80
        assert len(test) == 20


# =============================================================================
# Test M5 Loader (DatasetNotFoundError path)
# =============================================================================


class TestM5Loader:
    """Test M5 loader DatasetNotFoundError behavior."""

    def test_no_path_raises_not_found(self) -> None:
        """Should raise DatasetNotFoundError with no path."""
        from temporalcv.benchmarks.m5 import load_m5

        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_m5(path=None)

        assert "M5 Walmart" in str(exc_info.value)
        assert "kaggle.com" in str(exc_info.value)

    def test_missing_file_raises_not_found(self, tmp_path: pytest.TempPathFactory) -> None:
        """Should raise DatasetNotFoundError if file not found."""
        from temporalcv.benchmarks.m5 import load_m5

        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_m5(path=str(tmp_path))

        assert "sales_train_evaluation.csv not found" in str(exc_info.value)


# =============================================================================
# Test FRED Loader (API key path)
# =============================================================================


class TestFREDLoader:
    """Test FRED loader behavior."""

    def test_import_error_without_fredapi(self) -> None:
        """Should raise ImportError if fredapi not installed."""
        # This test only runs when fredapi is NOT installed
        try:
            import fredapi  # noqa: F401

            pytest.skip("fredapi is installed, skipping ImportError test")
        except ImportError:
            pass

        from temporalcv.benchmarks.fred import load_fred_rates

        with pytest.raises(ImportError, match="fredapi required"):
            load_fred_rates()

    def test_no_api_key_raises_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise DatasetNotFoundError without API key."""
        # This test requires fredapi to be installed
        pytest.importorskip("fredapi")

        # Ensure no API key in environment
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        from temporalcv.benchmarks.fred import load_fred_rates

        with pytest.raises(DatasetNotFoundError) as exc_info:
            load_fred_rates()

        assert "FRED" in str(exc_info.value)
        assert "API key" in str(exc_info.value)

    def test_list_available_series(self) -> None:
        """Should list available FRED series."""
        from temporalcv.benchmarks.fred import list_available_series

        series = list_available_series()

        assert "DGS10" in series
        assert "FEDFUNDS" in series


# =============================================================================
# Integration Tests
# =============================================================================


class TestBenchmarksIntegration:
    """Integration tests for benchmarks module."""

    def test_synthetic_workflow(self) -> None:
        """Complete workflow with synthetic data."""
        # Create dataset
        dataset = create_synthetic_dataset(
            n_obs=200, n_series=1, ar_coef=0.9, train_fraction=0.8
        )

        # Validate
        validate_dataset(dataset)

        # Split
        train, test = dataset.get_train_test_split()

        # Use for simple model
        # (just verify shapes work)
        assert len(train) == 160
        assert len(test) == 40

        # Compute simple forecast error (persistence = 0)
        persistence_mae = np.mean(np.abs(test))
        assert persistence_mae > 0  # Should have some error

    def test_multi_series_workflow(self) -> None:
        """Workflow with multiple series."""
        dataset = create_synthetic_dataset(n_obs=100, n_series=5, train_fraction=0.8)

        train, test = dataset.get_train_test_split()

        assert train.shape == (5, 80)
        assert test.shape == (5, 20)

        # Per-series MAE
        mae_per_series = np.mean(np.abs(test), axis=1)
        assert mae_per_series.shape == (5,)
