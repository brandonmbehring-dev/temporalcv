# =============================================================================
# Tests for Benchmarks.types
# =============================================================================
#
# Tests for core benchmark types: DatasetMetadata, TimeSeriesDataset.

using Test
using Dates

using TemporalValidation

# Import to_dict from Benchmarks explicitly (avoid conflict with Compare.to_dict)
import TemporalValidation.Benchmarks: to_dict as benchmark_to_dict


@testset "Benchmarks.types" begin

    # =========================================================================
    # DatasetNotFoundError
    # =========================================================================

    @testset "DatasetNotFoundError" begin
        err = DatasetNotFoundError(
            "M5",
            "https://kaggle.com/m5",
            "Download from Kaggle and extract to ~/data/m5/"
        )

        @test err.dataset_name == "M5"
        @test err.download_url == "https://kaggle.com/m5"
        @test contains(err.instructions, "Kaggle")

        # Should be throwable
        @test_throws DatasetNotFoundError throw(err)
    end

    # =========================================================================
    # DatasetMetadata
    # =========================================================================

    @testset "DatasetMetadata - basic construction" begin
        meta = DatasetMetadata(
            name = "test_dataset",
            frequency = "W",
            horizon = 4,
            n_series = 1,
            total_observations = 100
        )

        @test meta.name == "test_dataset"
        @test meta.frequency == "W"
        @test meta.horizon == 4
        @test meta.n_series == 1
        @test meta.total_observations == 100
        @test isnothing(meta.train_end_idx)
        @test meta.license == ""
        @test meta.official_split == false
    end

    @testset "DatasetMetadata - with optional fields" begin
        meta = DatasetMetadata(
            name = "full_dataset",
            frequency = "D",
            horizon = 7,
            n_series = 5,
            total_observations = 500,
            train_end_idx = 400,
            characteristics = Dict{String, Any}("high_persistence" => true),
            license = "MIT",
            source_url = "https://example.com",
            official_split = true,
            truncated = false,
            split_source = "Competition"
        )

        @test meta.train_end_idx == 400
        @test meta.characteristics["high_persistence"] == true
        @test meta.license == "MIT"
        @test meta.official_split == true
        @test meta.split_source == "Competition"
    end

    @testset "DatasetMetadata - validation" begin
        # Empty name
        @test_throws ErrorException DatasetMetadata(
            name = "",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100
        )

        # Invalid frequency
        @test_throws ErrorException DatasetMetadata(
            name = "test",
            frequency = "X",  # Invalid
            horizon = 2,
            n_series = 1,
            total_observations = 100
        )

        # Invalid horizon
        @test_throws ErrorException DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 0,  # Invalid
            n_series = 1,
            total_observations = 100
        )

        # Invalid n_series
        @test_throws ErrorException DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 0,  # Invalid
            total_observations = 100
        )

        # train_end_idx > total_observations
        @test_throws ErrorException DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100,
            train_end_idx = 150  # Invalid: exceeds total
        )
    end

    @testset "DatasetMetadata - valid frequencies" begin
        for freq in ["D", "W", "M", "H", "Y", "Q"]
            meta = DatasetMetadata(
                name = "test",
                frequency = freq,
                horizon = 1,
                n_series = 1,
                total_observations = 100
            )
            @test meta.frequency == freq
        end
    end

    @testset "DatasetMetadata - to_dict" begin
        meta = DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100,
            train_end_idx = 80
        )

        d = benchmark_to_dict(meta)

        @test d["name"] == "test"
        @test d["frequency"] == "W"
        @test d["horizon"] == 2
        @test d["train_end_idx"] == 80
        @test haskey(d, "characteristics")
    end

    # =========================================================================
    # TimeSeriesDataset
    # =========================================================================

    @testset "TimeSeriesDataset - single series" begin
        meta = DatasetMetadata(
            name = "single",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100,
            train_end_idx = 80
        )

        values = randn(100)
        dataset = TimeSeriesDataset(values=values, metadata=meta)

        @test dataset.values == values
        @test dataset.metadata === meta
        @test isnothing(dataset.timestamps)
        @test isnothing(dataset.exogenous)
        @test n_obs(dataset) == 100
        @test has_exogenous(dataset) == false
    end

    @testset "TimeSeriesDataset - multi series" begin
        meta = DatasetMetadata(
            name = "multi",
            frequency = "W",
            horizon = 2,
            n_series = 5,
            total_observations = 500,  # 5 * 100
            train_end_idx = 80
        )

        values = randn(5, 100)
        dataset = TimeSeriesDataset(values=values, metadata=meta)

        @test size(dataset.values) == (5, 100)
        @test n_obs(dataset) == 100  # Per series
    end

    @testset "TimeSeriesDataset - with timestamps" begin
        meta = DatasetMetadata(
            name = "with_dates",
            frequency = "D",
            horizon = 1,
            n_series = 1,
            total_observations = 30
        )

        values = randn(30)
        dates = [Date(2024, 1, 1) + Day(i-1) for i in 1:30]

        dataset = TimeSeriesDataset(
            values=values,
            metadata=meta,
            timestamps=dates
        )

        @test length(dataset.timestamps) == 30
        @test dataset.timestamps[1] == Date(2024, 1, 1)
    end

    @testset "TimeSeriesDataset - with exogenous" begin
        meta = DatasetMetadata(
            name = "with_exog",
            frequency = "D",
            horizon = 1,
            n_series = 1,
            total_observations = 50
        )

        values = randn(50)
        exog = randn(50, 3)  # 3 exogenous features

        dataset = TimeSeriesDataset(
            values=values,
            metadata=meta,
            exogenous=exog
        )

        @test has_exogenous(dataset) == true
        @test size(dataset.exogenous) == (50, 3)
    end

    @testset "TimeSeriesDataset - validation" begin
        # Mismatched values length
        meta = DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100
        )

        @test_throws ErrorException TimeSeriesDataset(
            values = randn(50),  # Wrong length
            metadata = meta
        )

        # Mismatched n_series
        meta_multi = DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 5,
            total_observations = 500
        )

        @test_throws ErrorException TimeSeriesDataset(
            values = randn(100),  # Should be Matrix for n_series=5
            metadata = meta_multi
        )
    end

    @testset "TimeSeriesDataset - get_train_test_split single" begin
        meta = DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100,
            train_end_idx = 80
        )

        values = collect(1.0:100.0)
        dataset = TimeSeriesDataset(values=values, metadata=meta)

        train, test = get_train_test_split(dataset)

        @test length(train) == 80
        @test length(test) == 20
        @test train[1] == 1.0
        @test train[end] == 80.0
        @test test[1] == 81.0
        @test test[end] == 100.0
    end

    @testset "TimeSeriesDataset - get_train_test_split multi" begin
        meta = DatasetMetadata(
            name = "test",
            frequency = "W",
            horizon = 2,
            n_series = 3,
            total_observations = 300,  # 3 * 100
            train_end_idx = 80
        )

        values = reshape(collect(1.0:300.0), 3, 100)
        dataset = TimeSeriesDataset(values=values, metadata=meta)

        train, test = get_train_test_split(dataset)

        @test size(train) == (3, 80)
        @test size(test) == (3, 20)
    end

    @testset "TimeSeriesDataset - get_train_test_split no split" begin
        meta = DatasetMetadata(
            name = "no_split",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100
            # No train_end_idx
        )

        dataset = TimeSeriesDataset(values=randn(100), metadata=meta)

        @test_throws ErrorException get_train_test_split(dataset)
    end

    # =========================================================================
    # validate_dataset
    # =========================================================================

    @testset "validate_dataset" begin
        # Valid dataset
        meta = DatasetMetadata(
            name = "valid",
            frequency = "W",
            horizon = 2,
            n_series = 1,
            total_observations = 100
        )
        dataset = TimeSeriesDataset(values=randn(100), metadata=meta)

        @test validate_dataset(dataset) == true
    end

end  # @testset "Benchmarks.types"
