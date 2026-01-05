# =============================================================================
# Tests for Benchmarks.synthetic
# =============================================================================
#
# Tests for synthetic dataset generation and benchmark integration.

using Test
using Statistics
using Random

# Import from parent module (assumes loaded via runtests.jl)
using TemporalValidation


@testset "Benchmarks.synthetic" begin

    # =========================================================================
    # AR(1) Series Generation
    # =========================================================================

    @testset "create_ar1_series - single series" begin
        # Basic generation
        series = create_ar1_series(100; ar_coef=0.9, noise_std=0.1, seed=42)
        @test length(series) == 100
        @test eltype(series) == Float64

        # Reproducibility
        series2 = create_ar1_series(100; ar_coef=0.9, noise_std=0.1, seed=42)
        @test series == series2

        # Different seed gives different result
        series3 = create_ar1_series(100; ar_coef=0.9, noise_std=0.1, seed=123)
        @test series != series3

        # No seed gives random result
        series_a = create_ar1_series(100; seed=nothing)
        series_b = create_ar1_series(100; seed=nothing)
        # Very unlikely to be equal (1 in 2^64 per element)
        @test series_a != series_b
    end

    @testset "create_ar1_series - multi series" begin
        # Matrix generation
        series = create_ar1_series(5, 100; seed=42)
        @test size(series) == (5, 100)
        @test eltype(series) == Float64

        # Each row is different (independent series)
        @test series[1, :] != series[2, :]
    end

    @testset "create_ar1_series - AR(1) properties" begin
        # Generate high-persistence series
        n = 1000
        ar_coef = 0.95
        series = create_ar1_series(n; ar_coef=ar_coef, noise_std=0.1, seed=42)

        # Compute lag-1 autocorrelation
        lag1_corr = cor(series[1:end-1], series[2:end])

        # Should be close to ar_coef (within statistical tolerance)
        @test 0.85 < lag1_corr < 0.99
    end

    @testset "create_ar1_series - validation" begin
        # n_obs must be >= 1
        @test_throws ErrorException create_ar1_series(0; seed=42)

        # noise_std must be > 0
        @test_throws ErrorException create_ar1_series(100; noise_std=0.0, seed=42)
        @test_throws ErrorException create_ar1_series(100; noise_std=-0.1, seed=42)

        # n_series must be >= 1
        @test_throws ErrorException create_ar1_series(0, 100; seed=42)

        # Non-stationary warning (doesn't throw, just warns)
        # ar_coef >= 1.0 should warn but not error
        @test_logs (:warn,) create_ar1_series(10; ar_coef=1.0, seed=42)
    end

    # =========================================================================
    # Synthetic Dataset Factory
    # =========================================================================

    @testset "create_synthetic_dataset - single series" begin
        dataset = create_synthetic_dataset(n_obs=100, seed=42)

        @test dataset.metadata.name == "synthetic_ar1"
        @test dataset.metadata.n_series == 1
        @test dataset.metadata.horizon == 2
        @test dataset.values isa Vector{Float64}
        @test length(dataset.values) == 100

        # Can split
        train, test = get_train_test_split(dataset)
        @test length(train) + length(test) == 100
        @test length(train) == 80  # 0.8 * 100
    end

    @testset "create_synthetic_dataset - multi series" begin
        dataset = create_synthetic_dataset(n_obs=50, n_series=3, seed=42)

        @test dataset.metadata.n_series == 3
        @test dataset.values isa Matrix{Float64}
        @test size(dataset.values) == (3, 50)

        # Can split
        train, test = get_train_test_split(dataset)
        @test size(train) == (3, 40)  # 0.8 * 50
        @test size(test) == (3, 10)
    end

    @testset "create_synthetic_dataset - custom parameters" begin
        dataset = create_synthetic_dataset(
            n_obs=200,
            frequency="D",
            horizon=5,
            train_fraction=0.7,
            ar_coef=0.95,
            noise_std=0.5,
            seed=123
        )

        @test dataset.metadata.frequency == "D"
        @test dataset.metadata.horizon == 5
        @test dataset.metadata.train_end_idx == 140  # 0.7 * 200

        # Characteristics preserved
        @test dataset.metadata.characteristics["ar_coef"] == 0.95
        @test dataset.metadata.characteristics["noise_std"] == 0.5
        @test dataset.metadata.characteristics["seed"] == 123
    end

    @testset "create_synthetic_dataset - validation" begin
        # Invalid train_fraction
        @test_throws ErrorException create_synthetic_dataset(train_fraction=0.0)
        @test_throws ErrorException create_synthetic_dataset(train_fraction=1.0)

        # Invalid horizon
        @test_throws ErrorException create_synthetic_dataset(horizon=0)

        # Train fraction too high (not enough test data for horizon)
        @test_throws ErrorException create_synthetic_dataset(
            n_obs=100, train_fraction=0.99, horizon=10
        )
    end

    @testset "create_synthetic_dataset - reproducibility" begin
        ds1 = create_synthetic_dataset(seed=42)
        ds2 = create_synthetic_dataset(seed=42)
        ds3 = create_synthetic_dataset(seed=123)

        @test ds1.values == ds2.values
        @test ds1.values != ds3.values
    end

    # =========================================================================
    # Electricity-like Dataset
    # =========================================================================

    @testset "create_electricity_like_dataset" begin
        dataset = create_electricity_like_dataset(n_obs=336, seed=42)

        @test dataset.metadata.name == "synthetic_electricity"
        @test dataset.metadata.frequency == "H"
        @test dataset.metadata.horizon == 24
        @test length(dataset.values) == 336

        # Values should be positive (base level 100)
        @test all(dataset.values .> 0)

        # Characteristics
        @test dataset.metadata.characteristics["seasonality"] == 24
        @test dataset.metadata.characteristics["synthetic"] == true
    end

    @testset "create_electricity_like_dataset - multi series" begin
        dataset = create_electricity_like_dataset(n_obs=100, n_series=3, seed=42)

        @test dataset.metadata.n_series == 3
        @test size(dataset.values) == (3, 100)
    end

    # =========================================================================
    # Bundled Test Datasets
    # =========================================================================

    @testset "create_bundled_test_datasets" begin
        datasets = create_bundled_test_datasets()

        @test length(datasets) == 3

        for ds in datasets
            @test ds.metadata.n_series == 1
            @test ds.metadata.horizon == 2
            @test n_obs(ds) == 150
        end

        # Different seeds should give different values
        @test datasets[1].values != datasets[2].values
        @test datasets[2].values != datasets[3].values
    end

    @testset "create_bundled_test_datasets - custom seeds" begin
        datasets = create_bundled_test_datasets(seeds=[1, 2])
        @test length(datasets) == 2
    end

    # =========================================================================
    # Benchmark Integration
    # =========================================================================

    @testset "to_benchmark_tuple - single series" begin
        dataset = create_synthetic_dataset(n_obs=100, seed=42)
        name, train, test = to_benchmark_tuple(dataset)

        @test name == "synthetic_ar1"
        @test train isa Vector{Float64}
        @test test isa Vector{Float64}
        @test length(train) == 80
        @test length(test) == 20
    end

    @testset "to_benchmark_tuple - multi series" begin
        dataset = create_synthetic_dataset(n_obs=100, n_series=3, seed=42)
        name, train, test = to_benchmark_tuple(dataset)

        # Should return first series only
        @test train isa Vector{Float64}
        @test test isa Vector{Float64}
        @test length(train) == 80
        @test length(test) == 20
    end

    @testset "to_benchmark_tuples - multi series expansion" begin
        dataset = create_synthetic_dataset(n_obs=50, n_series=3, seed=42)
        tuples = to_benchmark_tuples(dataset)

        @test length(tuples) == 3

        for (i, (name, train, test)) in enumerate(tuples)
            @test name == "synthetic_ar1_$i"
            @test length(train) == 40  # 0.8 * 50
            @test length(test) == 10
        end

        # Each tuple should have different values
        @test tuples[1][2] != tuples[2][2]
    end

    @testset "to_benchmark_tuples - single series" begin
        dataset = create_synthetic_dataset(n_obs=100, n_series=1, seed=42)
        tuples = to_benchmark_tuples(dataset)

        @test length(tuples) == 1
        @test tuples[1][1] == "synthetic_ar1"
    end

    # =========================================================================
    # Integration with Compare Module
    # =========================================================================

    @testset "integration with run_comparison" begin
        dataset = create_synthetic_dataset(n_obs=100, seed=42)
        name, train, test = to_benchmark_tuple(dataset)

        # Should work with run_comparison (must pass dataset_name explicitly)
        adapters = [NaiveAdapter()]
        result = run_comparison(
            train, test;
            adapters=adapters,
            dataset_name=name,  # Pass name from to_benchmark_tuple
            include_dm_test=false
        )

        @test result.dataset_name == name
        @test result.dataset_name == "synthetic_ar1"
        @test length(result.models) == 1
        @test haskey(result.models[1].metrics, "mae")
    end

    @testset "integration with run_benchmark_suite" begin
        datasets = create_bundled_test_datasets(seeds=[42, 123])
        tuples = [to_benchmark_tuple(ds) for ds in datasets]

        adapters = [NaiveAdapter(), SeasonalNaiveAdapter(4)]

        report = run_benchmark_suite(
            tuples;
            adapters=adapters,
            include_dm_test=false
        )

        @test length(report.results) == 2
        @test report.summary["n_datasets"] == 2
    end

end  # @testset "Benchmarks.synthetic"
