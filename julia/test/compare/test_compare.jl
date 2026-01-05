# =============================================================================
# Compare Module Tests
# =============================================================================

using Test
using Statistics

# Import parent module to get Compare submodule
using TemporalValidation
using TemporalValidation.Compare

@testset "Compare Module" begin

    # =========================================================================
    # ModelResult Tests
    # =========================================================================

    @testset "ModelResult" begin
        @testset "basic construction" begin
            result = ModelResult(
                model_name = "TestModel",
                package = "TestPackage",
                metrics = Dict("mae" => 0.5, "rmse" => 0.7),
                predictions = [1.0, 2.0, 3.0],
                runtime_seconds = 1.5
            )

            @test result.model_name == "TestModel"
            @test result.package == "TestPackage"
            @test result.metrics["mae"] == 0.5
            @test result.predictions == [1.0, 2.0, 3.0]
            @test result.runtime_seconds == 1.5
            @test isempty(result.model_params)
        end

        @testset "with model params" begin
            result = ModelResult(
                model_name = "Model",
                package = "Pkg",
                metrics = Dict("mae" => 0.1),
                predictions = [1.0],
                runtime_seconds = 0.1,
                model_params = Dict{String, Any}("order" => 1, "seasonal" => true)
            )

            @test result.model_params["order"] == 1
            @test result.model_params["seasonal"] == true
        end

        @testset "validation errors" begin
            # Empty model name
            @test_throws ErrorException ModelResult(
                model_name = "",
                package = "Pkg",
                metrics = Dict("mae" => 0.1),
                predictions = [1.0],
                runtime_seconds = 0.1
            )

            # Empty package
            @test_throws ErrorException ModelResult(
                model_name = "Model",
                package = "",
                metrics = Dict("mae" => 0.1),
                predictions = [1.0],
                runtime_seconds = 0.1
            )

            # Negative runtime
            @test_throws ErrorException ModelResult(
                model_name = "Model",
                package = "Pkg",
                metrics = Dict("mae" => 0.1),
                predictions = [1.0],
                runtime_seconds = -1.0
            )
        end

        @testset "get_metric" begin
            result = ModelResult(
                model_name = "Model",
                package = "Pkg",
                metrics = Dict("mae" => 0.5, "RMSE" => 0.7, "MaPe" => 10.0),
                predictions = [1.0],
                runtime_seconds = 0.1
            )

            # Exact match
            @test get_metric(result, "mae") == 0.5
            @test get_metric(result, "RMSE") == 0.7

            # Case-insensitive match
            @test get_metric(result, "MAE") == 0.5
            @test get_metric(result, "rmse") == 0.7
            @test get_metric(result, "mape") == 10.0

            # Not found
            @test_throws ErrorException get_metric(result, "unknown")
        end

        @testset "to_dict" begin
            result = ModelResult(
                model_name = "Model",
                package = "Pkg",
                metrics = Dict("mae" => 0.5),
                predictions = [1.0, 2.0],
                runtime_seconds = 1.0,
                model_params = Dict{String, Any}("p" => 1)
            )

            d = to_dict(result)

            @test d["model_name"] == "Model"
            @test d["package"] == "Pkg"
            @test d["metrics"]["mae"] == 0.5
            @test d["runtime_seconds"] == 1.0
            @test d["model_params"]["p"] == 1
        end
    end

    # =========================================================================
    # ComparisonResult Tests
    # =========================================================================

    @testset "ComparisonResult" begin
        @testset "basic construction" begin
            m1 = ModelResult(
                model_name = "Model1",
                package = "Pkg",
                metrics = Dict("mae" => 0.5, "rmse" => 0.7),
                predictions = [1.0],
                runtime_seconds = 1.0
            )
            m2 = ModelResult(
                model_name = "Model2",
                package = "Pkg",
                metrics = Dict("mae" => 0.3, "rmse" => 0.5),
                predictions = [1.1],
                runtime_seconds = 1.5
            )

            result = ComparisonResult(
                dataset_name = "test_data",
                models = [m1, m2],
                primary_metric = "mae"
            )

            @test result.dataset_name == "test_data"
            @test length(result.models) == 2
            @test result.primary_metric == "mae"
            @test result.best_model == "Model2"  # Lower MAE
        end

        @testset "best model by rmse" begin
            m1 = ModelResult(
                model_name = "A",
                package = "Pkg",
                metrics = Dict("mae" => 0.3, "rmse" => 0.8),
                predictions = [1.0],
                runtime_seconds = 1.0
            )
            m2 = ModelResult(
                model_name = "B",
                package = "Pkg",
                metrics = Dict("mae" => 0.5, "rmse" => 0.4),
                predictions = [1.0],
                runtime_seconds = 1.0
            )

            result = ComparisonResult(
                dataset_name = "data",
                models = [m1, m2],
                primary_metric = "rmse"
            )

            @test result.best_model == "B"  # Lower RMSE
        end

        @testset "validation errors" begin
            # Empty models
            @test_throws ErrorException ComparisonResult(
                dataset_name = "data",
                models = ModelResult[],
                primary_metric = "mae"
            )

            # Empty primary metric
            m1 = ModelResult(
                model_name = "M",
                package = "P",
                metrics = Dict("mae" => 0.1),
                predictions = [1.0],
                runtime_seconds = 0.1
            )
            @test_throws ErrorException ComparisonResult(
                dataset_name = "data",
                models = [m1],
                primary_metric = ""
            )

            # No model has metric
            @test_throws ErrorException ComparisonResult(
                dataset_name = "data",
                models = [m1],
                primary_metric = "unknown_metric"
            )
        end

        @testset "get_ranking" begin
            m1 = ModelResult(model_name="A", package="P", metrics=Dict("mae"=>0.5, "rmse"=>0.3),
                            predictions=[1.0], runtime_seconds=0.1)
            m2 = ModelResult(model_name="B", package="P", metrics=Dict("mae"=>0.3, "rmse"=>0.6),
                            predictions=[1.0], runtime_seconds=0.1)
            m3 = ModelResult(model_name="C", package="P", metrics=Dict("mae"=>0.4, "rmse"=>0.4),
                            predictions=[1.0], runtime_seconds=0.1)

            result = ComparisonResult(
                dataset_name = "data",
                models = [m1, m2, m3],
                primary_metric = "mae"
            )

            # Default ranking (by mae)
            ranking = get_ranking(result)
            @test length(ranking) == 3
            @test ranking[1] == ("B", 0.3)
            @test ranking[2] == ("C", 0.4)
            @test ranking[3] == ("A", 0.5)

            # Rank by rmse
            ranking_rmse = get_ranking(result; metric="rmse")
            @test ranking_rmse[1] == ("A", 0.3)
            @test ranking_rmse[2] == ("C", 0.4)
            @test ranking_rmse[3] == ("B", 0.6)
        end

        @testset "to_dict" begin
            m1 = ModelResult(model_name="M", package="P", metrics=Dict("mae"=>0.5),
                            predictions=[1.0], runtime_seconds=0.1)

            result = ComparisonResult(
                dataset_name = "data",
                models = [m1],
                primary_metric = "mae",
                statistical_tests = Dict{String, Any}("dm" => "test")
            )

            d = to_dict(result)
            @test d["dataset_name"] == "data"
            @test d["best_model"] == "M"
            @test length(d["models"]) == 1
            @test d["statistical_tests"]["dm"] == "test"
        end
    end

    # =========================================================================
    # ComparisonReport Tests
    # =========================================================================

    @testset "ComparisonReport" begin
        @testset "basic construction and summary" begin
            m1 = ModelResult(model_name="A", package="P", metrics=Dict("mae"=>0.5),
                            predictions=[1.0], runtime_seconds=0.1)
            m2 = ModelResult(model_name="B", package="P", metrics=Dict("mae"=>0.3),
                            predictions=[1.0], runtime_seconds=0.1)

            r1 = ComparisonResult(dataset_name="d1", models=[m1, m2], primary_metric="mae")
            r2 = ComparisonResult(dataset_name="d2", models=[m1, m2], primary_metric="mae")

            report = ComparisonReport(results = [r1, r2])

            @test length(report.results) == 2
            @test report.summary["n_datasets"] == 2
            @test report.summary["wins_by_model"]["B"] == 2  # B wins both
        end

        @testset "mixed wins" begin
            m1 = ModelResult(model_name="A", package="P", metrics=Dict("mae"=>0.2),
                            predictions=[1.0], runtime_seconds=0.1)
            m2 = ModelResult(model_name="B", package="P", metrics=Dict("mae"=>0.5),
                            predictions=[1.0], runtime_seconds=0.1)

            # A wins d1, B wins d2
            r1 = ComparisonResult(dataset_name="d1", models=[m1, m2], primary_metric="mae")

            m1_d2 = ModelResult(model_name="A", package="P", metrics=Dict("mae"=>0.6),
                               predictions=[1.0], runtime_seconds=0.1)
            m2_d2 = ModelResult(model_name="B", package="P", metrics=Dict("mae"=>0.3),
                               predictions=[1.0], runtime_seconds=0.1)
            r2 = ComparisonResult(dataset_name="d2", models=[m1_d2, m2_d2], primary_metric="mae")

            report = ComparisonReport(results = [r1, r2])

            @test report.summary["wins_by_model"]["A"] == 1
            @test report.summary["wins_by_model"]["B"] == 1
        end

        @testset "to_markdown" begin
            m1 = ModelResult(model_name="NaiveModel", package="P", metrics=Dict("mae"=>0.5),
                            predictions=[1.0], runtime_seconds=0.1)
            r1 = ComparisonResult(dataset_name="TestData", models=[m1], primary_metric="mae")
            report = ComparisonReport(results = [r1])

            md = to_markdown(report)

            @test occursin("# Model Comparison Report", md)
            @test occursin("## Summary", md)
            @test occursin("Datasets evaluated: 1", md)
            @test occursin("### Model Wins", md)
            @test occursin("NaiveModel", md)
            @test occursin("TestData", md)
            @test occursin("0.5", md)
        end
    end

    # =========================================================================
    # ForecastAdapter Tests
    # =========================================================================

    @testset "ForecastAdapter" begin
        @testset "NaiveAdapter" begin
            adapter = NaiveAdapter()

            @test model_name(adapter) == "Naive"
            @test package_name(adapter) == "TemporalValidation"
            @test isempty(get_params(adapter))

            # Basic prediction
            train = [1.0, 2.0, 3.0, 4.0, 5.0]
            preds = fit_predict(adapter, train, 3, 1)

            @test length(preds) == 3
            @test all(p == 5.0 for p in preds)

            # Single element
            preds_single = fit_predict(adapter, [10.0], 2, 1)
            @test all(p == 10.0 for p in preds_single)

            # Empty train should error
            @test_throws ErrorException fit_predict(adapter, Float64[], 2, 1)

            # Zero test_size should error
            @test_throws ErrorException fit_predict(adapter, train, 0, 1)
        end

        @testset "SeasonalNaiveAdapter" begin
            @testset "construction" begin
                adapter = SeasonalNaiveAdapter()
                @test adapter.season_length == 52

                adapter4 = SeasonalNaiveAdapter(4)
                @test adapter4.season_length == 4

                @test model_name(adapter4) == "SeasonalNaive_4"
                @test package_name(adapter4) == "TemporalValidation"
                @test get_params(adapter4)["season_length"] == 4

                # Invalid season length
                @test_throws ErrorException SeasonalNaiveAdapter(0)
                @test_throws ErrorException SeasonalNaiveAdapter(-1)
            end

            @testset "prediction with full season" begin
                # 2 full seasons of length 4
                train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                adapter = SeasonalNaiveAdapter(4)

                preds = fit_predict(adapter, train, 4, 1)

                # Should predict using values from last season
                @test preds[1] == 5.0  # Position 0 in next season -> index 5
                @test preds[2] == 6.0
                @test preds[3] == 7.0
                @test preds[4] == 8.0
            end

            @testset "prediction with partial season" begin
                # Less than 1 full season
                train = [1.0, 2.0, 3.0]
                adapter = SeasonalNaiveAdapter(4)

                preds = fit_predict(adapter, train, 2, 1)

                # Should fallback gracefully
                @test length(preds) == 2
            end

            @testset "longer forecast than season" begin
                train = collect(1.0:8.0)
                adapter = SeasonalNaiveAdapter(4)

                # Forecast 6 periods (wraps around)
                preds = fit_predict(adapter, train, 6, 1)
                @test length(preds) == 6
            end
        end
    end

    # =========================================================================
    # compute_comparison_metrics Tests
    # =========================================================================

    @testset "compute_comparison_metrics" begin
        @testset "perfect predictions" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            predictions = [1.0, 2.0, 3.0, 4.0, 5.0]

            metrics = compute_comparison_metrics(predictions, actuals)

            @test metrics["mae"] == 0.0
            @test metrics["rmse"] == 0.0
            @test metrics["mape"] == 0.0
            @test metrics["direction_accuracy"] == 1.0
        end

        @testset "constant error" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            predictions = [1.5, 2.5, 3.5, 4.5, 5.5]  # +0.5 error

            metrics = compute_comparison_metrics(predictions, actuals)

            @test metrics["mae"] ≈ 0.5
            @test metrics["rmse"] ≈ 0.5
            @test metrics["direction_accuracy"] == 1.0  # Same direction
        end

        @testset "opposite direction" begin
            actuals = [1.0, 3.0, 2.0, 4.0]  # up, down, up
            predictions = [1.0, 0.5, 2.5, 1.0]  # down, up, down

            metrics = compute_comparison_metrics(predictions, actuals)

            @test metrics["direction_accuracy"] == 0.0  # All wrong
        end

        @testset "zeros in actuals" begin
            actuals = [0.0, 1.0, 2.0]
            predictions = [0.5, 1.5, 2.5]

            metrics = compute_comparison_metrics(predictions, actuals)

            @test isfinite(metrics["mae"])
            # MAPE excludes zeros
            @test isfinite(metrics["mape"]) || isnan(metrics["mape"])
        end

        @testset "single element" begin
            metrics = compute_comparison_metrics([1.0], [1.5])

            @test metrics["mae"] == 0.5
            @test metrics["rmse"] == 0.5
            @test isnan(metrics["direction_accuracy"])  # Can't compute with 1 point
        end

        @testset "validation" begin
            # Length mismatch
            @test_throws ErrorException compute_comparison_metrics([1.0, 2.0], [1.0])

            # Empty arrays
            @test_throws ErrorException compute_comparison_metrics(Float64[], Float64[])
        end
    end

    # =========================================================================
    # run_comparison Tests
    # =========================================================================

    @testset "run_comparison" begin
        @testset "single adapter" begin
            train = collect(1.0:100.0)
            test = collect(101.0:110.0)

            result = run_comparison(train, test;
                adapters = [NaiveAdapter()],
                include_dm_test = false
            )

            @test result.dataset_name == "dataset"
            @test length(result.models) == 1
            @test result.best_model == "Naive"
            @test result.models[1].metrics["mae"] > 0
        end

        @testset "multiple adapters" begin
            train = collect(1.0:100.0)
            test = collect(101.0:110.0)

            adapters = [NaiveAdapter(), SeasonalNaiveAdapter(10)]

            result = run_comparison(train, test;
                adapters = adapters,
                include_dm_test = false
            )

            @test length(result.models) == 2
        end

        @testset "custom dataset name" begin
            train = [1.0, 2.0, 3.0]
            test = [4.0]

            result = run_comparison(train, test;
                adapters = [NaiveAdapter()],
                dataset_name = "my_dataset",
                include_dm_test = false
            )

            @test result.dataset_name == "my_dataset"
        end

        @testset "validation errors" begin
            # Empty adapters
            @test_throws ErrorException run_comparison([1.0], [2.0];
                adapters = ForecastAdapter[],
                include_dm_test = false
            )

            # Empty train
            @test_throws ErrorException run_comparison(Float64[], [1.0];
                adapters = [NaiveAdapter()],
                include_dm_test = false
            )

            # Empty test
            @test_throws ErrorException run_comparison([1.0], Float64[];
                adapters = [NaiveAdapter()],
                include_dm_test = false
            )
        end
    end

    # =========================================================================
    # run_benchmark_suite Tests
    # =========================================================================

    @testset "run_benchmark_suite" begin
        @testset "basic suite" begin
            datasets = [
                ("series_1", collect(1.0:50.0), collect(51.0:55.0)),
                ("series_2", collect(1.0:50.0), collect(51.0:55.0))
            ]

            report = run_benchmark_suite(datasets;
                adapters = [NaiveAdapter()],
                include_dm_test = false
            )

            @test length(report.results) == 2
            @test report.summary["n_datasets"] == 2
        end

        @testset "validation errors" begin
            # Empty datasets
            @test_throws ErrorException run_benchmark_suite(
                Tuple{String, Vector{Float64}, Vector{Float64}}[];
                adapters = [NaiveAdapter()]
            )

            # Empty adapters
            @test_throws ErrorException run_benchmark_suite(
                [("d", [1.0], [2.0])];
                adapters = ForecastAdapter[]
            )
        end
    end

    # =========================================================================
    # compare_to_baseline Tests
    # =========================================================================

    @testset "compare_to_baseline" begin
        @testset "model beats baseline" begin
            # Create data where seasonal naive might differ from naive
            train = collect(1.0:50.0)
            test = collect(51.0:55.0)

            # Use SeasonalNaiveAdapter as model (different from default NaiveAdapter baseline)
            result = compare_to_baseline(train, test, SeasonalNaiveAdapter(10))

            @test haskey(result, "model_name")
            @test haskey(result, "baseline_name")
            @test haskey(result, "improvement_pct")
            @test haskey(result, "model_is_better")
            @test result["baseline_name"] == "Naive"
            @test result["model_name"] == "SeasonalNaive_10"
        end

        @testset "custom baseline" begin
            train = collect(1.0:100.0)
            test = collect(101.0:105.0)

            result = compare_to_baseline(
                train, test,
                NaiveAdapter();
                baseline_adapter = SeasonalNaiveAdapter(10)
            )

            @test result["baseline_name"] == "SeasonalNaive_10"
        end
    end

    # =========================================================================
    # Edge Cases
    # =========================================================================

    @testset "edge cases" begin
        @testset "very short series" begin
            train = [1.0, 2.0]
            test = [3.0]

            result = run_comparison(train, test;
                adapters = [NaiveAdapter()],
                include_dm_test = false
            )

            @test length(result.models) == 1
        end

        @testset "all same values" begin
            train = fill(5.0, 50)
            test = fill(5.0, 10)

            result = run_comparison(train, test;
                adapters = [NaiveAdapter()],
                include_dm_test = false
            )

            @test result.models[1].metrics["mae"] == 0.0
        end

        @testset "with NaN in predictions" begin
            # This tests robustness of metrics computation
            # NaiveAdapter won't produce NaN, but metrics should handle it
            actuals = [1.0, 2.0, 3.0]
            predictions = [1.0, NaN, 3.0]

            # compute_comparison_metrics should handle NaN
            metrics = compute_comparison_metrics(predictions, actuals)

            # MAE will be NaN due to NaN in errors
            @test isnan(metrics["mae"])
        end

        @testset "with Inf values" begin
            actuals = [1.0, 2.0, 3.0]
            predictions = [1.0, Inf, 3.0]

            metrics = compute_comparison_metrics(predictions, actuals)

            # Should produce Inf or NaN
            @test !isfinite(metrics["mae"])
        end
    end

end
