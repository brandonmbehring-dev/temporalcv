@testset "CV Types" begin
    @testset "SplitInfo" begin
        @testset "Construction" begin
            info = SplitInfo(0, 1, 100, 103, 112)
            @test info.split_idx == 0
            @test info.train_start == 1
            @test info.train_end == 100
            @test info.test_start == 103
            @test info.test_end == 112
        end

        @testset "Properties" begin
            info = SplitInfo(0, 1, 100, 103, 112)
            @test train_size(info) == 100
            @test test_size(info) == 10
            @test gap(info) == 2  # 103 - 100 - 1 = 2
        end

        @testset "Temporal leakage detection" begin
            # train_end >= test_start should fail
            @test_throws ArgumentError SplitInfo(0, 1, 100, 100, 110)  # Equal
            @test_throws ArgumentError SplitInfo(0, 1, 100, 99, 110)   # Overlap
            @test_throws ArgumentError SplitInfo(0, 1, 100, 50, 60)    # Test before train_end
        end

        @testset "Gap = 0 is valid" begin
            # train_end + 1 == test_start (gap of 0)
            info = SplitInfo(0, 1, 100, 101, 110)
            @test gap(info) == 0
        end
    end

    @testset "SplitResult" begin
        @testset "Construction" begin
            predictions = [1.0, 1.1, 1.2, 1.3, 1.4]
            actuals = [1.05, 1.08, 1.25, 1.28, 1.45]

            result = SplitResult(0, 1, 100, 103, 107, predictions, actuals)
            @test result.split_idx == 0
            @test result.train_start == 1
            @test result.train_end == 100
            @test result.test_start == 103
            @test result.test_end == 107
            @test result.predictions == predictions
            @test result.actuals == actuals
        end

        @testset "Properties" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.5, 2.0, 2.5]

            result = SplitResult(0, 1, 50, 53, 55, predictions, actuals)

            @test train_size(result) == 50
            @test test_size(result) == 3
            @test gap(result) == 2

            errs = errors(result)
            @test errs ≈ [-0.5, 0.0, 0.5]

            @test absolute_errors(result) ≈ [0.5, 0.0, 0.5]
            @test mae(result) ≈ (0.5 + 0.0 + 0.5) / 3
            @test rmse(result) ≈ sqrt((0.25 + 0.0 + 0.25) / 3)
            @test bias(result) ≈ 0.0
        end

        @testset "to_split_info conversion" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.0, 2.0, 3.0]
            result = SplitResult(0, 1, 50, 53, 55, predictions, actuals)

            info = to_split_info(result)
            @test info.split_idx == 0
            @test info.train_start == 1
            @test info.train_end == 50
            @test info.test_start == 53
            @test info.test_end == 55
        end

        @testset "Temporal leakage detection" begin
            predictions = [1.0, 2.0]
            actuals = [1.0, 2.0]
            @test_throws ArgumentError SplitResult(0, 1, 100, 100, 101, predictions, actuals)
        end

        @testset "Length mismatch detection" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.0, 2.0]
            @test_throws ArgumentError SplitResult(0, 1, 50, 53, 55, predictions, actuals)
        end
    end

    @testset "WalkForwardResults" begin
        @testset "Construction" begin
            split1 = SplitResult(0, 1, 50, 53, 57, [1.0, 1.1, 1.2, 1.3, 1.4], [1.1, 1.2, 1.1, 1.4, 1.3])
            split2 = SplitResult(1, 1, 60, 63, 67, [2.0, 2.1, 2.2, 2.3, 2.4], [2.1, 2.0, 2.3, 2.2, 2.5])

            results = WalkForwardResults([split1, split2])
            @test n_splits(results) == 2
            @test length(results.splits) == 2
        end

        @testset "Empty splits rejected" begin
            @test_throws ArgumentError WalkForwardResults(SplitResult[])
        end

        @testset "Aggregate properties" begin
            # Create simple splits for testing aggregation
            split1 = SplitResult(0, 1, 50, 53, 54, [1.0, 2.0], [1.5, 2.0])  # errors: -0.5, 0.0
            split2 = SplitResult(1, 1, 60, 63, 64, [3.0, 4.0], [2.5, 4.5])  # errors: 0.5, -0.5

            results = WalkForwardResults([split1, split2])

            all_preds = predictions(results)
            @test all_preds ≈ [1.0, 2.0, 3.0, 4.0]

            all_acts = actuals(results)
            @test all_acts ≈ [1.5, 2.0, 2.5, 4.5]

            all_errs = errors(results)
            @test all_errs ≈ [-0.5, 0.0, 0.5, -0.5]

            @test absolute_errors(results) ≈ [0.5, 0.0, 0.5, 0.5]
            @test mae(results) ≈ (0.5 + 0.0 + 0.5 + 0.5) / 4
            @test total_samples(results) == 4
        end
    end
end
