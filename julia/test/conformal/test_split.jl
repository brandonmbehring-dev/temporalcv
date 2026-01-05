@testset "SplitConformalPredictor" begin
    @testset "Construction" begin
        cp = SplitConformalPredictor(alpha=0.1)
        @test cp.alpha == 0.1
        @test cp.calibrated == false

        # Default alpha
        cp2 = SplitConformalPredictor()
        @test cp2.alpha == 0.1

        # Invalid alpha
        @test_throws AssertionError SplitConformalPredictor(alpha=0.0)
        @test_throws AssertionError SplitConformalPredictor(alpha=1.0)
        @test_throws AssertionError SplitConformalPredictor(alpha=-0.1)
    end

    @testset "Calibration" begin
        cp = SplitConformalPredictor(alpha=0.1)

        # Simple calibration
        predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
        actuals = [1.1, 2.2, 2.8, 4.1, 5.2]

        calibrate!(cp, predictions, actuals)

        @test cp.calibrated == true
        @test length(cp.scores) == 5
        @test cp.quantile > 0

        # Scores are absolute residuals
        expected_scores = abs.(predictions .- actuals)
        @test cp.scores ≈ expected_scores
    end

    @testset "Quantile formula" begin
        # Test the finite-sample quantile formula: ceil((n+1)(1-α))/n
        # For n=10, α=0.1: ceil(11*0.9) = ceil(9.9) = 10 → take 10th sorted score
        rng = Random.MersenneTwister(42)
        predictions = randn(rng, 10)
        actuals = predictions .+ 0.1 .* randn(rng, 10)

        cp = SplitConformalPredictor(alpha=0.1)
        calibrate!(cp, predictions, actuals)

        scores = sort(abs.(predictions .- actuals))
        expected_idx = ceil(Int, 11 * 0.9)  # = 10
        expected_idx = min(expected_idx, 10)
        @test cp.quantile == scores[expected_idx]
    end

    @testset "Predict interval" begin
        rng = Random.MersenneTwister(42)
        n_cal = 100
        predictions_cal = randn(rng, n_cal)
        actuals_cal = predictions_cal .+ 0.5 .* randn(rng, n_cal)

        cp = SplitConformalPredictor(alpha=0.1)
        calibrate!(cp, predictions_cal, actuals_cal)

        # Generate intervals
        predictions_test = randn(rng, 50)
        intervals = predict_interval(cp, predictions_test)

        @test length(intervals) == 50
        @test all(intervals.upper .> intervals.lower)
        @test all(width(intervals) .≈ 2 * cp.quantile)

        # Intervals centered on predictions
        @test all((intervals.lower .+ intervals.upper) ./ 2 .≈ predictions_test)
    end

    @testset "Coverage guarantee on calibration data" begin
        # With exchangeable data, coverage should be at least 1-α on calibration set
        rng = Random.MersenneTwister(42)
        n = 200

        for alpha in [0.1, 0.2, 0.05]
            predictions = randn(rng, n)
            actuals = predictions .+ randn(rng, n)

            # Split into calibration and test
            n_cal = n ÷ 2
            cp = SplitConformalPredictor(alpha=alpha)
            calibrate!(cp, predictions[1:n_cal], actuals[1:n_cal])

            # Check coverage on held-out data (should be approximately 1-α)
            intervals = predict_interval(cp, predictions[n_cal+1:end])
            cov = coverage(intervals, actuals[n_cal+1:end])

            # Allow some margin for finite sample
            @test cov >= (1 - alpha) - 0.15
        end
    end

    @testset "Error handling" begin
        cp = SplitConformalPredictor(alpha=0.1)

        # Predict before calibration
        @test_throws AssertionError predict_interval(cp, [1.0, 2.0])

        # Mismatched lengths in calibration
        @test_throws AssertionError calibrate!(cp, [1.0, 2.0], [1.0])
    end
end
