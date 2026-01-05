@testset "AdaptiveConformalPredictor" begin
    @testset "Construction" begin
        acp = AdaptiveConformalPredictor(alpha=0.1, gamma=0.1)
        @test acp.alpha == 0.1
        @test acp.gamma == 0.1
        @test acp.initialized == false

        # Default values
        acp2 = AdaptiveConformalPredictor()
        @test acp2.alpha == 0.1
        @test acp2.gamma == 0.1

        # Invalid parameters
        @test_throws AssertionError AdaptiveConformalPredictor(alpha=0.0)
        @test_throws AssertionError AdaptiveConformalPredictor(alpha=1.0)
        @test_throws AssertionError AdaptiveConformalPredictor(gamma=0.0)
        @test_throws AssertionError AdaptiveConformalPredictor(gamma=-0.1)
    end

    @testset "Initialization" begin
        rng = Random.MersenneTwister(42)
        predictions = randn(rng, 50)
        actuals = predictions .+ 0.3 .* randn(rng, 50)

        acp = AdaptiveConformalPredictor(alpha=0.1)
        initialize!(acp, predictions, actuals)

        @test acp.initialized == true
        @test acp.quantile > 0
    end

    @testset "Predict interval" begin
        rng = Random.MersenneTwister(42)
        predictions = randn(rng, 50)
        actuals = predictions .+ 0.3 .* randn(rng, 50)

        acp = AdaptiveConformalPredictor(alpha=0.1)
        initialize!(acp, predictions, actuals)

        # Single prediction
        pred = 1.5
        lower, upper = predict_interval(acp, pred)

        @test lower < pred < upper
        @test upper - lower ≈ 2 * acp.quantile  # Use ≈ for floating point
        @test (lower + upper) / 2 ≈ pred
    end

    @testset "Update mechanism" begin
        rng = Random.MersenneTwister(42)
        predictions = randn(rng, 50)
        actuals = predictions .+ 0.3 .* randn(rng, 50)

        acp = AdaptiveConformalPredictor(alpha=0.1, gamma=0.2)
        initialize!(acp, predictions, actuals)
        initial_quantile = acp.quantile

        # When covered: quantile should decrease
        interval_covered = (0.0, 2.0)
        update!(acp, 1.0, interval_covered)  # actual=1.0 is within [0, 2]
        @test acp.quantile < initial_quantile
        @test acp.quantile ≈ initial_quantile - 0.2 * 0.1  # γ * α

        # When not covered: quantile should increase
        quantile_after_covered = acp.quantile
        interval_not_covered = (0.0, 0.5)
        update!(acp, 1.0, interval_not_covered)  # actual=1.0 is outside [0, 0.5]
        @test acp.quantile > quantile_after_covered
        @test acp.quantile ≈ quantile_after_covered + 0.2 * 0.9  # γ * (1-α)
    end

    @testset "Quantile stays non-negative" begin
        acp = AdaptiveConformalPredictor(alpha=0.1, gamma=1.0)

        # Initialize with small quantile
        acp.quantile = 0.05
        acp.initialized = true

        # Many covered updates should not make quantile negative
        for _ in 1:100
            update!(acp, 0.5, (0.0, 1.0))  # Always covered
        end

        @test acp.quantile >= 0.0
    end

    @testset "Convergence under stationarity" begin
        # Under stationary data, adaptive conformal should converge to target coverage
        rng = Random.MersenneTwister(42)
        alpha = 0.2
        gamma = 0.05

        # Generate stationary test data
        n_init = 50
        n_test = 500

        predictions_init = randn(rng, n_init)
        actuals_init = predictions_init .+ randn(rng, n_init)

        acp = AdaptiveConformalPredictor(alpha=alpha, gamma=gamma)
        initialize!(acp, predictions_init, actuals_init)

        # Online prediction and update
        coverage_count = 0
        for _ in 1:n_test
            pred = randn(rng)
            actual = pred + randn(rng)

            interval = predict_interval(acp, pred)
            if interval[1] <= actual <= interval[2]
                coverage_count += 1
            end
            update!(acp, actual, interval)
        end

        empirical_coverage = coverage_count / n_test
        # Should be approximately 1-α with some margin
        @test abs(empirical_coverage - (1 - alpha)) < 0.1
    end

    @testset "Error handling" begin
        acp = AdaptiveConformalPredictor(alpha=0.1)

        # Predict before initialization
        @test_throws AssertionError predict_interval(acp, 1.0)

        # Mismatched lengths in initialization
        @test_throws AssertionError initialize!(acp, [1.0, 2.0], [1.0])
    end
end
