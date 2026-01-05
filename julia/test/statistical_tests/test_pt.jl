@testset "PT Test" begin
    @testset "Construction - 2 class" begin
        rng = Random.MersenneTwister(42)
        n = 100

        # Generate data with some directional skill
        actual = randn(rng, n)
        predicted = 0.5 .* actual .+ 0.5 .* randn(rng, n)  # Positive correlation

        result = pt_test(actual, predicted)

        @test isa(result, PTTestResult)
        @test result.n_classes == 2
        @test 0 <= result.accuracy <= 1
        @test 0 <= result.expected <= 1
        @test 0 <= result.pvalue <= 1
        @test isfinite(result.statistic)
    end

    @testset "Construction - 3 class" begin
        rng = Random.MersenneTwister(42)
        n = 100

        actual = randn(rng, n)
        predicted = 0.5 .* actual .+ 0.5 .* randn(rng, n)

        result = pt_test(actual, predicted, move_threshold=0.5)

        @test result.n_classes == 3
        @test 0 <= result.accuracy <= 1
        @test 0 <= result.expected <= 1
    end

    @testset "Perfect prediction" begin
        rng = Random.MersenneTwister(42)
        n = 100

        actual = randn(rng, n)
        predicted = actual  # Perfect prediction

        result = pt_test(actual, predicted)

        # Accuracy should be 100%
        @test result.accuracy ≈ 1.0
        # Should be highly significant
        @test result.pvalue < 0.01
    end

    @testset "Random prediction" begin
        rng = Random.MersenneTwister(42)
        n = 200

        actual = randn(rng, n)
        predicted = randn(rng, n)  # Independent random prediction

        result = pt_test(actual, predicted)

        # Accuracy should be close to expected (around 50%)
        @test abs(result.accuracy - result.expected) < 0.15
        # Should not be significant
        @test result.pvalue > 0.05
    end

    @testset "Opposite prediction" begin
        rng = Random.MersenneTwister(42)
        n = 100

        actual = randn(rng, n)
        predicted = -actual  # Opposite prediction

        result = pt_test(actual, predicted)

        # Accuracy should be 0%
        @test result.accuracy ≈ 0.0
        # P-value should be high (worse than random)
        @test result.pvalue > 0.5
    end

    @testset "skill property" begin
        result = PTTestResult(1.5, 0.05, 0.65, 0.50, 100, 2)
        @test skill(result) ≈ 0.15
    end

    @testset "significant_at_05" begin
        result_sig = PTTestResult(2.0, 0.02, 0.70, 0.50, 100, 2)
        result_not_sig = PTTestResult(1.0, 0.15, 0.55, 0.50, 100, 2)

        @test significant_at_05(result_sig) == true
        @test significant_at_05(result_not_sig) == false
    end

    @testset "Input validation" begin
        rng = Random.MersenneTwister(42)
        actual = randn(rng, 50)

        # Different lengths
        @test_throws AssertionError pt_test(actual, randn(rng, 40))

        # Too few samples
        @test_throws AssertionError pt_test(randn(rng, 20), randn(rng, 20))
    end

    @testset "NaN handling" begin
        rng = Random.MersenneTwister(42)
        actual = randn(rng, 50)
        predicted = randn(rng, 50)
        actual[10] = NaN

        @test_throws AssertionError pt_test(actual, predicted)
    end

    @testset "3-class with threshold" begin
        rng = Random.MersenneTwister(42)
        n = 150

        # Create data with clear up/down/flat regions
        actual = randn(rng, n)
        # Predictions that tend to get direction right
        predicted = 0.7 .* actual .+ 0.3 .* randn(rng, n)

        threshold = 0.3
        result = pt_test(actual, predicted, move_threshold=threshold)

        @test result.n_classes == 3
        @test result.accuracy > result.expected  # Should have positive skill
    end

    @testset "Marginal probability calculation" begin
        rng = Random.MersenneTwister(42)
        n = 100

        # Create imbalanced data
        actual = randn(rng, n) .+ 0.5  # More positive values
        predicted = randn(rng, n)

        result = pt_test(actual, predicted)

        # Expected accuracy should account for marginal imbalance
        @test result.expected != 0.5  # Not balanced 50-50
    end

    @testset "show() method" begin
        result = PTTestResult(1.5, 0.05, 0.65, 0.50, 100, 2)
        str = string(result)

        @test occursin("PT", str)
        @test occursin("%", str)
        @test occursin("p=", str)
    end

    @testset "Edge case: all zeros in actual" begin
        rng = Random.MersenneTwister(42)
        n = 50

        actual = zeros(n)
        predicted = randn(rng, n)

        # 2-class mode should handle all zeros gracefully
        result = pt_test(actual, predicted)
        @test result.pvalue == 1.0  # Cannot reject null
    end

    @testset "Reproducibility" begin
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(42)
        n = 100

        actual = randn(rng1, n)
        predicted1 = randn(rng1, n)

        actual2 = randn(rng2, n)
        predicted2 = randn(rng2, n)

        result1 = pt_test(actual, predicted1)
        result2 = pt_test(actual2, predicted2)

        @test result1.statistic ≈ result2.statistic
        @test result1.pvalue ≈ result2.pvalue
    end
end
