@testset "DM Test" begin
    @testset "Construction" begin
        rng = Random.MersenneTwister(42)
        n = 100
        errors_1 = randn(rng, n)
        errors_2 = randn(rng, n) .+ 0.5  # Worse model (higher errors)

        result = dm_test(errors_1, errors_2)

        @test isa(result, DMTestResult)
        @test result.n == n
        @test result.h == 1
        @test result.loss == :squared
        @test result.alternative == :two_sided
        @test result.harvey_adjusted == true
        @test isfinite(result.statistic)
        @test 0 <= result.pvalue <= 1
    end

    @testset "Significant difference" begin
        rng = Random.MersenneTwister(42)
        n = 100

        # Model with small errors
        errors_1 = 0.1 .* randn(rng, n)
        # Much worse model
        errors_2 = randn(rng, n) .+ 1.0

        result = dm_test(errors_1, errors_2, alternative=:less)

        # Model 1 should be significantly better
        @test result.mean_loss_diff < 0  # Model 1 has lower loss
        @test significant_at_05(result)
    end

    @testset "No significant difference" begin
        rng = Random.MersenneTwister(42)
        n = 100

        # Two similar models
        errors_1 = randn(rng, n)
        errors_2 = randn(rng, n)

        result = dm_test(errors_1, errors_2)

        # Should not be significant (p-value should be reasonably high)
        # Just test that the test runs and produces sensible output
        @test isfinite(result.statistic)
        @test 0 <= result.pvalue <= 1
    end

    @testset "Loss functions" begin
        rng = Random.MersenneTwister(42)
        n = 50
        errors_1 = randn(rng, n)
        errors_2 = randn(rng, n) .+ 0.3

        result_sq = dm_test(errors_1, errors_2, loss=:squared)
        result_abs = dm_test(errors_1, errors_2, loss=:absolute)

        @test result_sq.loss == :squared
        @test result_abs.loss == :absolute
        # Results may differ between loss functions
        @test isfinite(result_sq.statistic)
        @test isfinite(result_abs.statistic)
    end

    @testset "Alternatives" begin
        rng = Random.MersenneTwister(42)
        n = 50
        errors_1 = 0.5 .* randn(rng, n)
        errors_2 = randn(rng, n) .+ 0.5

        result_two = dm_test(errors_1, errors_2, alternative=:two_sided)
        result_less = dm_test(errors_1, errors_2, alternative=:less)
        result_greater = dm_test(errors_1, errors_2, alternative=:greater)

        @test result_two.alternative == :two_sided
        @test result_less.alternative == :less
        @test result_greater.alternative == :greater

        # For one-sided tests with clear direction:
        # If model 1 is better, :less should have lower p-value
        if result_two.mean_loss_diff < 0
            @test result_less.pvalue < result_greater.pvalue
        end
    end

    @testset "Multi-step horizon" begin
        rng = Random.MersenneTwister(42)
        n = 100
        errors_1 = randn(rng, n)
        errors_2 = randn(rng, n) .+ 0.3

        result_h1 = dm_test(errors_1, errors_2, h=1)
        result_h2 = dm_test(errors_1, errors_2, h=2)
        result_h4 = dm_test(errors_1, errors_2, h=4)

        @test result_h1.h == 1
        @test result_h2.h == 2
        @test result_h4.h == 4

        # All should produce valid results
        @test isfinite(result_h1.statistic)
        @test isfinite(result_h2.statistic)
        @test isfinite(result_h4.statistic)
    end

    @testset "Harvey correction" begin
        rng = Random.MersenneTwister(42)
        n = 50
        errors_1 = randn(rng, n)
        errors_2 = randn(rng, n) .+ 0.3

        result_with = dm_test(errors_1, errors_2, harvey_correction=true)
        result_without = dm_test(errors_1, errors_2, harvey_correction=false)

        @test result_with.harvey_adjusted == true
        @test result_without.harvey_adjusted == false

        # Harvey adjustment typically reduces the test statistic magnitude
        # (not always, but test that they differ)
        @test result_with.statistic != result_without.statistic
    end

    @testset "Input validation" begin
        rng = Random.MersenneTwister(42)
        errors = randn(rng, 50)

        # Different lengths
        @test_throws AssertionError dm_test(errors, randn(rng, 40))

        # Too few samples
        @test_throws AssertionError dm_test(randn(rng, 20), randn(rng, 20))

        # Invalid horizon
        @test_throws AssertionError dm_test(errors, errors, h=0)

        # Invalid loss
        @test_throws AssertionError dm_test(errors, errors, loss=:invalid)

        # Invalid alternative
        @test_throws AssertionError dm_test(errors, errors, alternative=:invalid)
    end

    @testset "NaN handling" begin
        rng = Random.MersenneTwister(42)
        errors_1 = randn(rng, 50)
        errors_2 = randn(rng, 50)
        errors_1[10] = NaN

        @test_throws AssertionError dm_test(errors_1, errors_2)
    end

    @testset "show() method" begin
        rng = Random.MersenneTwister(42)
        errors_1 = randn(rng, 50)
        errors_2 = randn(rng, 50)

        result = dm_test(errors_1, errors_2)
        str = string(result)

        @test occursin("DM", str)
        @test occursin("p=", str)
    end

    @testset "significant_at methods" begin
        # Create a mock result with known p-values
        result_sig_01 = DMTestResult(3.0, 0.005, 1, 100, :squared, :two_sided, true, -0.1)
        result_sig_05 = DMTestResult(2.0, 0.03, 1, 100, :squared, :two_sided, true, -0.1)
        result_not_sig = DMTestResult(1.0, 0.2, 1, 100, :squared, :two_sided, true, -0.05)

        @test significant_at_01(result_sig_01) == true
        @test significant_at_05(result_sig_01) == true

        @test significant_at_01(result_sig_05) == false
        @test significant_at_05(result_sig_05) == true

        @test significant_at_01(result_not_sig) == false
        @test significant_at_05(result_not_sig) == false
    end
end
