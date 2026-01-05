@testset "Quantile Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_pinball_loss
    # ==========================================================================

    @testset "compute_pinball_loss" begin
        @testset "Perfect predictions" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            preds = copy(actuals)
            @test compute_pinball_loss(actuals, preds, 0.5) == 0.0
            @test compute_pinball_loss(actuals, preds, 0.9) == 0.0
            @test compute_pinball_loss(actuals, preds, 0.1) == 0.0
        end

        @testset "Over-prediction at tau=0.9" begin
            actuals = [1.0, 2.0, 3.0]
            preds = [2.0, 3.0, 4.0]  # All over-predicted by 1
            # For over-prediction (errors < 0): loss = (tau - 1) * error
            # error = actual - pred = -1, loss = (0.9 - 1) * (-1) = 0.1
            @test compute_pinball_loss(actuals, preds, 0.9) ≈ 0.1
        end

        @testset "Under-prediction at tau=0.9" begin
            actuals = [2.0, 3.0, 4.0]
            preds = [1.0, 2.0, 3.0]  # All under-predicted by 1
            # For under-prediction (errors > 0): loss = tau * error
            # error = actual - pred = 1, loss = 0.9 * 1 = 0.9
            @test compute_pinball_loss(actuals, preds, 0.9) ≈ 0.9
        end

        @testset "Asymmetry at different tau" begin
            actuals = [5.0]
            preds = [4.0]  # Under-prediction by 1

            # Higher tau penalizes under-prediction more
            loss_high_tau = compute_pinball_loss(actuals, preds, 0.9)
            loss_low_tau = compute_pinball_loss(actuals, preds, 0.1)
            @test loss_high_tau > loss_low_tau
        end

        @testset "Invalid tau" begin
            @test_throws ErrorException compute_pinball_loss([1.0], [1.0], 0.0)
            @test_throws ErrorException compute_pinball_loss([1.0], [1.0], 1.0)
            @test_throws ErrorException compute_pinball_loss([1.0], [1.0], -0.1)
            @test_throws ErrorException compute_pinball_loss([1.0], [1.0], 1.5)
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_pinball_loss([1.0, 2.0], [1.0], 0.5)
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_pinball_loss(Float64[], Float64[], 0.5)
        end
    end

    # ==========================================================================
    # compute_crps
    # ==========================================================================

    @testset "compute_crps" begin
        @testset "Perfect samples (point mass at actual)" begin
            actuals = [1.0, 2.0, 3.0]
            # Samples concentrated at actuals
            samples = [
                1.0 1.0 1.0 1.0 1.0;
                2.0 2.0 2.0 2.0 2.0;
                3.0 3.0 3.0 3.0 3.0
            ]
            # CRPS should be near 0 for perfect forecast
            @test compute_crps(actuals, samples) ≈ 0.0 atol=1e-10
        end

        @testset "Wider spread increases CRPS" begin
            actuals = [0.0, 0.0, 0.0]

            # Narrow samples
            narrow = [
                -0.1 0.0 0.1 0.0 0.0;
                -0.1 0.0 0.1 0.0 0.0;
                -0.1 0.0 0.1 0.0 0.0
            ]

            # Wide samples
            wide = [
                -2.0 0.0 2.0 1.0 -1.0;
                -2.0 0.0 2.0 1.0 -1.0;
                -2.0 0.0 2.0 1.0 -1.0
            ]

            crps_narrow = compute_crps(actuals, narrow)
            crps_wide = compute_crps(actuals, wide)

            @test crps_wide > crps_narrow
        end

        @testset "Biased forecast increases CRPS" begin
            actuals = [0.0, 0.0, 0.0]

            # Centered samples
            centered = randn(rng, 3, 100)

            # Biased samples (shifted by 5)
            biased = randn(rng, 3, 100) .+ 5.0

            crps_centered = compute_crps(actuals, centered)
            crps_biased = compute_crps(actuals, biased)

            @test crps_biased > crps_centered
        end

        @testset "Dimension mismatch" begin
            @test_throws ErrorException compute_crps([1.0, 2.0], rand(3, 10))
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_crps(Float64[], zeros(0, 10))
        end
    end

    # ==========================================================================
    # compute_interval_score
    # ==========================================================================

    @testset "compute_interval_score" begin
        @testset "All covered - only width penalty" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            lower = [0.5, 1.5, 2.5, 3.5, 4.5]
            upper = [1.5, 2.5, 3.5, 4.5, 5.5]
            # Width = 1.0 for all, no coverage penalty
            # IS = mean(width) = 1.0
            @test compute_interval_score(actuals, lower, upper, 0.1) ≈ 1.0
        end

        @testset "Below lower bound penalty" begin
            actuals = [0.0]  # Below lower
            lower = [1.0]
            upper = [2.0]
            alpha = 0.1
            # Width = 1.0
            # Penalty = (2/0.1) * (1.0 - 0.0) = 20
            # Total = 1.0 + 20 = 21
            @test compute_interval_score(actuals, lower, upper, alpha) ≈ 21.0
        end

        @testset "Above upper bound penalty" begin
            actuals = [3.0]  # Above upper
            lower = [1.0]
            upper = [2.0]
            alpha = 0.1
            # Width = 1.0
            # Penalty = (2/0.1) * (3.0 - 2.0) = 20
            # Total = 1.0 + 20 = 21
            @test compute_interval_score(actuals, lower, upper, alpha) ≈ 21.0
        end

        @testset "Higher alpha = smaller penalty" begin
            actuals = [0.0]
            lower = [1.0]
            upper = [2.0]

            score_10 = compute_interval_score(actuals, lower, upper, 0.1)
            score_20 = compute_interval_score(actuals, lower, upper, 0.2)

            @test score_10 > score_20  # Higher alpha = smaller penalty factor
        end

        @testset "Invalid alpha" begin
            @test_throws ErrorException compute_interval_score([1.0], [0.0], [2.0], 0.0)
            @test_throws ErrorException compute_interval_score([1.0], [0.0], [2.0], 1.0)
        end

        @testset "Lower > upper error" begin
            @test_throws ErrorException compute_interval_score([1.0], [2.0], [1.0], 0.1)
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_interval_score(Float64[], Float64[], Float64[], 0.1)
        end
    end

    # ==========================================================================
    # compute_winkler_score
    # ==========================================================================

    @testset "compute_winkler_score" begin
        @testset "Same as interval score" begin
            actuals = [1.0, 2.0, 3.0]
            lower = [0.5, 1.5, 2.0]
            upper = [1.5, 2.5, 4.0]
            alpha = 0.1

            is = compute_interval_score(actuals, lower, upper, alpha)
            ws = compute_winkler_score(actuals, lower, upper, alpha)

            @test is == ws
        end
    end

    # ==========================================================================
    # compute_quantile_coverage
    # ==========================================================================

    @testset "compute_quantile_coverage" begin
        @testset "All covered" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            lower = [0.5, 1.5, 2.5, 3.5, 4.5]
            upper = [1.5, 2.5, 3.5, 4.5, 5.5]
            @test compute_quantile_coverage(actuals, lower, upper) == 1.0
        end

        @testset "None covered" begin
            actuals = [0.0, 0.0, 0.0]
            lower = [1.0, 1.0, 1.0]
            upper = [2.0, 2.0, 2.0]
            @test compute_quantile_coverage(actuals, lower, upper) == 0.0
        end

        @testset "Partial coverage" begin
            actuals = [1.0, 0.0, 3.0]  # 1st and 3rd covered
            lower = [0.5, 1.0, 2.5]
            upper = [1.5, 2.0, 3.5]
            @test compute_quantile_coverage(actuals, lower, upper) ≈ 2/3
        end

        @testset "Boundary inclusion" begin
            actuals = [1.0, 2.0]  # At boundaries
            lower = [1.0, 1.0]
            upper = [2.0, 2.0]
            @test compute_quantile_coverage(actuals, lower, upper) == 1.0
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_quantile_coverage(Float64[], Float64[], Float64[])
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Conformal prediction workflow" begin
        rng_int = Random.MersenneTwister(456)
        n = 50

        # Simulate predictions with uncertainty
        true_values = randn(rng_int, n)
        predictions = true_values .+ 0.3 .* randn(rng_int, n)  # Noisy predictions

        # 90% prediction intervals (alpha = 0.1)
        # Assume some residual std for interval construction
        residual_std = 0.5
        alpha = 0.1
        z = 1.645  # Approximate z for 90%

        lower = predictions .- z * residual_std
        upper = predictions .+ z * residual_std

        # Compute metrics
        coverage = compute_quantile_coverage(true_values, lower, upper)
        score = compute_interval_score(true_values, lower, upper, alpha)

        # Coverage should be roughly 90% or better
        @test 0.7 <= coverage <= 1.0  # Within reasonable range

        # Score should be positive
        @test score > 0

        # Wider intervals should have higher coverage
        wider_lower = predictions .- 2 * z * residual_std
        wider_upper = predictions .+ 2 * z * residual_std
        coverage_wide = compute_quantile_coverage(true_values, wider_lower, wider_upper)
        @test coverage_wide >= coverage
    end
end
