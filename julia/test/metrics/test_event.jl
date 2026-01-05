@testset "Event Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_direction_brier - 2-class
    # ==========================================================================

    @testset "compute_direction_brier (2-class)" begin
        @testset "Perfect predictions" begin
            probs = [1.0, 0.0, 1.0, 0.0]  # Perfect confidence
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            @test result.brier_score ≈ 0.0
            @test result.n_samples == 4
            @test result.n_classes == 2
        end

        @testset "Worst predictions" begin
            probs = [0.0, 1.0, 0.0, 1.0]  # Completely wrong
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            @test result.brier_score ≈ 1.0
        end

        @testset "Random guessing (0.5 probability)" begin
            probs = [0.5, 0.5, 0.5, 0.5]
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            # Brier = mean((0.5 - y)^2) = mean(0.25) = 0.25
            @test result.brier_score ≈ 0.25
        end

        @testset "Partial confidence" begin
            probs = [0.7, 0.3, 0.8, 0.2]
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            # Errors: [0.3, 0.3, 0.2, 0.2], squared: [0.09, 0.09, 0.04, 0.04]
            expected = mean([0.09, 0.09, 0.04, 0.04])
            @test result.brier_score ≈ expected
        end

        @testset "Uncertainty calculation" begin
            probs = [0.5, 0.5, 0.5, 0.5]
            actuals = [1, 0, 1, 0]  # 50% positive
            result = compute_direction_brier(probs, actuals)

            # Uncertainty = p_bar * (1 - p_bar) = 0.5 * 0.5 = 0.25
            @test result.uncertainty ≈ 0.25
        end

        @testset "Empty arrays" begin
            result = compute_direction_brier(Float64[], Int[])
            @test isnan(result.brier_score)
            @test result.n_samples == 0
        end

        @testset "Invalid probability range" begin
            @test_throws ErrorException compute_direction_brier([1.5], [1])
            @test_throws ErrorException compute_direction_brier([-0.1], [0])
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_direction_brier([0.5, 0.5], [1])
        end
    end

    # ==========================================================================
    # compute_direction_brier - 3-class
    # ==========================================================================

    @testset "compute_direction_brier (3-class)" begin
        @testset "Perfect predictions" begin
            # probs: (n, 3) for [DOWN, FLAT, UP]
            probs = [
                0.0 0.0 1.0;  # UP
                1.0 0.0 0.0;  # DOWN
                0.0 1.0 0.0   # FLAT
            ]
            actuals = [2, 0, 1]  # UP, DOWN, FLAT
            result = compute_direction_brier(probs, actuals; n_classes=3)

            @test result.brier_score ≈ 0.0
            @test result.n_samples == 3
            @test result.n_classes == 3
        end

        @testset "Worst predictions" begin
            probs = [
                1.0 0.0 0.0;  # Predicts DOWN, actual UP
                0.0 0.0 1.0;  # Predicts UP, actual DOWN
                0.0 0.0 1.0   # Predicts UP, actual FLAT
            ]
            actuals = [2, 0, 1]
            result = compute_direction_brier(probs, actuals; n_classes=3)

            # Each wrong prediction: (0-1)^2 + (0-0)^2 + (1-0)^2 = 2 per sample
            @test result.brier_score ≈ 2.0
        end

        @testset "Uniform probabilities (random guessing)" begin
            probs = [
                1/3 1/3 1/3;
                1/3 1/3 1/3;
                1/3 1/3 1/3
            ]
            actuals = [0, 1, 2]
            result = compute_direction_brier(probs, actuals; n_classes=3)

            # For each sample: sum((1/3 - onehot)^2) = (1/3-1)^2 + (1/3)^2 + (1/3)^2
            # = 4/9 + 1/9 + 1/9 = 6/9 = 2/3
            @test result.brier_score ≈ 2/3
        end

        @testset "Probability sum validation" begin
            probs = [
                0.5 0.3 0.1;  # Sums to 0.9, not 1.0
            ]
            actuals = [0]
            @test_throws ErrorException compute_direction_brier(probs, actuals; n_classes=3)
        end

        @testset "Invalid direction values" begin
            probs = [1.0 0.0 0.0;]
            actuals = [3]  # Invalid: must be 0, 1, or 2
            @test_throws ErrorException compute_direction_brier(probs, actuals; n_classes=3)
        end
    end

    # ==========================================================================
    # skill_score
    # ==========================================================================

    @testset "skill_score" begin
        @testset "Perfect predictions give skill = 1" begin
            probs = [1.0, 0.0, 1.0, 0.0]
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            # BSS = 1 - (BS / uncertainty) = 1 - (0 / 0.25) = 1
            @test skill_score(result) ≈ 1.0
        end

        @testset "Random guessing gives skill = 0" begin
            probs = [0.5, 0.5, 0.5, 0.5]
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            # BS = 0.25, uncertainty = 0.25
            # BSS = 1 - (0.25 / 0.25) = 0
            @test skill_score(result) ≈ 0.0
        end

        @testset "Worse than random gives negative skill" begin
            # Overconfident wrong predictions
            probs = [0.1, 0.9, 0.1, 0.9]  # Predict opposite
            actuals = [1, 0, 1, 0]
            result = compute_direction_brier(probs, actuals)

            # BS > uncertainty → negative skill
            @test skill_score(result) < 0.0
        end

        @testset "Zero uncertainty returns 0" begin
            # All same class (uncertainty = 0)
            probs = [0.9, 0.8, 0.7]
            actuals = [1, 1, 1]  # All positive
            result = compute_direction_brier(probs, actuals)

            # Uncertainty = 1 * 0 = 0
            @test result.uncertainty ≈ 0.0
            @test skill_score(result) == 0.0
        end
    end

    # ==========================================================================
    # compute_calibrated_direction_brier
    # ==========================================================================

    @testset "compute_calibrated_direction_brier" begin
        @testset "Perfect calibration" begin
            # Perfectly calibrated: when p=0.2, 20% are positive
            probs = vcat(fill(0.2, 50), fill(0.8, 50))
            # 10 positives in first 50 (20%), 40 positives in last 50 (80%)
            actuals = vcat(
                vcat(zeros(Int, 40), ones(Int, 10)),  # 20% positive
                vcat(zeros(Int, 10), ones(Int, 40))   # 80% positive
            )

            brier, bin_means, bin_fracs = compute_calibrated_direction_brier(
                probs, actuals; n_bins=5
            )

            @test isfinite(brier)
            @test length(bin_means) == 5
            @test length(bin_fracs) == 5
        end

        @testset "Empty input" begin
            brier, bin_means, bin_fracs = compute_calibrated_direction_brier(
                Float64[], Int[]
            )
            @test isnan(brier)
            @test isempty(bin_means)
            @test isempty(bin_fracs)
        end

        @testset "Invalid n_bins" begin
            @test_throws ErrorException compute_calibrated_direction_brier(
                [0.5], [1]; n_bins=0
            )
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_calibrated_direction_brier(
                [0.5, 0.5], [1]
            )
        end
    end

    # ==========================================================================
    # Murphy decomposition validation
    # ==========================================================================

    @testset "Murphy decomposition" begin
        @testset "Decomposition sums correctly" begin
            # Random but valid probabilities and outcomes
            probs = [0.1, 0.4, 0.6, 0.9, 0.3, 0.7, 0.2, 0.8]
            actuals = [0, 0, 1, 1, 0, 1, 0, 1]

            result = compute_direction_brier(probs, actuals)

            # Brier = Reliability - Resolution + Uncertainty
            expected = result.reliability - result.resolution + result.uncertainty
            @test result.brier_score ≈ expected atol=0.05  # Allow some binning error

            # Use decomposition_valid helper
            @test decomposition_valid(result; tol=0.05)
        end

        @testset "Reliability = 0 for perfect calibration" begin
            # Perfect calibration: p = actual frequency
            # 10 samples at p=0.3, 3 positives
            probs = fill(0.3, 10)
            actuals = vcat(zeros(Int, 7), ones(Int, 3))

            result = compute_direction_brier(probs, actuals)

            # Reliability should be low (but may not be exactly 0 due to binning)
            @test result.reliability < 0.1
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Direction forecasting workflow" begin
        rng_int = Random.MersenneTwister(789)
        n = 100

        # Simulate direction forecasting
        # True probabilities
        true_probs = rand(rng_int, n)

        # Actual outcomes sampled from true probabilities
        actuals = Int.(rand(rng_int, n) .< true_probs)

        # Model predictions (somewhat calibrated)
        model_probs = clamp.(true_probs .+ 0.1 .* randn(rng_int, n), 0.01, 0.99)

        # Compute Brier score
        result = compute_direction_brier(model_probs, actuals)

        @test isfinite(result.brier_score)
        @test 0.0 <= result.brier_score <= 1.0
        @test result.n_samples == n

        # Compute calibrated version
        brier, bin_means, bin_fracs = compute_calibrated_direction_brier(
            model_probs, actuals; n_bins=10
        )

        @test brier ≈ result.brier_score

        # At least some bins should have data
        @test any(.!isnan.(bin_means))
    end
end
