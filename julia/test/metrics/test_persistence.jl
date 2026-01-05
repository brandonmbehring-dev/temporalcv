@testset "Persistence Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_move_threshold
    # ==========================================================================

    @testset "compute_move_threshold" begin
        @testset "70th percentile default" begin
            actuals = collect(-1.0:0.1:1.0)  # 21 values
            threshold = compute_move_threshold(actuals)
            # 70th percentile of absolute values
            expected = quantile(abs.(actuals), 0.70)
            @test threshold ≈ expected
        end

        @testset "Custom percentile" begin
            actuals = [0.1, 0.2, 0.3, 0.4, 0.5]
            threshold_50 = compute_move_threshold(actuals; percentile=50.0)
            threshold_90 = compute_move_threshold(actuals; percentile=90.0)
            @test threshold_50 < threshold_90
        end

        @testset "Symmetric distribution" begin
            actuals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            threshold = compute_move_threshold(actuals)
            @test threshold > 0
        end

        @testset "All positive" begin
            actuals = [0.1, 0.2, 0.3, 0.4, 0.5]
            threshold = compute_move_threshold(actuals; percentile=80.0)
            @test threshold ≈ quantile(actuals, 0.80)  # Already positive
        end

        @testset "Empty array error" begin
            @test_throws ErrorException compute_move_threshold(Float64[])
        end

        @testset "Invalid percentile" begin
            @test_throws ErrorException compute_move_threshold([1.0]; percentile=0.0)
            @test_throws ErrorException compute_move_threshold([1.0]; percentile=101.0)
        end

        @testset "Level mode error" begin
            @test_throws ErrorException compute_move_threshold([1.0]; target_mode=:level)
        end
    end

    # ==========================================================================
    # classify_moves
    # ==========================================================================

    @testset "classify_moves" begin
        @testset "Basic classification" begin
            values = [0.1, -0.1, 0.02, -0.02, 0.0]
            moves = classify_moves(values, 0.05)
            @test moves[1] == UP      # 0.1 > 0.05
            @test moves[2] == DOWN    # -0.1 < -0.05
            @test moves[3] == FLAT    # |0.02| <= 0.05
            @test moves[4] == FLAT    # |-0.02| <= 0.05
            @test moves[5] == FLAT    # 0.0 <= 0.05
        end

        @testset "Boundary conditions" begin
            values = [0.05, -0.05, 0.050001, -0.050001]
            moves = classify_moves(values, 0.05)
            @test moves[1] == FLAT     # exactly at threshold = FLAT
            @test moves[2] == FLAT     # exactly at -threshold = FLAT
            @test moves[3] == UP       # just above
            @test moves[4] == DOWN     # just below
        end

        @testset "Zero threshold - all become moves" begin
            values = [0.001, -0.001, 0.0]
            moves = classify_moves(values, 0.0)
            @test moves[1] == UP
            @test moves[2] == DOWN
            @test moves[3] == FLAT  # 0 is always FLAT
        end

        @testset "Large threshold - all become FLAT" begin
            values = [1.0, -1.0, 0.5, -0.5]
            moves = classify_moves(values, 10.0)
            @test all(moves .== FLAT)
        end

        @testset "Negative threshold error" begin
            @test_throws ErrorException classify_moves([1.0], -0.1)
        end
    end

    # ==========================================================================
    # compute_move_conditional_metrics
    # ==========================================================================

    @testset "compute_move_conditional_metrics" begin
        @testset "Basic calculation" begin
            predictions = [0.15, -0.12, 0.03, 0.08, -0.10]
            actuals = [0.10, -0.10, 0.01, 0.12, -0.08]
            threshold = 0.05

            result = compute_move_conditional_metrics(
                predictions, actuals;
                threshold=threshold
            )

            @test result isa MoveConditionalResult
            @test result.move_threshold == threshold
            @test n_total(result) == 5
        end

        @testset "Skill score interpretation" begin
            # Model that perfectly predicts changes
            actuals = [0.1, -0.1, 0.2, -0.2]
            predictions = copy(actuals)  # Perfect
            threshold = 0.05

            result = compute_move_conditional_metrics(
                predictions, actuals;
                threshold=threshold
            )

            # Perfect predictions on moves → MAE = 0, skill = 1
            @test result.skill_score ≈ 1.0
        end

        @testset "Persistence baseline skill" begin
            # Model that predicts 0 (no change) = persistence
            actuals = [0.1, -0.1, 0.2, -0.2]
            predictions = zeros(length(actuals))  # Persistence
            threshold = 0.05

            result = compute_move_conditional_metrics(
                predictions, actuals;
                threshold=threshold
            )

            # Persistence error = |actual|, model error = |actual|
            # skill = 1 - (model/persistence) = 1 - 1 = 0
            @test result.skill_score ≈ 0.0 atol=1e-10
        end

        @testset "Counts by direction" begin
            actuals = [0.1, 0.2, -0.1, -0.2, 0.01, 0.02]  # 2 UP, 2 DOWN, 2 FLAT
            predictions = zeros(6)
            threshold = 0.05

            result = compute_move_conditional_metrics(
                predictions, actuals;
                threshold=threshold
            )

            @test result.n_up == 2
            @test result.n_down == 2
            @test result.n_flat == 2
        end

        @testset "Auto-compute threshold" begin
            actuals = randn(rng, 100)
            predictions = randn(rng, 100)

            result = compute_move_conditional_metrics(predictions, actuals)

            # Should use 70th percentile
            expected_threshold = compute_move_threshold(actuals)
            @test result.move_threshold ≈ expected_threshold
        end

        @testset "Empty arrays" begin
            result = compute_move_conditional_metrics(Float64[], Float64[])
            @test isnan(result.skill_score)
            @test n_total(result) == 0
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_move_conditional_metrics(
                [1.0, 2.0], [1.0]
            )
        end

        @testset "NaN in data error" begin
            @test_throws ErrorException compute_move_conditional_metrics(
                [1.0, NaN], [1.0, 2.0]
            )
            @test_throws ErrorException compute_move_conditional_metrics(
                [1.0, 2.0], [1.0, NaN]
            )
        end

        @testset "Level mode error" begin
            @test_throws ErrorException compute_move_conditional_metrics(
                [1.0], [2.0];
                target_mode=:level
            )
        end

        @testset "is_reliable property" begin
            # Enough samples per direction
            actuals = vcat(fill(0.2, 15), fill(-0.2, 12), fill(0.01, 20))
            predictions = zeros(length(actuals))
            threshold = 0.05

            result = compute_move_conditional_metrics(
                predictions, actuals;
                threshold=threshold
            )

            @test is_reliable(result) == true

            # Not enough samples
            actuals_few = vcat(fill(0.2, 5), fill(-0.2, 3), fill(0.01, 20))
            predictions_few = zeros(length(actuals_few))

            result_few = compute_move_conditional_metrics(
                predictions_few, actuals_few;
                threshold=threshold
            )

            @test is_reliable(result_few) == false
        end
    end

    # ==========================================================================
    # compute_direction_accuracy
    # ==========================================================================

    @testset "compute_direction_accuracy" begin
        @testset "2-class (sign) comparison" begin
            predictions = [0.5, -0.5, 0.5, -0.5]
            actuals = [0.3, -0.3, -0.3, 0.3]  # 2 correct, 2 wrong
            @test compute_direction_accuracy(predictions, actuals) ≈ 0.5
        end

        @testset "Perfect 2-class accuracy" begin
            predictions = [0.5, -0.5, 0.3, -0.3]
            actuals = [0.1, -0.1, 0.2, -0.2]  # All same signs
            @test compute_direction_accuracy(predictions, actuals) ≈ 1.0
        end

        @testset "3-class with threshold" begin
            predictions = [0.1, -0.1, 0.02, 0.15, -0.08]
            actuals = [0.12, -0.08, 0.01, 0.2, -0.1]
            threshold = 0.05

            # Predictions: UP, DOWN, FLAT, UP, DOWN
            # Actuals:     UP, DOWN, FLAT, UP, DOWN
            # All match!
            @test compute_direction_accuracy(
                predictions, actuals;
                move_threshold=threshold
            ) ≈ 1.0
        end

        @testset "3-class partial accuracy" begin
            predictions = [0.1, -0.1, 0.02]  # UP, DOWN, FLAT
            actuals = [0.12, 0.08, 0.01]     # UP, UP, FLAT
            threshold = 0.05

            # Match on 1st (UP) and 3rd (FLAT), miss on 2nd
            @test compute_direction_accuracy(
                predictions, actuals;
                move_threshold=threshold
            ) ≈ 2/3
        end

        @testset "Empty arrays" begin
            @test compute_direction_accuracy(Float64[], Float64[]) == 0.0
        end

        @testset "All zero actuals in 2-class" begin
            predictions = [0.1, 0.2]
            actuals = [0.0, 0.0]
            @test compute_direction_accuracy(predictions, actuals) == 0.0
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_direction_accuracy(
                [1.0, 2.0], [1.0]
            )
        end
    end

    # ==========================================================================
    # compute_move_only_mae
    # ==========================================================================

    @testset "compute_move_only_mae" begin
        @testset "Basic calculation" begin
            predictions = [0.15, -0.12, 0.02, 0.08]
            actuals = [0.10, -0.10, 0.01, 0.12]  # 1, 2, 4 are moves; 3 is flat
            threshold = 0.05

            mae, n = compute_move_only_mae(predictions, actuals, threshold)

            @test n == 3  # 3 moves
            @test isfinite(mae)
        end

        @testset "Perfect predictions on moves" begin
            actuals = [0.1, -0.1, 0.02]  # 2 moves
            predictions = copy(actuals)
            threshold = 0.05

            mae, n = compute_move_only_mae(predictions, actuals, threshold)

            @test n == 2
            @test mae == 0.0
        end

        @testset "No moves returns NaN" begin
            predictions = [0.01, 0.02, -0.01]
            actuals = [0.01, 0.02, -0.01]  # All FLAT
            threshold = 0.05

            mae, n = compute_move_only_mae(predictions, actuals, threshold)

            @test n == 0
            @test isnan(mae)
        end

        @testset "Negative threshold error" begin
            @test_throws ErrorException compute_move_only_mae([1.0], [1.0], -0.1)
        end
    end

    # ==========================================================================
    # compute_persistence_mae
    # ==========================================================================

    @testset "compute_persistence_mae" begin
        @testset "Without threshold" begin
            actuals = [0.1, -0.2, 0.3, -0.4]
            # Persistence predicts 0, so error = |actual|
            expected = mean(abs.(actuals))
            @test compute_persistence_mae(actuals) ≈ expected
        end

        @testset "With threshold (moves only)" begin
            actuals = [0.1, -0.2, 0.02, 0.3]  # 1, 2, 4 are moves
            threshold = 0.05
            # Moves: [0.1, -0.2, 0.3]
            expected = mean([0.1, 0.2, 0.3])
            @test compute_persistence_mae(actuals; threshold=threshold) ≈ expected
        end

        @testset "Empty array" begin
            @test isnan(compute_persistence_mae(Float64[]))
        end

        @testset "No moves returns NaN" begin
            actuals = [0.01, 0.02, -0.01]  # All FLAT
            @test isnan(compute_persistence_mae(actuals; threshold=0.05))
        end
    end

    # ==========================================================================
    # Integration tests
    # ==========================================================================

    @testset "Integration: Full workflow" begin
        # Simulate a forecast evaluation workflow
        rng_workflow = Random.MersenneTwister(123)

        # Generate training data for threshold computation
        train_actuals = 0.1 .* randn(rng_workflow, 100)

        # Generate test data
        test_actuals = 0.1 .* randn(rng_workflow, 50)
        # Model that captures some signal
        test_predictions = 0.3 .* test_actuals .+ 0.05 .* randn(rng_workflow, 50)

        # Compute threshold from training data only (critical!)
        threshold = compute_move_threshold(train_actuals)

        # Evaluate on test data
        result = compute_move_conditional_metrics(
            test_predictions, test_actuals;
            threshold=threshold
        )

        # Basic sanity checks
        @test n_total(result) == 50
        @test isfinite(result.skill_score)
        @test isfinite(result.mae_up) || result.n_up == 0
        @test isfinite(result.mae_down) || result.n_down == 0

        # Direction accuracy
        dir_acc = compute_direction_accuracy(
            test_predictions, test_actuals;
            move_threshold=threshold
        )
        @test 0.0 <= dir_acc <= 1.0

        # Move-only MAE
        mae, n = compute_move_only_mae(
            test_predictions, test_actuals, threshold
        )
        @test n == n_moves(result)
    end
end
