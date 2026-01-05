@testset "Core Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # Point Forecast Metrics
    # ==========================================================================

    @testset "compute_mae" begin
        @testset "Basic calculation" begin
            predictions = [1.0, 2.0, 3.0, 4.0, 5.0]
            actuals = [1.5, 2.5, 2.5, 4.5, 4.5]
            # Errors: [0.5, 0.5, 0.5, 0.5, 0.5] → mean = 0.5
            @test compute_mae(predictions, actuals) ≈ 0.5
        end

        @testset "Perfect predictions" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.0, 2.0, 3.0]
            @test compute_mae(predictions, actuals) == 0.0
        end

        @testset "Single element" begin
            @test compute_mae([5.0], [3.0]) == 2.0
        end

        @testset "Symmetric errors" begin
            predictions = [0.0, 0.0]
            actuals = [1.0, -1.0]
            @test compute_mae(predictions, actuals) == 1.0
        end

        @testset "Input validation - length mismatch" begin
            @test_throws ErrorException compute_mae([1.0, 2.0], [1.0])
        end

        @testset "Input validation - NaN in predictions" begin
            @test_throws ErrorException compute_mae([1.0, NaN], [1.0, 2.0])
        end

        @testset "Input validation - NaN in actuals" begin
            @test_throws ErrorException compute_mae([1.0, 2.0], [1.0, NaN])
        end
    end

    @testset "compute_mse" begin
        @testset "Basic calculation" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [2.0, 2.0, 2.0]
            # Errors: [1, 0, 1], squared: [1, 0, 1] → mean = 2/3
            @test compute_mse(predictions, actuals) ≈ 2/3
        end

        @testset "Perfect predictions" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.0, 2.0, 3.0]
            @test compute_mse(predictions, actuals) == 0.0
        end

        @testset "Emphasizes large errors" begin
            predictions = [0.0, 0.0]
            actuals = [1.0, 2.0]
            # Errors: [1, 2], squared: [1, 4] → mean = 2.5
            @test compute_mse(predictions, actuals) == 2.5
        end
    end

    @testset "compute_rmse" begin
        @testset "Basic calculation" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [2.0, 2.0, 2.0]
            expected = sqrt(2/3)
            @test compute_rmse(predictions, actuals) ≈ expected
        end

        @testset "Perfect predictions" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.0, 2.0, 3.0]
            @test compute_rmse(predictions, actuals) == 0.0
        end

        @testset "RMSE >= MAE always" begin
            predictions = randn(rng, 100)
            actuals = randn(rng, 100)
            mae = compute_mae(predictions, actuals)
            rmse = compute_rmse(predictions, actuals)
            @test rmse >= mae
        end
    end

    @testset "compute_mape" begin
        @testset "Basic calculation" begin
            predictions = [11.0, 22.0, 33.0]
            actuals = [10.0, 20.0, 30.0]
            # Errors: [1, 2, 3], relative: [0.1, 0.1, 0.1] → 10%
            @test compute_mape(predictions, actuals) ≈ 10.0
        end

        @testset "Perfect predictions" begin
            predictions = [10.0, 20.0, 30.0]
            actuals = [10.0, 20.0, 30.0]
            @test compute_mape(predictions, actuals) == 0.0
        end

        @testset "Handles near-zero actuals with epsilon" begin
            predictions = [1.0]
            actuals = [0.0]  # Should use epsilon
            result = compute_mape(predictions, actuals)
            @test isfinite(result)
        end

        @testset "Can exceed 100%" begin
            predictions = [0.0]
            actuals = [1.0]
            @test compute_mape(predictions, actuals) ≈ 100.0
        end
    end

    @testset "compute_smape" begin
        @testset "Basic calculation" begin
            predictions = [10.0, 20.0]
            actuals = [11.0, 22.0]
            # SMAPE formula: 100 * 2|p-a|/(|p|+|a|)
            expected = 100 * mean([
                2 * 1 / (10 + 11),
                2 * 2 / (20 + 22)
            ])
            @test compute_smape(predictions, actuals) ≈ expected
        end

        @testset "Bounded between 0 and 200" begin
            predictions = [0.0]
            actuals = [100.0]
            result = compute_smape(predictions, actuals)
            @test 0 <= result <= 200
        end

        @testset "Symmetric" begin
            smape1 = compute_smape([10.0], [15.0])
            smape2 = compute_smape([15.0], [10.0])
            @test smape1 ≈ smape2
        end

        @testset "Both zero excluded" begin
            predictions = [0.0, 10.0]
            actuals = [0.0, 12.0]
            # First pair excluded, second pair used
            result = compute_smape(predictions, actuals)
            @test isfinite(result)
        end

        @testset "All zeros returns 0" begin
            predictions = [0.0, 0.0]
            actuals = [0.0, 0.0]
            @test compute_smape(predictions, actuals) == 0.0
        end
    end

    @testset "compute_bias" begin
        @testset "Positive bias (over-prediction)" begin
            predictions = [5.0, 6.0, 7.0]
            actuals = [4.0, 5.0, 6.0]
            @test compute_bias(predictions, actuals) == 1.0
        end

        @testset "Negative bias (under-prediction)" begin
            predictions = [3.0, 4.0, 5.0]
            actuals = [4.0, 5.0, 6.0]
            @test compute_bias(predictions, actuals) == -1.0
        end

        @testset "Zero bias" begin
            predictions = [3.0, 5.0, 7.0]
            actuals = [4.0, 5.0, 6.0]
            @test compute_bias(predictions, actuals) == 0.0
        end

        @testset "Mixed errors can cancel" begin
            predictions = [0.0, 2.0]
            actuals = [1.0, 1.0]
            @test compute_bias(predictions, actuals) == 0.0
        end
    end

    # ==========================================================================
    # Scale-Invariant Metrics
    # ==========================================================================

    @testset "compute_naive_error" begin
        @testset "Persistence method" begin
            values = [1.0, 2.0, 4.0, 7.0]
            # Diffs: [1, 2, 3] → mean = 2.0
            @test compute_naive_error(values) == 2.0
        end

        @testset "Mean method" begin
            values = [2.0, 4.0, 6.0]
            # Mean = 4, deviations: [2, 0, 2] → mean = 4/3
            @test compute_naive_error(values; method=:mean) ≈ 4/3
        end

        @testset "Constant series persistence" begin
            values = [5.0, 5.0, 5.0, 5.0]
            @test compute_naive_error(values) == 0.0
        end

        @testset "Too short series" begin
            @test_throws ErrorException compute_naive_error([1.0])
        end

        @testset "Invalid method" begin
            @test_throws ErrorException compute_naive_error([1.0, 2.0]; method=:invalid)
        end
    end

    @testset "compute_mase" begin
        @testset "Basic calculation" begin
            predictions = [2.0, 3.0, 4.0]
            actuals = [2.5, 3.5, 4.5]
            naive_mae = 1.0  # Hypothetical
            # MAE = 0.5, MASE = 0.5 / 1.0 = 0.5
            @test compute_mase(predictions, actuals, naive_mae) == 0.5
        end

        @testset "MASE = 1 when equal to naive" begin
            predictions = [2.0, 3.0, 4.0]
            actuals = [2.5, 3.5, 4.5]
            naive_mae = 0.5
            @test compute_mase(predictions, actuals, naive_mae) == 1.0
        end

        @testset "MASE < 1 means better than naive" begin
            predictions = [2.0, 3.0, 4.0]
            actuals = [2.1, 3.1, 4.1]  # Small errors
            naive_mae = 1.0
            @test compute_mase(predictions, actuals, naive_mae) < 1.0
        end

        @testset "MASE > 1 means worse than naive" begin
            predictions = [0.0, 0.0, 0.0]
            actuals = [2.0, 3.0, 4.0]  # Large errors
            naive_mae = 1.0
            @test compute_mase(predictions, actuals, naive_mae) > 1.0
        end

        @testset "Invalid naive_mae" begin
            @test_throws ErrorException compute_mase([1.0], [2.0], 0.0)
            @test_throws ErrorException compute_mase([1.0], [2.0], -1.0)
        end
    end

    @testset "compute_mrae" begin
        @testset "Basic calculation" begin
            predictions = [1.0, 2.0]
            actuals = [1.5, 2.5]
            naive = [2.0, 3.0]
            # Model errors: [0.5, 0.5], Naive errors: [0.5, 0.5]
            # Ratios: [1.0, 1.0] → mean = 1.0
            @test compute_mrae(predictions, actuals, naive) ≈ 1.0
        end

        @testset "Model better than naive" begin
            predictions = [1.0, 2.0]
            actuals = [1.1, 2.1]  # Small model errors
            naive = [0.0, 1.0]   # Larger naive errors
            @test compute_mrae(predictions, actuals, naive) < 1.0
        end

        @testset "Excludes zero naive errors" begin
            predictions = [1.0, 2.0]
            actuals = [1.5, 2.0]  # Second is perfect for naive
            naive = [0.0, 2.0]   # First is off, second is perfect
            # Only first point used (naive error = 1.5)
            result = compute_mrae(predictions, actuals, naive)
            @test isfinite(result)
        end

        @testset "All naive perfect returns NaN" begin
            predictions = [1.0, 2.0]
            actuals = [1.5, 2.5]
            naive = [1.5, 2.5]  # Perfect
            @test isnan(compute_mrae(predictions, actuals, naive))
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_mrae([1.0, 2.0], [1.0, 2.0], [1.0])
        end
    end

    @testset "compute_theils_u" begin
        @testset "Uses persistence baseline by default" begin
            predictions = [2.0, 3.0, 4.0, 5.0]
            actuals = [2.1, 3.2, 4.1, 5.0]
            result = compute_theils_u(predictions, actuals)
            @test isfinite(result)
        end

        @testset "U = 1 when equal to naive" begin
            # Model exactly equals persistence
            actuals = [1.0, 2.0, 3.0, 4.0]
            predictions = copy(actuals)  # Perfect model
            # But we're comparing to persistence...
            # Actually, if predictions = actuals, model RMSE = 0
            # and we get U = 0 (or inf if naive also perfect)
            # Let me test explicit naive
            naive = [1.0, 2.0, 3.0]  # Lagged actuals
            preds = [1.0, 2.0, 3.0]  # Same as naive
            acts = [1.5, 2.5, 3.5]
            @test compute_theils_u(preds, acts; naive_predictions=naive) ≈ 1.0
        end

        @testset "Custom naive predictions" begin
            predictions = [1.4, 2.4, 3.4]  # Close to actuals
            actuals = [1.5, 2.5, 3.5]
            naive = [0.0, 1.0, 2.0]  # Much worse than model
            result = compute_theils_u(predictions, actuals; naive_predictions=naive)
            @test result < 1.0  # Model better than naive
        end

        @testset "Perfect naive returns Inf" begin
            predictions = [1.0, 2.0]
            actuals = [1.5, 2.5]
            naive = [1.5, 2.5]  # Perfect
            @test compute_theils_u(predictions, actuals; naive_predictions=naive) == Inf
        end

        @testset "Too short for persistence" begin
            @test_throws ErrorException compute_theils_u([1.0], [2.0])
        end
    end

    # ==========================================================================
    # Correlation Metrics
    # ==========================================================================

    @testset "compute_forecast_correlation" begin
        @testset "Pearson - perfect positive" begin
            predictions = [1.0, 2.0, 3.0, 4.0]
            actuals = [2.0, 4.0, 6.0, 8.0]
            @test compute_forecast_correlation(predictions, actuals) ≈ 1.0
        end

        @testset "Pearson - perfect negative" begin
            predictions = [1.0, 2.0, 3.0, 4.0]
            actuals = [8.0, 6.0, 4.0, 2.0]
            @test compute_forecast_correlation(predictions, actuals) ≈ -1.0
        end

        @testset "Pearson - no correlation" begin
            predictions = [1.0, 2.0, 1.0, 2.0]
            actuals = [1.0, 1.0, 2.0, 2.0]
            result = compute_forecast_correlation(predictions, actuals)
            @test abs(result) < 0.1
        end

        @testset "Spearman - rank correlation" begin
            predictions = [1.0, 2.0, 3.0, 4.0]
            actuals = [10.0, 20.0, 30.0, 40.0]
            result = compute_forecast_correlation(predictions, actuals; method=:spearman)
            @test result ≈ 1.0
        end

        @testset "Single element returns NaN" begin
            @test isnan(compute_forecast_correlation([1.0], [2.0]))
        end

        @testset "Invalid method" begin
            @test_throws ErrorException compute_forecast_correlation([1.0, 2.0], [1.0, 2.0]; method=:invalid)
        end
    end

    @testset "compute_r_squared" begin
        @testset "Perfect predictions" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.0, 2.0, 3.0]
            @test compute_r_squared(predictions, actuals) == 1.0
        end

        @testset "Mean forecast gives R² = 0" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            mean_pred = mean(actuals)
            predictions = fill(mean_pred, length(actuals))
            @test compute_r_squared(predictions, actuals) ≈ 0.0 atol=1e-10
        end

        @testset "Worse than mean gives R² < 0" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            predictions = [10.0, 10.0, 10.0, 10.0, 10.0]  # Far from actuals
            @test compute_r_squared(predictions, actuals) < 0.0
        end

        @testset "Constant actuals - perfect prediction" begin
            predictions = [5.0, 5.0, 5.0]
            actuals = [5.0, 5.0, 5.0]
            @test compute_r_squared(predictions, actuals) == 1.0
        end

        @testset "Constant actuals - imperfect prediction" begin
            predictions = [4.0, 5.0, 6.0]
            actuals = [5.0, 5.0, 5.0]
            @test compute_r_squared(predictions, actuals) == -Inf
        end
    end
end
