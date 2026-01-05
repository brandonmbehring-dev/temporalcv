@testset "Asymmetric Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_linex_loss
    # ==========================================================================

    @testset "compute_linex_loss" begin
        @testset "Perfect predictions" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            preds = copy(actuals)
            @test compute_linex_loss(preds, actuals) ≈ 0.0 atol=1e-10
        end

        @testset "Under-prediction penalized more with a > 0" begin
            actuals = [10.0]
            over_pred = [12.0]  # Over-prediction by 2
            under_pred = [8.0]   # Under-prediction by 2

            # With a > 0, under-predictions are exponentially penalized
            loss_under = compute_linex_loss(under_pred, actuals; a=0.5)
            loss_over = compute_linex_loss(over_pred, actuals; a=0.5)

            @test loss_under > loss_over  # Under-prediction more costly
        end

        @testset "Over-prediction penalized more with a < 0" begin
            actuals = [10.0]
            over_pred = [12.0]
            under_pred = [8.0]

            # With a < 0, over-predictions are exponentially penalized
            loss_under = compute_linex_loss(under_pred, actuals; a=-0.5)
            loss_over = compute_linex_loss(over_pred, actuals; a=-0.5)

            @test loss_over > loss_under  # Over-prediction more costly
        end

        @testset "Scaling parameter b" begin
            actuals = [10.0]
            preds = [8.0]

            loss_b1 = compute_linex_loss(preds, actuals; a=0.5, b=1.0)
            loss_b2 = compute_linex_loss(preds, actuals; a=0.5, b=2.0)

            @test loss_b2 ≈ 2 * loss_b1
        end

        @testset "Invalid a = 0" begin
            @test_throws ErrorException compute_linex_loss([1.0], [1.0]; a=0.0)
        end

        @testset "Invalid b <= 0" begin
            @test_throws ErrorException compute_linex_loss([1.0], [1.0]; b=0.0)
            @test_throws ErrorException compute_linex_loss([1.0], [1.0]; b=-1.0)
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_linex_loss([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_linex_loss(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # compute_asymmetric_mape
    # ==========================================================================

    @testset "compute_asymmetric_mape" begin
        @testset "Symmetric at alpha=0.5" begin
            actuals = [100.0, 100.0]
            # One over, one under by same amount
            preds = [110.0, 90.0]

            amape = compute_asymmetric_mape(preds, actuals; alpha=0.5)
            # Both should contribute equally
            @test amape ≈ 0.05  # 10% error weighted by 0.5 each
        end

        @testset "Under-prediction weighted more with alpha > 0.5" begin
            actuals = [100.0, 100.0]
            preds = [110.0, 90.0]  # Over and under by 10

            amape_high = compute_asymmetric_mape(preds, actuals; alpha=0.7)
            amape_low = compute_asymmetric_mape(preds, actuals; alpha=0.3)

            # Under-prediction (90) should weigh more with high alpha
            @test amape_high > amape_low
        end

        @testset "Perfect predictions" begin
            actuals = [100.0, 200.0, 300.0]
            preds = copy(actuals)
            @test compute_asymmetric_mape(preds, actuals) ≈ 0.0 atol=1e-10
        end

        @testset "Invalid alpha" begin
            @test_throws ErrorException compute_asymmetric_mape([1.0], [1.0]; alpha=-0.1)
            @test_throws ErrorException compute_asymmetric_mape([1.0], [1.0]; alpha=1.5)
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_asymmetric_mape([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_asymmetric_mape(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # compute_directional_loss
    # ==========================================================================

    @testset "compute_directional_loss" begin
        @testset "Perfect direction predictions" begin
            pred_changes = [1.0, -1.0, 1.0, -1.0]
            actual_changes = [0.5, -0.5, 0.5, -0.5]  # Same direction
            @test compute_directional_loss(pred_changes, actual_changes) ≈ 0.0
        end

        @testset "All wrong directions" begin
            pred_changes = [1.0, -1.0, 1.0, -1.0]
            actual_changes = [-0.5, 0.5, -0.5, 0.5]  # Opposite direction

            # With default weights = 1.0, loss = 1.0 for each
            @test compute_directional_loss(pred_changes, actual_changes) ≈ 1.0
        end

        @testset "Custom weights for UP miss" begin
            # Predict DOWN, actual is UP
            pred_changes = [-1.0]
            actual_changes = [1.0]

            loss_high = compute_directional_loss(pred_changes, actual_changes;
                                                 up_miss_weight=2.0)
            loss_low = compute_directional_loss(pred_changes, actual_changes;
                                                up_miss_weight=0.5)

            @test loss_high > loss_low
            @test loss_high ≈ 2.0
            @test loss_low ≈ 0.5
        end

        @testset "Custom weights for DOWN miss" begin
            # Predict UP, actual is DOWN
            pred_changes = [1.0]
            actual_changes = [-1.0]

            loss_high = compute_directional_loss(pred_changes, actual_changes;
                                                 down_miss_weight=3.0)
            @test loss_high ≈ 3.0
        end

        @testset "With previous_actuals (levels)" begin
            previous = [100.0, 102.0, 101.0]
            predictions = [103.0, 101.0, 102.0]  # Predicted levels
            actuals = [101.0, 103.0, 100.0]  # Actual levels

            # Changes: pred=[3,-1,1], actual=[1,1,-1]
            # Miss UP at idx 2: pred=-1, actual=1
            # Miss DOWN at idx 3: pred=1, actual=-1
            loss = compute_directional_loss(predictions, actuals;
                                           previous_actuals=previous)
            @test loss ≈ 2/3  # 2 misses out of 3
        end

        @testset "Zero predictions treated conservatively" begin
            # Zero predictions are treated as non-positive (don't predict UP)
            # and non-negative (don't predict DOWN), so they can miss
            pred_changes = [0.0, 0.0]
            actual_changes = [1.0, -1.0]
            # Zero vs UP: miss_up (pred<=0, actual>0) = true
            # Zero vs DOWN: miss_down (pred>=0, actual<0) = true
            # Average loss = 1.0
            @test compute_directional_loss(pred_changes, actual_changes) ≈ 1.0
        end

        @testset "Negative weights error" begin
            @test_throws ErrorException compute_directional_loss([1.0], [1.0];
                                                                  up_miss_weight=-1.0)
            @test_throws ErrorException compute_directional_loss([1.0], [1.0];
                                                                  down_miss_weight=-1.0)
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_directional_loss([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_directional_loss(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # compute_squared_log_error
    # ==========================================================================

    @testset "compute_squared_log_error" begin
        @testset "Perfect predictions" begin
            actuals = [100.0, 1000.0, 10000.0]
            preds = copy(actuals)
            @test compute_squared_log_error(preds, actuals) ≈ 0.0 atol=1e-10
        end

        @testset "Scale invariance" begin
            # Same relative error at different scales
            actuals_small = [10.0, 20.0]
            preds_small = [11.0, 22.0]  # 10% over

            actuals_large = [1000.0, 2000.0]
            preds_large = [1100.0, 2200.0]  # 10% over

            msle_small = compute_squared_log_error(preds_small, actuals_small)
            msle_large = compute_squared_log_error(preds_large, actuals_large)

            # Should be similar (not exactly equal due to log(1+x) transform)
            @test abs(msle_small - msle_large) < 0.01
        end

        @testset "Under-prediction penalized more" begin
            actuals = [100.0]
            over_pred = [110.0]  # 10% over
            under_pred = [90.0]   # ~11% under

            msle_over = compute_squared_log_error(over_pred, actuals)
            msle_under = compute_squared_log_error(under_pred, actuals)

            # Under-prediction is slightly more penalized
            @test msle_under > msle_over
        end

        @testset "Negative values error" begin
            @test_throws ErrorException compute_squared_log_error([-1.0], [1.0])
            @test_throws ErrorException compute_squared_log_error([1.0], [-1.0])
        end

        @testset "Zero values handled" begin
            actuals = [0.0, 1.0, 2.0]
            preds = [0.0, 1.0, 2.0]
            @test compute_squared_log_error(preds, actuals) ≈ 0.0 atol=1e-10
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_squared_log_error([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_squared_log_error(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # compute_huber_loss
    # ==========================================================================

    @testset "compute_huber_loss" begin
        @testset "Perfect predictions" begin
            actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
            preds = copy(actuals)
            @test compute_huber_loss(preds, actuals) ≈ 0.0 atol=1e-10
        end

        @testset "Small errors are quadratic (like MSE)" begin
            actuals = [10.0]
            preds = [10.5]  # Error = 0.5, less than delta=1.0

            huber = compute_huber_loss(preds, actuals; delta=1.0)
            mse_like = 0.5 * 0.5^2  # 0.5 * e^2

            @test huber ≈ mse_like
        end

        @testset "Large errors are linear (like MAE)" begin
            actuals = [10.0]
            preds = [15.0]  # Error = 5, greater than delta=1.0

            huber = compute_huber_loss(preds, actuals; delta=1.0)
            # Linear: delta * (|e| - 0.5 * delta) = 1 * (5 - 0.5) = 4.5
            expected = 1.0 * (5.0 - 0.5 * 1.0)

            @test huber ≈ expected
        end

        @testset "Robustness to outliers vs MSE" begin
            actuals = [1.0, 2.0, 100.0]  # One outlier
            preds = [1.1, 1.9, 10.0]

            huber = compute_huber_loss(preds, actuals; delta=1.0)
            mse = mean((actuals .- preds).^2)

            # Huber should be much smaller due to outlier handling
            @test huber < mse / 10
        end

        @testset "Delta controls transition" begin
            actuals = [10.0]
            preds = [12.0]  # Error = -2 (actual - pred)

            huber_d1 = compute_huber_loss(preds, actuals; delta=1.0)  # Linear regime
            huber_d3 = compute_huber_loss(preds, actuals; delta=3.0)  # Quadratic regime

            # delta=1, |e|=2: linear zone → loss = 1*(2-0.5) = 1.5
            # delta=3, |e|=2: quadratic zone → loss = 0.5*2^2 = 2.0
            # So huber_d1 < huber_d3 (linear is gentler in tails)
            @test huber_d1 ≈ 1.5
            @test huber_d3 ≈ 2.0
        end

        @testset "Invalid delta" begin
            @test_throws ErrorException compute_huber_loss([1.0], [1.0]; delta=0.0)
            @test_throws ErrorException compute_huber_loss([1.0], [1.0]; delta=-1.0)
        end

        @testset "Length mismatch" begin
            @test_throws ErrorException compute_huber_loss([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays" begin
            @test_throws ErrorException compute_huber_loss(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Asymmetric loss comparison" begin
        rng_int = Random.MersenneTwister(123)
        n = 100

        # Generate predictions with some bias (under-predicting)
        true_values = 100.0 .+ 10.0 .* randn(rng_int, n)
        predictions = true_values .- 5.0 .+ 2.0 .* randn(rng_int, n)  # Systematic under

        # LinEx with a > 0 should give higher loss (penalizes under-prediction)
        linex_a_pos = compute_linex_loss(predictions, true_values; a=0.1)
        linex_a_neg = compute_linex_loss(predictions, true_values; a=-0.1)

        @test linex_a_pos > linex_a_neg

        # Asymmetric MAPE with high alpha should penalize more
        amape_high = compute_asymmetric_mape(predictions, true_values; alpha=0.8)
        amape_low = compute_asymmetric_mape(predictions, true_values; alpha=0.2)

        @test amape_high > amape_low

        # All metrics should be finite
        @test isfinite(linex_a_pos)
        @test isfinite(linex_a_neg)
        @test isfinite(amape_high)
        @test isfinite(amape_low)
    end
end
