@testset "Volatility Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # Volatility Estimator Types
    # ==========================================================================

    @testset "RollingVolatility" begin
        @testset "Basic estimation" begin
            values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            estimator = RollingVolatility(5, 2)
            vol = estimate(estimator, values)

            @test length(vol) == length(values)
            @test all(isfinite.(vol))
            @test all(vol .>= 0)
        end

        @testset "Window size validation" begin
            @test_throws ErrorException RollingVolatility(1)
        end

        @testset "Min periods default" begin
            est = RollingVolatility(10)
            @test est.min_periods == 5  # div(10, 2)
        end

        @testset "Empty input" begin
            estimator = RollingVolatility(5, 2)
            vol = estimate(estimator, Float64[])
            @test isempty(vol)
        end

        @testset "Forward-fill NaNs" begin
            values = [1.0, 2.0, 3.0, 4.0, 5.0]
            estimator = RollingVolatility(5, 4)  # Need 4 obs before valid
            vol = estimate(estimator, values)

            # First few should be forward-filled
            @test !any(isnan.(vol))
        end
    end

    @testset "EWMAVolatility" begin
        @testset "Basic estimation" begin
            values = randn(rng, 100)
            estimator = EWMAVolatility(13)
            vol = estimate(estimator, values)

            @test length(vol) == length(values)
            @test all(isfinite.(vol))
            @test all(vol .>= 0)
        end

        @testset "Span validation" begin
            @test_throws ErrorException EWMAVolatility(0)
        end

        @testset "Alpha calculation" begin
            est = EWMAVolatility(9)
            @test est.alpha ≈ 2.0 / 10.0  # 2/(span+1)
        end

        @testset "Empty input" begin
            estimator = EWMAVolatility(13)
            vol = estimate(estimator, Float64[])
            @test isempty(vol)
        end

        @testset "Responsiveness to changes" begin
            # EWMA should respond to volatility changes
            stable = fill(1.0, 50)
            volatile = 1.0 .+ 2.0 .* randn(rng, 50)
            values = vcat(stable, volatile)

            estimator = EWMAVolatility(10)
            vol = estimate(estimator, values)

            # Volatility should increase after the change
            @test mean(vol[55:end]) > mean(vol[1:45])
        end
    end

    # ==========================================================================
    # compute_local_volatility
    # ==========================================================================

    @testset "compute_local_volatility" begin
        @testset "Rolling std method" begin
            values = randn(rng, 100)
            vol = compute_local_volatility(values; window=13, method=:rolling_std)

            @test length(vol) == 100
            @test all(isfinite.(vol))
        end

        @testset "EWMA method" begin
            values = randn(rng, 100)
            vol = compute_local_volatility(values; window=13, method=:ewm)

            @test length(vol) == 100
            @test all(isfinite.(vol))
        end

        @testset "Different methods produce different results" begin
            values = randn(rng, 100)
            vol_rolling = compute_local_volatility(values; window=13, method=:rolling_std)
            vol_ewma = compute_local_volatility(values; window=13, method=:ewm)

            # Methods should give different results (not identical)
            @test vol_rolling != vol_ewma
        end

        @testset "Invalid method error" begin
            @test_throws ErrorException compute_local_volatility([1.0, 2.0]; method=:invalid)
        end

        @testset "Empty input" begin
            vol = compute_local_volatility(Float64[])
            @test isempty(vol)
        end
    end

    # ==========================================================================
    # compute_volatility_normalized_mae
    # ==========================================================================

    @testset "compute_volatility_normalized_mae" begin
        @testset "Perfect predictions" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = copy(predictions)
            volatility = [0.1, 0.2, 0.3]

            vnmae = compute_volatility_normalized_mae(predictions, actuals, volatility)
            @test vnmae ≈ 0.0 atol=1e-10
        end

        @testset "Errors relative to volatility" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.1, 2.1, 3.1]  # All errors = 0.1
            volatility = [0.1, 0.1, 0.1]  # Same volatility

            vnmae = compute_volatility_normalized_mae(predictions, actuals, volatility)
            # Error/volatility ≈ 1.0 (with epsilon adjustment in denominator)
            @test vnmae ≈ 1.0 atol=1e-6
        end

        @testset "Higher volatility reduces normalized error" begin
            predictions = [1.0, 2.0]
            actuals = [1.1, 2.1]  # Both errors = 0.1

            vol_low = [0.1, 0.1]
            vol_high = [1.0, 1.0]

            vnmae_low = compute_volatility_normalized_mae(predictions, actuals, vol_low)
            vnmae_high = compute_volatility_normalized_mae(predictions, actuals, vol_high)

            @test vnmae_low > vnmae_high  # Same error, higher vol = lower normalized
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_volatility_normalized_mae(
                [1.0, 2.0], [1.0], [0.1]
            )
        end

        @testset "Empty arrays error" begin
            @test_throws ErrorException compute_volatility_normalized_mae(
                Float64[], Float64[], Float64[]
            )
        end
    end

    # ==========================================================================
    # compute_volatility_weighted_mae
    # ==========================================================================

    @testset "compute_volatility_weighted_mae" begin
        @testset "Inverse weighting favors low-vol periods" begin
            predictions = [1.0, 2.0]
            actuals = [1.5, 2.1]  # Errors: 0.5, 0.1
            volatility = [0.1, 1.0]  # First low-vol, second high-vol

            wmae_inv = compute_volatility_weighted_mae(
                predictions, actuals, volatility; weighting=:inverse
            )

            # Low-vol period (0.1) has higher weight, so 0.5 error dominates
            # Simple MAE would be (0.5 + 0.1) / 2 = 0.3
            # With inverse weighting, 0.5 error gets more weight
            @test wmae_inv > 0.3
        end

        @testset "Importance weighting favors high-vol periods" begin
            predictions = [1.0, 2.0]
            actuals = [1.5, 2.1]  # Errors: 0.5, 0.1
            volatility = [0.1, 1.0]  # First low-vol, second high-vol

            wmae_imp = compute_volatility_weighted_mae(
                predictions, actuals, volatility; weighting=:importance
            )

            # High-vol period (1.0) has higher weight, so 0.1 error dominates
            @test wmae_imp < 0.3
        end

        @testset "Different weightings give different results" begin
            predictions = [1.0, 2.0, 3.0]
            actuals = [1.1, 2.5, 3.1]
            volatility = [0.1, 0.5, 1.0]

            wmae_inv = compute_volatility_weighted_mae(
                predictions, actuals, volatility; weighting=:inverse
            )
            wmae_imp = compute_volatility_weighted_mae(
                predictions, actuals, volatility; weighting=:importance
            )

            @test wmae_inv != wmae_imp
        end

        @testset "Invalid weighting error" begin
            @test_throws ErrorException compute_volatility_weighted_mae(
                [1.0], [1.0], [0.1]; weighting=:invalid
            )
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_volatility_weighted_mae(
                [1.0, 2.0], [1.0], [0.1]
            )
        end

        @testset "Empty arrays error" begin
            @test_throws ErrorException compute_volatility_weighted_mae(
                Float64[], Float64[], Float64[]
            )
        end
    end

    # ==========================================================================
    # compute_volatility_stratified_metrics
    # ==========================================================================

    @testset "compute_volatility_stratified_metrics" begin
        @testset "Basic stratification" begin
            rng_strat = Random.MersenneTwister(123)
            predictions = randn(rng_strat, 100)
            actuals = randn(rng_strat, 100)

            result = compute_volatility_stratified_metrics(predictions, actuals)

            @test result isa VolatilityStratifiedResult
            @test isfinite(result.mae_low) || isnan(result.mae_low)
            @test isfinite(result.mae_medium) || isnan(result.mae_medium)
            @test isfinite(result.mae_high) || isnan(result.mae_high)
            @test result.n_low + result.n_medium + result.n_high == 100
        end

        @testset "Tercile counts are balanced" begin
            rng_strat = Random.MersenneTwister(456)
            predictions = randn(rng_strat, 300)
            actuals = randn(rng_strat, 300)

            result = compute_volatility_stratified_metrics(predictions, actuals)

            # Roughly 1/3 in each tercile
            @test 80 < result.n_low < 120
            @test 80 < result.n_medium < 120
            @test 80 < result.n_high < 120
        end

        @testset "Pre-computed volatility" begin
            predictions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            actuals = [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
            volatility = [0.1, 0.2, 0.5, 0.6, 0.8, 1.0]

            result = compute_volatility_stratified_metrics(
                predictions, actuals; volatility=volatility
            )

            @test isfinite(result.mae_low) || isnan(result.mae_low)
            @test isfinite(result.mae_medium) || isnan(result.mae_medium)
            @test isfinite(result.mae_high) || isnan(result.mae_high)
        end

        @testset "Volatility thresholds" begin
            rng_strat = Random.MersenneTwister(789)
            predictions = randn(rng_strat, 100)
            actuals = randn(rng_strat, 100)

            result = compute_volatility_stratified_metrics(predictions, actuals)

            p33, p67 = result.vol_thresholds
            @test p33 < p67  # Thresholds in order
        end

        @testset "Different methods" begin
            rng_strat = Random.MersenneTwister(101)
            predictions = randn(rng_strat, 100)
            actuals = randn(rng_strat, 100)

            result_roll = compute_volatility_stratified_metrics(
                predictions, actuals; method=:rolling_std
            )
            result_ewm = compute_volatility_stratified_metrics(
                predictions, actuals; method=:ewm
            )

            # Methods should give different volatility estimates
            @test result_roll.vol_thresholds != result_ewm.vol_thresholds
        end

        @testset "Volatility length mismatch error" begin
            @test_throws ErrorException compute_volatility_stratified_metrics(
                [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]; volatility=[0.1, 0.2]
            )
        end

        @testset "Predictions/actuals length mismatch error" begin
            @test_throws ErrorException compute_volatility_stratified_metrics(
                [1.0, 2.0], [1.0]
            )
        end

        @testset "Empty arrays error" begin
            @test_throws ErrorException compute_volatility_stratified_metrics(
                Float64[], Float64[]
            )
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Volatility-adjusted evaluation" begin
        rng_int = Random.MersenneTwister(999)
        n = 200

        # Generate time series with varying volatility
        # First half: low vol, second half: high vol
        low_vol_returns = 0.01 .* randn(rng_int, 100)
        high_vol_returns = 0.05 .* randn(rng_int, 100)
        returns = vcat(low_vol_returns, high_vol_returns)

        # Predictions with similar pattern
        low_vol_preds = 0.01 .* randn(rng_int, 100)
        high_vol_preds = 0.05 .* randn(rng_int, 100)
        predictions = vcat(low_vol_preds, high_vol_preds)

        # Compute volatility
        vol = compute_local_volatility(returns; window=20)
        @test length(vol) == n

        # Volatility should be higher in second half
        @test mean(vol[120:end]) > mean(vol[1:80])

        # Compute normalized MAE
        vnmae = compute_volatility_normalized_mae(predictions, returns, vol)
        @test isfinite(vnmae)

        # Compute weighted MAE both ways
        wmae_inv = compute_volatility_weighted_mae(predictions, returns, vol; weighting=:inverse)
        wmae_imp = compute_volatility_weighted_mae(predictions, returns, vol; weighting=:importance)
        @test isfinite(wmae_inv)
        @test isfinite(wmae_imp)

        # Stratified metrics
        result = compute_volatility_stratified_metrics(predictions, returns; volatility=vol)
        @test isfinite(result.mae_low) || isnan(result.mae_low)
        @test isfinite(result.mae_medium) || isnan(result.mae_medium)
        @test isfinite(result.mae_high) || isnan(result.mae_high)

        # High vol MAE should be higher (larger errors in volatile periods)
        @test result.mae_high > result.mae_low
    end

    @testset "Integration: VolatilityStratifiedResult accessors" begin
        rng_int = Random.MersenneTwister(111)
        predictions = randn(rng_int, 100)
        actuals = randn(rng_int, 100)

        result = compute_volatility_stratified_metrics(predictions, actuals)

        # Check all fields are accessible
        @test result.mae_low >= 0 || isnan(result.mae_low)
        @test result.mae_medium >= 0 || isnan(result.mae_medium)
        @test result.mae_high >= 0 || isnan(result.mae_high)
        @test result.n_low >= 0
        @test result.n_medium >= 0
        @test result.n_high >= 0
        @test length(result.vol_thresholds) == 2
    end
end
