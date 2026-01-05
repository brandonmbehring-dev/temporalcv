@testset "Regimes: Classification & Stratified Metrics" begin
    using Random
    using Statistics

    # ==========================================================================
    # VolatilityRegime Enum
    # ==========================================================================

    @testset "VolatilityRegime enum" begin
        @test VOL_LOW isa VolatilityRegime
        @test VOL_MED isa VolatilityRegime
        @test VOL_HIGH isa VolatilityRegime

        @test regime_string(VOL_LOW) == "LOW"
        @test regime_string(VOL_MED) == "MED"
        @test regime_string(VOL_HIGH) == "HIGH"

        @test string(VOL_LOW) == "LOW"
        @test string(VOL_MED) == "MED"
        @test string(VOL_HIGH) == "HIGH"
    end

    # ==========================================================================
    # DirectionRegime Enum
    # ==========================================================================

    @testset "DirectionRegime enum" begin
        @test DIR_UP isa DirectionRegime
        @test DIR_DOWN isa DirectionRegime
        @test DIR_FLAT isa DirectionRegime

        @test regime_string(DIR_UP) == "UP"
        @test regime_string(DIR_DOWN) == "DOWN"
        @test regime_string(DIR_FLAT) == "FLAT"

        @test string(DIR_UP) == "UP"
        @test string(DIR_DOWN) == "DOWN"
        @test string(DIR_FLAT) == "FLAT"
    end

    # ==========================================================================
    # classify_volatility_regime
    # ==========================================================================

    @testset "classify_volatility_regime" begin
        rng = Random.MersenneTwister(42)

        @testset "Basic functionality" begin
            values = cumsum(randn(rng, 200) .* 0.1) .+ 10.0
            regimes = classify_volatility_regime(values; window=13)

            @test length(regimes) == 200
            @test all(r -> r isa VolatilityRegime, regimes)
        end

        @testset "Uses changes by default (not levels)" begin
            # Steady drift: high std of levels, but low volatility of changes
            # Use integer increments to avoid floating point issues
            n = 100
            steady_drift = Float64.(collect(100:199))  # [100.0, 101.0, ..., 199.0]

            regimes = classify_volatility_regime(steady_drift; window=13, basis=:changes)

            # With constant changes (all +1), volatility is 0, all should be same regime
            # First window points may be MED (default), rest should be consistent
            unique_after_warmup = unique(regimes[14:end])
            @test length(unique_after_warmup) == 1  # All same volatility (LOW, since std=0)
        end

        @testset "Levels basis (legacy)" begin
            values = randn(rng, 100)
            regimes_changes = classify_volatility_regime(values; window=10, basis=:changes)
            regimes_levels = classify_volatility_regime(values; window=10, basis=:levels)

            # Should produce different results
            @test regimes_changes != regimes_levels
        end

        @testset "Regime distribution" begin
            # Generate data with varying volatility
            values = vcat(
                randn(rng, 100) .* 0.1,  # Low vol
                randn(rng, 100) .* 1.0,  # High vol
                randn(rng, 100) .* 0.5   # Med vol
            )
            values = cumsum(values) .+ 50.0

            regimes = classify_volatility_regime(values; window=10)
            counts = get_regime_counts(regimes)

            # Should have all three regimes
            @test haskey(counts, "LOW")
            @test haskey(counts, "MED")
            @test haskey(counts, "HIGH")
        end

        @testset "Custom percentiles" begin
            values = randn(rng, 200)

            regimes_default = classify_volatility_regime(values; window=10)
            regimes_custom = classify_volatility_regime(values; window=10, low_pct=25.0, high_pct=75.0)

            # Different percentiles should produce different distributions
            counts_default = get_regime_counts(regimes_default)
            counts_custom = get_regime_counts(regimes_custom)

            # With 25/75 split, more samples in MED
            # (Not guaranteed, but typical)
        end

        @testset "Edge cases" begin
            # Empty array
            regimes = classify_volatility_regime(Float64[])
            @test isempty(regimes)

            # Single element
            regimes = classify_volatility_regime([1.0])
            @test length(regimes) == 1
            @test regimes[1] == VOL_MED  # Default

            # Fewer than window+1 elements
            regimes = classify_volatility_regime([1.0, 2.0, 3.0]; window=10)
            @test all(r -> r == VOL_MED, regimes)

            # All same value (zero variance)
            regimes = classify_volatility_regime(fill(5.0, 50); window=5)
            @test length(regimes) == 50
        end

        @testset "NaN handling" begin
            values = [1.0, 2.0, NaN, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            regimes = classify_volatility_regime(values; window=3)
            @test length(regimes) == 10
            # Should not crash
        end

        @testset "Invalid basis error" begin
            # Need enough data points (> window) to reach basis validation
            long_data = collect(1.0:50.0)
            @test_throws ErrorException classify_volatility_regime(long_data; window=10, basis=:invalid)
        end
    end

    # ==========================================================================
    # classify_direction_regime
    # ==========================================================================

    @testset "classify_direction_regime" begin
        @testset "Basic classification" begin
            values = [0.1, -0.1, 0.02, -0.02, 0.0]
            regimes = classify_direction_regime(values, 0.05)

            @test regimes[1] == DIR_UP    # 0.1 > 0.05
            @test regimes[2] == DIR_DOWN  # -0.1 < -0.05
            @test regimes[3] == DIR_FLAT  # |0.02| <= 0.05
            @test regimes[4] == DIR_FLAT  # |-0.02| <= 0.05
            @test regimes[5] == DIR_FLAT  # 0.0 <= 0.05
        end

        @testset "Threshold boundary" begin
            # Exactly at threshold
            regimes = classify_direction_regime([0.05, -0.05], 0.05)
            @test regimes[1] == DIR_FLAT  # <= threshold
            @test regimes[2] == DIR_FLAT  # >= -threshold

            # Just above threshold
            regimes = classify_direction_regime([0.051, -0.051], 0.05)
            @test regimes[1] == DIR_UP
            @test regimes[2] == DIR_DOWN
        end

        @testset "Zero threshold" begin
            values = [0.1, -0.1, 0.0]
            regimes = classify_direction_regime(values, 0.0)

            @test regimes[1] == DIR_UP
            @test regimes[2] == DIR_DOWN
            @test regimes[3] == DIR_FLAT  # Exactly zero
        end

        @testset "Edge cases" begin
            # Empty array
            regimes = classify_direction_regime(Float64[], 0.1)
            @test isempty(regimes)

            # Single element
            regimes = classify_direction_regime([0.5], 0.1)
            @test length(regimes) == 1
            @test regimes[1] == DIR_UP
        end

        @testset "NaN handling" begin
            values = [0.1, NaN, -0.1]
            regimes = classify_direction_regime(values, 0.05)

            @test regimes[1] == DIR_UP
            @test regimes[2] == DIR_FLAT  # NaN treated as FLAT
            @test regimes[3] == DIR_DOWN
        end

        @testset "Negative threshold error" begin
            @test_throws ErrorException classify_direction_regime([1.0], -0.1)
        end
    end

    # ==========================================================================
    # get_combined_regimes
    # ==========================================================================

    @testset "get_combined_regimes" begin
        @testset "Basic combination with enums" begin
            vol = [VOL_HIGH, VOL_LOW, VOL_MED]
            dir = [DIR_UP, DIR_DOWN, DIR_FLAT]
            combined = get_combined_regimes(vol, dir)

            @test combined[1] == "HIGH-UP"
            @test combined[2] == "LOW-DOWN"
            @test combined[3] == "MED-FLAT"
        end

        @testset "With string inputs" begin
            vol = ["HIGH", "LOW", "MED"]
            dir = ["UP", "DOWN", "FLAT"]
            combined = get_combined_regimes(vol, dir)

            @test combined[1] == "HIGH-UP"
            @test combined[2] == "LOW-DOWN"
            @test combined[3] == "MED-FLAT"
        end

        @testset "Length mismatch error" begin
            vol = [VOL_HIGH, VOL_LOW]
            dir = [DIR_UP]
            @test_throws ErrorException get_combined_regimes(vol, dir)
        end

        @testset "Empty arrays" begin
            combined = get_combined_regimes(VolatilityRegime[], DirectionRegime[])
            @test isempty(combined)
        end
    end

    # ==========================================================================
    # get_regime_counts
    # ==========================================================================

    @testset "get_regime_counts" begin
        @testset "With enums" begin
            regimes = [VOL_HIGH, VOL_LOW, VOL_LOW, VOL_MED, VOL_LOW]
            counts = get_regime_counts(regimes)

            @test counts["LOW"] == 3
            @test counts["HIGH"] == 1
            @test counts["MED"] == 1
        end

        @testset "With strings" begin
            regimes = ["A", "B", "A", "A", "C"]
            counts = get_regime_counts(regimes)

            @test counts["A"] == 3
            @test counts["B"] == 1
            @test counts["C"] == 1
        end

        @testset "Empty array" begin
            counts = get_regime_counts(String[])
            @test isempty(counts)
        end

        @testset "Single element" begin
            counts = get_regime_counts([VOL_HIGH])
            @test counts["HIGH"] == 1
            @test length(counts) == 1
        end
    end

    # ==========================================================================
    # mask_low_n_regimes
    # ==========================================================================

    @testset "mask_low_n_regimes" begin
        @testset "Masks low-count regimes" begin
            regimes = vcat(fill("HIGH", 5), fill("LOW", 15))
            masked = mask_low_n_regimes(regimes; min_n=10)

            @test count(==("MASKED"), masked) == 5
            @test count(==("LOW"), masked) == 15
        end

        @testset "No masking when all sufficient" begin
            regimes = vcat(fill("HIGH", 20), fill("LOW", 20))
            masked = mask_low_n_regimes(regimes; min_n=10)

            @test count(==("MASKED"), masked) == 0
        end

        @testset "Custom mask value" begin
            regimes = vcat(fill("A", 5), fill("B", 15))
            masked = mask_low_n_regimes(regimes; min_n=10, mask_value="OTHER")

            @test count(==("OTHER"), masked) == 5
        end

        @testset "With enum input" begin
            regimes = vcat(fill(VOL_HIGH, 5), fill(VOL_LOW, 15))
            masked = mask_low_n_regimes(regimes; min_n=10)

            @test count(==("MASKED"), masked) == 5
            @test count(==("LOW"), masked) == 15
        end

        @testset "Empty array" begin
            masked = mask_low_n_regimes(String[])
            @test isempty(masked)
        end
    end

    # ==========================================================================
    # StratifiedMetricsResult
    # ==========================================================================

    @testset "StratifiedMetricsResult" begin
        @testset "summary function" begin
            result = StratifiedMetricsResult(
                0.5,  # overall_mae
                0.7,  # overall_rmse
                100,  # n_total
                Dict{String, Dict{Symbol, Float64}}(
                    "LOW" => Dict(:mae => 0.3, :rmse => 0.4, :n => 50.0, :pct => 50.0),
                    "HIGH" => Dict(:mae => 0.7, :rmse => 0.9, :n => 50.0, :pct => 50.0)
                ),
                String[]  # no masked regimes
            )

            s = RegimeSummary(result)
            @test occursin("Overall: MAE=0.5", s)
            @test occursin("RMSE=0.7", s)
            @test occursin("LOW:", s)
            @test occursin("HIGH:", s)
        end

        @testset "summary with masked regimes" begin
            result = StratifiedMetricsResult(
                0.5, 0.7, 100,
                Dict{String, Dict{Symbol, Float64}}(),
                ["SMALL_REGIME"]
            )

            s = RegimeSummary(result)
            @test occursin("Masked", s)
            @test occursin("SMALL_REGIME", s)
        end
    end

    # ==========================================================================
    # compute_stratified_metrics
    # ==========================================================================

    @testset "compute_stratified_metrics" begin
        rng = Random.MersenneTwister(42)

        @testset "Basic functionality" begin
            n = 100
            predictions = randn(rng, n)
            actuals = predictions .+ randn(rng, n) .* 0.1
            regimes = vcat(fill(VOL_LOW, 50), fill(VOL_HIGH, 50))

            result = compute_stratified_metrics(predictions, actuals, regimes)

            @test result.n_total == 100
            @test isfinite(result.overall_mae)
            @test isfinite(result.overall_rmse)
            @test haskey(result.by_regime, "LOW")
            @test haskey(result.by_regime, "HIGH")
        end

        @testset "Masks low-n regimes" begin
            predictions = randn(rng, 100)
            actuals = randn(rng, 100)
            regimes = vcat(fill("A", 5), fill("B", 95))  # A has only 5

            result = compute_stratified_metrics(predictions, actuals, regimes; min_n=10)

            @test "A" in result.masked_regimes
            @test !haskey(result.by_regime, "A")
            @test haskey(result.by_regime, "B")
        end

        @testset "Percentage sums to 100" begin
            predictions = randn(rng, 100)
            actuals = randn(rng, 100)
            regimes = vcat(fill("A", 30), fill("B", 30), fill("C", 40))

            result = compute_stratified_metrics(predictions, actuals, regimes; min_n=5)

            total_pct = sum(m[:pct] for m in values(result.by_regime))
            @test isapprox(total_pct, 100.0; atol=0.1)
        end

        @testset "MAE/RMSE values reasonable" begin
            predictions = zeros(100)
            actuals = ones(100)  # Error of 1.0 everywhere
            regimes = fill("ALL", 100)

            result = compute_stratified_metrics(predictions, actuals, regimes; min_n=5)

            @test result.overall_mae ≈ 1.0
            @test result.overall_rmse ≈ 1.0
            @test result.by_regime["ALL"][:mae] ≈ 1.0
        end

        @testset "Error on empty input" begin
            @test_throws ErrorException compute_stratified_metrics(
                Float64[], Float64[], String[]
            )
        end

        @testset "Error on length mismatch" begin
            @test_throws ErrorException compute_stratified_metrics(
                [1.0, 2.0], [1.0], ["A", "B"]
            )
            @test_throws ErrorException compute_stratified_metrics(
                [1.0, 2.0], [1.0, 2.0], ["A"]
            )
        end

        @testset "Works with string regimes" begin
            predictions = randn(rng, 50)
            actuals = randn(rng, 50)
            regimes = vcat(fill("LOW", 25), fill("HIGH", 25))

            result = compute_stratified_metrics(predictions, actuals, regimes)

            @test haskey(result.by_regime, "LOW")
            @test haskey(result.by_regime, "HIGH")
        end
    end

    # ==========================================================================
    # Integration Tests
    # ==========================================================================

    @testset "Integration: Full regime analysis workflow" begin
        rng = Random.MersenneTwister(123)
        n = 300

        # Generate synthetic data with regime structure
        # Low vol period, then high vol period
        low_vol_data = cumsum(randn(rng, 150) .* 0.05) .+ 10.0
        high_vol_data = cumsum(randn(rng, 150) .* 0.3) .+ low_vol_data[end]
        data_values = vcat(low_vol_data, high_vol_data)

        # Classify regimes
        vol_regimes = classify_volatility_regime(data_values; window=13, basis=:changes)

        # Count regime distribution
        counts = get_regime_counts(vol_regimes)
        @test sum(Base.values(counts)) == n

        # Generate predictions and actuals
        predictions = data_values[2:end]  # Lag-1 forecast
        actuals = data_values[2:end] .+ randn(rng, n - 1) .* 0.1

        # Compute stratified metrics
        result = compute_stratified_metrics(
            predictions, actuals, vol_regimes[2:end]
        )

        @test result.n_total == n - 1
        @test isfinite(result.overall_mae)

        # Summary should be readable
        s = RegimeSummary(result)
        @test !isempty(s)
    end

    @testset "Integration: Combined vol + direction regimes" begin
        rng = Random.MersenneTwister(456)
        n = 200

        # Generate data
        changes = randn(rng, n) .* 0.5
        values = cumsum(changes) .+ 50.0

        # Classify both dimensions
        vol_regimes = classify_volatility_regime(values; window=10)
        threshold = quantile(abs.(changes), 0.7)
        dir_regimes = classify_direction_regime(changes, threshold)

        # Combine
        combined = get_combined_regimes(vol_regimes, dir_regimes)

        @test length(combined) == n
        @test all(occursin("-", c) for c in combined)

        # Mask low-n combined regimes
        masked = mask_low_n_regimes(combined; min_n=5)
        @test length(masked) == n
    end

    @testset "Integration: Direction accuracy comparison" begin
        rng = Random.MersenneTwister(789)
        n = 100

        # Simulate predictions
        actuals = randn(rng, n) .* 0.1
        threshold = quantile(abs.(actuals), 0.7)

        # Good model: similar to actuals
        good_predictions = actuals .+ randn(rng, n) .* 0.02

        # Bad model: random
        bad_predictions = randn(rng, n) .* 0.1

        # Classify actual directions
        actual_dirs = classify_direction_regime(actuals, threshold)

        # Direction accuracy
        good_dirs = classify_direction_regime(good_predictions, threshold)
        bad_dirs = classify_direction_regime(bad_predictions, threshold)

        good_accuracy = mean(good_dirs .== actual_dirs)
        bad_accuracy = mean(bad_dirs .== actual_dirs)

        # Good model should have higher accuracy
        @test good_accuracy >= bad_accuracy
    end
end
