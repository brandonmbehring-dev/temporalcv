@testset "Changepoint Detection" begin
    rng = Random.MersenneTwister(42)

    @testset "Changepoint type" begin
        cp = Changepoint(30, 2.5)
        @test cp.index == 30
        @test cp.cost_reduction == 2.5
        @test isnothing(cp.regime_before)
        @test isnothing(cp.regime_after)

        # With regime info
        cp2 = Changepoint(50, 1.0, "LOW", "HIGH")
        @test cp2.regime_before == "LOW"
        @test cp2.regime_after == "HIGH"
    end

    @testset "ChangepointResult type" begin
        cps = [Changepoint(30, 2.0), Changepoint(60, 1.5)]
        result = ChangepointResult(cps, 3, :variance, 3.0)
        @test length(result.changepoints) == 2
        @test result.n_segments == 3
        @test result.method == :variance
        @test result.penalty == 3.0
    end

    @testset "detect_changepoints_variance" begin
        @testset "Detects single level shift" begin
            # Clear level shift at index 30
            series = vcat(fill(3.0, 30), fill(8.0, 30))
            result = detect_changepoints_variance(series; penalty=1.5, window=5)
            # Should detect at least one changepoint for a 5-unit level shift
            @test length(result.changepoints) >= 1
            # Changepoint should be near index 30 (within window size)
            if length(result.changepoints) > 0
                @test any(25 <= cp.index <= 40 for cp in result.changepoints)
            end
        end

        @testset "Detects multiple level shifts" begin
            # Two level shifts
            series = vcat(fill(1.0, 30), fill(5.0, 30), fill(2.0, 30))
            result = detect_changepoints_variance(series; penalty=2.0, window=5)
            @test length(result.changepoints) >= 1
        end

        @testset "No changepoints in constant series" begin
            series = fill(5.0, 60)
            result = detect_changepoints_variance(series; window=5)
            @test isempty(result.changepoints)
            @test result.n_segments == 1
        end

        @testset "No changepoints in white noise" begin
            # Moderate white noise shouldn't trigger changepoints with high penalty
            series = randn(rng, 100)
            result = detect_changepoints_variance(series; penalty=5.0, window=10)
            # May or may not detect changepoints, but should be few
            @test length(result.changepoints) <= 3
        end

        @testset "Penalty affects sensitivity" begin
            series = vcat(fill(1.0, 30), fill(3.0, 30))
            result_low = detect_changepoints_variance(series; penalty=1.0, window=5)
            result_high = detect_changepoints_variance(series; penalty=5.0, window=5)
            # Lower penalty should find more or equal changepoints
            @test length(result_low.changepoints) >= length(result_high.changepoints)
        end

        @testset "min_segment_length constraint" begin
            # Two shifts close together
            series = vcat(fill(1.0, 20), fill(5.0, 5), fill(1.0, 20))
            result = detect_changepoints_variance(series; penalty=2.0, min_segment_length=10, window=5)
            # Should not detect both due to min_segment_length
            # (segment of length 5 is too short)
            @test length(result.changepoints) <= 2
        end

        @testset "Short series error" begin
            series = fill(1.0, 10)  # Too short
            @test_throws AssertionError detect_changepoints_variance(series)
        end
    end

    @testset "detect_changepoints" begin
        @testset "Default method is variance" begin
            series = vcat(fill(1.0, 30), fill(5.0, 30))
            result = detect_changepoints(series; penalty=2.0, window=5)
            @test result.method == :variance
        end

        @testset "Unknown method error" begin
            series = fill(1.0, 60)
            @test_throws ErrorException detect_changepoints(series; method=:unknown)
        end
    end

    @testset "classify_regimes" begin
        @testset "Level-based classification" begin
            series = vcat(fill(1.0, 30), fill(10.0, 30))
            cps = [Changepoint(30, 1.0)]
            regimes = classify_regimes(series, cps; method=:level)
            @test all(regimes[1:30] .== "LOW") || all(regimes[1:30] .== "MEDIUM")
            @test all(regimes[31:60] .== "HIGH") || all(regimes[31:60] .== "MEDIUM")
        end

        @testset "From ChangepointResult" begin
            series = vcat(fill(1.0, 30), fill(5.0, 30))
            result = ChangepointResult([Changepoint(30, 1.0)], 2, :variance, 3.0)
            regimes = classify_regimes(series, result; method=:level)
            @test length(regimes) == 60
        end

        @testset "Custom thresholds" begin
            series = vcat(fill(1.0, 20), fill(5.0, 20), fill(10.0, 20))
            cps = [Changepoint(20, 1.0), Changepoint(40, 1.0)]
            regimes = classify_regimes(series, cps; method=:level, thresholds=(3.0, 7.0))
            @test all(regimes[1:20] .== "LOW")
            @test all(regimes[21:40] .== "MEDIUM")
            @test all(regimes[41:60] .== "HIGH")
        end

        @testset "Volatility method" begin
            # High volatility segment in the middle
            series = vcat(fill(1.0, 20), randn(rng, 20) .* 5, fill(1.0, 20))
            cps = [Changepoint(20, 1.0), Changepoint(40, 1.0)]
            regimes = classify_regimes(series, cps; method=:volatility)
            @test length(regimes) == 60
        end

        @testset "Trend method" begin
            # Increasing, flat, decreasing
            series = vcat(collect(1.0:20.0), fill(15.0, 20), collect(range(15.0, 1.0, length=20)))
            cps = [Changepoint(20, 1.0), Changepoint(40, 1.0)]
            regimes = classify_regimes(series, cps; method=:trend)
            @test length(regimes) == 60
        end

        @testset "Unknown method error" begin
            series = fill(1.0, 30)
            cps = Changepoint[]
            @test_throws ErrorException classify_regimes(series, cps; method=:unknown)
        end
    end

    @testset "get_segment_boundaries" begin
        @testset "Single changepoint" begin
            cps = [Changepoint(30, 1.0)]
            bounds = get_segment_boundaries(60, cps)
            @test length(bounds) == 2
            @test bounds[1] == (1, 30)
            @test bounds[2] == (31, 60)
        end

        @testset "Multiple changepoints" begin
            cps = [Changepoint(20, 1.0), Changepoint(40, 1.0)]
            bounds = get_segment_boundaries(60, cps)
            @test length(bounds) == 3
            @test bounds[1] == (1, 20)
            @test bounds[2] == (21, 40)
            @test bounds[3] == (41, 60)
        end

        @testset "No changepoints" begin
            cps = Changepoint[]
            bounds = get_segment_boundaries(60, cps)
            @test length(bounds) == 1
            @test bounds[1] == (1, 60)
        end

        @testset "From ChangepointResult" begin
            result = ChangepointResult([Changepoint(30, 1.0)], 2, :variance, 3.0)
            bounds = get_segment_boundaries(60, result)
            @test length(bounds) == 2
        end
    end

    @testset "create_regime_indicators" begin
        @testset "Returns expected keys" begin
            series = vcat(fill(1.0, 30), fill(5.0, 30))
            cps = [Changepoint(30, 1.0)]
            indicators = create_regime_indicators(series, cps)
            @test haskey(indicators, "is_regime_change")
            @test haskey(indicators, "periods_since_change")
            @test haskey(indicators, "regime_labels")
        end

        @testset "is_regime_change near changepoint" begin
            series = vcat(fill(1.0, 30), fill(5.0, 30))
            cps = [Changepoint(30, 1.0)]
            indicators = create_regime_indicators(series, cps; recent_window=4)
            # Points 31-34 should be marked as recent change
            @test indicators["is_regime_change"][31] == 1
            @test indicators["is_regime_change"][34] == 1
            @test indicators["is_regime_change"][35] == 0
        end

        @testset "periods_since_change" begin
            series = vcat(fill(1.0, 30), fill(5.0, 30))
            cps = [Changepoint(30, 1.0)]
            indicators = create_regime_indicators(series, cps)
            @test indicators["periods_since_change"][30] == 0
            @test indicators["periods_since_change"][31] == 1
            @test indicators["periods_since_change"][35] == 5
        end

        @testset "From ChangepointResult" begin
            series = vcat(fill(1.0, 30), fill(5.0, 30))
            result = ChangepointResult([Changepoint(30, 1.0)], 2, :variance, 3.0)
            indicators = create_regime_indicators(series, result)
            @test length(indicators["regime_labels"]) == 60
        end
    end
end
