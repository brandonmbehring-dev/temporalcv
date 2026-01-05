@testset "Lag Selection" begin
    rng = Random.MersenneTwister(42)

    @testset "LagSelectionResult type" begin
        result = LagSelectionResult(
            2,
            Dict(1 => 100.0, 2 => 95.0, 3 => 98.0),
            :bic,
            [1, 2, 3]
        )
        @test result.optimal_lag == 2
        @test result.method == :bic
        @test length(result.all_lags_tested) == 3
        @test result.criterion_values[2] == 95.0
    end

    @testset "compute_pacf" begin
        # White noise: PACF should be near zero at all lags
        y = randn(rng, 200)
        pacf = compute_pacf(y, 10)
        @test length(pacf) == 11  # lags 0 to 10
        @test pacf[1] ≈ 1.0  # PACF at lag 0 is always 1
        @test all(abs.(pacf[2:end]) .< 0.2)  # Near zero for white noise

        # AR(1) with high phi: first PACF should be high
        ar1 = zeros(200)
        ar1[1] = randn(rng)
        for i in 2:200
            ar1[i] = 0.8 * ar1[i-1] + 0.2 * randn(rng)
        end
        pacf_ar1 = compute_pacf(ar1, 5)
        @test abs(pacf_ar1[2]) > 0.5  # High at lag 1
    end

    @testset "select_lag_pacf" begin
        @testset "White noise" begin
            y = randn(rng, 100)
            result = select_lag_pacf(y)
            @test result.method == :pacf
            @test result.optimal_lag <= 2  # Should be low for white noise
            @test haskey(result.criterion_values, 0)
        end

        @testset "AR(1) process" begin
            n = 200
            ar1 = zeros(n)
            ar1[1] = randn(rng)
            for i in 2:n
                ar1[i] = 0.7 * ar1[i-1] + randn(rng)
            end
            result = select_lag_pacf(ar1)
            @test result.optimal_lag >= 1
        end

        @testset "AR(2) process" begin
            n = 300
            ar2 = zeros(n)
            for i in 3:n
                ar2[i] = 0.5 * ar2[i-1] + 0.3 * ar2[i-2] + randn(rng)
            end
            result = select_lag_pacf(ar2)
            # Should detect around 2 lags
            @test result.optimal_lag in 1:4
        end

        @testset "Custom max_lag" begin
            y = randn(rng, 100)
            result = select_lag_pacf(y; max_lag=5)
            @test maximum(result.all_lags_tested) == 5
        end

        @testset "Custom alpha" begin
            y = randn(rng, 100)
            result_strict = select_lag_pacf(y; alpha=0.01)
            result_loose = select_lag_pacf(y; alpha=0.10)
            # Stricter alpha may find fewer significant lags
            @test result_strict.optimal_lag <= result_loose.optimal_lag + 1
        end

        @testset "Short series error" begin
            y = randn(rng, 5)
            @test_throws AssertionError select_lag_pacf(y)
        end
    end

    @testset "select_lag_aic" begin
        @testset "Returns valid result" begin
            y = randn(rng, 100)
            result = select_lag_aic(y)
            @test result.method == :aic
            @test result.optimal_lag >= 1
            @test !isempty(result.criterion_values)
        end

        @testset "AR(1) detection" begin
            # Use fresh RNG for reproducibility
            rng_ar = Random.MersenneTwister(123)
            n = 200
            ar1 = zeros(n)
            ar1[1] = randn(rng_ar)
            for i in 2:n
                ar1[i] = 0.7 * ar1[i-1] + randn(rng_ar)
            end
            result = select_lag_aic(ar1)
            # AIC should find low lag for AR(1), but may overfit slightly
            @test result.optimal_lag in 1:5
        end

        @testset "Custom max_lag" begin
            y = randn(rng, 100)
            result = select_lag_aic(y; max_lag=5)
            @test maximum(result.all_lags_tested) == 5
        end

        @testset "Short series error" begin
            y = randn(rng, 5)
            @test_throws AssertionError select_lag_aic(y)
        end
    end

    @testset "select_lag_bic" begin
        @testset "Returns valid result" begin
            y = randn(rng, 100)
            result = select_lag_bic(y)
            @test result.method == :bic
            @test result.optimal_lag >= 1
        end

        @testset "BIC more parsimonious than AIC" begin
            # BIC should generally prefer simpler models
            n = 200
            ar2 = zeros(n)
            for i in 3:n
                ar2[i] = 0.3 * ar2[i-1] + 0.2 * ar2[i-2] + randn(rng)
            end
            result_aic = select_lag_aic(ar2)
            result_bic = select_lag_bic(ar2)
            # BIC typically selects same or smaller lag
            @test result_bic.optimal_lag <= result_aic.optimal_lag + 1
        end

        @testset "AR(1) detection" begin
            n = 200
            ar1 = zeros(n)
            ar1[1] = randn(rng)
            for i in 2:n
                ar1[i] = 0.8 * ar1[i-1] + randn(rng)
            end
            result = select_lag_bic(ar1)
            @test result.optimal_lag in 1:2
        end
    end

    @testset "auto_select_lag" begin
        y = randn(rng, 100)

        @testset "Default method is BIC" begin
            lag = auto_select_lag(y)
            result = select_lag_bic(y)
            @test lag == result.optimal_lag
        end

        @testset "Method selection" begin
            lag_aic = auto_select_lag(y; method=:aic)
            lag_bic = auto_select_lag(y; method=:bic)
            lag_pacf = auto_select_lag(y; method=:pacf)
            # All should return valid lags
            @test lag_aic >= 1
            @test lag_bic >= 1
            @test lag_pacf >= 0
        end

        @testset "Invalid method error" begin
            @test_throws ErrorException auto_select_lag(y; method=:invalid)
        end
    end

    @testset "suggest_cv_gap" begin
        @testset "Returns at least horizon" begin
            y = randn(rng, 100)
            gap = suggest_cv_gap(y; horizon=3)
            @test gap >= 3
        end

        @testset "AR series gets larger gap" begin
            # AR(5) should suggest larger gap than white noise
            n = 200
            ar5 = zeros(n)
            for i in 6:n
                ar5[i] = 0.4 * ar5[i-1] + 0.1 * ar5[i-3] + 0.1 * ar5[i-5] + randn(rng)
            end
            gap_ar = suggest_cv_gap(ar5; horizon=1)
            gap_wn = suggest_cv_gap(randn(rng, 200); horizon=1)
            # AR series should have larger gap due to autocorrelation
            @test gap_ar >= gap_wn
        end

        @testset "Method option" begin
            y = randn(rng, 100)
            gap_aic = suggest_cv_gap(y; horizon=1, method=:aic)
            gap_bic = suggest_cv_gap(y; horizon=1, method=:bic)
            # Both should be valid
            @test gap_aic >= 1
            @test gap_bic >= 1
        end
    end

    @testset "compute_max_lag" begin
        # Test rule of thumb
        @test TemporalValidation.compute_max_lag(100, nothing) > 0
        @test TemporalValidation.compute_max_lag(100, nothing) <= 25  # n/4

        # Test custom max_lag
        @test TemporalValidation.compute_max_lag(100, 5) == 5
        @test TemporalValidation.compute_max_lag(100, 60) < 60  # Capped at n/2-1
    end
end
