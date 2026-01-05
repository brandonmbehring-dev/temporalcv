@testset "Stationarity Tests" begin
    rng = Random.MersenneTwister(42)

    @testset "adf_test" begin
        @testset "Stationary white noise" begin
            # White noise is stationary
            y = randn(rng, 100)
            result = adf_test(y)
            @test result.test_name == :ADF
            @test result.is_stationary == true
            @test result.pvalue < 0.05
            @test result.lags_used > 0
            @test haskey(result.critical_values, "5%")
        end

        @testset "Random walk is non-stationary" begin
            # Random walk has unit root
            y = cumsum(randn(rng, 100))
            result = adf_test(y)
            @test result.is_stationary == false
            @test result.pvalue > 0.05
        end

        @testset "AR(1) with low phi is stationary" begin
            # AR(1) with |phi| < 1 is stationary
            n = 100
            y = zeros(n)
            y[1] = randn(rng)
            for i in 2:n
                y[i] = 0.3 * y[i-1] + randn(rng)
            end
            result = adf_test(y)
            @test result.is_stationary == true
        end

        @testset "AR(1) with high phi is near-unit-root" begin
            # AR(1) with phi close to 1 is borderline
            n = 100
            y = zeros(n)
            y[1] = randn(rng)
            for i in 2:n
                y[i] = 0.99 * y[i-1] + randn(rng)
            end
            result = adf_test(y)
            # May or may not reject depending on realization
            @test result.test_name == :ADF
        end

        @testset "Regression options" begin
            y = randn(rng, 100)

            # Constant only (default)
            result_c = adf_test(y; regression=:c)
            @test result_c.regression == :c

            # Constant and trend
            result_ct = adf_test(y; regression=:ct)
            @test result_ct.regression == :ct

            # No constant
            result_n = adf_test(y; regression=:n)
            @test result_n.regression == :n
        end

        @testset "Custom max_lags" begin
            y = randn(rng, 100)
            result = adf_test(y; max_lags=5)
            @test result.lags_used == 5
        end

        @testset "Custom alpha" begin
            y = randn(rng, 100)

            # With low alpha, harder to reject H0
            result_low = adf_test(y; alpha=0.01)
            result_high = adf_test(y; alpha=0.10)

            # Same statistic, different conclusions possible
            @test result_low.statistic == result_high.statistic
        end

        @testset "Short series error" begin
            y = randn(rng, 15)  # Too short
            @test_throws AssertionError adf_test(y)
        end
    end

    @testset "kpss_test" begin
        @testset "Stationary white noise" begin
            # White noise should not reject stationarity
            y = randn(rng, 100)
            result = kpss_test(y)
            @test result.test_name == :KPSS
            @test result.is_stationary == true
            @test result.pvalue > 0.05
        end

        @testset "Random walk is non-stationary" begin
            # Random walk should reject stationarity
            y = cumsum(randn(rng, 100))
            result = kpss_test(y)
            @test result.is_stationary == false
            @test result.pvalue < 0.05
        end

        @testset "Regression options" begin
            y = randn(rng, 100)

            # Level stationarity
            result_c = kpss_test(y; regression=:c)
            @test result_c.regression == :c

            # Trend stationarity
            result_ct = kpss_test(y; regression=:ct)
            @test result_ct.regression == :ct
        end

        @testset "Custom nlags" begin
            y = randn(rng, 100)
            result = kpss_test(y; nlags=5)
            @test result.lags_used == 5
        end

        @testset "KPSS critical values" begin
            y = randn(rng, 100)
            result = kpss_test(y)
            @test haskey(result.critical_values, "5%")
            @test result.critical_values["5%"] > 0
        end

        @testset "Short series error" begin
            y = randn(rng, 15)
            @test_throws AssertionError kpss_test(y)
        end
    end

    @testset "check_stationarity (joint)" begin
        @testset "Stationary series" begin
            y = randn(rng, 100)
            result = check_stationarity(y)
            @test result.conclusion == STATIONARY
            @test occursin("stationary", lowercase(result.recommended_action))
        end

        @testset "Non-stationary series" begin
            y = cumsum(randn(rng, 100))
            result = check_stationarity(y)
            @test result.conclusion == NON_STATIONARY
            @test occursin("differencing", lowercase(result.recommended_action))
        end

        @testset "Components populated" begin
            y = randn(rng, 100)
            result = check_stationarity(y)
            @test result.adf_result.test_name == :ADF
            @test result.kpss_result.test_name == :KPSS
            @test result.conclusion isa StationarityConclusion
            @test !isempty(result.recommended_action)
        end

        @testset "Trend stationarity" begin
            # Series with linear trend
            t = 1:100
            y = 0.1 .* t .+ randn(rng, 100)
            result = check_stationarity(y; regression=:ct)
            # With trend regression, trend-stationary series may be detected
            @test result.conclusion isa StationarityConclusion
        end

        @testset "Custom alpha" begin
            y = randn(rng, 100)
            result_low = check_stationarity(y; alpha=0.01)
            result_high = check_stationarity(y; alpha=0.10)
            # Same data may have different conclusions at different alpha
            @test result_low.adf_result.statistic == result_high.adf_result.statistic
        end
    end

    @testset "difference_until_stationary" begin
        @testset "Already stationary" begin
            y = randn(rng, 100)
            diff_y, d = difference_until_stationary(y)
            @test d == 0  # No differencing needed
            @test diff_y == y
        end

        @testset "I(1) series" begin
            y = cumsum(randn(rng, 100))
            diff_y, d = difference_until_stationary(y)
            @test d == 1  # One difference needed
            @test length(diff_y) == 99
        end

        @testset "I(2) series" begin
            # Integrated of order 2
            y = cumsum(cumsum(randn(rng, 200)))
            diff_y, d = difference_until_stationary(y)
            @test d <= 2
        end

        @testset "Max diff constraint" begin
            # Very persistent series
            y = cumsum(cumsum(cumsum(randn(rng, 200))))
            @test_throws ErrorException difference_until_stationary(y; max_diff=1)
        end
    end

    @testset "integration_order" begin
        @testset "I(0) stationary" begin
            y = randn(rng, 100)
            @test integration_order(y) == 0
        end

        @testset "I(1) random walk" begin
            y = cumsum(randn(rng, 100))
            @test integration_order(y) == 1
        end
    end
end
