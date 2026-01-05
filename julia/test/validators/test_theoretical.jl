@testset "Theoretical Bounds" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_ar1_mse_bound
    # ==========================================================================

    @testset "compute_ar1_mse_bound" begin
        @testset "h=1 returns sigma_sq" begin
            # For any AR(1), 1-step MSE = sigma^2
            @test compute_ar1_mse_bound(0.0, 1.0; horizon=1) == 1.0
            @test compute_ar1_mse_bound(0.5, 1.0; horizon=1) == 1.0
            @test compute_ar1_mse_bound(0.9, 1.0; horizon=1) == 1.0
            @test compute_ar1_mse_bound(-0.9, 1.0; horizon=1) == 1.0
        end

        @testset "phi=0 (white noise)" begin
            # For white noise, MSE(h) = h * sigma^2
            @test compute_ar1_mse_bound(0.0, 1.0; horizon=1) == 1.0
            @test compute_ar1_mse_bound(0.0, 1.0; horizon=5) == 5.0
            @test compute_ar1_mse_bound(0.0, 2.0; horizon=5) == 10.0
        end

        @testset "MSE increases with horizon" begin
            phi = 0.9
            sigma_sq = 1.0

            mse1 = compute_ar1_mse_bound(phi, sigma_sq; horizon=1)
            mse2 = compute_ar1_mse_bound(phi, sigma_sq; horizon=2)
            mse5 = compute_ar1_mse_bound(phi, sigma_sq; horizon=5)
            mse10 = compute_ar1_mse_bound(phi, sigma_sq; horizon=10)

            @test mse1 < mse2 < mse5 < mse10
        end

        @testset "MSE converges to unconditional variance" begin
            phi = 0.9
            sigma_sq = 1.0

            # Var(y) = sigma^2 / (1 - phi^2) for stationary AR(1)
            unconditional_var = sigma_sq / (1 - phi^2)

            # Large horizon should approach unconditional variance
            mse_large = compute_ar1_mse_bound(phi, sigma_sq; horizon=100)
            @test isapprox(mse_large, unconditional_var, rtol=0.01)
        end

        @testset "Non-stationary phi error" begin
            @test_throws ErrorException compute_ar1_mse_bound(1.0, 1.0)
            @test_throws ErrorException compute_ar1_mse_bound(-1.0, 1.0)
            @test_throws ErrorException compute_ar1_mse_bound(1.5, 1.0)
        end

        @testset "Invalid sigma_sq error" begin
            @test_throws ErrorException compute_ar1_mse_bound(0.5, 0.0)
            @test_throws ErrorException compute_ar1_mse_bound(0.5, -1.0)
        end

        @testset "Invalid horizon error" begin
            @test_throws ErrorException compute_ar1_mse_bound(0.5, 1.0; horizon=0)
            @test_throws ErrorException compute_ar1_mse_bound(0.5, 1.0; horizon=-1)
        end
    end

    # ==========================================================================
    # compute_ar1_mae_bound
    # ==========================================================================

    @testset "compute_ar1_mae_bound" begin
        @testset "h=1 MAE = sigma * sqrt(2/pi)" begin
            # For h=1, RMSE = sigma, so MAE = sigma * sqrt(2/pi)
            expected = sqrt(2/pi)
            @test isapprox(compute_ar1_mae_bound(0.0, 1.0; horizon=1), expected, rtol=1e-10)
            @test isapprox(compute_ar1_mae_bound(0.5, 1.0; horizon=1), expected, rtol=1e-10)
            @test isapprox(compute_ar1_mae_bound(0.9, 1.0; horizon=1), expected, rtol=1e-10)
        end

        @testset "MAE scales with sigma" begin
            @test isapprox(
                compute_ar1_mae_bound(0.5, 2.0; horizon=1),
                2.0 * compute_ar1_mae_bound(0.5, 1.0; horizon=1),
                rtol=1e-10
            )
        end

        @testset "MAE increases with horizon" begin
            phi = 0.9
            sigma = 1.0

            mae1 = compute_ar1_mae_bound(phi, sigma; horizon=1)
            mae5 = compute_ar1_mae_bound(phi, sigma; horizon=5)
            mae10 = compute_ar1_mae_bound(phi, sigma; horizon=10)

            @test mae1 < mae5 < mae10
        end

        @testset "Invalid sigma error" begin
            @test_throws ErrorException compute_ar1_mae_bound(0.5, 0.0)
            @test_throws ErrorException compute_ar1_mae_bound(0.5, -1.0)
        end
    end

    # ==========================================================================
    # compute_ar1_rmse_bound
    # ==========================================================================

    @testset "compute_ar1_rmse_bound" begin
        @testset "RMSE = sqrt(MSE)" begin
            phi = 0.9
            sigma = 1.0

            for h in [1, 3, 5]
                mse = compute_ar1_mse_bound(phi, sigma^2; horizon=h)
                rmse = compute_ar1_rmse_bound(phi, sigma; horizon=h)
                @test isapprox(rmse, sqrt(mse), rtol=1e-10)
            end
        end
    end

    # ==========================================================================
    # compute_ar2_mse_bound
    # ==========================================================================

    @testset "compute_ar2_mse_bound" begin
        @testset "h=1 returns sigma_sq" begin
            # For any AR(2), 1-step MSE = sigma^2
            @test compute_ar2_mse_bound(0.5, 0.2, 1.0; horizon=1) == 1.0
            @test compute_ar2_mse_bound(0.3, -0.1, 1.0; horizon=1) == 1.0
        end

        @testset "MSE increases with horizon" begin
            phi1, phi2 = 0.5, 0.2
            sigma_sq = 1.0

            mse1 = compute_ar2_mse_bound(phi1, phi2, sigma_sq; horizon=1)
            mse2 = compute_ar2_mse_bound(phi1, phi2, sigma_sq; horizon=2)
            mse5 = compute_ar2_mse_bound(phi1, phi2, sigma_sq; horizon=5)

            @test mse1 < mse2 < mse5
        end

        @testset "Non-stationary coefficients error" begin
            # Violates phi1 + phi2 < 1
            @test_throws ErrorException compute_ar2_mse_bound(0.6, 0.5, 1.0)
            # Violates phi2 - phi1 < 1
            @test_throws ErrorException compute_ar2_mse_bound(-0.5, 0.6, 1.0)
            # Violates |phi2| < 1
            @test_throws ErrorException compute_ar2_mse_bound(0.5, 1.0, 1.0)
        end

        @testset "Invalid inputs error" begin
            @test_throws ErrorException compute_ar2_mse_bound(0.5, 0.2, 0.0)
            @test_throws ErrorException compute_ar2_mse_bound(0.5, 0.2, 1.0; horizon=0)
        end
    end

    # ==========================================================================
    # generate_ar1_series
    # ==========================================================================

    @testset "generate_ar1_series" begin
        @testset "Correct length" begin
            series = generate_ar1_series(0.9, 1.0, 100; rng=rng)
            @test length(series) == 100
        end

        @testset "Reproducible with same RNG" begin
            rng1 = Random.MersenneTwister(42)
            rng2 = Random.MersenneTwister(42)

            series1 = generate_ar1_series(0.9, 1.0, 50; rng=rng1)
            series2 = generate_ar1_series(0.9, 1.0, 50; rng=rng2)

            @test series1 == series2
        end

        @testset "Approximate sample statistics" begin
            rng_stats = Random.MersenneTwister(123)
            phi, sigma = 0.7, 1.5
            n = 5000

            series = generate_ar1_series(phi, sigma, n; rng=rng_stats)

            # Expected variance = sigma^2 / (1 - phi^2)
            expected_var = sigma^2 / (1 - phi^2)
            sample_var = var(series)

            @test isapprox(sample_var, expected_var, rtol=0.15)

            # Mean should be approximately 0
            @test isapprox(mean(series), 0.0, atol=0.5)
        end

        @testset "Approximate sample autocorrelation" begin
            rng_acf = Random.MersenneTwister(456)
            phi = 0.8
            sigma = 1.0
            n = 10000

            series = generate_ar1_series(phi, sigma, n; rng=rng_acf)

            # Lag-1 autocorrelation should approximate phi
            acf1 = cor(series[1:end-1], series[2:end])
            @test isapprox(acf1, phi, rtol=0.1)
        end

        @testset "Non-stationary phi error" begin
            @test_throws ErrorException generate_ar1_series(1.0, 1.0, 100)
            @test_throws ErrorException generate_ar1_series(-1.0, 1.0, 100)
        end

        @testset "Invalid inputs error" begin
            @test_throws ErrorException generate_ar1_series(0.5, 0.0, 100)
            @test_throws ErrorException generate_ar1_series(0.5, -1.0, 100)
            @test_throws ErrorException generate_ar1_series(0.5, 1.0, 0)
        end
    end

    # ==========================================================================
    # generate_ar2_series
    # ==========================================================================

    @testset "generate_ar2_series" begin
        @testset "Correct length" begin
            series = generate_ar2_series(0.5, 0.2, 1.0, 100; rng=rng)
            @test length(series) == 100
        end

        @testset "Reproducible with same RNG" begin
            rng1 = Random.MersenneTwister(42)
            rng2 = Random.MersenneTwister(42)

            series1 = generate_ar2_series(0.5, 0.2, 1.0, 50; rng=rng1)
            series2 = generate_ar2_series(0.5, 0.2, 1.0, 50; rng=rng2)

            @test series1 == series2
        end

        @testset "Non-stationary coefficients error" begin
            @test_throws ErrorException generate_ar2_series(0.6, 0.5, 1.0, 100)
        end

        @testset "Invalid inputs error" begin
            @test_throws ErrorException generate_ar2_series(0.5, 0.2, 0.0, 100)
            @test_throws ErrorException generate_ar2_series(0.5, 0.2, 1.0, 1)
        end
    end

    # ==========================================================================
    # estimate_ar1_params
    # ==========================================================================

    @testset "estimate_ar1_params" begin
        @testset "Recovers true parameters approximately" begin
            true_phi, true_sigma = 0.8, 1.0
            rng_est = Random.MersenneTwister(789)
            series = generate_ar1_series(true_phi, true_sigma, 1000; rng=rng_est)

            phi_hat, sigma_hat = estimate_ar1_params(series)

            @test isapprox(phi_hat, true_phi, atol=0.1)
            @test isapprox(sigma_hat, true_sigma, rtol=0.2)
        end

        @testset "Clamps to stationary region" begin
            # Create a near-unit-root series
            rng_ur = Random.MersenneTwister(101)
            series = generate_ar1_series(0.999, 1.0, 100; rng=rng_ur)

            phi_hat, _ = estimate_ar1_params(series)
            @test abs(phi_hat) < 1.0
        end

        @testset "Handles constant series" begin
            constant = fill(5.0, 100)
            phi_hat, sigma_hat = estimate_ar1_params(constant)
            @test phi_hat == 0.0  # Division by zero case
            @test sigma_hat == 0.0  # No variance
        end

        @testset "Insufficient data error" begin
            @test_throws ErrorException estimate_ar1_params([1.0, 2.0])
        end
    end

    # ==========================================================================
    # check_against_ar1_bounds
    # ==========================================================================

    @testset "check_against_ar1_bounds" begin
        @testset "HALT when beating bounds" begin
            # Model MSE = 0.5, theoretical = 1.0, ratio = 0.5 < 0.667
            result = check_against_ar1_bounds(
                model_mse=0.5, phi=0.9, sigma_sq=1.0
            )
            @test result.status == :HALT
            @test result.ratio ≈ 0.5
            @test result.theoretical_bound ≈ 1.0
        end

        @testset "WARN when suspiciously close" begin
            # Model MSE = 1.1, theoretical = 1.0, ratio = 1.1 < 1.2
            result = check_against_ar1_bounds(
                model_mse=1.1, phi=0.9, sigma_sq=1.0
            )
            @test result.status == :WARN
        end

        @testset "PASS when within expected range" begin
            # Model MSE = 1.5, theoretical = 1.0, ratio = 1.5 > 1.2
            result = check_against_ar1_bounds(
                model_mse=1.5, phi=0.9, sigma_sq=1.0
            )
            @test result.status == :PASS
        end

        @testset "SKIP when cannot compute bounds" begin
            # Non-stationary phi
            result = check_against_ar1_bounds(
                model_mse=1.0, phi=1.0, sigma_sq=1.0
            )
            @test result.status == :SKIP
        end

        @testset "Custom tolerance" begin
            # With tolerance=2.0, threshold = 0.5
            result_strict = check_against_ar1_bounds(
                model_mse=0.6, phi=0.9, sigma_sq=1.0, tolerance=1.5
            )
            result_loose = check_against_ar1_bounds(
                model_mse=0.6, phi=0.9, sigma_sq=1.0, tolerance=2.0
            )

            @test result_strict.status == :HALT
            @test result_loose.status == :WARN  # 0.6 > 0.5 threshold
        end

        @testset "Multi-horizon" begin
            # At h=5, theoretical MSE is higher
            result_h1 = check_against_ar1_bounds(
                model_mse=1.5, phi=0.9, sigma_sq=1.0, horizon=1
            )
            result_h5 = check_against_ar1_bounds(
                model_mse=1.5, phi=0.9, sigma_sq=1.0, horizon=5
            )

            # Same model MSE, but h=5 has higher theoretical bound
            @test result_h5.theoretical_bound > result_h1.theoretical_bound
        end
    end

    # ==========================================================================
    # check_against_ar2_bounds
    # ==========================================================================

    @testset "check_against_ar2_bounds" begin
        @testset "HALT when beating bounds" begin
            result = check_against_ar2_bounds(
                model_mse=0.5, phi1=0.5, phi2=0.2, sigma_sq=1.0
            )
            @test result.status == :HALT
        end

        @testset "PASS when within range" begin
            result = check_against_ar2_bounds(
                model_mse=1.5, phi1=0.5, phi2=0.2, sigma_sq=1.0
            )
            @test result.status == :PASS
        end

        @testset "SKIP for non-stationary" begin
            result = check_against_ar2_bounds(
                model_mse=1.0, phi1=0.6, phi2=0.5, sigma_sq=1.0
            )
            @test result.status == :SKIP
        end
    end

    # ==========================================================================
    # AR1Bounds and AR2Bounds convenience functions
    # ==========================================================================

    @testset "compute_ar1_bounds" begin
        bounds = compute_ar1_bounds(0.9, 1.0; horizon=1)

        @test bounds isa AR1Bounds
        @test bounds.mse ≈ 1.0
        @test bounds.rmse ≈ 1.0
        @test bounds.mae ≈ sqrt(2/pi)
        @test bounds.phi ≈ 0.9
        @test bounds.sigma ≈ 1.0
        @test bounds.horizon == 1
    end

    @testset "compute_ar2_bounds" begin
        bounds = compute_ar2_bounds(0.5, 0.2, 1.0; horizon=1)

        @test bounds isa AR2Bounds
        @test bounds.mse ≈ 1.0
        @test bounds.rmse ≈ 1.0
        @test bounds.mae ≈ sqrt(2/pi)
        @test bounds.phi1 ≈ 0.5
        @test bounds.phi2 ≈ 0.2
        @test bounds.sigma ≈ 1.0
        @test bounds.horizon == 1
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Synthetic AR(1) validation" begin
        rng_int = Random.MersenneTwister(999)
        phi, sigma = 0.85, 1.2
        n = 500

        # Generate AR(1) series
        series = generate_ar1_series(phi, sigma, n; rng=rng_int)

        # Estimate parameters
        phi_hat, sigma_hat = estimate_ar1_params(series)

        # Compute bounds with estimated params
        bounds = compute_ar1_bounds(phi_hat, sigma_hat; horizon=1)

        @test isfinite(bounds.mse)
        @test isfinite(bounds.mae)
        @test isfinite(bounds.rmse)

        # Simple 1-step forecast (persistence)
        predictions = series[1:end-1]
        actuals = series[2:end]
        errors = predictions .- actuals
        model_mse = mean(errors .^ 2)

        # Check against bounds
        result = check_against_ar1_bounds(
            model_mse=model_mse,
            phi=phi_hat,
            sigma_sq=sigma_hat^2,
            horizon=1
        )

        # Persistence should not beat theoretical bounds
        @test result.status in [:PASS, :WARN]
    end

    @testset "Integration: Multi-horizon bounds" begin
        phi = 0.9
        sigma = 1.0

        # Bounds should increase with horizon
        bounds_h1 = compute_ar1_bounds(phi, sigma; horizon=1)
        bounds_h3 = compute_ar1_bounds(phi, sigma; horizon=3)
        bounds_h5 = compute_ar1_bounds(phi, sigma; horizon=5)

        @test bounds_h1.mse < bounds_h3.mse < bounds_h5.mse
        @test bounds_h1.mae < bounds_h3.mae < bounds_h5.mae
        @test bounds_h1.rmse < bounds_h3.rmse < bounds_h5.rmse
    end
end
