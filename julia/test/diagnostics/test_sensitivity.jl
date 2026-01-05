@testset "Sensitivity Analysis" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_parameter_sensitivity
    # ==========================================================================

    @testset "compute_parameter_sensitivity" begin
        @testset "Linear relationship" begin
            # Metric = 2 * param, so sensitivity = 2
            result = compute_parameter_sensitivity(
                10.0,
                x -> 2 * x;
                perturbation=0.1,
                n_points=5
            )

            @test result isa SensitivityResult
            @test result.base_value ≈ 10.0
            @test result.base_metric ≈ 20.0
            @test isapprox(result.sensitivity, 2.0, rtol=0.01)
        end

        @testset "Quadratic relationship" begin
            # Metric = param^2, sensitivity at x=10 is d/dx(x^2) = 2x = 20
            result = compute_parameter_sensitivity(
                10.0,
                x -> x^2;
                perturbation=0.05,  # Small perturbation for better linear approx
                n_points=5
            )

            @test isapprox(result.sensitivity, 20.0, rtol=0.1)
        end

        @testset "Constant function has zero sensitivity" begin
            result = compute_parameter_sensitivity(
                5.0,
                x -> 42.0;
                perturbation=0.2
            )

            @test result.sensitivity ≈ 0.0
        end

        @testset "Perturbed values are correct" begin
            base = 10.0
            pert = 0.2  # ±20%

            result = compute_parameter_sensitivity(base, x -> x; perturbation=pert, n_points=5)

            @test minimum(result.perturbed_values) ≈ base * (1 - pert)
            @test maximum(result.perturbed_values) ≈ base * (1 + pert)
            @test length(result.perturbed_values) == 5
        end

        @testset "Custom parameter name" begin
            result = compute_parameter_sensitivity(
                5.0,
                x -> x;
                parameter_name=:window_size
            )

            @test result.parameter == :window_size
        end

        @testset "Handles function errors gracefully" begin
            # Function that fails for some values
            result = compute_parameter_sensitivity(
                10.0,
                x -> x < 9.5 ? error("too small") : x;
                perturbation=0.1
            )

            # Should still compute something
            @test isfinite(result.sensitivity) || isnan(result.sensitivity)
        end

        @testset "Minimum n_points validation" begin
            @test_throws ErrorException compute_parameter_sensitivity(
                5.0, x -> x; n_points=1
            )
        end
    end

    # ==========================================================================
    # compute_stability_report
    # ==========================================================================

    @testset "compute_stability_report" begin
        @testset "Multiple parameters" begin
            params = Dict(:a => 10.0, :b => 5.0, :c => 2.0)

            report = compute_stability_report(
                params,
                p -> p[:a] * 2 + p[:b] + p[:c]^2
            )

            @test report isa StabilityReport
            @test haskey(report.sensitivities, :a)
            @test haskey(report.sensitivities, :b)
            @test haskey(report.sensitivities, :c)
        end

        @testset "Identifies most sensitive parameter" begin
            # f = 10*a + b + c, so most sensitive is :a
            params = Dict(:a => 1.0, :b => 1.0, :c => 1.0)

            report = compute_stability_report(
                params,
                p -> 10 * p[:a] + p[:b] + p[:c]
            )

            @test report.most_sensitive == :a
        end

        @testset "Stability score range" begin
            params = Dict(:x => 5.0)

            report = compute_stability_report(params, p -> p[:x] * 2)

            @test 0.0 <= report.stability_score <= 1.0
        end

        @testset "Total variation computed" begin
            params = Dict(:a => 1.0, :b => 2.0)

            report = compute_stability_report(
                params,
                p -> p[:a] * 3 + p[:b] * 2
            )

            expected_total = abs(report.sensitivities[:a].sensitivity) +
                             abs(report.sensitivities[:b].sensitivity)
            @test report.total_variation ≈ expected_total
        end

        @testset "Constant function is perfectly stable" begin
            params = Dict(:a => 1.0, :b => 2.0)

            report = compute_stability_report(params, p -> 42.0)

            @test report.stability_score ≈ 1.0
            @test report.total_variation ≈ 0.0
        end
    end

    # ==========================================================================
    # bootstrap_metric_variance
    # ==========================================================================

    @testset "bootstrap_metric_variance" begin
        @testset "Returns point estimate, SE, and CI" begin
            errors = randn(rng, 100)

            estimate, se, ci = bootstrap_metric_variance(
                errors,
                e -> mean(abs.(e));
                n_bootstrap=500,
                rng=rng
            )

            @test isfinite(estimate)
            @test isfinite(se)
            @test se >= 0
            @test length(ci) == 2
            @test ci[1] <= estimate <= ci[2]
        end

        @testset "Larger samples have smaller SE" begin
            rng1 = Random.MersenneTwister(42)
            rng2 = Random.MersenneTwister(42)

            small_errors = randn(rng1, 50)
            large_errors = randn(rng2, 500)

            _, se_small, _ = bootstrap_metric_variance(
                small_errors, mean; n_bootstrap=500, rng=rng1
            )
            _, se_large, _ = bootstrap_metric_variance(
                large_errors, mean; n_bootstrap=500, rng=rng2
            )

            @test se_large < se_small
        end

        @testset "Block bootstrap with block_size > 1" begin
            errors = randn(rng, 100)

            estimate, se, ci = bootstrap_metric_variance(
                errors,
                mean;
                n_bootstrap=500,
                block_size=5,
                rng=rng
            )

            @test isfinite(estimate)
            @test isfinite(se)
        end

        @testset "Custom confidence level" begin
            errors = randn(rng, 100)

            _, _, ci_90 = bootstrap_metric_variance(
                errors, mean;
                n_bootstrap=1000,
                confidence_level=0.90,
                rng=rng
            )

            _, _, ci_99 = bootstrap_metric_variance(
                errors, mean;
                n_bootstrap=1000,
                confidence_level=0.99,
                rng=rng
            )

            # 99% CI should be wider than 90% CI
            width_90 = ci_90[2] - ci_90[1]
            width_99 = ci_99[2] - ci_99[1]
            @test width_99 >= width_90 * 0.9  # Allow some bootstrap variance
        end

        @testset "MAE metric" begin
            true_mae = 1.0
            errors = randn(rng, 1000) .* true_mae * sqrt(pi / 2)  # Adjust for expected MAE

            estimate, se, ci = bootstrap_metric_variance(
                errors,
                e -> mean(abs.(e));
                n_bootstrap=500,
                rng=rng
            )

            @test isapprox(estimate, true_mae, rtol=0.2)
            @test ci[1] < true_mae < ci[2]  # True value in CI
        end

        @testset "Insufficient data error" begin
            @test_throws ErrorException bootstrap_metric_variance([1.0], mean)
        end

        @testset "Reproducible with same RNG" begin
            errors = randn(rng, 50)

            rng1 = Random.MersenneTwister(123)
            rng2 = Random.MersenneTwister(123)

            est1, _, _ = bootstrap_metric_variance(errors, mean; n_bootstrap=100, rng=rng1)
            est2, _, _ = bootstrap_metric_variance(errors, mean; n_bootstrap=100, rng=rng2)

            @test est1 ≈ est2
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Parameter sweep analysis" begin
        rng_int = Random.MersenneTwister(789)

        # Simulate a scenario where metric depends on multiple parameters
        # f(a, b) = a^2 / b, so:
        # df/da = 2a/b, df/db = -a^2/b^2
        params = Dict(:a => 10.0, :b => 5.0)

        report = compute_stability_report(
            params,
            p -> p[:a]^2 / p[:b];
            perturbation=0.05  # Small perturbation for linear approx
        )

        # Expected sensitivities at a=10, b=5:
        # df/da = 2*10/5 = 4
        # df/db = -100/25 = -4
        @test isapprox(report.sensitivities[:a].sensitivity, 4.0, rtol=0.2)
        @test isapprox(report.sensitivities[:b].sensitivity, -4.0, rtol=0.2)

        # Both should have similar magnitude
        @test abs(report.sensitivities[:a].sensitivity) ≈
              abs(report.sensitivities[:b].sensitivity) rtol=0.5
    end

    @testset "Integration: Bootstrap + sensitivity" begin
        rng_int = Random.MersenneTwister(456)
        n = 200

        # Generate errors
        errors = randn(rng_int, n)

        # Bootstrap the MAE
        estimate, se, ci = bootstrap_metric_variance(
            errors,
            e -> mean(abs.(e));
            n_bootstrap=500,
            block_size=5,
            rng=rng_int
        )

        # Verify reasonable results
        @test isfinite(estimate)
        @test isfinite(se)
        @test ci[1] < ci[2]

        # Sensitivity of MAE to scaling
        result = compute_parameter_sensitivity(
            1.0,  # Scale factor
            s -> mean(abs.(s .* errors))
        )

        # Sensitivity should be approximately the mean absolute error
        @test isapprox(result.sensitivity, estimate, rtol=0.3)
    end
end
