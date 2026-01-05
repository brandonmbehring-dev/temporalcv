@testset "Guardrails: Validation Checks" begin
    using Random

    # ==========================================================================
    # Result Types
    # ==========================================================================

    @testset "Result types" begin
        @testset "pass_result" begin
            result = pass_result()

            @test result.passed == true
            @test isempty(result.warnings)
            @test isempty(result.errors)
            @test isempty(result.skipped)
        end

        @testset "pass_result with details" begin
            result = pass_result(details=Dict{Symbol, Any}(:value => 42))

            @test result.passed == true
            @test result.details[:value] == 42
        end

        @testset "fail_result" begin
            result = fail_result("Test failure")

            @test result.passed == false
            @test "Test failure" in result.errors
            @test isempty(result.warnings)
        end

        @testset "fail_result with recommendations" begin
            result = fail_result(
                "Error",
                recommendations=["Fix A", "Fix B"]
            )

            @test !result.passed
            @test "Fix A" in result.recommendations
            @test "Fix B" in result.recommendations
        end

        @testset "warn_result" begin
            result = warn_result("Something concerning")

            @test result.passed == true  # Warnings don't fail
            @test "Something concerning" in result.warnings
            @test isempty(result.errors)
        end

        @testset "skip_result" begin
            result = skip_result("Insufficient data")

            @test result.passed == true
            @test "Insufficient data" in result.skipped
        end
    end

    # ==========================================================================
    # check_suspicious_improvement
    # ==========================================================================

    @testset "check_suspicious_improvement" begin
        @testset "Clear pass (small improvement)" begin
            result = check_suspicious_improvement(0.95, 1.0)

            @test result.passed == true
            @test isempty(result.warnings)
            @test result.details[:improvement_pct] ≈ 5.0
        end

        @testset "Warning zone (10-20% improvement)" begin
            result = check_suspicious_improvement(0.85, 1.0)

            @test result.passed == true
            @test !isempty(result.warnings)
            @test result.details[:improvement_pct] ≈ 15.0
        end

        @testset "Fail (>20% improvement)" begin
            result = check_suspicious_improvement(0.75, 1.0)

            @test result.passed == false
            @test !isempty(result.errors)
            @test result.details[:improvement_pct] ≈ 25.0
        end

        @testset "Custom threshold" begin
            # 15% improvement fails with 10% threshold
            result = check_suspicious_improvement(0.85, 1.0; threshold=0.10)

            @test result.passed == false
        end

        @testset "Higher is better" begin
            # R² improved from 0.5 to 0.8 (60% improvement!)
            result = check_suspicious_improvement(
                0.8, 0.5;
                lower_is_better=false
            )

            @test result.passed == false
            @test result.details[:improvement_pct] ≈ 60.0
        end

        @testset "No improvement" begin
            result = check_suspicious_improvement(1.0, 1.0)

            @test result.passed == true
            @test result.details[:improvement_pct] ≈ 0.0
        end

        @testset "Model worse than baseline" begin
            result = check_suspicious_improvement(1.2, 1.0)

            @test result.passed == true
            @test result.details[:improvement_pct] ≈ -20.0
        end

        @testset "Zero baseline skipped" begin
            result = check_suspicious_improvement(0.5, 0.0)

            @test result.passed == true
            @test !isempty(result.skipped)
        end

        @testset "Recommendations on failure" begin
            result = check_suspicious_improvement(0.5, 1.0)

            @test !isempty(result.recommendations)
            @test any(occursin("leakage", r) for r in result.recommendations)
        end
    end

    # ==========================================================================
    # check_minimum_sample_size
    # ==========================================================================

    @testset "check_minimum_sample_size" begin
        @testset "Sufficient samples" begin
            result = check_minimum_sample_size(100)

            @test result.passed == true
            @test result.details[:n] == 100
        end

        @testset "Exactly at threshold" begin
            result = check_minimum_sample_size(50)

            @test result.passed == true
        end

        @testset "Insufficient samples" begin
            result = check_minimum_sample_size(30)

            @test result.passed == false
            @test !isempty(result.errors)
        end

        @testset "Custom minimum" begin
            result = check_minimum_sample_size(100; min_samples=200)

            @test result.passed == false
        end

        @testset "Recommendations on failure" begin
            result = check_minimum_sample_size(10)

            @test !isempty(result.recommendations)
        end
    end

    # ==========================================================================
    # check_stratified_sample_size
    # ==========================================================================

    @testset "check_stratified_sample_size" begin
        @testset "All strata sufficient" begin
            result = check_stratified_sample_size([50, 30, 20, 15])

            @test result.passed == true
            @test result.details[:n_strata] == 4
        end

        @testset "One stratum too small" begin
            result = check_stratified_sample_size([50, 30, 5, 15])

            @test result.passed == false
            @test 3 in result.details[:failing_strata]
        end

        @testset "Multiple strata too small" begin
            result = check_stratified_sample_size([50, 5, 3, 15])

            @test result.passed == false
            @test length(result.details[:failing_strata]) == 2
        end

        @testset "Custom minimum" begin
            result = check_stratified_sample_size([50, 30, 20]; min_per_stratum=25)

            @test result.passed == false
            @test 3 in result.details[:failing_strata]
        end

        @testset "Empty strata skipped" begin
            result = check_stratified_sample_size(Int[])

            @test result.passed == true
            @test !isempty(result.skipped)
        end

        @testset "Single stratum" begin
            result = check_stratified_sample_size([100])

            @test result.passed == true
            @test result.details[:n_strata] == 1
        end
    end

    # ==========================================================================
    # check_forecast_horizon_consistency
    # ==========================================================================

    @testset "check_forecast_horizon_consistency" begin
        @testset "Consistent degradation" begin
            # Reasonable: h=1 slightly better than h=2,3,4
            result = check_forecast_horizon_consistency([0.8, 1.0, 1.1, 1.2])

            @test result.passed == true
            @test isempty(result.warnings)
        end

        @testset "Suspicious h=1 advantage" begin
            # h=1 is 3x better than average of others
            result = check_forecast_horizon_consistency([0.5, 1.4, 1.5, 1.6])

            @test result.passed == true  # Warnings don't fail
            @test !isempty(result.warnings)
        end

        @testset "Custom ratio threshold" begin
            result = check_forecast_horizon_consistency(
                [0.5, 1.0, 1.1, 1.2];
                max_ratio=1.5
            )

            @test !isempty(result.warnings)
        end

        @testset "Higher is better metric" begin
            # R²: h=1 much better (suspicious)
            result = check_forecast_horizon_consistency(
                [0.9, 0.3, 0.35, 0.4];
                lower_is_better=false
            )

            @test !isempty(result.warnings)
        end

        @testset "Single horizon skipped" begin
            result = check_forecast_horizon_consistency([1.0])

            @test result.passed == true
            @test !isempty(result.skipped)
        end

        @testset "Recommendations on warning" begin
            result = check_forecast_horizon_consistency([0.1, 1.0, 1.0, 1.0])

            @test !isempty(result.recommendations)
            @test any(occursin("gap", r) for r in result.recommendations)
        end
    end

    # ==========================================================================
    # check_residual_autocorrelation
    # ==========================================================================

    @testset "check_residual_autocorrelation" begin
        rng = Random.MersenneTwister(42)

        @testset "White noise residuals pass" begin
            residuals = randn(rng, 200)

            result = check_residual_autocorrelation(residuals)

            @test result.passed == true
            # White noise should have low ACF
        end

        @testset "Autocorrelated residuals warn" begin
            # AR(1) residuals with high phi
            n = 200
            residuals = zeros(n)
            residuals[1] = randn(rng)
            for i in 2:n
                residuals[i] = 0.8 * residuals[i-1] + 0.2 * randn(rng)
            end

            result = check_residual_autocorrelation(residuals)

            @test !isempty(result.warnings)
            @test 1 in result.details[:significant_lags]
        end

        @testset "Custom threshold" begin
            residuals = randn(rng, 100)

            result = check_residual_autocorrelation(residuals; max_acf=0.05)

            # Stricter threshold may trigger more warnings
            @test result isa GuardrailResult
        end

        @testset "Insufficient data skipped" begin
            result = check_residual_autocorrelation([1.0, 2.0, 3.0])

            @test result.passed == true
            @test !isempty(result.skipped)
        end

        @testset "ACF values computed" begin
            residuals = randn(rng, 100)

            result = check_residual_autocorrelation(residuals; max_lag=5)

            @test length(result.details[:acf_values]) == 5
            @test all(isfinite, result.details[:acf_values])
        end

        @testset "Recommendations on warning" begin
            # Create autocorrelated residuals
            n = 100
            residuals = cumsum(randn(rng, n) .* 0.5)  # Random walk = high ACF

            result = check_residual_autocorrelation(residuals)

            if !isempty(result.warnings)
                @test !isempty(result.recommendations)
            end
        end
    end

    # ==========================================================================
    # run_all_guardrails
    # ==========================================================================

    @testset "run_all_guardrails" begin
        rng = Random.MersenneTwister(42)

        @testset "All checks pass" begin
            summary = run_all_guardrails(
                model_metric=0.95,
                baseline_metric=1.0,
                n_samples=200,
                strata_sizes=[50, 60, 40, 50],
                horizon_metrics=[1.0, 1.1, 1.2, 1.3],
                residuals=randn(rng, 100)
            )

            @test summary.passed == true
            @test summary.n_failed == 0
            @test isempty(summary.all_errors)
        end

        @testset "One check fails" begin
            summary = run_all_guardrails(
                model_metric=0.5,  # 50% improvement - suspicious!
                baseline_metric=1.0,
                n_samples=200
            )

            @test summary.passed == false
            @test summary.n_failed >= 1
            @test !isempty(summary.all_errors)
        end

        @testset "Multiple checks fail" begin
            summary = run_all_guardrails(
                model_metric=0.5,
                baseline_metric=1.0,
                n_samples=10,
                strata_sizes=[5, 3, 2]
            )

            @test summary.passed == false
            @test summary.n_failed >= 2
        end

        @testset "Warnings don't fail overall" begin
            summary = run_all_guardrails(
                model_metric=0.85,  # 15% improvement - warning
                baseline_metric=1.0,
                n_samples=100
            )

            @test summary.passed == true
            @test summary.n_warnings >= 1
            @test !isempty(summary.all_warnings)
        end

        @testset "Partial inputs" begin
            # Only run sample size check
            summary = run_all_guardrails(n_samples=100)

            @test :minimum_sample_size in keys(summary.results)
            @test !haskey(summary.results, :suspicious_improvement)
        end

        @testset "Custom thresholds" begin
            summary = run_all_guardrails(
                model_metric=0.85,
                baseline_metric=1.0,
                improvement_threshold=0.10  # Stricter
            )

            @test summary.passed == false
        end

        @testset "No inputs skips all" begin
            summary = run_all_guardrails()

            @test summary.passed == true
            @test isempty(summary.results)
        end

        @testset "Results contain all run checks" begin
            summary = run_all_guardrails(
                model_metric=0.9,
                baseline_metric=1.0,
                n_samples=100,
                residuals=randn(rng, 50)
            )

            @test haskey(summary.results, :suspicious_improvement)
            @test haskey(summary.results, :minimum_sample_size)
            @test haskey(summary.results, :residual_autocorrelation)
        end
    end

    # ==========================================================================
    # Integration Tests
    # ==========================================================================

    @testset "Integration: Full validation workflow" begin
        rng = Random.MersenneTwister(123)

        # Simulate a model that might be overfitting
        n = 200
        actuals = randn(rng, n)
        predictions = actuals .+ randn(rng, n) .* 0.3  # Very good predictions
        residuals = actuals .- predictions

        model_mae = mean(abs.(residuals))
        baseline_mae = std(actuals)  # Naive baseline

        summary = run_all_guardrails(
            model_metric=model_mae,
            baseline_metric=baseline_mae,
            n_samples=n,
            residuals=residuals
        )

        @test summary isa GuardrailSummary

        # Check details are populated
        if haskey(summary.results, :suspicious_improvement)
            @test haskey(summary.results[:suspicious_improvement].details, :improvement_pct)
        end
    end

    @testset "Integration: Multi-horizon validation" begin
        # Simulate metrics across horizons
        horizon_metrics = [0.5, 1.2, 1.4, 1.5, 1.6]  # h=1 suspiciously good

        result = check_forecast_horizon_consistency(horizon_metrics)

        @test !isempty(result.warnings)

        # This should be flagged in full validation
        summary = run_all_guardrails(
            horizon_metrics=horizon_metrics,
            n_samples=100
        )

        @test summary.n_warnings >= 1
    end

    @testset "Integration: Stratified model evaluation" begin
        # Different performance by regime
        high_vol_size = 50
        low_vol_size = 150
        small_regime_size = 8  # Too small!

        strata_sizes = [high_vol_size, low_vol_size, small_regime_size]

        summary = run_all_guardrails(
            strata_sizes=strata_sizes,
            n_samples=sum(strata_sizes)
        )

        # Small regime should fail stratified check
        @test summary.passed == false
        @test haskey(summary.results, :stratified_sample_size)
    end
end
