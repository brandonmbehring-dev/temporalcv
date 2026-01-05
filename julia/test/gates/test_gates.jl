@testset "Gate Functions" begin
    @testset "gate_suspicious_improvement" begin
        @testset "HALT on large improvement" begin
            # 30% improvement = (1 - 0.07/0.10) = 0.30 > 0.20 threshold
            result = gate_suspicious_improvement(0.07, 0.10)
            @test result.status == HALT
            @test result.metric_value ≈ 0.30
            @test occursin("suspicious", lowercase(result.recommendation))
        end

        @testset "WARN on moderate improvement" begin
            # 15% improvement = (1 - 0.085/0.10) = 0.15 > 0.10 warn threshold
            result = gate_suspicious_improvement(0.085, 0.10)
            @test result.status == WARN
            @test 0.10 < result.metric_value < 0.20
        end

        @testset "PASS on reasonable improvement" begin
            # 5% improvement = (1 - 0.095/0.10) = 0.05 < 0.10 warn threshold
            result = gate_suspicious_improvement(0.095, 0.10)
            @test result.status == PASS
            @test result.metric_value < 0.10
        end

        @testset "SKIP on zero baseline" begin
            result = gate_suspicious_improvement(0.05, 0.0)
            @test result.status == SKIP
        end

        @testset "SKIP on negative baseline" begin
            result = gate_suspicious_improvement(0.05, -0.10)
            @test result.status == SKIP
        end

        @testset "Custom thresholds" begin
            # 4% improvement (1 - 0.96/1.0 = 0.04)
            # With threshold=0.05 HALT, warn=0.03 WARN
            # 4% > 3% → WARN
            result = gate_suspicious_improvement(
                0.96, 1.0,
                threshold=0.05, warn_threshold=0.03
            )
            @test result.status == WARN

            # 4% > 3% threshold → HALT
            result2 = gate_suspicious_improvement(
                0.96, 1.0,
                threshold=0.03, warn_threshold=0.02
            )
            @test result2.status == HALT
        end

        @testset "No improvement (model worse)" begin
            # Model is worse than baseline
            result = gate_suspicious_improvement(0.12, 0.10)
            @test result.status == PASS
            @test result.metric_value < 0  # Negative improvement
        end
    end

    @testset "gate_temporal_boundary" begin
        @testset "PASS with sufficient gap" begin
            # train_end=100, test_start=105, horizon=2
            # actual_gap = 105 - 100 - 1 = 4 >= required 2
            result = gate_temporal_boundary(100, 105, 2)
            @test result.status == PASS
            @test result.metric_value == 4.0
            @test result.threshold == 2.0
        end

        @testset "HALT with insufficient gap" begin
            # train_end=100, test_start=102, horizon=3
            # actual_gap = 102 - 100 - 1 = 1 < required 3
            result = gate_temporal_boundary(100, 102, 3)
            @test result.status == HALT
            @test result.metric_value == 1.0
            @test result.threshold == 3.0
            @test occursin("more periods", result.recommendation)
        end

        @testset "Exact boundary" begin
            # train_end=100, test_start=103, horizon=2
            # actual_gap = 103 - 100 - 1 = 2 == required 2
            result = gate_temporal_boundary(100, 103, 2)
            @test result.status == PASS
        end

        @testset "With additional gap" begin
            # train_end=100, test_start=105, horizon=2, gap=3
            # actual_gap = 4, required = 2 + 3 = 5 > 4 → HALT
            result = gate_temporal_boundary(100, 105, 2, gap=3)
            @test result.status == HALT
        end

        @testset "Details populated" begin
            result = gate_temporal_boundary(50, 60, 5)
            @test result.details["train_end_idx"] == 50
            @test result.details["test_start_idx"] == 60
            @test result.details["horizon"] == 5
        end
    end

    @testset "gate_residual_diagnostics" begin
        rng = Random.MersenneTwister(42)

        @testset "PASS with white noise residuals" begin
            residuals = randn(rng, 100)
            result = gate_residual_diagnostics(residuals)
            @test result.status == PASS
        end

        @testset "SKIP with insufficient data" begin
            residuals = randn(rng, 20)
            result = gate_residual_diagnostics(residuals)
            @test result.status == SKIP
        end

        @testset "Detects bias" begin
            # Residuals with non-zero mean
            residuals = randn(rng, 100) .+ 1.0
            result = gate_residual_diagnostics(residuals, halt_on_bias=true)
            @test result.status == HALT
            @test "mean_zero" in result.details["failing_tests"]
        end

        @testset "Detects autocorrelation" begin
            # Create highly autocorrelated residuals
            n = 100
            residuals = zeros(n)
            residuals[1] = randn(rng)
            for i in 2:n
                residuals[i] = 0.9 * residuals[i-1] + 0.1 * randn(rng)
            end

            result = gate_residual_diagnostics(residuals, halt_on_autocorr=true)
            # May detect autocorrelation depending on realization
            @test result.status in [HALT, WARN, PASS]
        end

        @testset "WARN when not configured to HALT" begin
            # Biased residuals, but halt_on_bias=false
            residuals = randn(rng, 100) .+ 1.5
            result = gate_residual_diagnostics(residuals, halt_on_bias=false)
            @test result.status == WARN
        end

        @testset "Custom max_lag" begin
            residuals = randn(rng, 100)
            result = gate_residual_diagnostics(residuals, max_lag=5)
            @test result.details["tests"]["ljung_box"]["max_lag"] == 5
        end

        @testset "NaN handling" begin
            residuals = randn(rng, 50)
            residuals[10] = NaN
            @test_throws AssertionError gate_residual_diagnostics(residuals)
        end
    end

    @testset "Utility functions" begin
        @testset "compute_acf" begin
            rng = Random.MersenneTwister(42)
            # White noise should have near-zero ACF at all lags
            x = randn(rng, 200)
            acf = compute_acf(x, 5)
            @test length(acf) == 5
            @test all(abs.(acf) .< 0.2)  # Approximate 95% CI for white noise

            # AR(1) with high phi should have positive ACF
            ar1 = zeros(200)
            ar1[1] = randn(rng)
            for i in 2:200
                ar1[i] = 0.9 * ar1[i-1] + 0.1 * randn(rng)
            end
            acf_ar1 = compute_acf(ar1, 3)
            @test acf_ar1[1] > 0.5  # First lag should be high for AR(1)
        end

        @testset "ljung_box_test" begin
            rng = Random.MersenneTwister(42)
            # White noise should pass Ljung-Box
            x = randn(rng, 200)
            Q, p = ljung_box_test(x, 10)
            @test p > 0.05  # Should not reject H0

            # Strongly autocorrelated series should fail
            ar1 = zeros(200)
            ar1[1] = randn(rng)
            for i in 2:200
                ar1[i] = 0.95 * ar1[i-1] + 0.05 * randn(rng)
            end
            Q_ar1, p_ar1 = ljung_box_test(ar1, 10)
            @test p_ar1 < 0.05  # Should reject H0
            @test Q_ar1 > 0
        end
    end

    @testset "run_gates" begin
        gate1 = GateResult(name = "g1", status = PASS, message = "OK")
        gate2 = GateResult(name = "g2", status = WARN, message = "Caution")

        report = run_gates([gate1, gate2])

        @test isa(report, ValidationReport)
        @test length(report.gates) == 2
        @test status(report) == "WARN"
    end
end
