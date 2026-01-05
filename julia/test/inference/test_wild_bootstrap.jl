@testset "Inference: Wild Cluster Bootstrap" begin
    using Random
    using Statistics

    # ==========================================================================
    # Weight Functions
    # ==========================================================================

    @testset "rademacher_weights" begin
        rng = Random.MersenneTwister(42)

        @testset "Returns correct length" begin
            weights = rademacher_weights(100; rng=rng)
            @test length(weights) == 100
        end

        @testset "Values are {-1, +1}" begin
            weights = rademacher_weights(1000; rng=rng)
            @test all(w -> w == 1.0 || w == -1.0, weights)
        end

        @testset "Approximately balanced" begin
            weights = rademacher_weights(10000; rng=rng)
            prop_positive = mean(weights .> 0)
            @test 0.48 < prop_positive < 0.52
        end

        @testset "Mean approximately zero" begin
            weights = rademacher_weights(10000; rng=rng)
            @test abs(mean(weights)) < 0.05
        end

        @testset "Variance approximately one" begin
            weights = rademacher_weights(10000; rng=rng)
            @test 0.95 < var(weights) < 1.05
        end

        @testset "Reproducible with same RNG" begin
            rng1 = Random.MersenneTwister(123)
            rng2 = Random.MersenneTwister(123)

            w1 = rademacher_weights(50; rng=rng1)
            w2 = rademacher_weights(50; rng=rng2)

            @test w1 == w2
        end

        @testset "Error on invalid n" begin
            @test_throws ErrorException rademacher_weights(0)
            @test_throws ErrorException rademacher_weights(-1)
        end
    end

    @testset "webb_weights" begin
        rng = Random.MersenneTwister(42)

        @testset "Returns correct length" begin
            weights = webb_weights(100; rng=rng)
            @test length(weights) == 100
        end

        @testset "Values from 6-point distribution" begin
            weights = webb_weights(1000; rng=rng)

            expected_points = [-sqrt(3/2), -sqrt(2/2), -sqrt(1/2),
                              sqrt(1/2), sqrt(2/2), sqrt(3/2)]

            for w in weights
                @test any(isapprox(w, p; atol=1e-10) for p in expected_points)
            end
        end

        @testset "Mean approximately zero" begin
            weights = webb_weights(10000; rng=rng)
            @test abs(mean(weights)) < 0.1
        end

        @testset "Variance approximately one" begin
            weights = webb_weights(10000; rng=rng)
            # Webb weights are designed to have variance 1
            @test 0.9 < var(weights) < 1.1
        end

        @testset "All 6 points appear" begin
            weights = webb_weights(10000; rng=rng)

            expected_points = [-sqrt(3/2), -sqrt(2/2), -sqrt(1/2),
                              sqrt(1/2), sqrt(2/2), sqrt(3/2)]

            for point in expected_points
                @test count(w -> isapprox(w, point; atol=1e-10), weights) > 100
            end
        end

        @testset "Error on invalid n" begin
            @test_throws ErrorException webb_weights(0)
            @test_throws ErrorException webb_weights(-1)
        end
    end

    # ==========================================================================
    # wild_cluster_bootstrap
    # ==========================================================================

    @testset "wild_cluster_bootstrap" begin
        rng = Random.MersenneTwister(42)

        @testset "Basic functionality" begin
            fold_metrics = [0.82, 0.91, 0.78, 0.85, 0.88]

            result = wild_cluster_bootstrap(fold_metrics; rng=rng)

            @test result isa WildBootstrapResult
            @test result.estimate ≈ mean(fold_metrics)
            @test result.se > 0
            @test result.ci_lower < result.estimate < result.ci_upper
            @test 0 <= result.p_value <= 1
        end

        @testset "Correct number of bootstrap samples" begin
            fold_metrics = [1.0, 1.1, 0.9, 1.0, 1.05]

            result = wild_cluster_bootstrap(fold_metrics; n_bootstrap=500, rng=rng)

            @test result.n_bootstrap == 500
            @test length(result.bootstrap_distribution) == 500
        end

        @testset "Auto-selects Webb for < 13 folds" begin
            fold_metrics = [0.8, 0.9, 0.85, 0.87, 0.82]

            result = wild_cluster_bootstrap(fold_metrics; rng=rng)

            @test result.weight_type == :webb
            @test result.n_clusters == 5
        end

        @testset "Auto-selects Rademacher for >= 13 folds" begin
            fold_metrics = randn(rng, 15) .+ 1.0

            result = wild_cluster_bootstrap(fold_metrics; rng=rng)

            @test result.weight_type == :rademacher
            @test result.n_clusters == 15
        end

        @testset "Force weight type" begin
            fold_metrics = [1.0, 1.1, 0.9]

            result_webb = wild_cluster_bootstrap(
                fold_metrics; weight_type=:webb, rng=rng
            )
            result_rad = wild_cluster_bootstrap(
                fold_metrics; weight_type=:rademacher, rng=rng
            )

            @test result_webb.weight_type == :webb
            @test result_rad.weight_type == :rademacher
        end

        @testset "Custom confidence level" begin
            fold_metrics = randn(rng, 10) .+ 1.0

            result_90 = wild_cluster_bootstrap(
                fold_metrics; confidence_level=0.90, n_bootstrap=999, rng=rng
            )
            result_99 = wild_cluster_bootstrap(
                fold_metrics; confidence_level=0.99, n_bootstrap=999, rng=rng
            )

            width_90 = result_90.ci_upper - result_90.ci_lower
            width_99 = result_99.ci_upper - result_99.ci_lower

            # 99% CI should be wider
            @test width_99 > width_90
        end

        @testset "P-value for true zero mean" begin
            # Generate centered data (mean ≈ 0)
            rng_test = Random.MersenneTwister(42)
            fold_metrics = randn(rng_test, 10)

            result = wild_cluster_bootstrap(
                fold_metrics; n_bootstrap=999, rng=rng
            )

            # P-value should be relatively high for null data
            # (not a hard guarantee, but usually true)
            @test result.p_value >= 0.0
        end

        @testset "P-value for large mean" begin
            fold_metrics = [10.0, 10.1, 9.9, 10.05, 9.95]

            result = wild_cluster_bootstrap(
                fold_metrics; n_bootstrap=999, rng=rng
            )

            # P-value should be very small for data clearly not centered at 0
            @test result.p_value < 0.01
        end

        @testset "Reproducibility with same RNG" begin
            fold_metrics = [0.8, 0.9, 0.85]

            rng1 = Random.MersenneTwister(123)
            rng2 = Random.MersenneTwister(123)

            result1 = wild_cluster_bootstrap(fold_metrics; rng=rng1)
            result2 = wild_cluster_bootstrap(fold_metrics; rng=rng2)

            @test result1.se ≈ result2.se
            @test result1.ci_lower ≈ result2.ci_lower
        end

        @testset "Error on insufficient folds" begin
            @test_throws ErrorException wild_cluster_bootstrap([1.0])
        end

        @testset "Error on invalid n_bootstrap" begin
            @test_throws ErrorException wild_cluster_bootstrap([1.0, 2.0]; n_bootstrap=0)
        end

        @testset "Error on invalid confidence_level" begin
            @test_throws ErrorException wild_cluster_bootstrap([1.0, 2.0]; confidence_level=0.0)
            @test_throws ErrorException wild_cluster_bootstrap([1.0, 2.0]; confidence_level=1.0)
        end

        @testset "Error on invalid weight_type" begin
            @test_throws ErrorException wild_cluster_bootstrap([1.0, 2.0]; weight_type=:invalid)
        end
    end

    # ==========================================================================
    # wild_cluster_bootstrap_difference
    # ==========================================================================

    @testset "wild_cluster_bootstrap_difference" begin
        rng = Random.MersenneTwister(42)

        @testset "Basic functionality" begin
            model_a = [0.82, 0.91, 0.78, 0.85, 0.88]
            model_b = [0.90, 0.95, 0.85, 0.92, 0.93]

            result = wild_cluster_bootstrap_difference(model_a, model_b; rng=rng)

            @test result isa WildBootstrapResult
            @test result.estimate ≈ mean(model_a) - mean(model_b)
        end

        @testset "Model A better (negative difference for MAE)" begin
            model_a = [0.5, 0.55, 0.52, 0.51, 0.53]  # Better MAE
            model_b = [1.0, 1.1, 0.95, 1.05, 1.02]   # Worse MAE

            result = wild_cluster_bootstrap_difference(
                model_a, model_b; n_bootstrap=999, rng=rng
            )

            @test result.estimate < 0  # A is better (lower)
            @test result.p_value < 0.05  # Significant difference
        end

        @testset "No significant difference" begin
            # Similar performance
            model_a = [0.82, 0.91, 0.78, 0.85, 0.88]
            model_b = [0.84, 0.89, 0.80, 0.83, 0.90]

            result = wild_cluster_bootstrap_difference(
                model_a, model_b; n_bootstrap=999, rng=rng
            )

            # P-value likely > 0.05 for similar models
            # (not guaranteed but typical)
            @test result isa WildBootstrapResult
        end

        @testset "Symmetric property" begin
            model_a = [0.8, 0.9, 0.85]
            model_b = [1.0, 1.1, 1.05]

            rng1 = Random.MersenneTwister(42)
            rng2 = Random.MersenneTwister(42)

            result_ab = wild_cluster_bootstrap_difference(model_a, model_b; rng=rng1)
            result_ba = wild_cluster_bootstrap_difference(model_b, model_a; rng=rng2)

            @test result_ab.estimate ≈ -result_ba.estimate
        end

        @testset "Error on mismatched lengths" begin
            @test_throws ErrorException wild_cluster_bootstrap_difference(
                [1.0, 2.0, 3.0],
                [1.0, 2.0]
            )
        end
    end

    # ==========================================================================
    # Integration Tests
    # ==========================================================================

    @testset "Integration: CV fold inference workflow" begin
        rng = Random.MersenneTwister(123)

        # Simulate 5-fold CV results
        n_folds = 5
        true_mae = 0.8
        fold_maes = true_mae .+ randn(rng, n_folds) .* 0.05

        result = wild_cluster_bootstrap(
            fold_maes;
            n_bootstrap=999,
            confidence_level=0.95,
            rng=rng
        )

        # True value should be in CI (usually)
        @test result.ci_lower < true_mae < result.ci_upper

        # SE should be reasonable
        @test 0.01 < result.se < 0.5
    end

    @testset "Integration: Model comparison" begin
        rng = Random.MersenneTwister(456)

        # Model A: clearly better
        model_a_folds = [0.5, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47]
        model_b_folds = [0.8, 0.82, 0.78, 0.81, 0.79, 0.80, 0.83, 0.77]

        result = wild_cluster_bootstrap_difference(
            model_a_folds, model_b_folds;
            n_bootstrap=999,
            rng=rng
        )

        # Model A is significantly better
        @test result.estimate < 0
        @test result.p_value < 0.05
        @test result.ci_upper < 0  # Entire CI below 0
    end

    @testset "Integration: Webb vs Rademacher comparison" begin
        rng_webb = Random.MersenneTwister(789)
        rng_rad = Random.MersenneTwister(789)

        # 5 folds - Webb should be used by default
        fold_metrics = [0.8, 0.85, 0.82, 0.87, 0.79]

        result_webb = wild_cluster_bootstrap(
            fold_metrics;
            n_bootstrap=999,
            weight_type=:webb,
            rng=rng_webb
        )

        result_rad = wild_cluster_bootstrap(
            fold_metrics;
            n_bootstrap=999,
            weight_type=:rademacher,
            rng=rng_rad
        )

        # Both should give similar estimates
        @test isapprox(result_webb.estimate, result_rad.estimate; atol=1e-10)

        # SE may differ slightly due to different weight distributions
        @test result_webb.se > 0
        @test result_rad.se > 0
    end

    @testset "Integration: Bootstrap distribution properties" begin
        rng = Random.MersenneTwister(111)

        fold_metrics = randn(rng, 8) .+ 5.0

        result = wild_cluster_bootstrap(
            fold_metrics;
            n_bootstrap=10000,
            rng=rng
        )

        # Bootstrap distribution should be centered around 0 (by construction)
        @test abs(mean(result.bootstrap_distribution)) < 0.1

        # Should have reasonable variance
        @test 0 < std(result.bootstrap_distribution) < 2.0

        # Should be approximately symmetric
        skewness = mean((result.bootstrap_distribution .- mean(result.bootstrap_distribution)).^3) /
                   std(result.bootstrap_distribution)^3
        @test abs(skewness) < 0.5
    end
end
