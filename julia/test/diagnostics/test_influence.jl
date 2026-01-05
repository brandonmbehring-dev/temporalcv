@testset "Influence Diagnostics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_dm_influence
    # ==========================================================================

    @testset "compute_dm_influence" begin
        @testset "Basic functionality" begin
            n = 100
            errors1 = randn(rng, n)
            errors2 = randn(rng, n)

            diag = compute_dm_influence(errors1, errors2)

            @test diag isa InfluenceDiagnostic
            @test length(diag.observation_influence) == n
            @test length(diag.observation_high_mask) == n
            @test all(isfinite.(diag.observation_influence))
        end

        @testset "Squared vs absolute loss" begin
            n = 50
            errors1 = randn(rng, n)
            errors2 = randn(rng, n)

            diag_sq = compute_dm_influence(errors1, errors2; loss=:squared)
            diag_abs = compute_dm_influence(errors1, errors2; loss=:absolute)

            # Results should differ
            @test diag_sq.observation_influence != diag_abs.observation_influence
        end

        @testset "Block influence computed" begin
            n = 100
            errors1 = randn(rng, n)
            errors2 = randn(rng, n)

            diag = compute_dm_influence(errors1, errors2; horizon=5)

            @test length(diag.block_influence) > 0
            @test length(diag.block_indices) == length(diag.block_influence)
            @test all(isfinite.(diag.block_influence))
        end

        @testset "Block size matches horizon" begin
            n = 100
            horizon = 10
            errors1 = randn(rng, n)
            errors2 = randn(rng, n)

            diag = compute_dm_influence(errors1, errors2; horizon=horizon)

            # Should have n ÷ horizon blocks
            expected_blocks = n ÷ horizon
            @test length(diag.block_indices) == expected_blocks
        end

        @testset "Influence threshold effects" begin
            n = 100
            errors1 = randn(rng, n)
            errors2 = randn(rng, n)

            diag_strict = compute_dm_influence(errors1, errors2; influence_threshold=3.0)
            diag_loose = compute_dm_influence(errors1, errors2; influence_threshold=1.0)

            # Looser threshold should flag more points
            @test diag_loose.n_high_influence_obs >= diag_strict.n_high_influence_obs
        end

        @testset "Detects influential outlier" begin
            n = 100
            errors1 = randn(rng, n) .* 0.1
            errors2 = randn(rng, n) .* 0.1

            # Add an outlier that makes model 1 much better at position 50
            errors1[50] = 0.01  # Very small error
            errors2[50] = 10.0  # Very large error

            diag = compute_dm_influence(errors1, errors2; influence_threshold=2.0)

            # The outlier should be flagged
            @test diag.observation_high_mask[50] == true
            @test diag.n_high_influence_obs >= 1
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_dm_influence(
                randn(rng, 50), randn(rng, 40)
            )
        end

        @testset "Insufficient data error" begin
            @test_throws ErrorException compute_dm_influence(
                randn(rng, 5), randn(rng, 5)
            )
        end

        @testset "Invalid loss error" begin
            @test_throws ErrorException compute_dm_influence(
                randn(rng, 20), randn(rng, 20); loss=:invalid
            )
        end

        @testset "Identical errors give zero influence" begin
            n = 50
            errors = randn(rng, n)

            diag = compute_dm_influence(errors, errors)

            # All loss differentials are zero, so influence should be zero
            @test all(diag.observation_influence .== 0)
            @test diag.n_high_influence_obs == 0
        end
    end

    # ==========================================================================
    # compute_block_influence
    # ==========================================================================

    @testset "compute_block_influence" begin
        @testset "Basic functionality" begin
            n = 100
            loss_diff = randn(rng, n)

            influence, mask, indices = compute_block_influence(loss_diff; block_size=10)

            @test length(influence) == n ÷ 10
            @test length(mask) == length(influence)
            @test length(indices) == length(influence)
        end

        @testset "Block indices are correct" begin
            n = 20
            loss_diff = randn(rng, n)

            _, _, indices = compute_block_influence(loss_diff; block_size=5)

            @test indices[1] == (1, 5)
            @test indices[2] == (6, 10)
            @test indices[3] == (11, 15)
            @test indices[4] == (16, 20)
        end

        @testset "Single-element blocks" begin
            n = 50
            loss_diff = randn(rng, n)

            influence, mask, indices = compute_block_influence(loss_diff; block_size=1)

            @test length(influence) == n
            @test length(indices) == n
        end

        @testset "Invalid block_size error" begin
            @test_throws ErrorException compute_block_influence(randn(rng, 20); block_size=0)
        end
    end

    # ==========================================================================
    # identify_influential_points
    # ==========================================================================

    @testset "identify_influential_points" begin
        @testset "Returns correct indices" begin
            # Create influence with known outliers
            influence = zeros(100)
            influence[25] = 5.0  # High positive
            influence[75] = -5.0  # High negative

            indices = identify_influential_points(influence; threshold=2.0)

            @test 25 in indices
            @test 75 in indices
        end

        @testset "Empty result for uniform data" begin
            influence = ones(100)

            indices = identify_influential_points(influence)

            @test isempty(indices)
        end

        @testset "Threshold effects" begin
            influence = randn(rng, 100)

            indices_strict = identify_influential_points(influence; threshold=3.0)
            indices_loose = identify_influential_points(influence; threshold=1.0)

            @test length(indices_loose) >= length(indices_strict)
        end

        @testset "All zeros returns empty" begin
            influence = zeros(50)
            indices = identify_influential_points(influence)
            @test isempty(indices)
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Full influence workflow" begin
        rng_int = Random.MersenneTwister(999)
        n = 200

        # Simulate two model errors
        # Model 1: generally better
        errors1 = randn(rng_int, n) .* 0.5
        # Model 2: worse, with some periods where it's much worse
        errors2 = randn(rng_int, n) .* 0.8

        # Add regime where model 2 is extremely bad
        errors2[100:120] .*= 5.0

        diag = compute_dm_influence(errors1, errors2; horizon=10)

        # Should detect the extreme block
        @test diag.n_high_influence_blocks >= 1

        # The block containing indices 100-120 should be flagged
        extreme_block_flagged = any(
            diag.block_high_mask[i] &&
            any(start <= 110 <= stop for (start, stop) in [diag.block_indices[i]])
            for i in 1:length(diag.block_high_mask)
        )
        @test extreme_block_flagged

        # Observation-level should also flag points in that region
        high_obs_in_region = sum(diag.observation_high_mask[100:120])
        @test high_obs_in_region > 0
    end

    @testset "Integration: Symmetric errors" begin
        rng_int = Random.MersenneTwister(123)
        n = 100

        # Same distribution for both models
        errors1 = randn(rng_int, n)
        rng_int2 = Random.MersenneTwister(123)  # Reset for same sequence
        errors2 = randn(rng_int2, n)

        diag = compute_dm_influence(errors1, errors2)

        # With identical errors, all influence should be zero
        @test all(diag.observation_influence .≈ 0)
    end
end
