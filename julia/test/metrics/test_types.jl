@testset "Metrics Types" begin
    using Random
    rng = Random.MersenneTwister(42)

    @testset "MoveDirection enum" begin
        @test UP isa MoveDirection
        @test DOWN isa MoveDirection
        @test FLAT isa MoveDirection

        # Enum instances are distinct
        @test UP != DOWN
        @test DOWN != FLAT
        @test UP != FLAT
    end

    @testset "MoveConditionalResult struct" begin
        @testset "Constructor and fields" begin
            result = MoveConditionalResult(
                0.1, 0.2, 0.05,  # mae_up, mae_down, mae_flat
                15, 12, 23,      # n_up, n_down, n_flat
                0.35, 0.05       # skill_score, move_threshold
            )

            @test result.mae_up == 0.1
            @test result.mae_down == 0.2
            @test result.mae_flat == 0.05
            @test result.n_up == 15
            @test result.n_down == 12
            @test result.n_flat == 23
            @test result.skill_score == 0.35
            @test result.move_threshold == 0.05
        end

        @testset "n_total accessor" begin
            result = MoveConditionalResult(0.1, 0.2, 0.05, 10, 8, 22, 0.3, 0.05)
            @test n_total(result) == 40
        end

        @testset "n_moves accessor" begin
            result = MoveConditionalResult(0.1, 0.2, 0.05, 10, 8, 22, 0.3, 0.05)
            @test n_moves(result) == 18
        end

        @testset "is_reliable - both directions >= 10" begin
            reliable = MoveConditionalResult(0.1, 0.2, 0.05, 15, 12, 20, 0.3, 0.05)
            @test is_reliable(reliable) == true

            not_reliable_up = MoveConditionalResult(0.1, 0.2, 0.05, 5, 12, 20, 0.3, 0.05)
            @test is_reliable(not_reliable_up) == false

            not_reliable_down = MoveConditionalResult(0.1, 0.2, 0.05, 15, 7, 20, 0.3, 0.05)
            @test is_reliable(not_reliable_down) == false

            edge_case = MoveConditionalResult(0.1, 0.2, 0.05, 10, 10, 20, 0.3, 0.05)
            @test is_reliable(edge_case) == true
        end

        @testset "move_fraction" begin
            result = MoveConditionalResult(0.1, 0.2, 0.05, 10, 10, 80, 0.3, 0.05)
            @test move_fraction(result) ≈ 0.2

            all_moves = MoveConditionalResult(0.1, 0.2, NaN, 50, 50, 0, 0.3, 0.05)
            @test move_fraction(all_moves) ≈ 1.0

            no_samples = MoveConditionalResult(NaN, NaN, NaN, 0, 0, 0, NaN, 0.0)
            @test move_fraction(no_samples) == 0.0
        end

        @testset "show method" begin
            result = MoveConditionalResult(0.1, 0.2, 0.05, 15, 12, 23, 0.35, 0.05)
            str = sprint(show, result)
            @test occursin("MoveConditionalResult", str)
            @test occursin("skill_score=0.35", str)
            @test occursin("n_moves=27", str)
        end
    end

    @testset "BrierScoreResult struct" begin
        @testset "Constructor and fields" begin
            result = BrierScoreResult(
                0.15,    # brier_score
                0.02,    # reliability
                0.08,    # resolution
                0.21,    # uncertainty
                100,     # n_samples
                2        # n_classes
            )

            @test result.brier_score == 0.15
            @test result.reliability == 0.02
            @test result.resolution == 0.08
            @test result.uncertainty == 0.21
            @test result.n_samples == 100
            @test result.n_classes == 2
        end

        @testset "decomposition_valid" begin
            # Valid decomposition: brier = reliability - resolution + uncertainty
            valid = BrierScoreResult(0.15, 0.02, 0.08, 0.21, 100, 2)
            # 0.02 - 0.08 + 0.21 = 0.15 ✓
            @test decomposition_valid(valid)

            # Invalid decomposition
            invalid = BrierScoreResult(0.20, 0.02, 0.08, 0.21, 100, 2)
            @test decomposition_valid(invalid) == false
        end

        @testset "show method" begin
            result = BrierScoreResult(0.15, 0.02, 0.08, 0.21, 100, 2)
            str = sprint(show, result)
            @test occursin("BrierScoreResult", str)
            @test occursin("brier=0.15", str)
        end
    end

    @testset "VolatilityStratifiedResult struct" begin
        @testset "Constructor and fields" begin
            result = VolatilityStratifiedResult(
                0.05, 0.10, 0.20,  # mae_low, mae_medium, mae_high
                30, 40, 30,        # n_low, n_medium, n_high
                (0.02, 0.08)       # vol_thresholds
            )

            @test result.mae_low == 0.05
            @test result.mae_medium == 0.10
            @test result.mae_high == 0.20
            @test result.n_low == 30
            @test result.n_medium == 40
            @test result.n_high == 30
            @test result.vol_thresholds == (0.02, 0.08)
        end

        @testset "n_total accessor" begin
            result = VolatilityStratifiedResult(0.05, 0.10, 0.20, 30, 40, 30, (0.02, 0.08))
            @test n_total(result) == 100
        end

        @testset "show method" begin
            result = VolatilityStratifiedResult(0.05, 0.10, 0.20, 30, 40, 30, (0.02, 0.08))
            str = sprint(show, result)
            @test occursin("VolatilityStratifiedResult", str)
        end
    end

    @testset "IntervalScoreResult struct" begin
        @testset "Constructor and fields" begin
            result = IntervalScoreResult(
                0.5,    # score
                0.2,    # width_penalty
                0.3,    # coverage_penalty
                0.85,   # coverage
                0.15,   # mean_width
                0.1,    # alpha
                100     # n_samples
            )

            @test result.score == 0.5
            @test result.width_penalty == 0.2
            @test result.coverage_penalty == 0.3
            @test result.coverage == 0.85
            @test result.mean_width == 0.15
            @test result.alpha == 0.1
            @test result.n_samples == 100
        end

        @testset "show method" begin
            result = IntervalScoreResult(0.5, 0.2, 0.3, 0.85, 0.15, 0.1, 100)
            str = sprint(show, result)
            @test occursin("IntervalScoreResult", str)
            @test occursin("coverage=0.85", str)
        end
    end
end
