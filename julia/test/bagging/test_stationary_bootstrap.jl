@testset "StationaryBootstrap" begin
    @testset "Construction" begin
        sb = StationaryBootstrap(expected_block_length=10.0)
        @test sb.expected_block_length == 10.0

        # Default (auto)
        sb2 = StationaryBootstrap()
        @test isnothing(sb2.expected_block_length)

        # Invalid expected block length
        @test_throws AssertionError StationaryBootstrap(expected_block_length=0.5)
        @test_throws AssertionError StationaryBootstrap(expected_block_length=-1.0)
    end

    @testset "Auto expected block length" begin
        rng = Random.MersenneTwister(42)
        n = 100
        X = randn(rng, n, 3)
        y = randn(rng, n)

        sb = StationaryBootstrap()  # Auto
        samples = bootstrap_sample(sb, X, y, 1, rng)

        # Expected block length should be n^(1/3) ≈ 4.64
        expected_exp_len = n^(1/3)
        @test expected_exp_len ≈ 4.64 atol=0.01

        # Sample should have correct dimensions
        X_boot, y_boot = samples[1]
        @test size(X_boot) == size(X)
        @test length(y_boot) == length(y)
    end

    @testset "Sample dimensions" begin
        rng = Random.MersenneTwister(42)
        n, p = 50, 4
        X = randn(rng, n, p)
        y = randn(rng, n)

        sb = StationaryBootstrap(expected_block_length=8.0)
        samples = bootstrap_sample(sb, X, y, 20, rng)

        @test length(samples) == 20

        for (X_boot, y_boot) in samples
            @test size(X_boot) == (n, p)
            @test length(y_boot) == n
        end
    end

    @testset "Geometric block lengths" begin
        # With high expected block length, most steps should continue
        rng = Random.MersenneTwister(42)
        n = 1000
        y = collect(1.0:n)
        X = reshape(y, n, 1)

        # Very high expected block length → fewer jumps
        sb_long = StationaryBootstrap(expected_block_length=100.0)
        samples_long = bootstrap_sample(sb_long, X, y, 1, rng)
        _, y_boot_long = samples_long[1]

        # Count jumps (where diff != 1 and != -(n-1) for circular wrap)
        diffs = diff(y_boot_long)
        jumps_long = count(d -> d != 1.0 && d != -(n-1), diffs)

        # Short expected block length → more jumps
        rng2 = Random.MersenneTwister(42)
        sb_short = StationaryBootstrap(expected_block_length=5.0)
        samples_short = bootstrap_sample(sb_short, X, y, 1, rng2)
        _, y_boot_short = samples_short[1]
        diffs_short = diff(y_boot_short)
        jumps_short = count(d -> d != 1.0 && d != -(n-1), diffs_short)

        # Should have significantly more jumps with shorter expected length
        @test jumps_short > jumps_long
    end

    @testset "Circular wrap" begin
        rng = Random.MersenneTwister(42)
        n = 50
        y = collect(1.0:n)
        X = reshape(y, n, 1)

        sb = StationaryBootstrap(expected_block_length=20.0)

        # Run many samples to check we sometimes wrap around
        found_wrap = false
        for _ in 1:100
            samples = bootstrap_sample(sb, X, y, 1, rng)
            _, y_boot = samples[1]
            diffs = diff(y_boot)

            # Wrap would show as diff == -(n-1) (from n back to 1)
            if any(d -> d == -(n-1), diffs)
                found_wrap = true
                break
            end
        end

        # Should find at least one wrap in 100 trials
        @test found_wrap
    end

    @testset "Reproducibility" begin
        n, p = 50, 3
        X = randn(Random.MersenneTwister(1), n, p)
        y = randn(Random.MersenneTwister(1), n)

        sb = StationaryBootstrap(expected_block_length=8.0)

        # Same seed → same samples
        samples1 = bootstrap_sample(sb, X, y, 10, Random.MersenneTwister(42))
        samples2 = bootstrap_sample(sb, X, y, 10, Random.MersenneTwister(42))

        for i in 1:10
            @test samples1[i][1] == samples2[i][1]
            @test samples1[i][2] == samples2[i][2]
        end
    end

    @testset "Edge cases" begin
        rng = Random.MersenneTwister(42)

        # Very small expected block length (most steps are jumps)
        n = 20
        X = randn(rng, n, 2)
        y = randn(rng, n)
        sb = StationaryBootstrap(expected_block_length=1.0)
        samples = bootstrap_sample(sb, X, y, 5, rng)
        @test length(samples) == 5

        # Very large expected block length
        sb2 = StationaryBootstrap(expected_block_length=1000.0)
        samples2 = bootstrap_sample(sb2, X, y, 5, rng)
        @test length(samples2) == 5

        # Single observation
        X_single = randn(rng, 1, 2)
        y_single = [1.0]
        samples3 = bootstrap_sample(sb, X_single, y_single, 3, rng)
        @test length(samples3) == 3
        @test all(s -> s[1] == X_single, samples3)
    end
end
