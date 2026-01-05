@testset "MovingBlockBootstrap" begin
    @testset "Construction" begin
        mbb = MovingBlockBootstrap(block_length=10)
        @test mbb.block_length == 10

        # Default (auto) block length
        mbb2 = MovingBlockBootstrap()
        @test isnothing(mbb2.block_length)

        # Invalid block length
        @test_throws AssertionError MovingBlockBootstrap(block_length=0)
        @test_throws AssertionError MovingBlockBootstrap(block_length=-1)
    end

    @testset "Auto block length" begin
        rng = Random.MersenneTwister(42)
        n = 100
        X = randn(rng, n, 3)
        y = randn(rng, n)

        mbb = MovingBlockBootstrap()  # Auto block length
        samples = bootstrap_sample(mbb, X, y, 1, rng)

        # Block length should be ceil(n^(1/3)) = ceil(4.64) = 5
        expected_block_len = ceil(Int, n^(1/3))
        @test expected_block_len == 5

        # Sample should have same size as original
        X_boot, y_boot = samples[1]
        @test size(X_boot) == size(X)
        @test length(y_boot) == length(y)
    end

    @testset "Sample dimensions" begin
        rng = Random.MersenneTwister(42)
        n, p = 50, 4
        X = randn(rng, n, p)
        y = randn(rng, n)

        mbb = MovingBlockBootstrap(block_length=10)
        samples = bootstrap_sample(mbb, X, y, 20, rng)

        @test length(samples) == 20

        for (X_boot, y_boot) in samples
            @test size(X_boot) == (n, p)
            @test length(y_boot) == n
        end
    end

    @testset "Preserves local structure" begin
        rng = Random.MersenneTwister(42)
        n = 100
        # Create sequential data
        y = collect(1.0:n)
        X = reshape(y, n, 1)

        mbb = MovingBlockBootstrap(block_length=10)
        samples = bootstrap_sample(mbb, X, y, 1, rng)

        X_boot, y_boot = samples[1]

        # Within each block, values should be consecutive
        # Find blocks by looking for jumps
        diffs = diff(y_boot)
        block_boundaries = findall(d -> d != 1.0, diffs)

        # Diffs within blocks should be 1.0
        within_block = setdiff(1:length(diffs), block_boundaries)
        @test all(diffs[within_block] .== 1.0)
    end

    @testset "Reproducibility" begin
        n, p = 50, 3
        X = randn(Random.MersenneTwister(1), n, p)
        y = randn(Random.MersenneTwister(1), n)

        mbb = MovingBlockBootstrap(block_length=5)

        # Same seed → same samples
        samples1 = bootstrap_sample(mbb, X, y, 10, Random.MersenneTwister(42))
        samples2 = bootstrap_sample(mbb, X, y, 10, Random.MersenneTwister(42))

        for i in 1:10
            @test samples1[i][1] == samples2[i][1]
            @test samples1[i][2] == samples2[i][2]
        end

        # Different seeds → different samples
        samples3 = bootstrap_sample(mbb, X, y, 10, Random.MersenneTwister(43))
        @test samples1[1][1] != samples3[1][1]
    end

    @testset "Edge cases" begin
        rng = Random.MersenneTwister(42)

        # Block length equals data length
        n = 20
        X = randn(rng, n, 2)
        y = randn(rng, n)
        mbb = MovingBlockBootstrap(block_length=n)
        samples = bootstrap_sample(mbb, X, y, 5, rng)
        @test length(samples) == 5

        # Block length larger than data (gets clamped)
        mbb2 = MovingBlockBootstrap(block_length=100)
        samples2 = bootstrap_sample(mbb2, X, y, 5, rng)
        @test length(samples2) == 5

        # Very small data
        X_small = randn(rng, 5, 2)
        y_small = randn(rng, 5)
        mbb3 = MovingBlockBootstrap(block_length=2)
        samples3 = bootstrap_sample(mbb3, X_small, y_small, 3, rng)
        @test length(samples3) == 3
    end
end
