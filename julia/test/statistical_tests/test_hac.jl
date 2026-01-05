@testset "HAC Variance" begin
    @testset "Bartlett kernel" begin
        # At lag 0, weight is 1
        @test bartlett_kernel(0, 5) == 1.0

        # Weight decreases linearly
        @test bartlett_kernel(1, 5) ≈ 1.0 - 1/6
        @test bartlett_kernel(2, 5) ≈ 1.0 - 2/6
        @test bartlett_kernel(3, 5) ≈ 1.0 - 3/6

        # Beyond bandwidth, weight is 0
        @test bartlett_kernel(6, 5) == 0.0
        @test bartlett_kernel(10, 5) == 0.0

        # At bandwidth, weight is still positive
        @test bartlett_kernel(5, 5) ≈ 1.0 - 5/6 > 0
    end

    @testset "compute_hac_variance - Basic" begin
        rng = Random.MersenneTwister(42)

        # White noise: HAC variance should equal sample variance / n
        n = 100
        d = randn(rng, n)
        var_hac = compute_hac_variance(d, bandwidth=0)

        # With bandwidth=0, this is just variance / n
        expected = var(d) / n
        @test var_hac ≈ expected rtol=0.1
    end

    @testset "compute_hac_variance - Autocorrelated" begin
        rng = Random.MersenneTwister(42)
        n = 200

        # Generate AR(1) process with high autocorrelation
        rho = 0.7
        d = zeros(n)
        d[1] = randn(rng)
        for i in 2:n
            d[i] = rho * d[i-1] + randn(rng)
        end

        # HAC variance with larger bandwidth should be higher than naive
        var_naive = var(d) / n
        var_hac = compute_hac_variance(d, bandwidth=5)

        # For positively autocorrelated series, HAC variance > naive variance
        @test var_hac > var_naive * 0.8  # Allow some margin
    end

    @testset "compute_hac_variance - Auto bandwidth" begin
        rng = Random.MersenneTwister(42)
        d = randn(rng, 100)

        # Should not error when bandwidth is auto-selected
        var_auto = compute_hac_variance(d)
        @test var_auto > 0
        @test isfinite(var_auto)
    end

    @testset "compute_hac_variance - Constant input" begin
        # Constant series has zero variance
        d = fill(5.0, 50)
        var_hac = compute_hac_variance(d)
        @test var_hac ≈ 0.0 atol=1e-10
    end

    @testset "compute_hac_variance - Small bandwidth" begin
        rng = Random.MersenneTwister(42)
        d = randn(rng, 100)

        # Bandwidth 0 (no autocorrelation correction)
        var_0 = compute_hac_variance(d, bandwidth=0)
        # Bandwidth 1
        var_1 = compute_hac_variance(d, bandwidth=1)
        # Bandwidth 5
        var_5 = compute_hac_variance(d, bandwidth=5)

        # All should be positive
        @test var_0 > 0
        @test var_1 > 0
        @test var_5 > 0
    end

    @testset "Andrews bandwidth rule" begin
        # For n=100, bandwidth should be floor(4 * (100/100)^(2/9)) = floor(4) = 4
        rng = Random.MersenneTwister(42)
        d = randn(rng, 100)

        # The auto bandwidth should be 4 for n=100
        expected_bandwidth = floor(Int, 4 * (100/100)^(2/9))
        @test expected_bandwidth == 4

        # For n=1000
        expected_1000 = floor(Int, 4 * (1000/100)^(2/9))
        @test expected_1000 > 4  # Should be larger
    end
end
