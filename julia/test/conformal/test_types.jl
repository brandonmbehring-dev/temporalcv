@testset "PredictionInterval" begin
    @testset "Construction" begin
        pi = PredictionInterval([1.0, 2.0], [3.0, 4.0])
        @test length(pi) == 2
        @test pi.lower == [1.0, 2.0]
        @test pi.upper == [3.0, 4.0]
    end

    @testset "Width" begin
        pi = PredictionInterval([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        @test width(pi) == [3.0, 3.0, 3.0]
        @test mean_width(pi) == 3.0
    end

    @testset "Coverage" begin
        pi = PredictionInterval([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

        # All covered
        @test coverage(pi, [1.0, 1.0, 1.0]) == 1.0

        # None covered
        @test coverage(pi, [3.0, 3.0, 3.0]) == 0.0

        # Partial coverage
        @test coverage(pi, [1.0, 1.0, 3.0]) ≈ 2/3

        # Boundary cases
        @test coverage(pi, [0.0, 2.0, 1.0]) == 1.0  # Boundaries included
    end

    @testset "Edge cases" begin
        # Single observation
        pi = PredictionInterval([1.0], [2.0])
        @test length(pi) == 1
        @test coverage(pi, [1.5]) == 1.0

        # Zero-width interval
        pi = PredictionInterval([1.0], [1.0])
        @test width(pi) == [0.0]
        @test coverage(pi, [1.0]) == 1.0  # Exactly on boundary
        @test coverage(pi, [1.1]) == 0.0
    end
end
