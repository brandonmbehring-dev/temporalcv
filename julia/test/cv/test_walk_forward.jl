@testset "WalkForwardCV" begin
    @testset "Construction" begin
        @testset "Default values" begin
            cv = WalkForwardCV()
            @test cv.n_splits == 5
            @test cv.test_size == 1
            @test cv.gap == 0
            @test cv.window_type == :expanding
            @test isnothing(cv.window_size)
            @test isnothing(cv.horizon)
        end

        @testset "Custom values" begin
            cv = WalkForwardCV(
                n_splits=10,
                test_size=5,
                gap=2,
                window_type=:sliding,
                window_size=50,
                horizon=2
            )
            @test cv.n_splits == 10
            @test cv.test_size == 5
            @test cv.gap == 2
            @test cv.window_type == :sliding
            @test cv.window_size == 50
            @test cv.horizon == 2
        end

        @testset "Invalid parameters" begin
            @test_throws AssertionError WalkForwardCV(n_splits=0)
            @test_throws AssertionError WalkForwardCV(n_splits=-1)
            @test_throws AssertionError WalkForwardCV(test_size=0)
            @test_throws AssertionError WalkForwardCV(gap=-1)
            @test_throws AssertionError WalkForwardCV(window_type=:invalid)
            @test_throws AssertionError WalkForwardCV(window_size=0)
        end

        @testset "Sliding requires window_size" begin
            @test_throws ArgumentError WalkForwardCV(window_type=:sliding)
        end

        @testset "Gap >= horizon enforcement" begin
            # gap < horizon should fail
            @test_throws ArgumentError WalkForwardCV(horizon=3, gap=2)
            @test_throws ArgumentError WalkForwardCV(horizon=5, gap=0)

            # gap >= horizon should work
            cv = WalkForwardCV(horizon=3, gap=3)
            @test cv.horizon == 3
            @test cv.gap == 3

            cv2 = WalkForwardCV(horizon=2, gap=5)
            @test cv2.horizon == 2
            @test cv2.gap == 5
        end
    end

    @testset "split() - Expanding window" begin
        @testset "Basic split" begin
            cv = WalkForwardCV(n_splits=3, test_size=10, gap=0)
            n = 100
            splits = split(cv, n)

            @test length(splits) == 3

            for (train_idx, test_idx) in splits
                @test length(test_idx) == 10
                @test last(train_idx) < first(test_idx)
            end
        end

        @testset "Gap enforcement" begin
            cv = WalkForwardCV(n_splits=3, test_size=10, gap=5)
            n = 200
            splits = split(cv, n)

            for (train_idx, test_idx) in splits
                # train_end + gap < test_start (1-based indexing)
                @test last(train_idx) + cv.gap < first(test_idx)
                actual_gap = first(test_idx) - last(train_idx) - 1
                @test actual_gap >= cv.gap
            end
        end

        @testset "Expanding window grows" begin
            cv = WalkForwardCV(n_splits=3, test_size=10, gap=0)
            n = 100
            splits = split(cv, n)

            train_sizes = [length(train) for (train, _) in splits]
            # Each split should have more or equal training data (expanding)
            for i in 2:length(train_sizes)
                @test train_sizes[i] >= train_sizes[i-1]
            end
        end

        @testset "Chronological order" begin
            cv = WalkForwardCV(n_splits=4, test_size=10, gap=2)
            n = 200
            splits = split(cv, n)

            # Test sets should be in chronological order
            for i in 2:length(splits)
                @test first(splits[i][2]) > first(splits[i-1][2])
            end
        end
    end

    @testset "split() - Sliding window" begin
        @testset "Fixed window size" begin
            cv = WalkForwardCV(
                n_splits=3,
                test_size=10,
                gap=0,
                window_type=:sliding,
                window_size=50
            )
            n = 200
            splits = split(cv, n)

            @test length(splits) == 3

            for (train_idx, test_idx) in splits
                @test length(train_idx) == 50
                @test length(test_idx) == 10
            end
        end

        @testset "Window slides forward" begin
            cv = WalkForwardCV(
                n_splits=4,
                test_size=10,
                gap=2,
                window_type=:sliding,
                window_size=50
            )
            n = 200
            splits = split(cv, n)

            train_starts = [first(train) for (train, _) in splits]
            # Each split should start later (window slides)
            for i in 2:length(train_starts)
                @test train_starts[i] > train_starts[i-1]
            end

            # All train windows same size
            for (train_idx, _) in splits
                @test length(train_idx) == 50
            end
        end
    end

    @testset "get_n_splits()" begin
        @testset "Basic" begin
            cv = WalkForwardCV(n_splits=5, test_size=10, gap=0)
            @test get_n_splits(cv, 200) == 5
        end

        @testset "Fewer valid splits than requested" begin
            cv = WalkForwardCV(n_splits=10, test_size=20, gap=5)
            # With limited data, might get fewer splits
            @test get_n_splits(cv, 100) <= 10
        end

        @testset "strict=false returns 0 on failure" begin
            cv = WalkForwardCV(n_splits=5, test_size=100, gap=50)
            # Impossible with small n
            @test get_n_splits(cv, 50; strict=false) == 0
        end

        @testset "strict=true raises on failure" begin
            cv = WalkForwardCV(n_splits=5, test_size=100, gap=50)
            @test_throws ArgumentError get_n_splits(cv, 50; strict=true)
        end
    end

    @testset "get_split_info()" begin
        cv = WalkForwardCV(n_splits=3, test_size=10, gap=2)
        n = 100
        infos = get_split_info(cv, n)

        @test length(infos) == 3

        for (i, info) in enumerate(infos)
            @test info.split_idx == i - 1  # 0-based
            @test gap(info) >= 2
            @test test_size(info) == 10
            @test info.train_end < info.test_start
        end
    end

    @testset "Edge cases" begin
        @testset "Minimum data" begin
            cv = WalkForwardCV(n_splits=1, test_size=1, gap=0)
            splits = split(cv, 10)
            @test length(splits) == 1
        end

        @testset "Large gap" begin
            cv = WalkForwardCV(n_splits=2, test_size=10, gap=50)
            splits = split(cv, 200)
            @test length(splits) >= 1

            for (train_idx, test_idx) in splits
                @test first(test_idx) - last(train_idx) - 1 >= 50
            end
        end

        @testset "Insufficient data" begin
            cv = WalkForwardCV(n_splits=5, test_size=100, gap=20)
            @test_throws ArgumentError split(cv, 50)
        end
    end

    @testset "Reproducibility and consistency" begin
        cv = WalkForwardCV(n_splits=5, test_size=10, gap=3)
        n = 200

        splits1 = split(cv, n)
        splits2 = split(cv, n)

        @test length(splits1) == length(splits2)
        for i in 1:length(splits1)
            @test splits1[i] == splits2[i]
        end
    end

    @testset "show() method" begin
        cv = WalkForwardCV(n_splits=5, test_size=10, gap=2)
        str = string(cv)
        @test occursin("WalkForwardCV", str)
        @test occursin("n_splits=5", str)
        @test occursin("gap=2", str)
    end
end
