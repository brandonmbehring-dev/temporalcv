@testset "CrossFitCV" begin
    @testset "Construction" begin
        @testset "Default values" begin
            cv = CrossFitCV()
            @test cv.n_splits == 5
            @test cv.gap == 0
            @test isnothing(cv.test_size)
        end

        @testset "Custom values" begin
            cv = CrossFitCV(n_splits=10, gap=2, test_size=20)
            @test cv.n_splits == 10
            @test cv.gap == 2
            @test cv.test_size == 20
        end

        @testset "Invalid parameters" begin
            @test_throws AssertionError CrossFitCV(n_splits=1)  # n_splits >= 2
            @test_throws AssertionError CrossFitCV(n_splits=0)
            @test_throws AssertionError CrossFitCV(gap=-1)
            @test_throws AssertionError CrossFitCV(test_size=0)
        end
    end

    @testset "split() - Basic" begin
        @testset "Forward-only splits" begin
            cv = CrossFitCV(n_splits=5, gap=0)
            n = 100
            splits = split(cv, n)

            # Skips fold 0, so n_splits - 1 actual splits
            @test length(splits) == 4

            for (train_idx, test_idx) in splits
                # Train always starts from 1
                @test first(train_idx) == 1
                # Train ends before test starts
                @test last(train_idx) < first(test_idx)
            end
        end

        @testset "Train grows with each fold" begin
            cv = CrossFitCV(n_splits=5, gap=0)
            n = 100
            splits = split(cv, n)

            train_sizes = [length(train) for (train, _) in splits]
            # Each fold should have more training data
            for i in 2:length(train_sizes)
                @test train_sizes[i] > train_sizes[i-1]
            end
        end

        @testset "Chronological order" begin
            cv = CrossFitCV(n_splits=4, gap=0)
            n = 100
            splits = split(cv, n)

            # Test sets should be in chronological order
            for i in 2:length(splits)
                @test first(splits[i][2]) > first(splits[i-1][2])
            end
        end
    end

    @testset "split() - With gap" begin
        @testset "Gap respected" begin
            cv = CrossFitCV(n_splits=5, gap=5)
            n = 200
            splits = split(cv, n)

            for (train_idx, test_idx) in splits
                actual_gap = first(test_idx) - last(train_idx) - 1
                @test actual_gap >= cv.gap
            end
        end

        @testset "Large gap reduces splits" begin
            cv_no_gap = CrossFitCV(n_splits=5, gap=0)
            cv_with_gap = CrossFitCV(n_splits=5, gap=20)
            n = 100

            splits_no_gap = split(cv_no_gap, n)
            splits_with_gap = split(cv_with_gap, n)

            # Large gap may result in fewer valid splits
            @test length(splits_with_gap) <= length(splits_no_gap)
        end
    end

    @testset "split() - With test_size" begin
        @testset "Fixed test size" begin
            cv = CrossFitCV(n_splits=5, gap=0, test_size=15)
            n = 100
            splits = split(cv, n)

            for (_, test_idx) in splits
                @test length(test_idx) == 15 || last(test_idx) == n  # Last fold may be larger
            end
        end
    end

    @testset "get_n_splits()" begin
        @testset "Basic" begin
            cv = CrossFitCV(n_splits=5, gap=0)
            n = 100
            @test get_n_splits(cv, n) == 4  # n_splits - 1
        end

        @testset "strict=false returns 0 on failure" begin
            cv = CrossFitCV(n_splits=10, gap=0, test_size=50)
            @test get_n_splits(cv, 10; strict=false) == 0
        end
    end

    @testset "get_fold_indices()" begin
        cv = CrossFitCV(n_splits=5, gap=0)
        n = 100
        folds = get_fold_indices(cv, n)

        @test length(folds) == 5

        # Folds should be consecutive
        for i in 2:length(folds)
            @test folds[i][1] == folds[i-1][2] + 1
        end

        # Last fold ends at n
        @test folds[end][2] == n
    end

    @testset "Edge cases" begin
        @testset "Minimum splits" begin
            cv = CrossFitCV(n_splits=2, gap=0)
            n = 100
            splits = split(cv, n)
            @test length(splits) == 1  # Only fold 1 (fold 0 skipped)
        end

        @testset "Very large gap" begin
            cv = CrossFitCV(n_splits=5, gap=50)
            n = 100

            splits = split(cv, n)
            # May have very few or no valid splits due to large gap
            @test length(splits) <= 4
        end
    end

    @testset "Reproducibility" begin
        cv = CrossFitCV(n_splits=5, gap=2)
        n = 200

        splits1 = split(cv, n)
        splits2 = split(cv, n)

        @test length(splits1) == length(splits2)
        for i in 1:length(splits1)
            @test splits1[i] == splits2[i]
        end
    end

    @testset "show() method" begin
        cv = CrossFitCV(n_splits=5, gap=2, test_size=20)
        str = string(cv)
        @test occursin("CrossFitCV", str)
        @test occursin("n_splits=5", str)
        @test occursin("gap=2", str)
    end
end
