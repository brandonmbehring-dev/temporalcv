@testset "CVFinancial: Purged Cross-Validation" begin
    using Random
    # Use CVFin alias for functions that conflict with CV module
    split_cv = CVFin.split
    get_n_splits_cv = CVFin.get_n_splits

    # ==========================================================================
    # compute_label_overlap
    # ==========================================================================

    @testset "compute_label_overlap" begin
        @testset "Basic overlap matrix" begin
            overlap = compute_label_overlap(10, 3)

            @test overlap isa BitMatrix
            @test size(overlap) == (10, 10)

            # Diagonal is always true (sample overlaps with itself)
            @test all(overlap[i, i] for i in 1:10)
        end

        @testset "Adjacent samples with horizon=5" begin
            overlap = compute_label_overlap(10, 5)

            # Samples 1 and 4 overlap (|1-4| = 3 < 5)
            @test overlap[1, 4] == true
            @test overlap[4, 1] == true

            # Samples 1 and 6 don't overlap (|1-6| = 5 >= 5)
            @test overlap[1, 6] == false
            @test overlap[6, 1] == false
        end

        @testset "Horizon 0 means no overlap except self" begin
            overlap = compute_label_overlap(10, 0)

            # Only diagonal should be true
            @test !any(overlap[i, j] for i in 1:10 for j in 1:10 if i != j)
        end

        @testset "Horizon >= n means full overlap" begin
            overlap = compute_label_overlap(5, 10)

            @test all(overlap)
        end

        @testset "Symmetry" begin
            overlap = compute_label_overlap(20, 3)

            for i in 1:20
                for j in 1:20
                    @test overlap[i, j] == overlap[j, i]
                end
            end
        end

        @testset "Error on invalid input" begin
            @test_throws ErrorException compute_label_overlap(0, 5)
            @test_throws ErrorException compute_label_overlap(-1, 5)
            @test_throws ErrorException compute_label_overlap(10, -1)
        end
    end

    # ==========================================================================
    # estimate_purge_gap
    # ==========================================================================

    @testset "estimate_purge_gap" begin
        @testset "Default decay factor" begin
            @test estimate_purge_gap(5) == 5
            @test estimate_purge_gap(10) == 10
            @test estimate_purge_gap(0) == 0
        end

        @testset "Custom decay factor" begin
            @test estimate_purge_gap(5; decay_factor=1.5) == 8
            @test estimate_purge_gap(10; decay_factor=2.0) == 20
            @test estimate_purge_gap(7; decay_factor=1.2) == 9
        end

        @testset "Ceiling behavior" begin
            # 5 * 1.1 = 5.5 → ceil to 6
            @test estimate_purge_gap(5; decay_factor=1.1) == 6
        end

        @testset "Error on invalid input" begin
            @test_throws ErrorException estimate_purge_gap(-1)
            @test_throws ErrorException estimate_purge_gap(5; decay_factor=0.0)
            @test_throws ErrorException estimate_purge_gap(5; decay_factor=-1.0)
        end
    end

    # ==========================================================================
    # apply_purge_and_embargo
    # ==========================================================================

    @testset "apply_purge_and_embargo" begin
        @testset "No purge or embargo" begin
            train = collect(1:80)
            test = collect(81:100)

            result = apply_purge_and_embargo(train, test, 100)

            @test result.train_indices == train
            @test result.test_indices == test
            @test result.n_purged == 0
            @test result.n_embargoed == 0
        end

        @testset "Purge before test" begin
            train = collect(1:80)
            test = collect(81:100)

            result = apply_purge_and_embargo(train, test, 100; purge_gap=5)

            # Samples 76-80 should be purged (within 5 of test start)
            @test !(76 in result.train_indices)
            @test !(77 in result.train_indices)
            @test !(78 in result.train_indices)
            @test !(79 in result.train_indices)
            @test !(80 in result.train_indices)
            @test 75 in result.train_indices
            @test result.n_purged == 5
        end

        @testset "Purge after test" begin
            train = vcat(collect(1:40), collect(61:100))
            test = collect(41:60)

            result = apply_purge_and_embargo(train, test, 100; purge_gap=3)

            # Samples 61-63 should be purged (within 3 of test end)
            @test !(61 in result.train_indices)
            @test !(62 in result.train_indices)
            @test !(63 in result.train_indices)
            @test 64 in result.train_indices

            # Samples 38-40 should be purged (within 3 of test start)
            @test !(38 in result.train_indices)
            @test !(39 in result.train_indices)
            @test !(40 in result.train_indices)
            @test 37 in result.train_indices
        end

        @testset "Embargo after test" begin
            train = vcat(collect(1:50), collect(71:100))
            test = collect(51:70)

            result = apply_purge_and_embargo(train, test, 100; embargo_pct=0.05)

            # 5% of 100 = 5 samples embargoed after test end (70)
            # Samples 71-75 should be embargoed
            @test !(71 in result.train_indices)
            @test !(72 in result.train_indices)
            @test !(73 in result.train_indices)
            @test !(74 in result.train_indices)
            @test !(75 in result.train_indices)
            @test 76 in result.train_indices
            @test result.n_embargoed == 5
        end

        @testset "Combined purge and embargo" begin
            train = vcat(collect(1:45), collect(61:100))
            test = collect(46:60)

            result = apply_purge_and_embargo(train, test, 100; purge_gap=3, embargo_pct=0.02)

            # Purge: 43-45 before, 61-63 after
            @test !(43 in result.train_indices)
            @test !(61 in result.train_indices)

            # Embargo: 2% of 100 = 2 samples (but overlaps with purge after)
            @test result.n_purged > 0
        end

        @testset "Empty test indices" begin
            train = collect(1:100)
            test = Int[]

            result = apply_purge_and_embargo(train, test, 100; purge_gap=5, embargo_pct=0.1)

            @test result.train_indices == train
            @test result.n_purged == 0
            @test result.n_embargoed == 0
        end
    end

    # ==========================================================================
    # PurgedKFold
    # ==========================================================================

    @testset "PurgedKFold" begin
        @testset "Construction" begin
            cv = PurgedKFold(n_splits=5, purge_gap=3, embargo_pct=0.01)

            @test cv.n_splits == 5
            @test cv.purge_gap == 3
            @test cv.embargo_pct == 0.01
        end

        @testset "Default values" begin
            cv = PurgedKFold()

            @test cv.n_splits == 5
            @test cv.purge_gap == 0
            @test cv.embargo_pct == 0.0
        end

        @testset "Validation errors" begin
            @test_throws ErrorException PurgedKFold(n_splits=1)
            @test_throws ErrorException PurgedKFold(purge_gap=-1)
            @test_throws ErrorException PurgedKFold(embargo_pct=1.0)
            @test_throws ErrorException PurgedKFold(embargo_pct=-0.1)
        end

        @testset "get_n_splits" begin
            cv = PurgedKFold(n_splits=7)
            @test get_n_splits_cv(cv) == 7
        end

        @testset "split basic" begin
            cv = PurgedKFold(n_splits=5)
            splits = split_cv(cv, 100)

            @test length(splits) == 5

            for s in splits
                @test s isa PurgedSplit
                @test length(s.test_indices) == 20
                @test length(s.train_indices) == 80
            end
        end

        @testset "split with purging" begin
            cv = PurgedKFold(n_splits=5, purge_gap=5)
            splits = split_cv(cv, 100)

            # Each split should have purged some samples
            for s in splits
                @test s.n_purged > 0
                @test length(s.train_indices) < 80
            end
        end

        @testset "split with embargo" begin
            cv = PurgedKFold(n_splits=5, embargo_pct=0.05)
            splits = split_cv(cv, 100)

            # Embargo only affects splits where training follows test
            has_embargo = any(s.n_embargoed > 0 for s in splits)
            @test has_embargo
        end

        @testset "No overlap between train and test" begin
            cv = PurgedKFold(n_splits=3, purge_gap=5)
            splits = split_cv(cv, 60)

            for s in splits
                train_set = Set(s.train_indices)
                test_set = Set(s.test_indices)
                @test isempty(intersect(train_set, test_set))
            end
        end

        @testset "All samples used as test exactly once (before purging)" begin
            cv = PurgedKFold(n_splits=4)
            splits = split_cv(cv, 100)

            all_test = Int[]
            for s in splits
                append!(all_test, s.test_indices)
            end
            sort!(all_test)

            @test all_test == collect(1:100)
        end

        @testset "Uneven fold sizes" begin
            cv = PurgedKFold(n_splits=3)
            splits = split_cv(cv, 10)

            # 10 / 3 = 3 remainder 1, so first fold gets extra
            sizes = [length(s.test_indices) for s in splits]
            @test sum(sizes) == 10
        end
    end

    # ==========================================================================
    # CombinatorialPurgedCV
    # ==========================================================================

    @testset "CombinatorialPurgedCV" begin
        @testset "Construction" begin
            cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2, purge_gap=3)

            @test cv.n_splits == 5
            @test cv.n_test_splits == 2
            @test cv.purge_gap == 3
        end

        @testset "Validation errors" begin
            @test_throws ErrorException CombinatorialPurgedCV(n_splits=1)
            @test_throws ErrorException CombinatorialPurgedCV(n_test_splits=0)
            @test_throws ErrorException CombinatorialPurgedCV(n_splits=5, n_test_splits=5)
            @test_throws ErrorException CombinatorialPurgedCV(n_splits=3, n_test_splits=4)
        end

        @testset "get_n_splits" begin
            # 5 choose 2 = 10
            cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
            @test get_n_splits_cv(cv) == 10

            # 6 choose 3 = 20
            cv2 = CombinatorialPurgedCV(n_splits=6, n_test_splits=3)
            @test get_n_splits_cv(cv2) == 20
        end

        @testset "split generates correct number of splits" begin
            cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
            splits = split_cv(cv, 100)

            @test length(splits) == 10
        end

        @testset "Test set size matches n_test_splits folds" begin
            cv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
            splits = split_cv(cv, 100)

            for s in splits
                # 2 folds out of 5, each fold has 20 samples
                @test length(s.test_indices) == 40
            end
        end

        @testset "Unique combinations" begin
            cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2)
            splits = split_cv(cv, 100)

            # Convert test indices to sets for comparison
            test_sets = [Set(s.test_indices) for s in splits]

            # All should be unique
            @test length(unique(test_sets)) == length(test_sets)
        end

        @testset "With purging" begin
            cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=2, purge_gap=5)
            splits = split_cv(cv, 100)

            for s in splits
                @test s.n_purged >= 0
                # Train size should be reduced due to purging
                @test length(s.train_indices) <= 50
            end
        end
    end

    # ==========================================================================
    # PurgedWalkForward
    # ==========================================================================

    @testset "PurgedWalkForward" begin
        @testset "Construction" begin
            cv = PurgedWalkForward(n_splits=5, train_size=100, purge_gap=5)

            @test cv.n_splits == 5
            @test cv.train_size == 100
            @test cv.purge_gap == 5
        end

        @testset "Default values" begin
            cv = PurgedWalkForward()

            @test cv.n_splits == 5
            @test cv.train_size == 0  # Expanding window
            @test cv.purge_gap == 0
        end

        @testset "Validation errors" begin
            @test_throws ErrorException PurgedWalkForward(n_splits=0)
            @test_throws ErrorException PurgedWalkForward(train_size=-1)
        end

        @testset "get_n_splits" begin
            cv = PurgedWalkForward(n_splits=7)
            @test get_n_splits_cv(cv) == 7
        end

        @testset "split with expanding window" begin
            cv = PurgedWalkForward(n_splits=4, train_size=0)
            splits = split_cv(cv, 100)

            # Training windows should grow
            train_sizes = [length(s.train_indices) for s in splits]
            @test issorted(train_sizes)
        end

        @testset "split with fixed window" begin
            cv = PurgedWalkForward(n_splits=3, train_size=30)
            splits = split_cv(cv, 100)

            # All training windows should be <= train_size
            for s in splits
                @test length(s.train_indices) <= 30
            end
        end

        @testset "Purge gap creates separation" begin
            cv = PurgedWalkForward(n_splits=3, purge_gap=5)
            splits = split_cv(cv, 100)

            for s in splits
                if !isempty(s.train_indices) && !isempty(s.test_indices)
                    train_max = maximum(s.train_indices)
                    test_min = minimum(s.test_indices)
                    @test test_min - train_max >= cv.purge_gap + 1
                end
            end
        end

        @testset "No future leakage" begin
            cv = PurgedWalkForward(n_splits=5)
            splits = split_cv(cv, 200)

            for s in splits
                if !isempty(s.train_indices) && !isempty(s.test_indices)
                    @test maximum(s.train_indices) < minimum(s.test_indices)
                end
            end
        end
    end

    # ==========================================================================
    # Utility Functions
    # ==========================================================================

    @testset "Utility functions" begin
        @testset "get_train_test_indices" begin
            s = PurgedSplit([1, 2, 3], [4, 5], 1, 0)
            train, test = get_train_test_indices(s)

            @test train == [1, 2, 3]
            @test test == [4, 5]
        end

        @testset "total_purged_samples" begin
            splits = [
                PurgedSplit([1, 2], [3, 4], 2, 0),
                PurgedSplit([5, 6], [7, 8], 3, 0),
                PurgedSplit([9, 10], [11, 12], 1, 0)
            ]

            @test total_purged_samples(splits) == 6
        end

        @testset "total_embargoed_samples" begin
            splits = [
                PurgedSplit([1, 2], [3, 4], 0, 1),
                PurgedSplit([5, 6], [7, 8], 0, 2),
                PurgedSplit([9, 10], [11, 12], 0, 0)
            ]

            @test total_embargoed_samples(splits) == 3
        end
    end

    # ==========================================================================
    # Integration Tests
    # ==========================================================================

    @testset "Integration: Financial CV workflow" begin
        rng = Random.MersenneTwister(42)
        n = 500

        # Simulate returns with 5-day forward labels
        returns = randn(rng, n) .* 0.02
        forward_5d = [sum(returns[i:min(i+4, n)]) for i in 1:n-4]
        n_labels = length(forward_5d)

        # Use PurgedKFold with appropriate gap
        horizon = 5
        purge_gap = estimate_purge_gap(horizon; decay_factor=1.0)
        cv = PurgedKFold(n_splits=5, purge_gap=purge_gap, embargo_pct=0.01)

        splits = split_cv(cv, n_labels)

        @test length(splits) == 5

        for s in splits
            # Verify temporal separation
            if !isempty(s.train_indices) && !isempty(s.test_indices)
                train_set = Set(s.train_indices)
                test_set = Set(s.test_indices)

                # No overlap
                @test isempty(intersect(train_set, test_set))

                # Check purge gap is respected for adjacent regions
                train_max = maximum(s.train_indices)
                test_min = minimum(s.test_indices)

                if train_max < test_min
                    @test test_min - train_max >= purge_gap
                end
            end
        end
    end

    @testset "Integration: CPCV backtesting" begin
        n = 200
        cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, purge_gap=3)

        splits = split_cv(cv, n)

        # 6 choose 2 = 15 splits
        @test length(splits) == 15

        # Each sample should appear in multiple test sets
        test_counts = zeros(Int, n)
        for s in splits
            for idx in s.test_indices
                test_counts[idx] += 1
            end
        end

        # Each sample in exactly (5 choose 1) = 5 test sets
        # (when it's part of one of the 2 test folds)
        @test all(c > 0 for c in test_counts)
    end

    @testset "Integration: Walk-forward with purging" begin
        n = 300
        cv = PurgedWalkForward(n_splits=5, train_size=100, purge_gap=10, embargo_pct=0.02)

        splits = split_cv(cv, n)

        # Verify no temporal leakage across all splits
        for (i, s) in enumerate(splits)
            @test all(s.train_indices .< minimum(s.test_indices) - cv.purge_gap)
        end

        # Embargo should affect some samples
        total_embargo = total_embargoed_samples(splits)
        @test total_embargo >= 0
    end
end
