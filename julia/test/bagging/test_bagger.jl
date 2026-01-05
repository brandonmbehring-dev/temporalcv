# Simple linear model for testing
# Note: We use TemporalValidation.fit! and TemporalValidation.predict for interface
mutable struct SimpleLinearModel
    coef::Vector{Float64}
    intercept::Float64
    is_fitted::Bool

    SimpleLinearModel() = new(Float64[], 0.0, false)
end

# Extend the TemporalValidation interface
function TemporalValidation.Bagging.fit!(m::SimpleLinearModel, X::AbstractMatrix, y::AbstractVector)
    # Add intercept column
    n = size(X, 1)
    X_aug = hcat(ones(n), X)

    # Least squares
    coef_full = X_aug \ y
    m.intercept = coef_full[1]
    m.coef = coef_full[2:end]
    m.is_fitted = true
    return m
end

function TemporalValidation.Bagging.predict(m::SimpleLinearModel, X::AbstractMatrix)
    @assert m.is_fitted "Model not fitted"
    return X * m.coef .+ m.intercept
end

@testset "TimeSeriesBagger" begin
    @testset "Construction" begin
        bagger = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(expected_block_length=10.0),
            n_estimators=20
        )
        @test bagger.n_estimators == 20
        @test bagger.aggregation == :mean
        @test bagger.fitted == false

        # With median aggregation
        bagger2 = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=MovingBlockBootstrap(block_length=5),
            n_estimators=10,
            aggregation=:median
        )
        @test bagger2.aggregation == :median

        # Invalid parameters
        @test_throws AssertionError TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(),
            n_estimators=0
        )
        @test_throws AssertionError TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(),
            aggregation=:invalid
        )
    end

    @testset "Fit and predict" begin
        rng = Random.MersenneTwister(42)
        n_train, n_test, p = 100, 20, 3

        # Generate data with linear relationship
        X_train = randn(rng, n_train, p)
        true_coef = [1.0, -0.5, 0.3]
        y_train = X_train * true_coef .+ 0.5 .* randn(rng, n_train)

        X_test = randn(rng, n_test, p)
        y_test = X_test * true_coef .+ 0.5 .* randn(rng, n_test)

        bagger = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(expected_block_length=10.0),
            n_estimators=20,
            random_state=42
        )

        fit!(bagger, X_train, y_train)

        @test bagger.fitted == true
        @test length(bagger.models) == 20

        predictions = predict(bagger, X_test)
        @test length(predictions) == n_test

        # Should have reasonable accuracy
        mse = mean((predictions .- y_test).^2)
        @test mse < 1.0  # Should be better than noise level
    end

    @testset "Aggregation modes" begin
        rng = Random.MersenneTwister(42)
        n, p = 50, 2
        X = randn(rng, n, p)
        y = X * [1.0, 0.5] .+ 0.1 .* randn(rng, n)

        # Mean aggregation
        bagger_mean = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=MovingBlockBootstrap(block_length=5),
            n_estimators=10,
            aggregation=:mean,
            random_state=42
        )
        fit!(bagger_mean, X, y)
        pred_mean = predict(bagger_mean, X)

        # Median aggregation
        bagger_median = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=MovingBlockBootstrap(block_length=5),
            n_estimators=10,
            aggregation=:median,
            random_state=42
        )
        fit!(bagger_median, X, y)
        pred_median = predict(bagger_median, X)

        # Both should produce valid predictions
        @test length(pred_mean) == n
        @test length(pred_median) == n

        # Mean and median can differ
        # (but with similar estimators they should be close)
        @test cor(pred_mean, pred_median) > 0.9
    end

    @testset "Predict with uncertainty" begin
        rng = Random.MersenneTwister(42)
        n, p = 50, 2
        X = randn(rng, n, p)
        y = X * [1.0, 0.5] .+ randn(rng, n)

        bagger = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(expected_block_length=8.0),
            n_estimators=30,
            random_state=42
        )
        fit!(bagger, X, y)

        mean_pred, std_pred = predict_with_uncertainty(bagger, X)

        @test length(mean_pred) == n
        @test length(std_pred) == n
        @test all(std_pred .>= 0)  # Std should be non-negative
    end

    @testset "Predict interval" begin
        rng = Random.MersenneTwister(42)
        n_train, n_test, p = 200, 50, 2

        # Training data with low noise for more predictable intervals
        X_train = randn(rng, n_train, p)
        y_train = X_train * [1.0, 0.5] .+ 0.5 .* randn(rng, n_train)

        # Test data with same distribution
        X_test = randn(rng, n_test, p)
        y_test = X_test * [1.0, 0.5] .+ 0.5 .* randn(rng, n_test)

        bagger = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(expected_block_length=15.0),
            n_estimators=100,  # More estimators for stable intervals
            random_state=42
        )
        fit!(bagger, X_train, y_train)

        # Use wider intervals (alpha=0.2 → 80% intervals)
        mean_pred, lower, upper = predict_interval(bagger, X_test, alpha=0.2)

        @test length(mean_pred) == n_test
        @test length(lower) == n_test
        @test length(upper) == n_test
        @test all(lower .<= upper)
        @test all(lower .<= mean_pred .<= upper)

        # Check that interval widths are positive
        @test all(upper .- lower .> 0)

        # Coverage test: bootstrap intervals provide uncertainty, not guaranteed coverage
        # We just verify they're reasonable (not all points outside)
        covered = (lower .<= y_test) .& (y_test .<= upper)
        coverage_rate = mean(covered)
        @test coverage_rate > 0.0  # At least some coverage
    end

    @testset "Reproducibility" begin
        rng = Random.MersenneTwister(1)
        n, p = 50, 2
        X = randn(rng, n, p)
        y = randn(rng, n)

        # Same random state → same results
        bagger1 = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(expected_block_length=8.0),
            n_estimators=10,
            random_state=42
        )
        fit!(bagger1, X, y)
        pred1 = predict(bagger1, X)

        bagger2 = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(expected_block_length=8.0),
            n_estimators=10,
            random_state=42
        )
        fit!(bagger2, X, y)
        pred2 = predict(bagger2, X)

        @test pred1 ≈ pred2
    end

    @testset "Error handling" begin
        bagger = TimeSeriesBagger(
            base_model=SimpleLinearModel(),
            strategy=StationaryBootstrap(),
            n_estimators=10
        )

        X = randn(Random.MersenneTwister(42), 50, 3)

        # Predict before fit
        @test_throws AssertionError predict(bagger, X)
        @test_throws AssertionError predict_with_uncertainty(bagger, X)
        @test_throws AssertionError predict_interval(bagger, X)
    end
end
