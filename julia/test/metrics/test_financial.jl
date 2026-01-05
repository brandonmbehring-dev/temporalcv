@testset "Financial Metrics" begin
    using Random
    rng = Random.MersenneTwister(42)

    # ==========================================================================
    # compute_sharpe_ratio
    # ==========================================================================

    @testset "compute_sharpe_ratio" begin
        @testset "Positive returns give positive Sharpe" begin
            returns = [0.01, 0.02, 0.015, 0.01, 0.025]  # All positive
            sharpe = compute_sharpe_ratio(returns)
            @test sharpe > 0
        end

        @testset "Negative returns give negative Sharpe" begin
            returns = [-0.01, -0.02, -0.015, -0.01, -0.025]  # All negative
            sharpe = compute_sharpe_ratio(returns)
            @test sharpe < 0
        end

        @testset "Zero volatility edge cases" begin
            # All same return
            returns = [0.01, 0.01, 0.01]
            @test compute_sharpe_ratio(returns) == Inf

            # All zero
            returns = [0.0, 0.0, 0.0]
            @test compute_sharpe_ratio(returns) == 0.0

            # All same negative
            returns = [-0.01, -0.01, -0.01]
            @test compute_sharpe_ratio(returns) == -Inf
        end

        @testset "Risk-free rate adjustment" begin
            returns = [0.02, 0.03, 0.01, 0.02, 0.025]  # Varying returns
            rf = 0.01

            sharpe_no_rf = compute_sharpe_ratio(returns; risk_free_rate=0.0)
            sharpe_with_rf = compute_sharpe_ratio(returns; risk_free_rate=rf)

            @test sharpe_with_rf < sharpe_no_rf
        end

        @testset "Annualization factor" begin
            returns = [0.01, 0.02, 0.015, 0.01, 0.025]

            sharpe_daily = compute_sharpe_ratio(returns; annualization=252.0)
            sharpe_weekly = compute_sharpe_ratio(returns; annualization=52.0)

            @test sharpe_daily > sharpe_weekly  # sqrt(252) > sqrt(52)
        end

        @testset "Empty returns error" begin
            @test_throws ErrorException compute_sharpe_ratio(Float64[])
        end

        @testset "Single return error" begin
            @test_throws ErrorException compute_sharpe_ratio([0.01])
        end
    end

    # ==========================================================================
    # compute_max_drawdown
    # ==========================================================================

    @testset "compute_max_drawdown" begin
        @testset "From cumulative returns" begin
            # Peak at 120, trough at 100 = 16.67% drawdown
            cumulative = [100.0, 110.0, 120.0, 100.0, 115.0]
            mdd = compute_max_drawdown(cumulative_returns=cumulative)
            @test mdd ≈ (120.0 - 100.0) / 120.0  # 16.67%
        end

        @testset "From period returns" begin
            returns = [0.10, -0.10, 0.05]  # 10%, -10%, 5%
            # Cumulative: 1.10, 0.99, 1.0395
            # Peak at 1.10, trough at 0.99
            mdd = compute_max_drawdown(returns=returns)
            @test mdd ≈ (1.10 - 0.99) / 1.10  # 10%
        end

        @testset "No drawdown (monotonic increase)" begin
            cumulative = [100.0, 105.0, 110.0, 115.0, 120.0]
            mdd = compute_max_drawdown(cumulative_returns=cumulative)
            @test mdd ≈ 0.0
        end

        @testset "Full drawdown" begin
            cumulative = [100.0, 50.0]  # 50% drawdown
            mdd = compute_max_drawdown(cumulative_returns=cumulative)
            @test mdd ≈ 0.5
        end

        @testset "Must provide input" begin
            @test_throws ErrorException compute_max_drawdown()
        end

        @testset "Empty returns error" begin
            @test_throws ErrorException compute_max_drawdown(returns=Float64[])
        end
    end

    # ==========================================================================
    # compute_cumulative_return
    # ==========================================================================

    @testset "compute_cumulative_return" begin
        @testset "Geometric compounding" begin
            returns = [0.10, 0.10]  # Two 10% gains
            cum = compute_cumulative_return(returns; method=:geometric)
            # (1.1) * (1.1) - 1 = 0.21
            @test cum ≈ 0.21
        end

        @testset "Arithmetic sum" begin
            returns = [0.10, 0.10]
            cum = compute_cumulative_return(returns; method=:arithmetic)
            @test cum ≈ 0.20  # Simple sum
        end

        @testset "Geometric with losses" begin
            returns = [0.10, -0.10]  # Up 10%, down 10%
            cum = compute_cumulative_return(returns; method=:geometric)
            # (1.1) * (0.9) - 1 = -0.01
            @test cum ≈ -0.01
        end

        @testset "Invalid method error" begin
            @test_throws ErrorException compute_cumulative_return([0.1]; method=:invalid)
        end

        @testset "Empty returns error" begin
            @test_throws ErrorException compute_cumulative_return(Float64[])
        end
    end

    # ==========================================================================
    # compute_information_ratio
    # ==========================================================================

    @testset "compute_information_ratio" begin
        @testset "Outperforming benchmark" begin
            portfolio = [0.02, 0.03, 0.01, 0.02, 0.03]
            benchmark = [0.01, 0.02, 0.01, 0.01, 0.02]  # Lower returns

            ir = compute_information_ratio(portfolio, benchmark)
            @test ir > 0  # Positive alpha
        end

        @testset "Underperforming benchmark" begin
            portfolio = [0.01, 0.01, 0.01, 0.01, 0.01]
            benchmark = [0.02, 0.02, 0.02, 0.02, 0.02]

            ir = compute_information_ratio(portfolio, benchmark)
            @test ir < 0  # Negative alpha
        end

        @testset "Zero tracking error (identical)" begin
            returns = [0.01, 0.02, 0.01]

            @test compute_information_ratio(returns, returns) == 0.0
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_information_ratio([0.01, 0.02], [0.01])
        end

        @testset "Empty returns error" begin
            @test_throws ErrorException compute_information_ratio(Float64[], Float64[])
        end

        @testset "Single return error" begin
            @test_throws ErrorException compute_information_ratio([0.01], [0.01])
        end
    end

    # ==========================================================================
    # compute_hit_rate
    # ==========================================================================

    @testset "compute_hit_rate" begin
        @testset "Perfect hit rate" begin
            predicted = [1.0, -1.0, 1.0, -1.0]
            actual = [0.5, -0.5, 0.5, -0.5]  # Same direction
            @test compute_hit_rate(predicted, actual) ≈ 1.0
        end

        @testset "Zero hit rate" begin
            predicted = [1.0, -1.0, 1.0, -1.0]
            actual = [-0.5, 0.5, -0.5, 0.5]  # Opposite direction
            @test compute_hit_rate(predicted, actual) ≈ 0.0
        end

        @testset "50% hit rate" begin
            predicted = [1.0, 1.0, -1.0, -1.0]
            actual = [0.5, -0.5, 0.5, -0.5]  # Two right, two wrong
            @test compute_hit_rate(predicted, actual) ≈ 0.5
        end

        @testset "Zero predictions are conservative hits" begin
            predicted = [0.0, 0.0]
            actual = [1.0, -1.0]
            # Zeros count as matches (no prediction = no miss)
            @test compute_hit_rate(predicted, actual) ≈ 1.0
        end

        @testset "Zero actuals are conservative hits" begin
            predicted = [1.0, -1.0]
            actual = [0.0, 0.0]
            @test compute_hit_rate(predicted, actual) ≈ 1.0
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_hit_rate([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays error" begin
            @test_throws ErrorException compute_hit_rate(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # compute_profit_factor
    # ==========================================================================

    @testset "compute_profit_factor" begin
        @testset "All winning trades" begin
            predicted = [1.0, 1.0, 1.0]  # All long
            actual = [0.02, 0.01, 0.03]  # All positive returns

            pf = compute_profit_factor(predicted, actual)
            @test pf == Inf  # No losses
        end

        @testset "All losing trades" begin
            predicted = [1.0, 1.0, 1.0]  # All long
            actual = [-0.02, -0.01, -0.03]  # All negative returns

            pf = compute_profit_factor(predicted, actual)
            @test pf == 0.0  # No profits
        end

        @testset "Balanced trades" begin
            predicted = [1.0, 1.0]  # Both long
            actual = [0.02, -0.02]  # Win and lose same amount

            pf = compute_profit_factor(predicted, actual)
            @test pf ≈ 1.0  # Break even
        end

        @testset "Profitable strategy" begin
            predicted = [1.0, -1.0, 1.0, -1.0]
            actual = [0.02, -0.01, 0.01, 0.02]
            # Long on +2%: profit
            # Short on -1%: profit (neg × neg = pos)
            # Long on +1%: profit
            # Short on +2%: loss (neg × pos = neg)

            pf = compute_profit_factor(predicted, actual)
            @test pf > 1.0
        end

        @testset "Short positions" begin
            predicted = [-1.0, -1.0]  # Both short
            actual = [-0.02, 0.01]
            # Short on -2%: profit = 0.02
            # Short on +1%: loss = 0.01

            pf = compute_profit_factor(predicted, actual)
            @test pf ≈ 2.0  # Profit 0.02 / Loss 0.01
        end

        @testset "Length mismatch error" begin
            @test_throws ErrorException compute_profit_factor([1.0, 2.0], [1.0])
        end

        @testset "Empty arrays error" begin
            @test_throws ErrorException compute_profit_factor(Float64[], Float64[])
        end
    end

    # ==========================================================================
    # compute_calmar_ratio
    # ==========================================================================

    @testset "compute_calmar_ratio" begin
        @testset "Positive returns with drawdown" begin
            # Positive overall with some volatility
            returns = [0.05, -0.02, 0.03, -0.01, 0.04]

            calmar = compute_calmar_ratio(returns; annualization=252.0)
            @test isfinite(calmar)
            @test calmar > 0  # Positive returns / positive drawdown
        end

        @testset "No drawdown (monotonic gains)" begin
            returns = [0.01, 0.01, 0.01, 0.01, 0.01]

            calmar = compute_calmar_ratio(returns)
            @test calmar == Inf  # No drawdown
        end

        @testset "Negative cumulative return" begin
            returns = [-0.05, -0.05, 0.01, -0.03]

            calmar = compute_calmar_ratio(returns)
            @test calmar < 0  # Negative return
        end

        @testset "Empty returns error" begin
            @test_throws ErrorException compute_calmar_ratio(Float64[])
        end
    end

    # ==========================================================================
    # Integration
    # ==========================================================================

    @testset "Integration: Portfolio evaluation workflow" begin
        rng_int = Random.MersenneTwister(456)
        n = 252  # One year of daily data

        # Simulate portfolio returns (slightly positive trend)
        portfolio_returns = 0.0005 .+ 0.015 .* randn(rng_int, n)
        benchmark_returns = 0.0003 .+ 0.012 .* randn(rng_int, n)

        # Compute various metrics
        sharpe = compute_sharpe_ratio(portfolio_returns)
        ir = compute_information_ratio(portfolio_returns, benchmark_returns)
        mdd = compute_max_drawdown(returns=portfolio_returns)
        calmar = compute_calmar_ratio(portfolio_returns)
        cum_ret = compute_cumulative_return(portfolio_returns)

        # All should be finite
        @test isfinite(sharpe)
        @test isfinite(ir)
        @test isfinite(mdd)
        @test 0.0 <= mdd <= 1.0  # Drawdown is a fraction
        @test isfinite(cum_ret)

        # Calmar may be Inf if no drawdown, but should not be NaN
        @test !isnan(calmar)

        # Directional analysis
        pred_changes = portfolio_returns[2:end]  # Use current return as prediction
        actual_changes = portfolio_returns[2:end]

        hr = compute_hit_rate(pred_changes, actual_changes)
        @test hr ≈ 1.0  # Same as prediction

        pf = compute_profit_factor(pred_changes, actual_changes)
        @test isfinite(pf) || pf == Inf
    end

    @testset "Integration: Trading strategy evaluation" begin
        rng_int = Random.MersenneTwister(789)
        n = 100

        # Generate random predictions and returns (mostly independent)
        predictions = randn(rng_int, n)
        actual_returns = randn(rng_int, n) .* 0.02

        hr = compute_hit_rate(predictions, actual_returns)
        # Random predictions should give ~50% hit rate
        @test 0.3 < hr < 0.7

        pf = compute_profit_factor(predictions, actual_returns)
        @test isfinite(pf) || pf == Inf || pf == 0.0
    end
end
