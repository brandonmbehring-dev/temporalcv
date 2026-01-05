# =============================================================================
# Statistical Tests - Result Types
# =============================================================================

"""
    DMTestResult

Result from Diebold-Mariano test for equal predictive accuracy.

# Fields
- `statistic::Float64`: DM test statistic (asymptotically N(0,1) under H0)
- `pvalue::Float64`: P-value for the test
- `h::Int`: Forecast horizon used
- `n::Int`: Number of observations
- `loss::Symbol`: Loss function used (:squared or :absolute)
- `alternative::Symbol`: Alternative hypothesis (:two_sided, :less, :greater)
- `harvey_adjusted::Bool`: Whether Harvey et al. (1997) adjustment was applied
- `mean_loss_diff::Float64`: Mean loss differential (positive = model 1 has higher loss)

# Example
```julia
result = dm_test(model_errors, baseline_errors, h=2)
if significant_at_05(result)
    println("Model significantly different from baseline")
end
```
"""
struct DMTestResult
    statistic::Float64
    pvalue::Float64
    h::Int
    n::Int
    loss::Symbol
    alternative::Symbol
    harvey_adjusted::Bool
    mean_loss_diff::Float64
end

"""Is result significant at alpha=0.05?"""
significant_at_05(r::DMTestResult) = r.pvalue < 0.05

"""Is result significant at alpha=0.01?"""
significant_at_01(r::DMTestResult) = r.pvalue < 0.01

function Base.show(io::IO, r::DMTestResult)
    sig = r.pvalue < 0.01 ? "***" : r.pvalue < 0.05 ? "**" : r.pvalue < 0.10 ? "*" : ""
    print(io, "DM($(r.h)): $(round(r.statistic, digits=3)) (p=$(round(r.pvalue, digits=4)))$(sig)")
end


"""
    PTTestResult

Result from Pesaran-Timmermann test for directional accuracy.

# Fields
- `statistic::Float64`: PT test statistic (z-score, asymptotically N(0,1) under H0)
- `pvalue::Float64`: P-value (one-sided: testing if better than random)
- `accuracy::Float64`: Observed directional accuracy
- `expected::Float64`: Expected accuracy under null hypothesis (independence)
- `n::Int`: Number of observations
- `n_classes::Int`: Number of direction classes (2 or 3)

# Example
```julia
result = pt_test(actual_changes, predicted_changes)
println("Accuracy: \$(round(result.accuracy * 100, digits=1))%")
```
"""
struct PTTestResult
    statistic::Float64
    pvalue::Float64
    accuracy::Float64
    expected::Float64
    n::Int
    n_classes::Int
end

"""Is directional accuracy significantly better than random?"""
significant_at_05(r::PTTestResult) = r.pvalue < 0.05

"""Directional skill = accuracy - expected."""
skill(r::PTTestResult) = r.accuracy - r.expected

function Base.show(io::IO, r::PTTestResult)
    sig = r.pvalue < 0.01 ? "***" : r.pvalue < 0.05 ? "**" : r.pvalue < 0.10 ? "*" : ""
    acc_pct = round(r.accuracy * 100, digits=1)
    exp_pct = round(r.expected * 100, digits=1)
    print(io, "PT: $(acc_pct)% vs $(exp_pct)% expected (z=$(round(r.statistic, digits=3)), p=$(round(r.pvalue, digits=4)))$(sig)")
end
