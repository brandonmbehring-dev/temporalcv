# =============================================================================
# Gate Runner
# =============================================================================

"""
    run_gates(gates::Vector{GateResult}) -> ValidationReport

Aggregate gate results into a validation report.

# Arguments
- `gates`: Pre-computed gate results

# Returns
`ValidationReport` with aggregated results.

# Example
```julia
results = [
    gate_suspicious_improvement(model_mae, persistence_mae),
    gate_temporal_boundary(train_end, test_start, horizon)
]
report = run_gates(results)
if status(report) == "HALT"
    println(summary(report))
end
```
"""
function run_gates(gates::Vector{GateResult})
    return ValidationReport(gates)
end
