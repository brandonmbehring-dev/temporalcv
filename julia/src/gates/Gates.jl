"""
    Gates

Validation Gates Module.

Three-stage validation framework with HALT/PASS/WARN/SKIP decisions:

1. **External validation**: Shuffled target, Synthetic AR(1)
2. **Internal validation**: Suspicious improvement detection
3. **Statistical validation**: Residual diagnostics

The key insight: if a model shows suspiciously large improvement or
significantly outperforms theoretical bounds, it's likely learning from leakage.

# Knowledge Tiers
- [T1] Walk-forward validation framework (Tashman 2000)
- [T1] Ljung-Box test for autocorrelation (Ljung & Box 1978)
- [T2] "External-first" validation ordering (synthetic → shuffled → internal)
- [T3] 20% improvement threshold = "too good to be true" heuristic

# Example
```julia
using TemporalValidation.Gates

report = run_gates([
    gate_suspicious_improvement(model_mae, baseline_mae),
    gate_temporal_boundary(train_end, test_start, horizon),
    gate_residual_diagnostics(residuals)
])

if status(report) == "HALT"
    println(summary(report))
end
```

# References
- Hewamalage, H., Bergmeir, C. & Bandara, K. (2023). Forecast evaluation
  for data scientists: Common pitfalls and best practices.
- Tashman, L.J. (2000). Out-of-sample tests of forecasting accuracy.
"""
module Gates

using Statistics
using Distributions

# =============================================================================
# Types
# =============================================================================

include("types.jl")

# =============================================================================
# Gate Functions
# =============================================================================

include("suspicious.jl")
include("temporal_boundary.jl")
include("residual.jl")
include("runner.jl")

# =============================================================================
# Exports
# =============================================================================

# Enums and types
export GateStatus, HALT, WARN, PASS, SKIP
export GateResult, ValidationReport

# Type accessors
export status, failures, warnings, summary

# Gate functions
export gate_suspicious_improvement
export gate_temporal_boundary
export gate_residual_diagnostics

# Utilities
export compute_acf, ljung_box_test
export run_gates

end # module Gates
