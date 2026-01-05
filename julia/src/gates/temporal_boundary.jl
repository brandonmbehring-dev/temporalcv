# =============================================================================
# Temporal Boundary Gate
# =============================================================================

"""
    gate_temporal_boundary(train_end_idx, test_start_idx, horizon; gap=0)

Verify temporal boundary enforcement.

Ensures proper gap between training end and test start for h-step forecasts.

# Arguments
- `train_end_idx`: Last index of training data (inclusive, 1-based)
- `test_start_idx`: First index of test data (1-based)
- `horizon`: Forecast horizon (h)
- `gap`: Additional gap beyond horizon requirement (default 0)

# Returns
`GateResult`:
- HALT if temporal boundary is violated
- PASS if gap is sufficient

# Notes
For h-step ahead forecasting, the last training observation should be
at least h periods before the first test observation to prevent leakage.

Required: `test_start_idx >= train_end_idx + horizon + gap + 1`
(the +1 accounts for Julia's 1-based indexing)

# Knowledge Tier
[T1] Gap >= horizon prevents information leakage for h-step forecasts

# Example
```julia
# With training ending at index 100, test starting at 103, horizon 2
result = gate_temporal_boundary(100, 103, 2)
# actual_gap = 103 - 100 - 1 = 2 >= required 2 → PASS
```
"""
function gate_temporal_boundary(
    train_end_idx::Int,
    test_start_idx::Int,
    horizon::Int;
    gap::Int = 0
)
    required_gap = horizon + gap
    actual_gap = test_start_idx - train_end_idx - 1

    details = Dict{String, Any}(
        "train_end_idx" => train_end_idx,
        "test_start_idx" => test_start_idx,
        "horizon" => horizon,
        "gap" => gap,
        "required_gap" => required_gap,
        "actual_gap" => actual_gap
    )

    if actual_gap < required_gap
        return GateResult(
            name = "temporal_boundary",
            status = HALT,
            message = "Gap $actual_gap < required $required_gap for h=$horizon",
            metric_value = Float64(actual_gap),
            threshold = Float64(required_gap),
            details = details,
            recommendation = "Increase gap between train and test. Need $(required_gap - actual_gap) more periods."
        )
    end

    return GateResult(
        name = "temporal_boundary",
        status = PASS,
        message = "Gap $actual_gap >= required $required_gap",
        metric_value = Float64(actual_gap),
        threshold = Float64(required_gap),
        details = details
    )
end
