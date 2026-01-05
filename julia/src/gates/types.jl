# =============================================================================
# Gates - Types and Enums
# =============================================================================

"""
    GateStatus

Validation gate status enum.

- `HALT`: Critical failure - stop and investigate
- `WARN`: Caution - continue but verify
- `PASS`: Validation passed
- `SKIP`: Insufficient data to run gate
"""
@enum GateStatus begin
    HALT
    WARN
    PASS
    SKIP
end


"""
    GateResult

Result from a validation gate.

# Fields
- `name::String`: Gate identifier (e.g., "shuffled_target", "suspicious_improvement")
- `status::GateStatus`: HALT, WARN, PASS, or SKIP
- `message::String`: Human-readable description of result
- `metric_value::Union{Float64, Nothing}`: Primary metric for this gate
- `threshold::Union{Float64, Nothing}`: Threshold used for decision
- `recommendation::String`: What to do if not PASS
- `details::Dict{String, Any}`: Additional metrics and diagnostics

# Example
```julia
result = gate_suspicious_improvement(0.05, 0.10)
if result.status == HALT
    println("Investigation required: \$(result.recommendation)")
end
```
"""
struct GateResult
    name::String
    status::GateStatus
    message::String
    metric_value::Union{Float64, Nothing}
    threshold::Union{Float64, Nothing}
    recommendation::String
    details::Dict{String, Any}

    function GateResult(;
        name::String,
        status::GateStatus,
        message::String,
        metric_value::Union{Float64, Nothing} = nothing,
        threshold::Union{Float64, Nothing} = nothing,
        recommendation::String = "",
        details::Dict{String, Any} = Dict{String, Any}()
    )
        new(name, status, message, metric_value, threshold, recommendation, details)
    end
end

function Base.show(io::IO, r::GateResult)
    print(io, "[$(r.status)] $(r.name): $(r.message)")
end


"""
    ValidationReport

Complete validation report across all gates.

# Fields
- `gates::Vector{GateResult}`: Results from all gates run

# Properties
- `status`: Overall status (HALT if any HALT, WARN if any WARN, else PASS)
- `failures`: Gates that HALTed
- `warnings`: Gates that WARNed
"""
struct ValidationReport
    gates::Vector{GateResult}
end

"""Overall status: HALT if any HALT, WARN if any WARN, else PASS."""
function status(r::ValidationReport)
    if any(g.status == HALT for g in r.gates)
        return "HALT"
    elseif any(g.status == WARN for g in r.gates)
        return "WARN"
    else
        return "PASS"
    end
end

"""Return gates that HALTed."""
failures(r::ValidationReport) = filter(g -> g.status == HALT, r.gates)

"""Return gates that WARNed."""
warnings(r::ValidationReport) = filter(g -> g.status == WARN, r.gates)

"""Return human-readable summary."""
function summary(r::ValidationReport)
    lines = [
        "=" ^ 60,
        "VALIDATION REPORT",
        "=" ^ 60,
        ""
    ]

    for gate in r.gates
        push!(lines, "  $(gate)")
    end

    push!(lines, "")
    push!(lines, "=" ^ 60)
    push!(lines, "OVERALL STATUS: $(status(r))")
    push!(lines, "=" ^ 60)

    fs = failures(r)
    if !isempty(fs)
        push!(lines, "")
        push!(lines, "HALTED GATES (require investigation):")
        for gate in fs
            push!(lines, "  - $(gate.name): $(gate.recommendation)")
        end
    end

    return join(lines, "\n")
end
