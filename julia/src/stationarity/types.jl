# =============================================================================
# Types for Stationarity Testing
# =============================================================================

"""
    StationarityConclusion

Joint interpretation of ADF + KPSS stationarity tests.

Values:
- `STATIONARY`: ADF rejects unit root + KPSS fails to reject stationarity
- `NON_STATIONARY`: ADF fails to reject + KPSS rejects stationarity
- `DIFFERENCE_STATIONARY`: Both tests reject (may need differencing)
- `INSUFFICIENT_EVIDENCE`: Neither test conclusive
"""
@enum StationarityConclusion begin
    STATIONARY
    NON_STATIONARY
    DIFFERENCE_STATIONARY
    INSUFFICIENT_EVIDENCE
end

"""
    StationarityTestResult

Result of a unit root or stationarity test.

# Fields
- `test_name::Symbol`: Name of the test (:ADF, :KPSS, :PP, :AR1)
- `statistic::Float64`: Test statistic value
- `pvalue::Float64`: P-value for the test
- `is_stationary::Bool`: Stationarity conclusion at given alpha
- `lags_used::Int`: Number of lags used in the test
- `regression::Symbol`: Regression type (:c, :ct, :n)
- `critical_values::Dict{String, Float64}`: Critical values at standard levels
"""
struct StationarityTestResult
    test_name::Symbol
    statistic::Float64
    pvalue::Float64
    is_stationary::Bool
    lags_used::Int
    regression::Symbol
    critical_values::Dict{String, Float64}
end

# Constructor with defaults
function StationarityTestResult(;
    test_name::Symbol,
    statistic::Float64,
    pvalue::Float64,
    is_stationary::Bool,
    lags_used::Int = 0,
    regression::Symbol = :c,
    critical_values::Dict{String, Float64} = Dict{String, Float64}()
)
    StationarityTestResult(
        test_name, statistic, pvalue, is_stationary,
        lags_used, regression, critical_values
    )
end

function Base.show(io::IO, r::StationarityTestResult)
    status = r.is_stationary ? "STATIONARY" : "NON-STATIONARY"
    print(io, "$(r.test_name) Test: $(status) (stat=$(round(r.statistic, digits=4)), p=$(round(r.pvalue, digits=4)))")
end

"""
    JointStationarityResult

Result of joint ADF + KPSS stationarity check.

# Fields
- `adf_result::StationarityTestResult`: ADF test result
- `kpss_result::StationarityTestResult`: KPSS test result
- `conclusion::StationarityConclusion`: Joint interpretation
- `recommended_action::String`: Suggested next step
"""
struct JointStationarityResult
    adf_result::StationarityTestResult
    kpss_result::StationarityTestResult
    conclusion::StationarityConclusion
    recommended_action::String
end

function Base.show(io::IO, r::JointStationarityResult)
    println(io, "Joint Stationarity Analysis")
    println(io, "───────────────────────────")
    println(io, "  ADF:  $(r.adf_result.is_stationary ? "rejects unit root" : "fails to reject")")
    println(io, "  KPSS: $(r.kpss_result.is_stationary ? "fails to reject stationarity" : "rejects stationarity")")
    println(io, "  Conclusion: $(r.conclusion)")
    print(io, "  Action: $(r.recommended_action)")
end
