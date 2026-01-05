@testset "Gates Types" begin
    @testset "GateStatus enum" begin
        @test HALT isa GateStatus
        @test WARN isa GateStatus
        @test PASS isa GateStatus
        @test SKIP isa GateStatus
    end

    @testset "GateResult construction" begin
        result = GateResult(
            name = "test_gate",
            status = PASS,
            message = "Test passed"
        )
        @test result.name == "test_gate"
        @test result.status == PASS
        @test result.message == "Test passed"
        @test isnothing(result.metric_value)
        @test isnothing(result.threshold)
        @test result.recommendation == ""
        @test isempty(result.details)
    end

    @testset "GateResult with all fields" begin
        details = Dict{String, Any}("key" => "value")
        result = GateResult(
            name = "full_gate",
            status = HALT,
            message = "Test failed",
            metric_value = 0.25,
            threshold = 0.20,
            recommendation = "Investigate",
            details = details
        )
        @test result.metric_value == 0.25
        @test result.threshold == 0.20
        @test result.recommendation == "Investigate"
        @test result.details["key"] == "value"
    end

    @testset "GateResult show" begin
        result = GateResult(
            name = "test_gate",
            status = HALT,
            message = "Something went wrong"
        )
        str = string(result)
        @test occursin("HALT", str)
        @test occursin("test_gate", str)
        @test occursin("Something went wrong", str)
    end

    @testset "ValidationReport" begin
        gate1 = GateResult(name = "gate1", status = PASS, message = "OK")
        gate2 = GateResult(name = "gate2", status = WARN, message = "Caution")
        gate3 = GateResult(name = "gate3", status = HALT, message = "Failed")

        @testset "All PASS" begin
            report = ValidationReport([gate1])
            @test status(report) == "PASS"
            @test isempty(failures(report))
            @test isempty(warnings(report))
        end

        @testset "Contains WARN" begin
            report = ValidationReport([gate1, gate2])
            @test status(report) == "WARN"
            @test isempty(failures(report))
            @test length(warnings(report)) == 1
        end

        @testset "Contains HALT" begin
            report = ValidationReport([gate1, gate2, gate3])
            @test status(report) == "HALT"
            @test length(failures(report)) == 1
            @test length(warnings(report)) == 1
        end

        @testset "summary" begin
            report = ValidationReport([gate1, gate3])
            s = Gates.summary(report)  # Use qualified name
            @test occursin("VALIDATION REPORT", s)
            @test occursin("HALT", s)
            @test occursin("gate3", s)
        end
    end
end
