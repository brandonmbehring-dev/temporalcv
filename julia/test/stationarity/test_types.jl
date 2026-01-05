@testset "Stationarity Types" begin
    @testset "StationarityConclusion enum" begin
        @test STATIONARY isa StationarityConclusion
        @test NON_STATIONARY isa StationarityConclusion
        @test DIFFERENCE_STATIONARY isa StationarityConclusion
        @test INSUFFICIENT_EVIDENCE isa StationarityConclusion
    end

    @testset "StationarityTestResult construction" begin
        result = StationarityTestResult(
            test_name = :ADF,
            statistic = -3.5,
            pvalue = 0.01,
            is_stationary = true,
            lags_used = 2,
            regression = :c,
            critical_values = Dict("5%" => -2.86)
        )
        @test result.test_name == :ADF
        @test result.statistic == -3.5
        @test result.pvalue == 0.01
        @test result.is_stationary == true
        @test result.lags_used == 2
        @test result.regression == :c
        @test result.critical_values["5%"] == -2.86
    end

    @testset "StationarityTestResult show" begin
        result = StationarityTestResult(
            test_name = :ADF,
            statistic = -3.5,
            pvalue = 0.01,
            is_stationary = true
        )
        str = string(result)
        @test occursin("ADF", str)
        @test occursin("STATIONARY", str)
    end

    @testset "JointStationarityResult construction" begin
        adf = StationarityTestResult(
            test_name = :ADF,
            statistic = -4.0,
            pvalue = 0.001,
            is_stationary = true
        )
        kpss = StationarityTestResult(
            test_name = :KPSS,
            statistic = 0.2,
            pvalue = 0.5,
            is_stationary = true
        )
        joint = JointStationarityResult(
            adf, kpss, STATIONARY, "Safe to model"
        )
        @test joint.adf_result === adf
        @test joint.kpss_result === kpss
        @test joint.conclusion == STATIONARY
        @test joint.recommended_action == "Safe to model"
    end

    @testset "JointStationarityResult show" begin
        adf = StationarityTestResult(
            test_name = :ADF,
            statistic = -4.0,
            pvalue = 0.001,
            is_stationary = true
        )
        kpss = StationarityTestResult(
            test_name = :KPSS,
            statistic = 0.2,
            pvalue = 0.5,
            is_stationary = true
        )
        joint = JointStationarityResult(
            adf, kpss, STATIONARY, "Safe to model"
        )
        str = string(joint)
        @test occursin("Joint", str)
        @test occursin("STATIONARY", str)
    end
end
