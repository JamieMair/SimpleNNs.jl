using TestItemRunner
using TestItems

@run_package_tests

include("chain.jl")
include("forward.jl")

@testitem "Basic Test" begin
    @test 1+1 == 2
end