using TestItemRunner
using TestItems

@run_package_tests

@testitem "Basic Test" begin
    @test 1+1 == 2
end