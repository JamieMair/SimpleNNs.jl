using TestItemRunner
using TestItems

@run_package_tests

include("chain.jl")
include("forward.jl")
include("conv.jl")
include("backward.jl")
include("losses.jl")
include("initialisers.jl")
include("performance.jl")