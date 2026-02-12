using TestItemRunner
using TestItems

@run_package_tests

include("backward.jl")
include("chain.jl")
include("conv.jl")
include("edge_cases.jl")
include("forward.jl")
include("gpu.jl")
include("gradient_checking.jl")
include("initialisers.jl")
include("loss_api.jl")
include("losses.jl")
include("optimisers.jl")
include("performance.jl")