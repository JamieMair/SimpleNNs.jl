module SimpleNNs
using Logging
using Requires

include("layers/layers.jl")
include("utils.jl")
include("chain.jl")
include("forward/forward.jl")
include("backprop/backprop.jl")
include("gpu.jl")
include("optimisers/optimisers.jl")

# API
export Static, Dense, Conv, MaxPool, Flatten, chain, sigmoid, relu, tanh_fast, parameters, gradients
export MSELoss, LogitCrossEntropyLoss
export forward!, preallocate, preallocate_grads, set_inputs!, get_outputs, backprop!
export truncate
export gpu


# Backwards compatibility for older Julia versions
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
            @require NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd" begin
                @require cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd" begin
                    include("../ext/SimpleNNsCUDAExt.jl")
                end
            end
        end
    end
end

end
