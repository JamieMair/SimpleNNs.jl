# Defines the forward pass of a model
using LinearAlgebra
import NNlib
import NNlibCUDA
using CUDA
import CUDA: i32
include("preallocation.jl")
include("activations.jl")
include("dense.jl")
include("conv.jl")
include("flatten.jl")
include("maxpool.jl")
function forward_inner!(layer_output, layer, current_input)
    inner_layer = _inner_layer(layer)
    params = parameters(layer)
    forward!(layer_output, inner_layer, params, current_input)
    current_input = layer_output
    return current_input
end

forward!(cache::ForwardPassCache, model::Model) = _forward!(cache, model.layers)

@generated function _forward!(cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}) where {N}
    first_line = :(current_input = cache.input)
    calls = [:(current_input = forward_inner!(cache.layer_outputs[$i - 1], layers[$i], current_input)) for i in 2:N]
    Expr(:block, first_line, calls...)
end

export forward!