# Defines the forward pass of a model
using LinearAlgebra
include("preallocation.jl")
include("activations.jl")
include("dense.jl")
include("conv.jl")
include("flatten.jl")
include("maxpool.jl")
include("losses.jl")

function forward_inner!(layer_output, layer::AbstractParameterisedLayer, current_input)
    inner_layer = _inner_layer(layer)
    params = parameters(layer)
    forward!(layer_output, inner_layer, params, current_input)
    current_input = layer_output
    return current_input
end

"""
    forward!(cache::ForwardPassCache, model::Model)

Execute a forward pass through the neural network model.

This function computes the forward propagation through all layers of the model,
storing intermediate results in the pre-allocated cache. This is a zero-allocation
operation when used with properly pre-allocated caches.

# Arguments
- `cache::ForwardPassCache`: Pre-allocated cache containing input data and space for intermediate results
- `model::Model`: The neural network model to evaluate

# Returns
- The cache object (for convenience), with updated intermediate and output values

# Examples
```julia
# Create model and data
model = chain(Static(4), Dense(8, activation_fn=relu), Dense(1))
inputs = randn(Float32, 4, 32)  # 32 samples, 4 features each

# Pre-allocate cache and set inputs
cache = preallocate(model, 32)
set_inputs!(cache, inputs)

# Execute forward pass
forward!(cache, model)

# Get outputs
outputs = get_outputs(cache)
```

# Notes
- Requires pre-allocated cache from `preallocate(model, batch_size)`
- Input data must be set using `set_inputs!(cache, inputs)` before calling
- This is a mutating operation that modifies the cache in-place
- Designed for zero allocations when properly used
- Works on both CPU and GPU when model and data are on the same device

See also: [`preallocate`](@ref), [`set_inputs!`](@ref), [`get_outputs`](@ref)
"""
forward!(cache::ForwardPassCache, model::Model) = _forward!(cache, model.layers)

@generated function _forward!(cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}) where {N}
    first_line = :(current_input = cache.input)
    calls = [:(current_input = forward_inner!(cache.layer_outputs[$i - 1], layers[$i], current_input)) for i in 2:N]
    Expr(:block, first_line, calls...)
end


export forward!