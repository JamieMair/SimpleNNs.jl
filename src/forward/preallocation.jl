struct ForwardPassCache{A<:AbstractArray, C}
    input::A
    layer_outputs::C
end

"""
    get_outputs(cache::ForwardPassCache)

Gets the last output from the forward pass buffer.
"""
get_outputs(cache::ForwardPassCache) = last(cache.layer_outputs)
import CUDA: zeros as cu_zeros, CuArray
function zeros_fn(model::Model)
    if typeof(model.parameters) <: CuArray
        return cu_zeros
    else
        return zeros
    end
end
function _get_preallocation_size(layer, batch_size)
    if should_preallocate(layer)
        return (unbatched_output_size(layer)..., batch_size)
    else
        return 0
    end
end

"""
    preallocate(model::Model, batch_size::Integer)

Creates a buffer to store the intermediate layer outputs of a forward pass, along with the input.

The inputs can be set using [`set_inputs!`](@ref) and the outputs can be retrieved using [`get_outputs`](@ref).
"""
function preallocate(model::Model, batch_size::Integer)
    (input_layer, network_layers) = Iterators.peel(model.layers)
    input_size = (unbatched_output_size(input_layer)..., batch_size)
    device_zeros_fn = zeros_fn(model)
    input_array = device_zeros_fn(datatype(input_layer), input_size)
    layer_outputs = Vector{Any}(undef, length(model.layers)-1)
    prev_output = input_array
    for (i, layer) in enumerate(network_layers)
        if typeof(layer) <: Flatten
            layer_outputs[i] = prev_output = reshape(prev_output, layer.output_size..., batch_size)
            continue
        end
        layer_outputs[i] = prev_output = device_zeros_fn(datatype(layer), _get_preallocation_size(layer, batch_size))
    end

    return ForwardPassCache(input_array, Tuple(layer_outputs))
end
# TODO: Add preallocate method that takes an input array and uses that to preallocate.

"""
    set_inputs!(cache::ForwardPassCache, inputs)

Sets the input array in the forward pass cache.
"""
function set_inputs!(cache::ForwardPassCache, inputs)
    cache.input .= inputs
end

export preallocate, set_inputs!, get_outputs