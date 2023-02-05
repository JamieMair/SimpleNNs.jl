struct ForwardPassCache{A<:AbstractArray, B<:AbstractArray, C<:AbstractArray{B}}
    input::A
    layer_outputs::C
end
get_outputs(cache::ForwardPassCache) = last(cache.layer_outputs)
import CUDA: zeros as cu_zeros, CuArray
function zeros_fn(model::Model)
    if typeof(model.parameters) <: CuArray
        return cu_zeros
    else
        return zeros
    end
end
function preallocate(model::Model, batch_size::Integer)
    (input_layer, network_layers) = Iterators.peel(model.layers)
    input_size = (unbatched_output_size(input_layer)..., batch_size)
    device_zeros_fn = zeros_fn(model)
    input_array = device_zeros_fn(datatype(input_layer), input_size)
    layer_outputs = [device_zeros_fn(datatype(layer), (unbatched_output_size(layer)..., batch_size)) for layer in network_layers]
    return ForwardPassCache(input_array, layer_outputs)
end
# TODO: Add preallocate method that takes an input array and uses that to preallocate.

function set_inputs!(cache::ForwardPassCache, inputs)
    cache.input .= inputs
end

export preallocate, set_inputs!, get_outputs