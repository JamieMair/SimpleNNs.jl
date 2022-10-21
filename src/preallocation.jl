struct ForwardPassCache{A<:AbstractArray, B<:AbstractArray}
    input::A
    layer_outputs::AbstractArray{B}
end
get_outputs(cache::ForwardPassCache) = last(cache.layer_outputs)
function preallocate(model::Model, batch_size::Integer)
    (input_layer, network_layers) = Iterators.peel(model.layers)
    input_size = (outputcount(input_layer), batch_size)
    input_array = zeros(datatype(input_layer), input_size)
    layer_outputs = [zeros(datatype(layer), (outputcount(layer), batch_size)) for layer in network_layers]
    return ForwardPassCache(input_array, layer_outputs)
end
