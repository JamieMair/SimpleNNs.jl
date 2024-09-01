struct BackpropagationCache{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray{B}, D}
    parameter_gradients::A
    parameter_gradient_views::C
    layer_partials::D
end

"""
preallocate_grads(model::Model, batch_size::Integer)

Creates a buffer to store the intermediate arrays needed for backpropagation.

The gradients can be retrieved from the buffer using [`gradients`](@ref) on the buffer.
"""
function preallocate_grads(model::Model, batch_size::Integer)
    (_, network_layers) = Iterators.peel(model.layers)

    parameter_offsets = cumsum(num_parameters.(model.layers))
    layer_indices = [parameter_indices(layer, offset-num_parameters(layer)) for (layer, offset) in Iterators.zip(model.layers, parameter_offsets)]
    parameter_gradients = similar(model.parameters)
    fill!(parameter_gradients, zero(eltype(parameter_gradients)))
    parameter_gradient_views = _map_views(layer_indices, parameter_gradients)
    device_zeros_fn = zeros_fn(model)

    layer_partials = Vector{Any}(undef, length(model.layers)-1)
    for (i, layer) in enumerate(network_layers)
        if typeof(layer) <: Flatten
            layer_partials[i] = reshape(layer_partials[i-1], layer.output_size..., batch_size)
            continue
        end
        layer_partials[i] = device_zeros_fn(datatype(layer), _get_preallocation_size(layer, batch_size))
    end
    # TODO: Switch to tuple and allow for a flatten right at the start
    return BackpropagationCache(parameter_gradients, parameter_gradient_views, Tuple(layer_partials))
end


function truncate(cache::BackpropagationCache, batch_size)
    partials = map(x->_truncate_batch(x, batch_size), cache.layer_partials)

    return BackpropagationCache(cache.parameter_gradients, cache.parameter_gradient_views, partials)
end

"""
    gradients(cache::BackpropagationCache)

Extracts the gradient array from the backwards pass buffer, filled from use of the [`backprop!`](@ref) function.
"""
function gradients(cache::BackpropagationCache)
    return cache.parameter_gradients
end