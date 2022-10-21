# Defines the forward pass of a model
using LinearAlgebra

function forward!(output, layer::Dense, parameters, input)
    w = reshape(first(parameters), layer.outputs, layer.inputs)
    mul!(output, w, input)
    if layer.use_bias
        b = last(parameters)  # Flat vector of biases
        output .+= b # Automatically broadcast column-wise
    end
    if layer.activation_fn !== identity
        output .= layer.activation_fn.(output)
    end
    nothing
end

function forward!(cache::ForwardPassCache, model::Model)
    (_, network_layers) = Iterators.peel(model.layers)

    current_input = cache.input
    for (layer_output, layer) in zip(cache.layer_outputs, network_layers)
        inner_layer = _inner_layer(layer)
        params = parameters(layer)
        forward!(layer_output, inner_layer, params, current_input)
        current_input = layer_output
    end
    nothing
end

export forward!