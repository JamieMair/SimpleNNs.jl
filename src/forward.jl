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

function forward_inner!(layer_output, layer, current_input)
    inner_layer = _inner_layer(layer)
    params = parameters(layer)
    forward!(layer_output, inner_layer, params, current_input)
    current_input = layer_output
    return current_input
end

function forward!(cache::ForwardPassCache, model::Model)
    current_input = cache.input
    for (i, layer) in enumerate(model.layers)
        i == 1 && continue
        layer_output = cache.layer_outputs[i-1]
        current_input = forward_inner!(layer_output, layer, current_input)
    end
    nothing
end

export forward!