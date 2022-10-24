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

forward!(cache::ForwardPassCache, model::Model) = _forward!(cache, model.layers)

@generated function _forward!(cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}) where {N}
    first_line = :(current_input = cache.input)
    calls = [:(current_input = forward_inner!(cache.layer_outputs[$i - 1], layers[$i], current_input)) for i in 2:N]
    Expr(:block, first_line, calls...)
end

export forward!