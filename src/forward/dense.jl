function forward!(output, layer::Dense, parameters, input)
    w = reshape(first(parameters), layer.outputs, layer.inputs)
    if has_bias(layer)
        b = last(parameters)  # Flat vector of biases
        output .= b # Automatically broadcast column-wise
        mul!(output, w, input, true, true)
    else
        mul!(output, w, input)
    end


    if typeof(layer.activation_fn) !== typeof(identity)
        output .= layer.activation_fn.(output)
    end
    nothing
end