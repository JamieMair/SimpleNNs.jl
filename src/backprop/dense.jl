function pullback!(input_partials, output_partials, layer::ParameterisedLayer{T}) where {T<:Dense}
    layer_weights = weights(layer)
    mul!(input_partials, transpose(layer_weights), output_partials)
    return input_partials
end
function backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer::Dense)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = activation_gradient_fn(layer.activation_fn)
        partials_buffer .*= activation_derivative.(outputs)
    end

    w_grads = reshape(first(gradient_buffer), layer.outputs, layer.inputs)
    mul!(w_grads, partials_buffer, transpose(inputs))

    if has_bias(layer)
        b_grads = reshape(last(gradient_buffer), :, 1)
        sum!(b_grads, partials_buffer) # TODO: Remove allocations
    end
    
    return partials_buffer
end