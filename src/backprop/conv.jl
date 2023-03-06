function backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer::Conv)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = activation_gradient_fn(layer)
        partials_buffer .*= activation_derivative.(outputs)
    end

    # Kernel weights gradients
    k_grads = kernel_weights(layer, gradient_buffer)
    spatial_dims = length(layer.kernel_size)
    output_dimensions = CartesianIndices(size(outputs)[1:spatial_dims])
    kernel_indices = CartesianIndices(layer.kernel_size)
    one_one = CartesianIndex((1 for _ in 1:spatial_dims)...)
    fill!(k_grads, zero(eltype(k_grads)))
    @inbounds for n in axes(outputs, length(size(outputs)))
        for c_out in 1:layer.out_channels
            for c_in in 1:layer.in_channels
                for o_i in output_dimensions
                    for k_i in kernel_indices
                        k_grads[k_i, c_in, c_out] += partials_buffer[o_i, c_out, n] * inputs[o_i + k_i - one_one, c_in, n]
                    end
                end
            end
        end
    end
    # Kernel bias gradients
    if has_bias(layer)
        k_biases = kernel_biases(layer, gradient_buffer)
        fill!(k_biases, zero(eltype(k_biases)))
        @inbounds for n in axes(outputs, length(size(outputs)))
            for c_out in 1:layer.out_channels
                for o_i in output_dimensions
                    k_biases[c_out] += partials_buffer[o_i, c_out, n]
                end
            end
        end
    end

    return partials_buffer
end

function pullback!(input_partials, output_partials, layer::ParameterisedLayer{T}) where {T<:Conv}
    params = parameters(layer)
    conv_layer = _inner_layer(layer)::Conv
    kernel = kernel_weights(conv_layer, params)
    spatial_dims = length(conv_layer.kernel_size)
    input_dimensions = CartesianIndices(size(input_partials)[1:spatial_dims])
    kernel_indices = CartesianIndices(conv_layer.kernel_size)
    one_one = CartesianIndex((1 for _ in 1:spatial_dims)...)
    # Zero out input
    fill!(input_partials, zero(eltype(input_partials)))
    @inbounds for n in axes(output_partials, length(size(output_partials)))
        for c_out in 1:conv_layer.out_channels
            for c_in in 1:conv_layer.in_channels
                for x_i in input_dimensions
                    for k_i in kernel_indices

                        out_index = x_i + one_one - k_i
                        grad_contribution = if checkbounds(Bool, output_partials, out_index, c_out, n)
                            output_partials[out_index, c_out, n] * kernel[k_i, c_in, c_out]
                        else
                            zero(eltype(input_partials))
                        end
                        
                        input_partials[x_i, c_in, n] += grad_contribution
                    end
                end
            end
        end
    end

    return input_partials
end