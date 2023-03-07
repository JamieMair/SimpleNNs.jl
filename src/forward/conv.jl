function forward!(output::AbstractArray, layer::Conv, parameters, input::AbstractArray)
    kernel = kernel_weights(layer, parameters)
    spatial_dims = length(layer.kernel_size)
    
    # TODO: Change below to use a stride other than 1
    output_dimensions = CartesianIndices(size(output)[1:spatial_dims])
    kernel_indices = CartesianIndices(layer.kernel_size)
    one_offset = CartesianIndex(1,1)

    # TODO: Add bounds checking and make indexing in-bounds
    @inbounds for n in axes(output, length(size(output)))
        for c_out in 1:layer.out_channels
            for o_i in output_dimensions
                s = if has_bias(layer)
                    kernel_biases(layer, parameters)[c_out]
                else
                    zero(eltype(output))
                end

                for c_in in 1:layer.in_channels
                    for k_i in kernel_indices
                        s += kernel[k_i, c_in, c_out] * input[k_i+o_i-one_offset, c_in, n]
                    end
                end
                if typeof(layer.activation_fn) !== typeof(identity)
                    s = layer.activation_fn(s)
                end

                output[o_i, c_out, n] = s
            end
        end
    end

    nothing
end