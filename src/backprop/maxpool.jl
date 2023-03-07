has_combined_backprop_and_pullback(::MaxPool) = true

function backprop_and_pullback!(input_partials, output_partials, inputs, outputs, layer::MaxPool)    
    spatial_dims = length(layer.pool_size)
    kernel_indices = CartesianIndices(layer.pool_size)
    output_dimensions = CartesianIndices(size(outputs)[1:spatial_dims])
    channels = size(input_partials, spatial_dims+1)
    stride_dims = CartesianIndex(layer.stride)

    fill!(input_partials, zero(eltype(input_partials)))
    @inbounds for n in axes(outputs, length(size(outputs)))
        for c in 1:channels
            for o_i in output_dimensions
                out_val = outputs[o_i, c, n]
                offset = CartesianIndex(Tuple((i-1)*sx for (i, sx) in zip(Tuple(o_i), Tuple(stride_dims))))
                for k_i in kernel_indices
                    in_val = inputs[k_i+offset, c, n]
                    input_partials[k_i+offset, c, n] += (in_val==out_val) * output_partials[o_i, c, n]
                end
            end
        end
    end

    return input_partials
end