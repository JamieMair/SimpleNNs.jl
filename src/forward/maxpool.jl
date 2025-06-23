function forward_inner!(output::AbstractArray, layer::MaxPool, input::AbstractArray)
    spatial_dims = length(layer.pool_size)
    
    # TODO: Change below to use a stride other than 1
    output_dimensions = CartesianIndices(size(output)[1:spatial_dims])
    kernel_indices = CartesianIndices(layer.pool_size)
    stride_dims = CartesianIndex(layer.stride)
    channels = size(input, spatial_dims+1)
    pool_fn = max

    # TODO: Add bounds checking and make indexing in-bounds
    @inbounds for n in axes(output, length(size(output)))
        for c in 1:channels
            for o_i in output_dimensions
                offset = compute_offset(o_i, stride_dims)
                s = typemin(eltype(output))
                @simd for k_i in kernel_indices
                    s = pool_fn(s, input[k_i+offset, c, n])
                end

                output[o_i, c, n] = s
            end
        end
    end

    return output
end