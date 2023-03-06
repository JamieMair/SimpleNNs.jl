function forward!(output::AbstractArray, layer::MaxPool, _, input::AbstractArray)
    spatial_dims = length(layer.pool_size)
    
    # TODO: Change below to use a stride other than 1
    output_dimensions = CartesianIndices(size(output)[1:spatial_dims])
    kernel_indices = CartesianIndices(layer.kernel_size)
    stride_dims = CartesianIndex(layer.stride)
    channels = size(input, spatial_dims+1)
    pool_fn = max

    # TODO: Add bounds checking and make indexing in-bounds
    for n in axes(output, length(size(output)))
        for c in 1:channels
            for o_i in output_dimensions
                offset = CartesianIndex(((i-1)*sx for (i, sx) in zip(o_i, stride_dims)))
                s = typemin(eltype(output))
                @simd for k_i in kernel_indices
                    s = pool_fn(s, input[k_i+offset, c, n])
                end

                output[o_i, c, n] = s
            end
        end
    end

    nothing
end