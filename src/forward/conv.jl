
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
function _forward_conv_2d_no_bias(outputs, inputs, kernel, activation_fn = identity)
    s_x = convert(Int32, size(outputs, 1))
    s_y = convert(Int32, size(outputs, 2))
    out_channels = convert(Int32, size(outputs, 3))
    n = convert(Int32, size(outputs, 4))

    w = convert(Int32, size(kernel, 1))
    h = convert(Int32, size(kernel, 2))
    in_channels = convert(Int32, size(kernel, 3))


    x = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1i32) * blockDim().y
    i = threadIdx().z + (blockIdx().z - 1i32) * blockDim().z

    @inbounds if x <= s_x && y <= s_y && i <= n
        for c_out in 1i32:out_channels
            s = zero(eltype(outputs))
            for c_in in 1i32:in_channels
                for ky in 1i32:h
                    for kx in 1i32:w
                        s = CUDA.fma(kernel[kx, ky, c_in, c_out], inputs[x + kx - 1i32, y + ky - 1i32, c_in, i], s)
                    end
                end
            end
            outputs[x, y, c_out, i] = activation_fn(s)
        end
    end

    nothing
end
function _forward_conv_2d(outputs, inputs, kernel, bias, activation_fn = identity)
    s_x = convert(Int32, size(outputs, 1))
    s_y = convert(Int32, size(outputs, 2))
    out_channels = convert(Int32, size(outputs, 3))
    n = convert(Int32, size(outputs, 4))

    w = convert(Int32, size(kernel, 1))
    h = convert(Int32, size(kernel, 2))
    in_channels = convert(Int32, size(kernel, 3))


    x = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    y = threadIdx().y + (blockIdx().y - 1i32) * blockDim().y
    i = threadIdx().z + (blockIdx().z - 1i32) * blockDim().z

    @inbounds if x <= s_x && y <= s_y && i <= n
        for c_out in 1i32:out_channels
            s = bias[c_out]
            for c_in in 1i32:in_channels
                for ky in 1i32:h
                    for kx in 1i32:w
                        s = CUDA.fma(kernel[kx, ky, c_in, c_out], inputs[x + kx - 1i32, y + ky - 1i32, c_in, i], s)
                    end
                end
            end
            outputs[x, y, c_out, i] = activation_fn(s)
        end
    end

    nothing
end
function forward!(output::CuArray, layer::Conv, parameters, input::CuArray)
    kernel = kernel_weights(layer, parameters)

    if ndims(kernel)==4
        # Use custom 2D kernel
        w, h, _, n = size(output)
        x_threads = min(32, w)
        y_threads = min(32, h)
        z_threads = min(1024 ÷ x_threads ÷ y_threads, n)
        num_threads = (x_threads, y_threads, z_threads)
        num_blocks = (cld(w, x_threads), cld(h, y_threads), cld(n, z_threads))
        activation = layer.activation_fn

        if has_bias(layer)
            biases = kernel_biases(layer, parameters)
            @cuda blocks=num_blocks threads=num_threads _forward_conv_2d(output, input, kernel, biases, activation)
        else
            @cuda blocks=num_blocks threads=num_threads _forward_conv_2d_no_bias(output, input, kernel, activation)
        end
    else
        kernel = kernel_weights(layer, parameters)
        @assert !has_bias(layer) "Convolutions with $(ndims(kernel)-2) spatial dimensions with a bias are not supported yet."
        
        biases = kernel_biases(layer, parameters)
        conv_params = NNlib.DenseConvDims(size(input), size(kernel); flipkernel=true)
        activation_fn = if typeof(layer.activation_fn) === typeof(relu)
            NNlib.relu
        else
            identity
        end
    
        channel_dim = ndims(output)-1
        biases = reshape(biases, ntuple(i->i==channel_dim ? size(output, i) : 1, ndims(output)))
        NNlib.conv_bias_act!(output, input, kernel, conv_params, biases, activation_fn)
    
        if typeof(activation_fn) != typeof(NNlib.relu) && typeof(activation_fn) != typeof(layer.activation_fn)
            # Apply activation
            output .= layer.activation_fn.(output)
        end
    end

    nothing
end