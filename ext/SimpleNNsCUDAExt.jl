module SimpleNNsCUDAExt


if isdefined(Base, :get_extension)
    using CUDA
    import CUDA: i32
    import cuDNN
    import NNlib
    using SimpleNNs
    import SimpleNNs: Model, num_parameters, parameter_indices, _map_views, ParameterisedLayer, _inner_layer, AbstractParameterisedLayer
    import SimpleNNs: forward!, forward_inner!, pullback!, backprop!, backprop_and_pullback!
    import SimpleNNs: Conv, MaxPool, Dense, Flatten, has_bias, kernel_weights, kernel_biases, weights, biases
    import SimpleNNs: LogitCrossEntropyLoss, _truncate_batch, activation_gradient_fn
    import SimpleNNs: zeros_fn, ForwardPassCache, BackpropagationCache
else
    using ..CUDA
    import ..CUDA: i32
    using ..SimpleNNs
    import ..cuDNN
    import ..NNlib

    import ..SimpleNNs: Model, num_parameters, parameter_indices, _map_views, ParameterisedLayer, _inner_layer, AbstractParameterisedLayer
    import ..SimpleNNs: forward!, forward_inner!, pullback!, backprop!, backprop_and_pullback!
    import ..SimpleNNs: Conv, MaxPool, Dense, Flatten, has_bias, kernel_weights, kernel_biases, weights, biases
    import ..SimpleNNs: LogitCrossEntropyLoss, _truncate_batch, activation_gradient_fn
    import ..SimpleNNs: zeros_fn, ForwardPassCache, BackpropagationCache
end

using Logging

function SimpleNNs.gpu(model::SimpleNNs.Model)
    parameter_offsets = cumsum(SimpleNNs.num_parameters.(model.layers))
    layer_indices = [SimpleNNs.parameter_indices(layer, offset-SimpleNNs.num_parameters(layer)) for (layer, offset) in Iterators.zip(model.layers, parameter_offsets)]
    gpu_parameters = CuArray(model.parameters)
    gpu_parameters_views = SimpleNNs._map_views(layer_indices, gpu_parameters)
    function _inner_or_same(l)
        if typeof(l) <: SimpleNNs.AbstractParameterisedLayer
            return SimpleNNs._inner_layer(l)
        else
            return l
        end
    end
    unwrapped_layers = Tuple(_inner_or_same(l) for l in model.layers)
    model_layers = Tuple((SimpleNNs.num_parameters(l) > 0 ? SimpleNNs.ParameterisedLayer(l, v) : l) for (l,v) in zip(unwrapped_layers, gpu_parameters_views))
    return SimpleNNs.Model(gpu_parameters, model_layers)
end
SimpleNNs.gpu(arr::AbstractArray) = CuArray(arr)
SimpleNNs.gpu(arr::CuArray) = arr


# Forward pass implementations for GPU
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

function SimpleNNs.forward!(output::CuArray, layer::SimpleNNs.Conv, parameters, input::CuArray)
    kernel = SimpleNNs.kernel_weights(layer, parameters)

    if ndims(kernel)==4
        # Use custom 2D kernel
        w, h, _, n = size(output)
        x_threads = min(32, w)
        y_threads = min(32, h)
        max_threads_per_block = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
        z_threads = min(max_threads_per_block ÷ x_threads ÷ y_threads, n)
        num_threads = (x_threads, y_threads, z_threads)

        num_blocks = (cld(w, x_threads), cld(h, y_threads), cld(n, z_threads))
        activation = layer.activation_fn


        if SimpleNNs.has_bias(layer)
            biases = SimpleNNs.kernel_biases(layer, parameters)
            @cuda blocks=num_blocks threads=num_threads _forward_conv_2d(output, input, kernel, biases, activation)
        else
            @cuda blocks=num_blocks threads=num_threads _forward_conv_2d_no_bias(output, input, kernel, activation)
        end
    else
        kernel = SimpleNNs.kernel_weights(layer, parameters)
        @assert !SimpleNNs.has_bias(layer) "Convolutions with $(ndims(kernel)-2) spatial dimensions with a bias are not supported yet."
        
        biases = SimpleNNs.kernel_biases(layer, parameters)
        conv_params = NNlib.DenseConvDims(size(input), size(kernel); flipkernel=true)
        activation_fn = if typeof(layer.activation_fn) === typeof(SimpleNNs.relu)
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

function SimpleNNs.forward_inner!(output::CuArray, layer::SimpleNNs.MaxPool, input::CuArray)
    pool_dims = NNlib.PoolDims(size(input), layer.pool_size; stride=layer.stride)
    NNlib.maxpool!(output, input, pool_dims)
    return output
end

function _cross_entropy_loss_batch_forward_kernel!(outputs, inputs, targets)
    j = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    n = length(outputs)
    if j <= n
        z = zero(eltype(outputs))
        for i in axes(inputs, 1)
            z += exp(inputs[i, j])
        end
        true_class = targets[j]
        outputs[j] = log(z) - inputs[true_class, j]
    end
    nothing
end

function SimpleNNs.forward_inner!(layer_output::CuArray, layer::SimpleNNs.BatchCrossEntropyLoss, current_input::CuArray)
    Base.require_one_based_indexing(current_input, layer_output)
    n_samples = size(current_input, ndims(current_input))

    threads = min(length(layer_output), 256)
    blocks = cld(n_samples, threads)

    @cuda threads=threads blocks=blocks _cross_entropy_loss_batch_forward_kernel!(layer_output, current_input, layer.targets)

    return layer_output
end

# Backpropagation implementations for GPU
function SimpleNNs.pullback!(input_partials::CuArray, output_partials::CuArray, layer::SimpleNNs.ParameterisedLayer{T}) where {T<:SimpleNNs.Conv}
    params = SimpleNNs.parameters(layer)
    conv_layer = SimpleNNs._inner_layer(layer)::SimpleNNs.Conv
    kernel = SimpleNNs.kernel_weights(conv_layer, params)
    conv_params = NNlib.DenseConvDims(size(input_partials), size(kernel); flipkernel=true)
    NNlib.∇conv_data!(input_partials, output_partials, kernel, conv_params)

    return input_partials
end

function SimpleNNs.backprop!(partials_buffer::CuArray, gradient_buffer, inputs::CuArray, outputs::CuArray, layer::SimpleNNs.Conv)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = SimpleNNs.activation_gradient_fn(layer)
        partials_buffer .*= activation_derivative.(outputs)
    end
    k_grads = SimpleNNs.kernel_weights(layer, gradient_buffer)
    conv_params = NNlib.DenseConvDims(size(inputs), size(k_grads); flipkernel=true)
    
    NNlib.∇conv_filter!(k_grads, inputs, partials_buffer, conv_params)
    # Kernel bias gradients
    if SimpleNNs.has_bias(layer)
        k_biases = SimpleNNs.kernel_biases(layer, gradient_buffer)
        channel_dim = ndims(outputs) - 1
        k_biases = reshape(k_biases, ntuple(i->i==channel_dim ? size(outputs, i) : 1, ndims(outputs)))
        
        partials_buffer .+= k_biases
    end

    return partials_buffer
end

function SimpleNNs.pullback!(partials_buffer::CuArray, inputs::CuArray, loss::SimpleNNs.LogitCrossEntropyLoss{T, N}) where {T, N}
    @assert length(size(inputs)) == 2
    partials_buffer .= exp.(inputs)

    n = size(inputs, ndims(inputs))
    targets = loss.targets
    threads = 128
    blocks = cld(n, threads)
    gpu_total_loss = CUDA.zeros(eltype(partials_buffer), 1)
    @cuda blocks=blocks threads=threads _cross_entropy_pullback_kernel!(gpu_total_loss, partials_buffer, targets, Val(threads))
    total_loss = zero(eltype(partials_buffer))
    CUDA.@allowscalar begin
        total_loss = gpu_total_loss[]
    end
    CUDA.unsafe_free!(gpu_total_loss)
    return (-total_loss)
end

const gpu_threads_per_block = 128
function _cross_entropy_pullback_kernel!(total_loss, partials_buffer, targets, ::Val{NT}) where {NT}
    # Partials buffer should already contain exp.(inputs)
    # NT is the number of threads per block
    @assert blockDim().x == NT
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    N = size(partials_buffer, ndims(partials_buffer))

    partial_total_loss = CUDA.@cuStaticSharedMem(eltype(total_loss), NT)

    idx = threadIdx().x
    if i <= N
        true_class = targets[i]
        z = zero(eltype(partials_buffer))
        for j in axes(partials_buffer, 1)
            z += partials_buffer[j, i]
        end

        for j in axes(partials_buffer, 1)
            e_y = partials_buffer[j, i]
            e_y_over_z = ifelse(isfinite(e_y), e_y / z, one(eltype(partials_buffer)))
            partials_buffer[j, i] = (e_y_over_z - (j==true_class))
            if j==true_class
                partial_total_loss[idx] = log(e_y_over_z)
            end
        end
    else
        partial_total_loss[idx] = zero(eltype(partial_total_loss))
    end

    # Reduce the partial cross entropy losses together
    step = NT ÷ 2
    while step != 0
        CUDA.sync_threads()
        if (idx <= step)
            partial_total_loss[idx] += partial_total_loss[idx+step]
        end
        step ÷= 2
    end

    if idx == 1
        CUDA.@atomic total_loss[1] += partial_total_loss[1]
    end

    nothing
end

function SimpleNNs.backprop_and_pullback!(input_partials::CuArray, output_partials::CuArray, inputs::CuArray, outputs::CuArray, layer::SimpleNNs.MaxPool)    
    pool_size = layer.pool_size
    stride_size = layer.stride

    pool_dims = NNlib.PoolDims(size(inputs), pool_size; stride=stride_size)
    NNlib.∇maxpool!(input_partials, output_partials, outputs, inputs, pool_dims)

    return input_partials
end

# Export gpu function
export gpu

end
