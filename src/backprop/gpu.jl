import NNlib

function pullback!(input_partials::CuArray, output_partials::CuArray, layer::ParameterisedLayer{T}) where {T<:Conv}
    params = parameters(layer)
    conv_layer = _inner_layer(layer)::Conv
    kernel = kernel_weights(conv_layer, params)
    conv_params = NNlib.DenseConvDims(size(input_partials), size(kernel); flipkernel=true)
    NNlib.∇conv_data!(input_partials, output_partials, kernel, conv_params)

    return input_partials
end
function backprop!(partials_buffer::CuArray, gradient_buffer, inputs::CuArray, outputs::CuArray, layer::Conv)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = activation_gradient_fn(layer)
        partials_buffer .*= activation_derivative.(outputs)
    end
    k_grads = kernel_weights(layer, gradient_buffer)
    conv_params = NNlib.DenseConvDims(size(inputs), size(k_grads); flipkernel=true)
    

    NNlib.∇conv_filter!(k_grads, inputs, partials_buffer, conv_params)
    # Kernel bias gradients
    if has_bias(layer)
        k_biases = kernel_biases(layer, gradient_buffer)
        channel_dim = ndims(outputs) - 1
        k_biases = reshape(k_biases, ntuple(i->i==channel_dim ? size(outputs, i) : 1, ndims(outputs)))
        NNlib.∇conv_bias!(k_biases, partials_buffer)
    end

    return partials_buffer
end
function pullback!(partials_buffer::CuArray, inputs::CuArray, loss::LogitCrossEntropyLoss{T, N}) where {T, N}
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
using CUDA
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

function backprop_and_pullback!(input_partials::CuArray, output_partials::CuArray, inputs::CuArray, outputs::CuArray, layer::MaxPool)    
    pool_size = layer.pool_size
    stride_size = layer.stride

    pool_dims = NNlib.PoolDims(size(inputs), pool_size; stride=stride_size)
    NNlib.∇maxpool!(input_partials, output_partials, outputs, inputs, pool_dims)

    return input_partials
end