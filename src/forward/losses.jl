function forward_inner!(layer_output::AbstractArray, layer::BatchCrossEntropyLoss, current_input::AbstractArray)
    Base.require_one_based_indexing(current_input, layer_output)
    n_samples = size(current_input, ndims(current_input))
    n_classes = layer.num_classes

    for j in 1:n_samples
        z = zero(eltype(layer_output))
        @simd for i in 1:n_classes
            z += exp(current_input[i, j])
        end
        true_class = layer.targets[j]
        layer_output[1, j] = log(z) - current_input[true_class, j]
    end

    return layer_output
end
function _cross_entropy_loss_batch_forward_kernel!(outputs, inputs, targets)
    j = threadIdx().x + (blockIdx().x - 1i32) * blockDim().x
    n = length(outputs)
    n_classes = convert(Int32, size(inputs, 1))
    T_out = eltype(outputs)
    if j <= n
        z = zero(T_out)
        max_class = 1i32
        max_value = typemin(eltype(inputs))
        for i in 1i32:n_classes
            input = inputs[i, j]
            if max_value < input
                max_class = i
                max_value = input
            end
            # outtype_input = convert(T_out, input)
            z += exp(convert(T_out, input))
        end
        # BUG: Does not work if all outputs are large
        true_class = targets[j]
        if isinf(z)
            outputs[1, j] = ifelse(true_class==max_class, zero(eltype(outputs)), typemax(eltype(outputs)))
        else
            outputs[1, j] = log(z) - convert(T_out, inputs[true_class, j])
        end
    end
    nothing
end
function forward_inner!(layer_output::CuArray, layer::BatchCrossEntropyLoss, current_input::CuArray)
    Base.require_one_based_indexing(current_input, layer_output)
    n_samples = size(current_input, ndims(current_input))

    threads = min(length(layer_output), 256)
    blocks = cld(n_samples, threads)

    @cuda threads=threads blocks=blocks _cross_entropy_loss_batch_forward_kernel!(layer_output, current_input, layer.targets)

    return layer_output
end