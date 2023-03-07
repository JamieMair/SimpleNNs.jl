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
function forward_inner!(layer_output::CuArray, layer::BatchCrossEntropyLoss, current_input::CuArray)
    Base.require_one_based_indexing(current_input, layer_output)
    n_samples = size(current_input, ndims(current_input))

    threads = min(length(layer_output), 256)
    blocks = cld(n_samples, threads)

    @cuda threads=threads blocks=blocks _cross_entropy_loss_batch_forward_kernel!(layer_output, current_input, layer.targets)

    return layer_output
end