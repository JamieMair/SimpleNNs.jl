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