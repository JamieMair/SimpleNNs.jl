function forward_inner!(layer_output, layer::Flatten, current_input)
    n_samples = size(current_input, ndims(current_input))
    next_output = reshape(current_input, layer.output_size..., n_samples)
    return next_output
end