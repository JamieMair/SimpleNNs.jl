function pullback!(input_partials, output_partials, layer::Flatten)
    n_samples = size(output_partials, ndims(output_partials))
    input_partials = reshape(output_partials, layer.input_size..., n_samples)
    return input_partials
end