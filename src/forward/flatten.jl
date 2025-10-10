function forward_inner!(layer_output, ::Flatten, _)
    # `layer_output` is already reshaped during pre-allocation
    return layer_output
end