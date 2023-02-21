# Defines the forward pass of a model
using LinearAlgebra
import NNlib
import NNlibCUDA


function forward!(output, layer::Dense, parameters, input)
    w = reshape(first(parameters), layer.outputs, layer.inputs)
    mul!(output, w, input)
    if has_bias(layer)
        # TODO: Switch to fma for performance
        b = last(parameters)  # Flat vector of biases
        output .+= b # Automatically broadcast column-wise
    end
    if typeof(layer.activation_fn) !== typeof(identity)
        output .= layer.activation_fn.(output)
    end
    nothing
end
function forward!(output::AbstractArray, layer::Conv, parameters, input::AbstractArray)
    kernel = kernel_weights(layer, parameters)
    spatial_dims = length(layer.kernel_size)
    
    # TODO: Change below to use a stride other than 1
    output_dimensions = CartesianIndices(size(output)[1:spatial_dims])
    kernel_indices = CartesianIndices(layer.kernel_size)
    one_offset = CartesianIndex(1,1)

    # TODO: Add bounds checking and make indexing in-bounds
    for n in axes(output, length(size(output)))
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
function forward!(output::CuArray, layer::Conv, parameters, input::CuArray)
    kernel = kernel_weights(layer, parameters)
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

    nothing
end

function forward_inner!(layer_output, layer, current_input)
    inner_layer = _inner_layer(layer)
    params = parameters(layer)
    forward!(layer_output, inner_layer, params, current_input)
    current_input = layer_output
    return current_input
end
function forward_inner!(layer_output, layer::Flatten, current_input)
    next_output = reshape(current_input, layer.output_size..., :)
    return next_output
end

forward!(cache::ForwardPassCache, model::Model) = _forward!(cache, model.layers)

@generated function _forward!(cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}) where {N}
    first_line = :(current_input = cache.input)
    calls = [:(current_input = forward_inner!(cache.layer_outputs[$i - 1], layers[$i], current_input)) for i in 2:N]
    Expr(:block, first_line, calls...)
end

export forward!