# Defines the forward pass of a model
using LinearAlgebra


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
    kernel_centres = CartesianIndices(map((x,y)->(1+x ÷ 2):(y-x÷2), layer.kernel_size, size(input)[1:spatial_dims]))
    output_dimensions = CartesianIndices(size(output)[1:spatial_dims])
    kernel_offsets = CartesianIndices(map(x->(-x÷2):(x÷2), layer.kernel_size))
    fixed_offset = CartesianIndex(map(x->(x÷2+1), layer.kernel_size))


    for n in axes(output, length(output))
        for c_out in 1:layer.out_channels
            for (KI, OI) in zip(kernel_centres, output_dimensions)
                s = zero(eltype(output))
                if has_bias(layer)
                    s = kernel_biases(layer, parameters)[c_out]
                end
                for c_in in 1:layer.in_channels
                    for KO in kernel_offsets
                        s += input[KI+KO, c_in, n] * kernel[KO+fixed_offset, c_in, c_out]
                    end
                end
                if typeof(layer.activation_fn) !== typeof(identity)
                    s = layer.activation_fn(s)
                end

                output[OI, c_out, n] = s
            end
        end
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