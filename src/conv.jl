Base.@kwdef struct Conv{DT<:Real, KSIZE<:Union{Symbol,NTuple{<:Any, Int}}, K<:Union{Symbol,Int}, T<:Function, B, KD, KS<:NTuple{KD, Int}} <: AbstractLayer
    out_channels::Int
    use_bias::Val{B} = Val(true)
    parameter_type::Val{DT} = Val(Float32)
    in_size::KSIZE = :infer
    in_channels::K = :infer
    kernel_size::KS
    activation_fn::T = identity
end
Conv(kernel_size::NTuple{N, Int}, out_channels::Int; kwargs...) where {N} = Conv(;kernel_size, out_channels, kwargs...)
datatype(::Conv{DT}) where {DT} = DT
has_bias(layer) = typeof(layer.use_bias).parameters[begin]
function inputsize(layer::Conv)
    layer.in_size isa Symbol && error("Conv layer inputs should be set to a number.")
    return layer.in_size
end

num_parameters(layer::Conv) = reduce(*, layer.kernel_size) * layer.in_channels * layer.out_channels + (has_bias(layer) ? 1 : 0) * layer.out_channels
function reconstruct_layer(layer::Conv, previous_layer_size::NTuple{D, Int}, previous_datatype) where {D}
    if layer.in_channels isa Symbol || layer.in_size isa Symbol
        spatial_channels = length(layer.kernel_size)
        if spatial_channels != length(previous_layer_size) - 1
            error("Convolution layer with a $(layer.kernel_size) kernel size cannot accept an input layer size of $(previous_layer_size), must have at least one channel.")
        end
        in_channels = previous_layer_size[spatial_channels+1]

        return Conv(layer.out_channels, layer.use_bias, layer.parameter_type, previous_layer_size, in_channels, layer.kernel_size, layer.activation_fn)
    else
        return layer
    end
end

function parameter_array_size(layer::Conv)
    matrix_size = (layer.kernel_size..., layer.in_channels, layer.out_channels)
    if !has_bias(layer)
        return matrix_size
    else
        return (matrix_size, (layer.out_channels,))
    end
end
function _conv_dim_size(in_size, kernel_size, padding, stride)
    dilation = 1 
    return fld(in_size + 2*padding - dilation * (kernel_size-1) - 1, stride) + 1
end
# TODO: Add stride, padding and the padding type (and possibly dilation too)
function unbatched_output_size(layer::Conv)
    in_dims = layer.in_size[begin:end-1]
    # TODO: Remove hardcoded stride, padding and dilation
    return (map((in_dim, kernel_size)->_conv_dim_size(in_dim, kernel_size, 0, 1), in_dims, layer.kernel_size)..., layer.out_channels)
end


# TODO: We also need some pooling layers to be able to condense the results
function weights(layer::ParameterisedLayer{T, Q}) where {T<:Conv, Q}
    return kernel_weights(layer.layer, parameters(layer))
end
function biases(layer::ParameterisedLayer{T, Q}) where {T<:Conv, Q}
    return kernel_biases(layer.layer, parameters(layer))
end

function kernel_weights(layer::Conv, params)
    weights = first(params)
    new_shape = (layer.kernel_size..., layer.in_channels, layer.out_channels)
    if size(weights)==new_shape
        return weights
    end
    # TODO: FIGURE OUT WHY RESHAPE IS ALLOCATING
    return Base.ReshapedArray(weights, new_shape, ())
end
function kernel_biases(::Conv, params)
    return last(params)
end