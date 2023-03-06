Base.@kwdef struct Dense{DT<:Real, K<:Union{InferSize, Int}, T<:Function, B} <: AbstractLayer
    outputs::Int
    inputs::K = InferSize()
    use_bias::Val{B} = Val(true)
    activation_fn::T = identity
    parameter_type::Val{DT} = Val(Float32)
end
Dense(outputs::Integer; kwargs...) = Dense(;outputs=outputs, kwargs...)

has_bias(::Dense{KT, K, T, B}) where {KT, K, T, B} = B
datatype(::Dense{DT, K, T}) where {DT, K, T} = DT
unbatched_output_size(layer::Dense) = layer.outputs
function inputsize(layer::Dense)
    layer.inputs isa InferSize && error("Layer inputs $(layer.inputs) should be set to a number.")
    return layer.inputs
end
function parameter_array_size(layer::Dense)
    matrix_size = (layer.outputs, layer.inputs)
    if !has_bias(layer)
        return matrix_size
    else
        return (matrix_size, (layer.outputs,))
    end
end
num_parameters(layer::Dense) = layer.outputs * (layer.inputs + (has_bias(layer) ? 1 : 0))
function weights(layer::ParameterisedLayer{T, Q}) where {T<:Dense, Q}
    weights = first(parameters(layer))
    return reshape(weights, layer.layer.outputs, layer.layer.inputs)
end
function biases(layer::ParameterisedLayer{T, Q}) where {T<:Dense, Q}
    biases = last(parameters(layer))
    return biases
end
function reconstruct_layer(layer::Dense, previous_layer_size::Int, current_datatype)
    if layer.inputs isa InferSize
        return Dense(layer.outputs, previous_layer_size, layer.use_bias, layer.activation_fn, layer.parameter_type)
    else
        return layer
    end
end