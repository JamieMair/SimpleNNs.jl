Base.@kwdef struct Dense{DT<:Real, K<:Union{Infer, Int}, T<:Function, B} <: AbstractLayer
    outputs::Int
    inputs::K = Infer()
    use_bias::Val{B} = Val(true)
    activation_fn::T = identity
    parameter_type::Val{DT} = Val(Float32)
end

"""
    Dense(outputs::Integer; kwargs...)

A representation of a dense layer. By default this can be constructed by 
specifying the desired number of outputs. The input size can be inferred
from the rest of the chain when constructing a model.

# Keyword Arguments

- `use_bias` (default: `Val(true)`) - Whether or not to add a bias vector to the output. Wrapped in a `Val` for optimisation.
- `activation_fn` (default: `identity`) - A custom activation function. Note that not all functions are supported by backpropagation.
- `parameter_type` (default: `Val(Float32)`) - The datatype to use for the parameters, wrapped in a `Val` type.
- `inputs` (default: `Infer()`) - Specify the number of inputs, or infer them from the rest of the model.
"""
Dense(outputs::Integer; kwargs...) = Dense(;outputs=outputs, kwargs...)

has_bias(::Dense{KT, K, T, B}) where {KT, K, T, B} = B
datatype(::Dense{DT, K, T}) where {DT, K, T} = DT
unbatched_output_size(layer::Dense) = layer.outputs
function inputsize(layer::Dense)
    layer.inputs isa Infer && error("Layer inputs $(layer.inputs) should be set to a number.")
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
    if layer.inputs isa Infer
        return Dense(layer.outputs, previous_layer_size, layer.use_bias, layer.activation_fn, layer.parameter_type)
    else
        return layer
    end
end