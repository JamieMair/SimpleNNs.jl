# Abstract types
abstract type AbstractLayer end
datatype(::AbstractLayer) = unimplemented()
outputcount(::AbstractLayer) = unimplemented()
parameter_array_size(::AbstractLayer) = unimplemented()
num_parameters(layer::AbstractLayer) = flatten_size(parameter_array_size(layer))
parameter_indices(::AbstractLayer) = unimplemented()

abstract type AbstractParameterisedLayer <: AbstractLayer end
_inner_layer(::AbstractParameterisedLayer) = unimplemented()
parameters(::AbstractParameterisedLayer) = unimplemented()
# Forward all definitions to the "layer" property of the abstract layer
datatype(layer::AbstractParameterisedLayer) = datatype(_inner_layer(layer))
outputcount(layer::AbstractParameterisedLayer) = outputcount(_inner_layer(layer))
parameter_array_size(layer::AbstractParameterisedLayer) = parameter_array_size(_inner_layer(layer))
num_parameters(layer::AbstractParameterisedLayer) = num_parameters(_inner_layer(layer))
parameter_indices(layer::AbstractParameterisedLayer) = parameter_indices(_inner_layer(layer))

# Layer types
Base.@kwdef struct Static{DT, S} <: AbstractLayer
    inputs::S
    datatype::Val{DT} = Val(Float32)
end
Static(inputs::Integer; kwargs...) = Static(;inputs=inputs, kwargs...)
Base.@kwdef struct Dense{DT<:Real, K<:Union{Symbol, Int}, T<:Function, B} <: AbstractLayer
    outputs::Int
    inputs::K = :infer
    use_bias::Val{B} = Val(true)
    activation_fn::T = identity
    parameter_type::Val{DT} = Val(Float32)
end
Dense(outputs::Integer; kwargs...) = Dense(;outputs=outputs, kwargs...)
has_bias(::Dense{KT, K, T, B}) where {KT, K, T, B} = B

datatype(::Static{DT, S}) where {DT, S} = DT
datatype(::Dense{DT, K, T}) where {DT, K, T} = DT
outputcount(layer::Static) = flatten_size(layer.inputs)
outputcount(layer::Dense) = layer.outputs
inputsize(layer::Static) = 0
function inputsize(layer::Dense)
    layer.inputs isa Symbol && error("Layer inputs $(layer.inputs) should be set to a number.")
    return layer.inputs
end
parameter_array_size(::Static) = 0
function parameter_array_size(layer::Dense)
    matrix_size = (layer.outputs, layer.inputs)
    if !has_bias(layer)
        return matrix_size
    else
        return (matrix_size, (layer.outputs,))
    end
end
num_parameters(::Static) = 0
num_parameters(layer::Dense) = layer.outputs * (layer.inputs + (has_bias(layer) ? 1 : 0))
function parameter_indices(layer, current_offset::Integer)::Vector{UnitRange{Int}}
    parameter_sizes = parameter_array_size(layer)
    if eltype(parameter_sizes) <: Tuple
        sizes = flatten_size.(parameter_sizes)
        offsets = cumsum(sizes) .- sizes
        ranges = [(current_offset+1+offset:current_offset+offset+s) for (s, offset) in zip(sizes, offsets)]
        return ranges
    end

    num_parameters = flatten_size(parameter_sizes)

    @assert typeof(num_parameters) <: Integer
    return [current_offset+1:current_offset+num_parameters]
end

# Model
struct Model{T<:AbstractArray,Q}
    parameters::T
    layers::Q
end
struct ParameterisedLayer{T, Q} <: AbstractParameterisedLayer
    layer::T
    parameter_views::Q
end
_inner_layer(layer::ParameterisedLayer) = layer.layer
parameters(layer::ParameterisedLayer) = layer.parameter_views

function weights(layer::ParameterisedLayer{T, Q}) where {T<:Dense, Q}
    weights = first(parameters(layer))
    return reshape(weights, layer.layer.outputs, layer.layer.inputs)
end
function biases(layer::ParameterisedLayer{T, Q}) where {T<:Dense, Q}
    biases = last(parameters(layer))
    return biases
end

reconstruct_layer(layer::AbstractLayer, previous_layer_size) = layer
function reconstruct_layer(layer::Dense, previous_layer_size::Int)
    if layer.inputs isa Symbol
        return Dense(layer.outputs, previous_layer_size, layer.use_bias, layer.activation_fn, layer.parameter_type)
    else
        return layer
    end
end