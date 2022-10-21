module SimpleNNs
using Logging

# Utility
unimplemented() = error("Unimplemented function")

flatten_size(size::Tuple) = reduce(*, size)
flatten_size(size::Number) = size

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
Base.@kwdef struct Dense{DT<:Real, K<:Union{Symbol, Int}, T<:Function} <: AbstractLayer
    outputs::Int
    inputs::K = :infer
    use_bias::Bool = true
    activation_fn::T = identity
    parameter_type::Val{DT} = Val(Float32)
end
Dense(outputs::Integer; kwargs...) = Dense(;outputs=outputs, kwargs...)

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
    if !layer.use_bias
        return matrix_size
    else
        return (matrix_size, (layer.outputs,))
    end
end
num_parameters(::Static) = 0
num_parameters(layer::Dense) = layer.outputs * (layer.inputs + (layer.use_bias ? 1 : 0))
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
struct Model{T<:AbstractArray,Q<:AbstractParameterisedLayer}
    parameters::T
    layers::AbstractArray{Q}
end
struct ParameterisedLayer{T, Q} <: AbstractParameterisedLayer
    layer::T
    parameter_views::Q
end
_inner_layer(layer::ParameterisedLayer) = layer.layer
parameters(layer::ParameterisedLayer) = layer.parameter_views

# Activation functions
sigmoid(x) = inv(one(typeof(x) + exp(-x)))
relu(x) = ifelse(x>=zero(typeof(x)), x, zero(typeof(x)))

reconstruct_layer(layer::AbstractLayer, previous_layer_size) = layer
function reconstruct_layer(layer::Dense, previous_layer_size::Int)
    if layer.inputs isa Symbol
        return Dense(layer.outputs, previous_layer_size, layer.use_bias, layer.activation_fn, layer.parameter_type)
    else
        return layer
    end
end

function _map_views(indices::AbstractArray{Q}, array::AbstractArray) where {Q<:UnitRange}
    return (x->view(array, x)).(indices)
end
function _map_views(indices::AbstractArray{T}, array::AbstractArray) where {Q<:UnitRange, T<:AbstractArray{Q}}
    return (x->_map_views(x, array)).(indices)
end

parameters(model::Model) = model.parameters


function chain(layers...)::Model
    (input_layer, network_layers) = Iterators.peel(layers)
    if !(input_layer isa Static)
        @error "The first layer should always be a static layer, specifying the input size."
    end
    previous_layer_size::Integer = outputcount(input_layer)
    input_datatype = datatype(input_layer)
    
    overall_datatype = input_datatype

    total_parameter_size = 0
    # Create a mapping from layers to parameters
    layer_indices = Vector([parameter_indices(input_layer, total_parameter_size)])
    reconstructed_layers = AbstractLayer[input_layer]
    for layer in network_layers
        # Reconstruct the layer, adding in the previous layer size
        layer = reconstruct_layer(layer, previous_layer_size)
        push!(reconstructed_layers, layer)

        # Check consistency of datatypes
        current_datatype = datatype(layer)
        if current_datatype != overall_datatype
            overall_datatype = promote_type(current_datatype, overall_datatype)
            @warn "Datatypes are mismatched between two adjacent layers (expected $previous_datatype, got $current_datatype)"
            @warn "Switching to $overall_datatype for the datatype of the parameters."
            current_datatype = overall_datatype
        end

        # Check consistency of input and output sizes between layers
        expected_inputs = inputsize(layer)
        if expected_inputs != previous_layer_size
            error("Layer expected $(expected_inputs), but previous layer has a size of $(previous_layer_size)")
        end

        layer_size = outputcount(layer)

        num_params = num_parameters(layer)
        push!(layer_indices, parameter_indices(layer, total_parameter_size))

        total_parameter_size += num_params
        previous_layer_size = layer_size
    end

    parameter_array = zeros(overall_datatype, total_parameter_size)
    parameter_views = _map_views(layer_indices, parameter_array)
    
    model_layers = [ParameterisedLayer(l, v) for (l,v) in zip(reconstructed_layers, parameter_views)]
    return Model(parameter_array, model_layers)
end


# API
export Static, Dense, chain, sigmoid, relu, parameters

include("preallocation.jl")
include("forward.jl")

end
