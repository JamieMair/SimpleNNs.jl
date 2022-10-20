module SimpleNNs
using Logging

# Utility
unimplemented() = error("Unimplemented function")

flatten_size(size::Tuple) = reduce(*, size)
flatten_size(size::Number) = size

# Abstract types
abstract type AbstractLayer end
datatype(::AbstractLayer) = unimplemented()
outputsize(::AbstractLayer) = unimplemented()
parameter_array_size(::AbstractLayer) = unimplemented()
num_parameters(layer::AbstractLayer) = flatten_size(parameter_array_size(layer))
parameter_indices(::AbstractLayer) = unimplemented()

abstract type AbstractParameterisedLayer <: AbstractLayer end


# Layer types
struct Static{DT, S} <: AbstractLayer
    inputs::S
    datatype::DT = Val(Float32)
end
Base.@kwdef struct Dense{DT<:Real, K<:Union{Symbol, Int}, T<:Function} <: AbstractLayer
    outputs::Int
    inputs::K = :infer
    use_bias::Bool = true
    activation_fn::T = identity
    parameter_type::Val{DT} = Val(Float32)
end

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
function parameter_indices(layer, current_offset::Integer)
    parameter_sizes = parameter_array_size(layer)
    if eltype(parameter_sizes) isa Tuple
        sizes = flatten_size.(parameter_sizes)
        offsets = cumsum(sizes) .- sizes
        ranges = ((current_offset+1+offset:current_offset+offset+s) for (s, offset) in zip(sizes, offsets))
        return ranges
    end

    return (current_offset+1:current_offset+parameter_sizes)
end

# Model
struct Model{T,Q,K}
    parameters::T
    layers::Q
end

# Activation functions
sigmoid(x) = inv(one(typeof(x) + exp(-x)))
relu(x) = ifelse(x>=zero(typeof(x)), x, zero(typeof(x)))

reconstruct_layer(layer::AbstractLayer, previous_layer_size) = layer
function reconstruct_layer(layer::Dense, previous_layer_size::Int)
    if layer.inputs isa Symbol
        return Dense(layer.outputs, previous_layer_size, layer.use_bias, layer.activation_fn)
    else
        return layer
    end
end

function chain(layers...)
    (input_layer, network_layers) = Iterators.peel(layers)
    if !(input_layer isa Static)
        @error "The first layer should always be a static layer, specifying the input size."
    end
    previous_layer_size::Integer = outputsize(input_layer)
    input_datatype = datatype(first_layer)
    
    overall_datatype = input_datatype

    total_parameter_size = 0
    # Create a mapping from layers to parameters
    layer_indices = Any[parameter_indices(input_layer, total_parameter_size)]
    
    for layer in network_layers
        # Reconstruct the layer, adding in the previous layer size
        layer = reconstruct_layer(layer, previous_layer_size)

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

        layer_size = outputsize(layer)

        num_params = num_parameters(layer)
        push!(layer_indices, parameter_indices(layer, total_parameter_size))

        total_parameter_size += num_params
        previous_layer_size = layer_size
    end

    parameter_array = zeros(overall_datatype, total_parameter_size)
    # todo
    """
    1 - map views onto the parameter array
    2 - create a model object
    """


end

end
