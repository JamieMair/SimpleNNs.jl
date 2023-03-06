# Abstract types
abstract type AbstractLayer end
datatype(::AbstractLayer) = unimplemented()
unbatched_output_size(::AbstractLayer) = unimplemented()
parameter_array_size(::AbstractLayer) = unimplemented()
num_parameters(layer::AbstractLayer) = flatten_size(parameter_array_size(layer))
parameter_indices(::AbstractLayer) = unimplemented()
should_preallocate(::AbstractLayer) = true


abstract type AbstractParameterisedLayer <: AbstractLayer end
_inner_layer(::AbstractParameterisedLayer) = unimplemented()
parameters(::AbstractParameterisedLayer) = unimplemented()
# Forward all definitions to the "layer" property of the abstract layer
datatype(layer::AbstractParameterisedLayer) = datatype(_inner_layer(layer))
unbatched_output_size(layer::AbstractParameterisedLayer) = unbatched_output_size(_inner_layer(layer))
parameter_array_size(layer::AbstractParameterisedLayer) = parameter_array_size(_inner_layer(layer))
num_parameters(layer::AbstractParameterisedLayer) = num_parameters(_inner_layer(layer))
parameter_indices(layer::AbstractParameterisedLayer) = parameter_indices(_inner_layer(layer))



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

struct Infer end
