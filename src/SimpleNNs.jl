module SimpleNNs
using Logging

# Utility
unimplemented() = error("Unimplemented function")
unimplemented(msg) = error("Unimplemented function: $msg")

flatten_size(size::Tuple) = reduce(*, size)
flatten_size(size::Number) = size

include("layers/layers.jl")


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
    previous_layer_size = unbatched_output_size(input_layer)
    input_datatype = datatype(input_layer)
    
    overall_datatype = input_datatype

    total_parameter_size = 0
    # Create a mapping from layers to parameters
    layer_indices = Vector([parameter_indices(input_layer, total_parameter_size)])
    reconstructed_layers = AbstractLayer[input_layer]
    for layer in network_layers
        # Reconstruct the layer, adding in the previous layer size
        layer = reconstruct_layer(layer, previous_layer_size, overall_datatype)
        push!(reconstructed_layers, layer)

        # Check consistency of datatypes
        current_datatype = datatype(layer)
        if current_datatype != overall_datatype
            @warn "Datatypes are mismatched between two adjacent layers (expected $overall_datatype, got $current_datatype)"
            overall_datatype = promote_type(current_datatype, overall_datatype)
            if overall_datatype != current_datatype
                @warn "Switching to $overall_datatype for the datatype of the parameters."
                current_datatype = overall_datatype
            end
        end

        # Check consistency of input and output sizes between layers
        expected_inputs = inputsize(layer)
        if expected_inputs != previous_layer_size
            error("Layer expected $(expected_inputs), but previous layer has a size of $(previous_layer_size)")
        end

        layer_size = unbatched_output_size(layer)

        num_params = num_parameters(layer)
        push!(layer_indices, parameter_indices(layer, total_parameter_size))

        total_parameter_size += num_params
        previous_layer_size = layer_size
    end

    parameter_array = zeros(overall_datatype, total_parameter_size)
    parameter_views = _map_views(layer_indices, parameter_array)
    
    model_layers = Tuple((num_parameters(l) > 0 ? ParameterisedLayer(l, v) : l) for (l,v) in zip(reconstructed_layers, parameter_views))
    return Model(parameter_array, model_layers)
end


# API
export Static, Dense, Conv, MaxPool, Flatten, chain, sigmoid, relu, tanh_fast, parameters


include("forward/forward.jl")
include("backprop/backprop.jl")

include("gpu.jl")

end
