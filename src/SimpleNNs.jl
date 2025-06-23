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

"""
    parameters(model::Model)

Returns the array used to store the parameters of the model.

Modifying this array will change the parameters of the model.
"""
parameters(model::Model) = model.parameters

"""
    chain(layers...)

Combines the given layer definitions into a single model and propagates the layer sizes through the network.

The first layer must always be a `Static` layer which specifies the feature size. If this is a simple fully
connection network, then the first layer should be `Static(nf)` where `nf` is the number of features in your
input matrix. Do not specify the batch size in this static input.

The default datatype for most layers is `Float32`, but this may be changed. The parameters of the entire model
must be of the same datatype. This function will create a flat parameter vector for the model which can be
accessed using the [`parameters`](@ref) function.

# Examples

A simple dense, fully-connected, neural network which has 3 input features:
```julia
model = chain(
    Static(3),
    Dense(10, activation_fn=tanh),
    Dense(10, activation_fn=sigmoid),
    Dense(1, activation_fn=identity),
);
```

An example convolutional neural network:
```julia
# Image size is (WIDTH, HEIGHT, CHANNELS)
img_size = (28, 28, 1)
model = chain(
    Static(img_size),
    Conv((5,5), 16; activation_fn=relu),
    MaxPool((2,2)),
    Conv((3,3), 8; activation_fn=relu),
    MaxPool((4,4)),
    Flatten(),
    Dense(10, activation_fn=identity)
)
```

See also [`Static`](@ref), [`Dense`](@ref), [`Conv`](@ref), [`MaxPool`](@ref), [`Flatten`](@ref) and [`preallocate`](@ref).
"""
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
export Static, Dense, Conv, MaxPool, Flatten, chain, sigmoid, relu, tanh_fast, parameters, gradients
export MSELoss, LogitCrossEntropyLoss

include("forward/forward.jl")
include("backprop/backprop.jl")

include("gpu.jl")

# Backwards compatibility for older Julia versions
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
            @require NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd" begin
                include("../ext/SimpleNNsCUDAExt.jl")
            end
        end
    end
end

export gpu

end
