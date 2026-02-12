

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

"""
    has_loss(model::Model)

Check whether the model has a loss layer (a layer extending `AbstractTargetsLayer`) as its final layer.

Returns `true` if the last layer is a loss layer, `false` otherwise.

# Examples
```julia
model = chain(Static(10), Dense(5, activation_fn=relu))
has_loss(model)  # false

model_with_loss = add_loss(model, BatchCrossEntropyLoss(targets=zeros(Int, 32), num_classes=5))
has_loss(model_with_loss)  # true
```

See also [`add_loss`](@ref), [`remove_loss`](@ref), [`get_predictions`](@ref).
"""
function has_loss(model::Model)
    last_layer = model.layers[end]
    # Handle ParameterisedLayer wrapper
    layer = last_layer isa ParameterisedLayer ? _inner_layer(last_layer) : last_layer
    return layer isa AbstractTargetsLayer
end

"""
    get_loss(model::Model)

Returns the loss layer if the model has one, otherwise returns `nothing`.

If the model has a loss layer (a layer extending `AbstractTargetsLayer`) as its final layer,
this function returns that layer. Otherwise, it returns `nothing`.

# Returns
- The loss layer if present
- `nothing` if the model has no loss layer

# Examples
```julia
model = chain(Static(10), Dense(5, activation_fn=relu))
get_loss(model)  # nothing

model_with_loss = add_loss(model, BatchCrossEntropyLoss(targets=zeros(Int, 32), num_classes=5))
loss_layer = get_loss(model_with_loss)  # Returns the BatchCrossEntropyLoss layer
```

See also [`has_loss`](@ref), [`add_loss`](@ref), [`remove_loss`](@ref).
"""
function get_loss(model::Model)
    last_layer = model.layers[end]
    # Handle ParameterisedLayer wrapper
    layer = last_layer isa ParameterisedLayer ? _inner_layer(last_layer) : last_layer
    return layer isa AbstractTargetsLayer ? layer : nothing
end

"""
    add_loss(model::Model, loss_layer::AbstractTargetsLayer)

Create a new model with the given loss layer appended to the end of the existing model.

This function reconstructs the entire model chain with the loss layer added as the final layer.
The original model's parameters are copied to the new model.

# Arguments
- `model::Model`: The existing model to extend
- `loss_layer::AbstractTargetsLayer`: The loss layer to append (e.g., `BatchCrossEntropyLoss`)

# Returns
A new `Model` with the loss layer appended.

# Examples
```julia
model = chain(
    Static(10),
    Dense(32, activation_fn=relu),
    Dense(5, activation_fn=identity)
)

# Add a loss layer
targets = zeros(Int, batch_size)
loss_layer = BatchCrossEntropyLoss(targets=targets, num_classes=5)
model_with_loss = add_loss(model, loss_layer)
```

See also [`remove_loss`](@ref), [`has_loss`](@ref), [`get_predictions`](@ref).
"""
function add_loss(model::Model, loss_layer::AbstractTargetsLayer)
    if has_loss(model)
        @warn "Model already has a loss layer. The existing loss will be removed first."
        model = remove_loss(model)
    end
    
    # Extract all layers from the existing model
    layers = collect(model.layers)
    
    # Unwrap ParameterisedLayers to get the underlying layer definitions
    unwrapped_layers = map(layers) do layer
        layer isa ParameterisedLayer ? _inner_layer(layer) : layer
    end
    
    # Create new model with loss appended
    new_model = chain(unwrapped_layers..., loss_layer)
    
    # Copy the existing parameters to the new model
    # The new model should have the same parameters (loss layers have no parameters)
    if length(new_model.parameters) == length(model.parameters)
        copyto!(new_model.parameters, model.parameters)
    else
        @error "Parameter mismatch when adding loss layer. Expected $(length(model.parameters)), got $(length(new_model.parameters))"
    end
    
    return new_model
end

"""
    remove_loss(model::Model)

Create a new model with the loss layer removed from the end, if one exists.

This function reconstructs the model chain without the final loss layer.
The original model's parameters are copied to the new model.

# Arguments
- `model::Model`: The model to remove the loss layer from

# Returns
A new `Model` without the loss layer. If the model doesn't have a loss layer, returns the original model unchanged.

# Examples
```julia
model_with_loss = chain(
    Static(10),
    Dense(5, activation_fn=identity),
    BatchCrossEntropyLoss(targets=zeros(Int, 32), num_classes=5)
)

model = remove_loss(model_with_loss)
has_loss(model)  # false
```

See also [`add_loss`](@ref), [`has_loss`](@ref), [`get_predictions`](@ref).
"""
function remove_loss(model::Model)
    if !has_loss(model)
        @warn "Model does not have a loss layer. Returning original model."
        return model
    end
    
    # Extract all layers except the last one
    layers = collect(model.layers)[1:end-1]
    
    # Unwrap ParameterisedLayers to get the underlying layer definitions
    unwrapped_layers = map(layers) do layer
        layer isa ParameterisedLayer ? _inner_layer(layer) : layer
    end
    
    # Create new model without the loss layer
    new_model = chain(unwrapped_layers...)
    
    # Copy the existing parameters to the new model
    if length(new_model.parameters) == length(model.parameters)
        copyto!(new_model.parameters, model.parameters)
    else
        @error "Parameter mismatch when removing loss layer. Expected $(length(model.parameters)), got $(length(new_model.parameters))"
    end
    
    return new_model
end

"""
    get_predictions(model::Model, forward_cache)

Extract predictions from the forward cache based on whether the model has a loss layer.

If the model does not have a loss layer, returns the final output from the cache.
If the model has a loss layer, returns the input to the loss layer (i.e., the output of the second-to-last layer).

# Arguments
- `model::Model`: The model that was used for the forward pass
- `forward_cache`: The forward cache containing layer outputs

# Returns
An array containing the model's predictions (before the loss computation if applicable).

# Examples
```julia
model = chain(Static(10), Dense(5, activation_fn=identity))
forward_cache = preallocate(model, batch_size)
set_inputs!(forward_cache, inputs)
forward!(forward_cache, model)

predictions = get_predictions(model, forward_cache)  # Returns final layer output

# With loss layer
model_with_loss = add_loss(model, loss_layer)
forward_cache_with_loss = preallocate(model_with_loss, batch_size)
set_inputs!(forward_cache_with_loss, inputs)
forward!(forward_cache_with_loss, model_with_loss)

predictions = get_predictions(model_with_loss, forward_cache_with_loss)  # Returns output before loss
```

See also [`add_loss`](@ref), [`remove_loss`](@ref), [`has_loss`](@ref), [`get_outputs`](@ref).
"""
function get_predictions(model::Model, forward_cache)
    if has_loss(model)
        # Return the output of the second-to-last layer (input to the loss)
        # Forward cache stores outputs for each layer (excluding the Static input layer)
        return forward_cache.layer_outputs[end-1]
    else
        # Return the final output
        return forward_cache.layer_outputs[end]
    end
end