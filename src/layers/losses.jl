abstract type AbstractTargetsLayer <: AbstractLayer end


Base.@kwdef struct BatchCrossEntropyLoss{DT, S<:AbstractArray} <: AbstractTargetsLayer
    targets::S
    num_classes::Int
    datatype::Val{DT} = Val(Float32)
end

datatype(::BatchCrossEntropyLoss{DT, S}) where {DT, S} = DT
unbatched_output_size(::BatchCrossEntropyLoss) = 1
parameter_array_size(::BatchCrossEntropyLoss) = 0
num_parameters(::BatchCrossEntropyLoss) = 0
should_preallocate(::BatchCrossEntropyLoss) = true
inputsize(layer::BatchCrossEntropyLoss) = layer.num_classes

function set_targets!(layer::BatchCrossEntropyLoss, targets)
    layer.targets .= targets
    nothing
end

function reconstruct_layer(layer::BatchCrossEntropyLoss, previous_layer_size, current_datatype)
    @assert length(previous_layer_size) == 1 "Layer before the batch cross entropy loss should be flattened."
    @assert layer.num_classes == previous_layer_size[1] "Expected $(layer.num_classes) but previous output indicates $(previous_layer_size[1])"

    if current_datatype != datatype(layer)
        @warn "Changing datatype of batch cross entropy loss layer to $(current_datatype)."
        return BatchCrossEntropyLoss(layer.targets, layer.num_classes, Val(current_datatype))
    else
        return layer
    end
end