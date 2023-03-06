
Base.@kwdef struct Flatten{DT, S1<:Union{Symbol, NTuple},S2<:Union{Symbol, Int}} <: AbstractLayer 
    input_size::S1 = :infer
    output_size::S2 = :infer
    datatype::Val{DT} = Val(Float32)
end

datatype(::Flatten{DT}) where {DT} = DT


unbatched_output_size(layer::Flatten) = layer.output_size


function inputsize(layer::Flatten)
    layer.input_size isa Symbol && error("Flatten layer inputs should be set automatically already.")
    return layer.input_size
end

parameter_array_size(::Flatten) = 0


num_parameters(::Flatten) = 0
should_preallocate(::Flatten) = false

function reconstruct_layer(layer::Flatten, previous_layer_size, current_datatype)
    if layer.input_size isa Symbol || layer.output_size isa Symbol
        return Flatten(previous_layer_size, reduce(*, previous_layer_size), Val(current_datatype))
    else
        return layer
    end
end