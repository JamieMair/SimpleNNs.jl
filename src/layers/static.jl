Base.@kwdef struct Static{DT, S} <: AbstractLayer
    inputs::S
    datatype::Val{DT} = Val(Float32)
end
Static(inputs::Union{Int, NTuple}; kwargs...) = Static(;inputs=inputs, kwargs...)

inputsize(layer::Static) = 0
unbatched_output_size(layer::Static) = layer.inputs
datatype(::Static{DT, S}) where {DT, S} = DT
parameter_array_size(::Static) = 0
num_parameters(::Static) = 0