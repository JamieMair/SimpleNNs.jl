include("types.jl")
include("static.jl")
include("dense.jl")
include("flatten.jl")
include("conv.jl")
include("maxpool.jl")

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
reconstruct_layer(layer::AbstractLayer, previous_layer_size, current_datatype) = layer