include("types.jl")
include("static.jl")
include("dense.jl")
include("flatten.jl")
include("conv.jl")
include("maxpool.jl")
include("losses.jl")

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



# Layers without parameters don't need initialisation
initialise_layer!(layer::AbstractLayer) = nothing
# Layer-specific initialisation dispatch
function initialise_layer!(layer::ParameterisedLayer{T, Q}) where {T<:Dense, Q}
    inner = _inner_layer(layer)
    params = parameters(layer)
    
    # Initialise weights
    weight_view = first(params)
    fan_in = inner.inputs
    fan_out = inner.outputs
    init_weights!(weight_view, inner.init, fan_in, fan_out)
    
    # Initialise bias if present
    if has_bias(inner) && length(params) > 1
        bias_view = last(params)
        init_bias!(bias_view)
    end
end

function initialise_layer!(layer::ParameterisedLayer{T, Q}) where {T<:Conv, Q}
    inner = _inner_layer(layer)
    params = parameters(layer)
    
    # Initialise kernel weights
    weight_view = first(params)
    kernel_size = reduce(*, inner.kernel_size)
    fan_in = kernel_size * inner.in_channels
    fan_out = kernel_size * inner.out_channels
    init_weights!(weight_view, inner.init, fan_in, fan_out)
    
    # Initialise bias if present
    if has_bias(inner) && length(params) > 1
        bias_view = last(params)
        init_bias!(bias_view)
    end
end