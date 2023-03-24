Base.@kwdef struct MaxPool{DT, S1<:NTuple, S2<:Union{Infer, NTuple}, S3<:Union{Infer, NTuple}, S4<:Union{Infer, NTuple}} <: AbstractLayer 
    pool_size::S1
    stride::S2 = Infer()
    input_size::S3 = Infer()
    output_size::S4 = Infer()
    datatype::Val{DT} = Val(Infer())
end
MaxPool(pool_size::NTuple{N, Int}; kwargs...) where {N} = MaxPool(;pool_size, kwargs...)

function reconstruct_layer(layer::MaxPool, previous_layer_size, current_datatype)
    if layer.stride isa Infer || layer.input_size isa Infer || layer.output_size isa Infer || datatype(layer) isa Infer
        # Default stride is same size as the pool size
        stride = ifelse(layer.stride isa Infer, Tuple(i for i in layer.pool_size), layer.stride)

        input_size = ifelse(layer.input_size isa Infer, previous_layer_size, layer.input_size)
        num_channels = input_size[end]
        expected_output_size = map((in_dim, kernel_size, str)->_conv_dim_size(in_dim, kernel_size, 0, str), input_size[begin:end-1], layer.pool_size, stride)
        expected_output_size = (expected_output_size..., num_channels)
        if !(layer.output_size isa Infer)
            @assert expected_output_size == layer.output_size "Expected output size was $(expected_output_size), but layer had an output size of $(layer.output_size)"
        end

        dt = ifelse(datatype(layer) isa Infer, Val(current_datatype), layer.datatype)
        return MaxPool(layer.pool_size, stride, input_size, expected_output_size, dt)
    else
        return layer
    end
end

datatype(::MaxPool{DT}) where {DT} = DT
unbatched_output_size(layer::MaxPool) = layer.output_size
function inputsize(layer::MaxPool)
    layer.input_size isa Infer && error("Max pool layer inputs should be set automatically already.")
    return layer.input_size
end
parameter_array_size(::MaxPool) = 0
num_parameters(::MaxPool) = 0