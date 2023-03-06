Base.@kwdef struct MaxPool{DT, S1<:Union{InferSize, NTuple},S2<:Union{InferSize, Int}} <: AbstractLayer 
    pool_size::S1
    stride::S2
    input_size::S3
    output_size::S4
    datatype::Val{DT}
end