Base.@kwdef struct MaxPool{DT, S1<:Union{Symbol, NTuple},S2<:Union{Symbol, Int}} <: AbstractLayer 
    kernel_size::S1
    stride::S2
    input_size::S3 = :infer
    output_size::S4 = :infer
    datatype::Val{DT} = :infer
end