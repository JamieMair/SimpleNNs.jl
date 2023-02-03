Base.@kwdef struct Conv{DT<:Real, K<:Union{Symbol, Int}, T<:Function, B} <: AbstractLayer
    outputs::Int
    inputs::K = :infer
    use_bias::Val{B} = Val(true)
    activation_fn::T = identity
    parameter_type::Val{DT} = Val(Float32)
end

"""
Conv2D should require a kernel size and an activation function.

Kernel size: Tuple of ints
Stride is the spacing between sizes
Padding is the strategy to pad the outside
We need to know the number of channels coming in and the number coming out
The conv layer can also have a bias as well

The conv also needs to know what type of padding is available.

We may need to do some conversion on this one
"""

"""
We also need some pooling layers to be able to condense the results
"""