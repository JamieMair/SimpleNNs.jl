module SimpleNNs

# Abstract types
abstract type AbstractLayer end

# Layer types
struct Static{N} <: AbstractLayer
    inputs::NTuple{N, Int}
end
Base.@kwdef struct Dense{T<:Function} <: AbstractLayer
    outputs::Int
    inputs::Union{Symbol, Int} = :infer
    use_bias::Bool = true
    activation_fn::T = identity
end

# Model
struct Model{T,Q}
    parameters::T
    layers::Q
end

# Activation functions
sigmoid(x) = inv(one(typeof(x) + exp(-x)))
relu(x) = ifelse(x>=zero(typeof(x)), x, zero(typeof(x)))

function chain(layers...)
    first_layer = first(layers)
    if !(first_layer isa Static)
        @error "The first layer should always be a static layer, specifying the input size."
    end
    input_size = first_layer.inputs
    # todo - continue this work!
end

end
