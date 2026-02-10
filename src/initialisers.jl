# Weight Initialisation Methods
using Random

"""
Abstract type for weight initialisation strategies.
"""
abstract type Initialiser end

"""
    GlorotUniform()

Glorot uniform initialisation (also called Xavier uniform).
Samples weights from a uniform distribution in the range [-limit, limit]
where limit = √(6 / (fan_in + fan_out)).

Best suited for layers with sigmoid or tanh activations.
"""
struct GlorotUniform <: Initialiser end

"""
    GlorotNormal()

Glorot normal initialisation (also called Xavier normal).
Samples weights from a normal distribution with mean 0 and 
standard deviation √(2 / (fan_in + fan_out)).

Best suited for layers with sigmoid or tanh activations.
"""
struct GlorotNormal <: Initialiser end

"""
    HeUniform()

He uniform initialisation (also called Kaiming uniform).
Samples weights from a uniform distribution in the range [-limit, limit]
where limit = √(6 / fan_in).

Best suited for layers with ReLU activations.
"""
struct HeUniform <: Initialiser end

"""
    HeNormal()

He normal initialisation (also called Kaiming normal).
Samples weights from a normal distribution with mean 0 and
standard deviation √(2 / fan_in).

Best suited for layers with ReLU activations.
"""
struct HeNormal <: Initialiser end

"""
    LeCunNormal()

LeCun normal initialisation.
Samples weights from a normal distribution with mean 0 and
standard deviation √(1 / fan_in).

Best suited for layers with SELU activations.
"""
struct LeCunNormal <: Initialiser end

"""
    Zeros()

Initialise all weights to zero. Note: This is generally not recommended
for training as it breaks symmetry.
"""
struct Zeros <: Initialiser end

# Initialisation functions for weight matrices
function init_weights!(weights::AbstractArray, init::GlorotUniform, fan_in::Int, fan_out::Int)
    T = eltype(weights)
    limit = convert(T, sqrt(6 / (fan_in + fan_out)))
    rand!(weights)
    weights .= weights .* (2 * limit) .- limit
end

function init_weights!(weights::AbstractArray, init::GlorotNormal, fan_in::Int, fan_out::Int)
    T = eltype(weights)
    std = convert(T, sqrt(2 / (fan_in + fan_out)))
    randn!(weights)
    weights .*= std
end

function init_weights!(weights::AbstractArray, init::HeUniform, fan_in::Int, fan_out::Int)
    T = eltype(weights)
    limit = convert(T, sqrt(6 / fan_in))
    rand!(weights)
    weights .= weights .* (2 * limit) .- limit
end

function init_weights!(weights::AbstractArray, init::HeNormal, fan_in::Int, fan_out::Int)
    T = eltype(weights)
    std = convert(T, sqrt(2 / fan_in))
    randn!(weights)
    weights .*= std
end

function init_weights!(weights::AbstractArray, init::LeCunNormal, fan_in::Int, fan_out::Int)
    T = eltype(weights)
    std = convert(T, sqrt(1 / fan_in))
    randn!(weights)
    weights .*= std
end

function init_weights!(weights::AbstractArray, init::Zeros, fan_in::Int, fan_out::Int)
    fill!(weights, zero(eltype(weights)))
end

# Bias initialisation (typically zeros)
function init_bias!(bias::AbstractArray)
    fill!(bias, zero(eltype(bias)))
end

