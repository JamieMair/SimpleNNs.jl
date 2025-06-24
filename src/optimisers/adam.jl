mutable struct AdamOptimiser{T, X<:AbstractArray{T}} <: AbstractOptimiser
    lr::T
    beta_1::T
    beta_2::T
    m::X
    v::X
    epoch::Int
end

"""
    AdamOptimiser(gradients::AbstractArray{T}; lr = Float32(1e-3), beta_1 = 0.9f0, beta_2 = 0.999f0) where {T}

Create an Adam optimiser for gradient-based parameter updates.

# Arguments
- `gradients` (AbstractArray{T}): Template array matching the shape of gradients to be optimised

# Keyword Arguments
- `lr` (T): Learning rate (default: 1e-3)
- `beta_1` (T): Exponential decay rate for first moment estimates (default: 0.9)
- `beta_2` (T): Exponential decay rate for second moment estimates (default: 0.999)
"""
function AdamOptimiser(gradients::AbstractArray{T}; lr = Float32(1e-3), beta_1 = 0.9f0, beta_2 = 0.999f0) where {T}
    m = similar(gradients)
    v = similar(gradients)
    fill!(m, zero(T))
    fill!(v, zero(T))

    return AdamOptimiser(convert(T, lr), convert(T, beta_1), convert(T, beta_2), m, v, 1)
end

function reset!(opt::AdamOptimiser)
    fill!(opt.m, zero(eltype(opt.m)))
    fill!(opt.v, zero(eltype(opt.v)))
    opt.epoch = 1
    nothing
end

function update!(parameters, gradients, opt::AdamOptimiser)
    m = opt.m
    v = opt.v

    _one = one(eltype(m))

    m .= @. opt.beta_1 * m + (_one-opt.beta_1) * gradients
    v .= @. opt.beta_2 * v + (_one-opt.beta_2) * gradients * gradients

    denom_1 = inv(_one - opt.beta_1 ^ opt.epoch)
    denom_2 = inv(_one - opt.beta_2 ^ opt.epoch)
    _eps = convert(eltype(m), 1e-6)

    parameters .-= opt.lr .* (m .* denom_1) ./ (sqrt.(v.*denom_2) .+ _eps)

    opt.epoch += 1
    nothing
end