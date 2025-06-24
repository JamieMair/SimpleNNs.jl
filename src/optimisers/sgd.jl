"""
    SGDOptimiser{T} <: AbstractOptimiser

Stochastic Gradient Descent optimiser with optional momentum.

# Fields
- `lr::T`: Learning rate
- `momentum::T`: Momentum coefficient (0.0 for no momentum)
- `velocity::AbstractArray{T}`: Velocity buffer for momentum (internal state)
"""
mutable struct SGDOptimiser{T, X<:AbstractArray{T}} <: AbstractOptimiser
    lr::T
    momentum::T
    velocity::X
end

"""
    SGDOptimiser(gradients::AbstractArray{T}; lr = Float32(1e-3), momentum = 0.0f0) where {T}

Create an SGD optimiser for gradient-based parameter updates.

# Arguments
- `gradients` (AbstractArray{T}): Template array matching the shape of gradients to be optimised

# Keyword Arguments
- `lr` (T): Learning rate (default: 1e-3)
- `momentum` (T): Momentum coefficient, 0.0 for standard SGD (default: 0.0)

# Examples
```julia
# Standard SGD
opt = SGDOptimiser(gradients; lr=0.01f0)

# SGD with momentum
opt = SGDOptimiser(gradients; lr=0.01f0, momentum=0.9f0)
```
"""
function SGDOptimiser(gradients::AbstractArray{T}; lr = Float32(1e-3), momentum = 0.0f0) where {T}
    velocity = similar(gradients)
    fill!(velocity, zero(T))
    
    return SGDOptimiser(convert(T, lr), convert(T, momentum), velocity)
end

function reset!(opt::SGDOptimiser)
    fill!(opt.velocity, zero(eltype(opt.velocity)))
    nothing
end

function update!(parameters, gradients, opt::SGDOptimiser)
    if opt.momentum == zero(eltype(opt.velocity))
        # Standard SGD without momentum
        parameters .-= opt.lr .* gradients
    else
        # SGD with momentum
        opt.velocity .= opt.momentum .* opt.velocity .- opt.lr .* gradients
        parameters .+= opt.velocity
    end
    nothing
end
