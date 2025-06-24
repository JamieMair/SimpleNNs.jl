"""
    RMSPropOptimiser{T, X<:AbstractArray{T}} <: AbstractOptimiser

RMSProp optimiser with exponential moving average of squared gradients.

# Fields
- `lr::T`: Learning rate
- `rho::T`: Exponential decay rate for moving average
- `eps::T`: Small constant for numerical stability
- `v::X`: Exponential moving average of squared gradients
"""
mutable struct RMSPropOptimiser{T, X<:AbstractArray{T}} <: AbstractOptimiser
    lr::T
    rho::T
    eps::T
    v::X
end

"""
    RMSPropOptimiser(gradients::AbstractArray{T}; lr = Float32(1e-3), rho = 0.9f0, eps = Float32(1e-8)) where {T}

Create an RMSProp optimiser for gradient-based parameter updates.

RMSProp maintains a moving average of squared gradients to adaptively scale the learning rate.

# Arguments
- `gradients` (AbstractArray{T}): Template array matching the shape of gradients to be optimised

# Keyword Arguments
- `lr` (T): Learning rate (default: 1e-3)
- `rho` (T): Exponential decay rate for moving average of squared gradients (default: 0.9)
- `eps` (T): Small constant added to denominator for numerical stability (default: 1e-8)

# Examples
```julia
opt = RMSPropOptimiser(gradients; lr=0.001f0, rho=0.9f0)
```
"""
function RMSPropOptimiser(gradients::AbstractArray{T}; lr = Float32(1e-3), rho = 0.9f0, eps = Float32(1e-8)) where {T}
    v = similar(gradients)
    fill!(v, zero(T))
    
    return RMSPropOptimiser(convert(T, lr), convert(T, rho), convert(T, eps), v)
end

function reset!(opt::RMSPropOptimiser)
    fill!(opt.v, zero(eltype(opt.v)))
    nothing
end

function update!(parameters, gradients, opt::RMSPropOptimiser)
    v = opt.v
    _one = one(eltype(v))
    
    # Update exponential moving average of squared gradients
    v .= opt.rho .* v .+ (_one - opt.rho) .* (gradients .* gradients)
    
    # Update parameters - note the parentheses ensure correct order of operations
    parameters .-= (opt.lr ./ (sqrt.(v .+ opt.eps))) .* gradients
    
    nothing
end
