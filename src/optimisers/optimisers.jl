export reset!, update!
export AdamOptimiser

abstract type AbstractOptimiser end

"""
    reset!(opt::AbstractOptimiser)

Reset the internal state of the optimiser to its initial values.
"""
function reset!(opt::AbstractOptimiser)
    throw(error("Unimplemented"))
end

"""
    update!(parameters, gradients, opt::AbstractOptimiser)

Update the parameters using the provided gradients and optimiser.

# Arguments
- `parameters`: Model parameters to be updated
- `gradients`: Gradients computed from the loss function
- `opt` (AbstractOptimiser): The optimiser instance containing update rules
"""
function update!(parameters, gradients, opt::AbstractOptimiser)
    throw(error("Unimplemented"))
end

include("adam.jl")