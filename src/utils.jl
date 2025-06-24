# Utility
unimplemented() = error("Unimplemented function")
unimplemented(msg) = error("Unimplemented function: $msg")

flatten_size(size::Tuple) = reduce(*, size)
flatten_size(size::Number) = size

function _map_views(indices::AbstractArray{Q}, array::AbstractArray) where {Q<:UnitRange}
    return (x->view(array, x)).(indices)
end
function _map_views(indices::AbstractArray{T}, array::AbstractArray) where {Q<:UnitRange, T<:AbstractArray{Q}}
    return (x->_map_views(x, array)).(indices)
end


"""
    relu(x)

Rectified Linear Unit (ReLU) activation function.

Computes `max(0, x)` element-wise.

# Arguments
- `x`: Input value or array

# Returns
- Output with negative values set to zero

# Examples
```julia
relu(-2.0)  # Returns 0.0
relu(3.0)   # Returns 3.0
relu([-1, 2, -3, 4])  # Returns [0, 2, 0, 4]
```

# Mathematical Definition
```math
\\text{ReLU}(x) = \\max(0, x)
```
"""
relu(x) = max(zero(x), x)

"""
    sigmoid(x)

Logistic sigmoid activation function.

Computes the sigmoid function: `1 / (1 + exp(-x))`.

# Arguments
- `x`: Input value or array

# Returns
- Output in range (0, 1)

# Examples
```julia
sigmoid(0.0)    # Returns 0.5
sigmoid(1.0)    # Returns ~0.731
sigmoid(-1.0)   # Returns ~0.269
```

# Mathematical Definition
```math
\\sigma(x) = \\frac{1}{1 + e^{-x}}
```
"""
sigmoid(x) = one(x) / (one(x) + exp(-x))

"""
    tanh_fast(x)

Fast hyperbolic tangent activation function.

Computes an optimized version of the hyperbolic tangent function.
This may use approximations for better performance compared to the standard `tanh`.

# Arguments
- `x`: Input value or array

# Returns
- Output in range (-1, 1)

# Examples
```julia
tanh_fast(0.0)   # Returns 0.0
tanh_fast(1.0)   # Returns ~0.761
tanh_fast(-1.0)  # Returns ~-0.761
```

# Mathematical Definition
```math
\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}}
```

# Notes
- May use optimized implementations for better performance
- Output range is (-1, 1), making it zero-centered unlike sigmoid
"""
tanh_fast(x) = tanh(x)  # Default implementation, may be optimized in specific contexts