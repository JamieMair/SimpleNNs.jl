# Backpropagation algorithm for calculating gradients
function tanh_derivative(y) # w.r.t the output
    return (one(typeof(y))-y*y)
end
relu_derivative(x) = ifelse(x>zero(x), one(x), zero(x)) # w.r.t the output
function sigmoid_derivative(y) # w.r.t the output y
    return (one(typeof(y))-y)*y
end

"""
    activation_gradient_fn

The gradient function of a layer w.r.t to its OUTPUT (not the activation).
"""
activation_gradient_fn(::Dense{DT, K, T}) where {DT, K, T} = activation_gradient_fn(Val(T))
activation_gradient_fn(c) = activation_gradient_fn(Val(typeof(c.activation_fn)))
function activation_gradient_fn(::Val{T}) where {T}
    if T === typeof(identity)
        return one
    elseif T === typeof(tanh)
        return tanh_derivative
    elseif T === typeof(relu)
        return relu_derivative
    elseif T === typeof(sigmoid)
        return sigmoid_derivative
    elseif T === typeof(tanh_fast)
        return tanh_derivative
    else
        return unimplemented()
    end
end