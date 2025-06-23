function tanh_derivative(y)
    return (one(typeof(y))-y*y)
end
relu_derivative(y) = ifelse(y>zero(y), one(y), zero(y))
function sigmoid_derivative(y)
    return (one(typeof(y))-y)*y
end

"""
Dertivatives are used to backpropagate the gradients of the layer outputs back to
the activations of that layer. To save space, these are calculated exclusively
using the outputs of the layer. Instead of functions written as dy/dx=f(x), we 
instead write dy/dx = g(y). This can be done for the 3 major functions.

Whenever `y` is used below, assume this is a function of the output, not the input.
"""
activation_gradient_fn(::Dense{DT, K, T}) where {DT, K, T} = activation_gradient_fn(Val(T))
activation_gradient_fn(c::AbstractLayer) = activation_gradient_fn(Val(typeof(c.activation_fn)))
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