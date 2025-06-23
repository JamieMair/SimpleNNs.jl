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
activation_gradient_fn(c) = activation_gradient_fn(c.activation_fn)
activation_gradient_fn(::typeof(identity)) = one
activation_gradient_fn(::typeof(tanh)) = tanh_derivative
activation_gradient_fn(::typeof(tanh_fast)) = tanh_derivative
activation_gradient_fn(::typeof(sigmoid)) = sigmoid_derivative
activation_gradient_fn(::typeof(relu)) = relu_derivative
activation_gradient_fn(::Function) = unimplemented()