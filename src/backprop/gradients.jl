# Backpropagation algorithm for calculating gradients
function sech2(x)
    ex = exp(x)
    sum_exs = (inv(ex)+ex)
    return convert(typeof(x), 4) / (sum_exs*sum_exs)
end
relu_derivative(x) = ifelse(x>zero(x), one(x), zero(x))
function sigmoid_derivative(x)
    ex = exp(x)
    inv_ex_plusone = inv(ex + one(x))
    return ex * inv_ex_plusone * inv_ex_plusone 
end

activation_gradient_fn(::Dense{DT, K, T}) where {DT, K, T} = activation_gradient_fn(Val(T))
activation_gradient_fn(c) = activation_gradient_fn(Val(typeof(c.activation_fn)))
function activation_gradient_fn(::Val{T}) where {T}
    if T === typeof(identity)
        return one
    elseif T === typeof(tanh)
        return sech2
    elseif T === typeof(relu)
        return relu_derivative
    elseif T === typeof(sigmoid)
        return sigmoid_derivative
    elseif T === typeof(tanh_fast)
        return sech2
    else
        return unimplemented()
    end
end