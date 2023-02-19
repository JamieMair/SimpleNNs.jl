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

function backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer)
    return partials_buffer
end
function backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer::Dense)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = activation_gradient_fn(layer)
        partials_buffer .*= activation_derivative.(outputs)
    end

    w_grads = reshape(first(gradient_buffer), layer.outputs, layer.inputs)
    mul!(w_grads, partials_buffer, transpose(inputs))

    if has_bias(layer)
        b_grads = reshape(last(gradient_buffer), :, 1)
        sum!(b_grads, partials_buffer) # TODO: Remove allocations
    end
    
    return partials_buffer
end

function pullback!(input_partials, output_partials, layer::AbstractLayer)
    return input_partials
end
function pullback!(input_partials, output_partials, layer::ParameterisedLayer{T}) where {T<:Dense}
    layer_weights = weights(layer)
    mul!(input_partials, transpose(layer_weights), output_partials)
    return input_partials
end
function pullback!(input_partials, output_partials, layer::Flatten)
    
    input_partials = reshape(output_partials, layer.input_size..., :)
    return input_partials
end

struct BackpropagationCache{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray{B}, D<:AbstractArray, E<:AbstractArray{D}}
    parameter_gradients::A
    parameter_gradient_views::C
    layer_partials::E
end
function preallocate_grads(model::Model, batch_size::Integer)
    (_, network_layers) = Iterators.peel(model.layers)

    parameter_offsets = cumsum(num_parameters.(model.layers))
    layer_indices = [parameter_indices(layer, offset-num_parameters(layer)) for (layer, offset) in Iterators.zip(model.layers, parameter_offsets)]
    parameter_gradients = similar(model.parameters)
    fill!(parameter_gradients, zero(eltype(parameter_gradients)))
    parameter_gradient_views = _map_views(layer_indices, parameter_gradients)
    device_zeros_fn = zeros_fn(model)
    layer_partials = [device_zeros_fn(datatype(layer), _get_preallocation_size(layer, batch_size)) for layer in network_layers]
    return BackpropagationCache(parameter_gradients, parameter_gradient_views, layer_partials)
end

abstract type AbstractLoss end
struct MSELoss{T<:AbstractVector}
    targets::T
end

function pullback!(partials_buffer, inputs, loss::MSELoss)
    partials_buffer .= inputs .- loss.targets'
    partials_buffer
end

function backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, model::Model, loss::MSELoss)
    _backprop!(backprop_cache, forward_cache, model.layers, loss)
    nothing
end

@generated function _backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}, loss::MSELoss) where {N}
    setup_block = quote 
        current_partial = last(backprop_cache.layer_partials)
        current_partial = pullback!(current_partial, last(forward_cache.layer_outputs), loss)
    end
    
    layer_blocks = map((N-1):-1:1) do i
        quote
            inputs = if $i==1
                forward_cache.input
            else 
                forward_cache.layer_outputs[$i-1]
            end
            outputs = forward_cache.layer_outputs[$i]
            gradient_buffer = backprop_cache.parameter_gradient_views[$i+1]
            current_layer = layers[$i+1]
            
            current_partial = backprop!(current_partial, gradient_buffer, inputs, outputs, _inner_layer(current_layer))
    
            if $i > 1
                next_partials = backprop_cache.layer_partials[$i - 1]
                current_partial = pullback!(next_partials, current_partial, current_layer)
            end
        end
    end

    return Expr(:block, setup_block, layer_blocks...)
end

export preallocate_grads, backprop!, MSELoss