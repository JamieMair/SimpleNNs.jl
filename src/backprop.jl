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


function activation_gradient_fn(::Dense{DT, K, T}) where {DT, K, T}
    if T === typeof(identity)
        return one
    elseif T === typeof(tanh)
        return sech2
    elseif T === relu
        return typeof(relu_derivative)
    elseif T === sigmoid
        return typeof(sigmoid_derivative)
    else
        return activation_gradient_fn(Val(T))
    end
end
activation_gradient_fn(::Val) = unimplemented()


function backprop!(partials_buffer, layer::Dense, next_layer_weights, next_layer_partials, current_layer_output)
    grad_fn = activation_gradient_fn(layer)
    
    mul!(partials_buffer, transpose(next_layer_weights), next_layer_partials)
    partials_buffer .*= grad_fn.(current_layer_output)

    nothing
end

function calc_grads!(gradient_buffer, partials_buffer, layer::Dense, previous_layer_output)
    w_grads = reshape(first(gradient_buffer), layer.outputs, layer.inputs)
    mul!(w_grads, partials_buffer, transpose(previous_layer_output))

    if layer.use_bias
        b_grads = reshape(last(gradient_buffer), :, 1)
        sum!(b_grads, partials_buffer)
    end
end

struct BackpropagationCache{A,B,C}
    parameter_gradients::AbstractArray{A}
    parameter_gradient_views::AbstractArray{B}
    layer_partials::AbstractArray{C}
end
function preallocate_grads(model::Model, batch_size::Integer)
    (_, network_layers) = Iterators.peel(model.layers)

    parameter_offsets = cumsum(num_parameters.(model.layers))
    layer_indices = [parameter_indices(layer, offset-num_parameters(layer)) for (layer, offset) in Iterators.zip(model.layers, parameter_offsets)]
    parameter_gradients = similar(model.parameters)
    fill!(parameter_gradients, zero(eltype(parameter_gradients)))
    parameter_gradient_views = _map_views(layer_indices, parameter_gradients)

    layer_partials = [zeros(datatype(layer), (outputcount(layer), batch_size)) for layer in network_layers]
    return BackpropagationCache(parameter_gradients, parameter_gradient_views, layer_partials)
end

abstract type AbstractLoss end
struct MSELoss{T<:AbstractVector}
    targets::T
end

function backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, model::Model, loss::MSELoss)
    network_layers = view(model.layers, 2:length(model.layers))
    previous_layer_outputs = (forward_cache.input, Iterators.take(forward_cache.layer_outputs, length(forward_cache.layer_outputs)-1)...)

    current_layer_output = last(forward_cache.layer_outputs)
    relevant_parameter_gradient_views = view(backprop_cache.parameter_gradient_views, 2:length(backprop_cache.parameter_gradient_views))
    iter = Iterators.reverse(Iterators.zip(relevant_parameter_gradient_views, backprop_cache.layer_partials, previous_layer_outputs, network_layers))

    current_partial = last(backprop_cache.layer_partials)
    current_outputs = last(forward_cache.layer_outputs)
    
    last_layer_derivative_fn = activation_gradient_fn(_inner_layer(last(model.layers)))
    current_partial .= (current_outputs .- loss.targets') .* last_layer_derivative_fn.(current_outputs)
    
    current_weights = weights(last(model.layers))

    is_last_layer = true
    for (layer_param_grads, layer_partials, previous_layer_output, layer) in iter

        if !is_last_layer
            backprop!(layer_partials, _inner_layer(layer), current_weights, current_partial, current_layer_output)
        else
            is_last_layer = false
        end
        calc_grads!(layer_param_grads, layer_partials, _inner_layer(layer), previous_layer_output)

        current_layer_output = previous_layer_output
        current_partial = layer_partials
        current_weights = weights(layer)
    end
    nothing
end

export preallocate_grads, backprop!, MSELoss