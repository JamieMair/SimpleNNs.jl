include("preallocation.jl")
include("gradients.jl")


has_combined_backprop_and_pullback(::AbstractLayer) = false
"""
    backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer)

Backpropagates the partial gradients of the outputs of the current `layer`
into the parameters of the current layer. `partial_buffers` is used as a buffer
for the gradients of the output of this layer. `gradient_buffer` should be 
filled up with the gradients of the parameters of the current layer, using
the chain rule. `inputs` is the array fed into the layer and `outputs` is the
output of this layer in the forward pass. `layer` is the struct containing
information about the layer.
"""
function backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer)
    return partials_buffer
end
"""
    pullback!(input_partials, output_partials, layer)

Here, we complete the backpropagation of the partial gradients to the inputs
of the current layer. This should be called after `backprop!`. This method
will fill the `input_partials` buffer with partial gradients calculated via
the chain rule from the gradients of the partials from this layer's output.
"""
function pullback!(input_partials, output_partials, layer::AbstractLayer)
    return input_partials
end

include("losses.jl")

include("dense.jl")
include("flatten.jl")
include("conv.jl")
include("maxpool.jl")

include("gpu.jl")

function _deprecated_backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, model::Model, loss)
    current_partial = last(backprop_cache.layer_partials)
    total_loss = pullback!(current_partial, last(forward_cache.layer_outputs), loss)
    layers = model.layers
    N = length(layers)
    for i in N-1:-1:1
        inputs = if i==1
            forward_cache.input
        else
            forward_cache.layer_outputs[i-1]
        end

        outputs = forward_cache.layer_outputs[i]
        gradient_buffer = backprop_cache.parameter_gradient_views[i+1]
        current_layer = layers[i+1]
        
        if has_combined_backprop_and_pullback(current_layer)
            if i > 1
                next_partials = backprop_cache.layer_partials[i - 1]
                current_partial = backprop_and_pullback!(next_partials, current_partial, inputs, outputs, current_layer)
            end
        else
            if typeof(current_layer) <: AbstractParameterisedLayer
                current_partial = backprop!(current_partial, gradient_buffer, inputs, outputs, _inner_layer(current_layer))
            end

            if i > 1
                next_partials = backprop_cache.layer_partials[i - 1]
                current_partial = pullback!(next_partials, current_partial, current_layer)
            end
        end
    end

    return total_loss
end

function backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, model::Model, loss)
    return _backprop!(backprop_cache, forward_cache, model.layers, loss)
end

@generated function _backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}, loss) where {N}
    setup_block = quote 
        current_partial = last(backprop_cache.layer_partials)
        total_loss = pullback!(current_partial, last(forward_cache.layer_outputs), loss)
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
            
            if has_combined_backprop_and_pullback(current_layer)
                if $i > 1
                    next_partials = backprop_cache.layer_partials[$i - 1]
                    current_partial = backprop_and_pullback!(next_partials, current_partial, inputs, outputs, current_layer)
                end
            else
                if typeof(current_layer) <: AbstractParameterisedLayer
                    current_partial = backprop!(current_partial, gradient_buffer, inputs, outputs, _inner_layer(current_layer))
                end
    
                if $i > 1
                    next_partials = backprop_cache.layer_partials[$i - 1]
                    current_partial = pullback!(next_partials, current_partial, current_layer)
                end
            end
        end
    end

    end_block = quote

        return total_loss
    end

    return Expr(:block, setup_block, layer_blocks..., end_block)
end

export preallocate_grads, backprop!, MSELoss, LogitCrossEntropyLoss