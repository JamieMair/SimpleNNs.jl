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
function backprop!(partials_buffer, gradient_buffer, inputs, outputs, layer::Conv)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = activation_gradient_fn(layer)
        partials_buffer .*= activation_derivative.(outputs)
    end

    # Kernel weights gradients
    k_grads = kernel_weights(layer, gradient_buffer)
    spatial_dims = length(layer.kernel_size)
    output_dimensions = CartesianIndices(size(outputs)[1:spatial_dims])
    kernel_indices = CartesianIndices(layer.kernel_size)
    one_one = CartesianIndex((1 for _ in 1:spatial_dims)...)
    fill!(k_grads, zero(eltype(k_grads)))
    @inbounds for n in axes(outputs, length(size(outputs)))
        for c_out in 1:layer.out_channels
            for c_in in 1:layer.in_channels
                for o_i in output_dimensions
                    for k_i in kernel_indices
                        k_grads[k_i, c_in, c_out] += partials_buffer[o_i, c_out, n] * inputs[o_i + k_i - one_one, c_in, n]
                    end
                end
            end
        end
    end
    # Kernel bias gradients
    if has_bias(layer)
        k_biases = kernel_biases(layer, gradient_buffer)
        fill!(k_biases, zero(eltype(k_biases)))
        @inbounds for n in axes(outputs, length(size(outputs)))
            for c_out in 1:layer.out_channels
                for o_i in output_dimensions
                    k_biases[c_out] += partials_buffer[o_i, c_out, n]
                end
            end
        end
    end

    return partials_buffer
end
function backprop!(partials_buffer::CuArray, gradient_buffer::CuArray, inputs::CuArray, outputs::CuArray, layer::Conv)
    # Apply activation backprop
    if typeof(layer.activation_fn) !== typeof(identity)
        activation_derivative = activation_gradient_fn(layer)
        partials_buffer .*= activation_derivative.(outputs)
    end
    conv_params = NNlib.DenseConvDims(size(input), size(kernel); flipkernel=true)
    k_grads = kernel_weights(layer, gradient_buffer)

    NNlib.∇conv_filter!(k_grads, inputs, partials_buffer, conv_params)
    # Kernel bias gradients
    if has_bias(layer)
        k_biases = kernel_biases(layer, gradient_buffer)
        NNlibCUDA.∇conv_bias!(k_biases, partials_buffer)
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
function pullback!(input_partials, output_partials, layer::ParameterisedLayer{T}) where {T<:Conv}
    params = parameters(layer)
    conv_layer = _inner_layer(layer)::Conv
    kernel = kernel_weights(conv_layer, params)
    spatial_dims = length(conv_layer.kernel_size)
    input_dimensions = CartesianIndices(size(input_partials)[1:spatial_dims])
    kernel_indices = CartesianIndices(conv_layer.kernel_size)
    one_one = CartesianIndex((1 for _ in 1:spatial_dims)...)
    # Zero out input
    fill!(input_partials, zero(eltype(input_partials)))
    @inbounds for n in axes(output_partials, length(size(output_partials)))
        for c_out in 1:conv_layer.out_channels
            for c_in in 1:conv_layer.in_channels
                for x_i in input_dimensions
                    for k_i in kernel_indices

                        out_index = x_i + one_one - k_i
                        grad_contribution = if checkbounds(Bool, output_partials, out_index, c_out, n)
                            output_partials[out_index, c_out, n] * kernel[k_i, c_in, c_out]
                        else
                            zero(eltype(input_partials))
                        end
                        
                        input_partials[x_i, c_in, n] += grad_contribution
                    end
                end
            end
        end
    end

    return input_partials
end
function pullback!(input_partials::CuArray, output_partials::CuArray, layer::ParameterisedLayer{T}) where {T<:Conv}
    params = parameters(layer)
    conv_layer = _inner_layer(layer)::Conv
    kernel = kernel_weights(conv_layer, params)
    conv_params = NNlib.DenseConvDims(size(input), size(kernel); flipkernel=true)
    NNlib.∇conv_data!(input_partials, output_partials, kernel, conv_params)

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

    layer_partials = Vector{Any}(undef, length(model.layers)-1)
    for (i, layer) in enumerate(network_layers)
        if typeof(layer) <: Flatten
            layer_partials[i] = reshape(layer_partials[i-1], layer.output_size..., batch_size)
            continue
        end
        layer_partials[i] = device_zeros_fn(datatype(layer), _get_preallocation_size(layer, batch_size))
    end
    # TODO: Switch to tuple and allow for a flatten right at the start
    return BackpropagationCache(parameter_gradients, parameter_gradient_views, [layer_partials...])
end

abstract type AbstractLoss end
struct MSELoss{T<:AbstractVector}
    targets::T
end
struct LogitCrossEntropyLoss{T<:AbstractVector, N}
    targets::T
    num_classes::Val{N}
end
LogitCrossEntropyLoss(targets::AbstractVector, n::Integer) = LogitCrossEntropyLoss(targets, Val(n))

function pullback!(partials_buffer, inputs, loss::MSELoss)
    partials_buffer .= inputs .- loss.targets'
    return sum(partials_buffer) / (2*length(loss.targets))
end
function pullback!(partials_buffer, inputs, loss::LogitCrossEntropyLoss{T, N}) where {T, N}
    @assert length(size(inputs)) == 2

    partials_buffer .= exp.(inputs)

    total_loss = zero(eltype(partials_buffer))
    @inbounds for i in axes(inputs, length(size(inputs)))
        true_class = loss.targets[i]
        z = zero(eltype(partials_buffer))
        @simd for k in 1:N
            z += partials_buffer[k, i]
        end
        for j in axes(inputs, 1)
            e_y = partials_buffer[j, i]
            e_y_over_z = ifelse(isfinite(e_y), e_y / z, one(eltype(partials_buffer)))
            total_loss -= ifelse(j==true_class, log(e_y_over_z), zero(typeof(total_loss)))
            partials_buffer[j, i] = (e_y_over_z - (j==true_class))
        end
    end

    return total_loss
end

function backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, model::Model, loss)
    # _backprop!(backprop_cache, forward_cache, model.layers, loss)

    N = length(model.layers)
    current_partial = last(backprop_cache.layer_partials)
    total_loss = pullback!(current_partial, last(forward_cache.layer_outputs), loss)
    
    
    for index in (N-1):-1:1
        inputs = if index==1
            forward_cache.input
        else
            forward_cache.layer_outputs[index-1]
        end

        outputs = forward_cache.layer_outputs[index]
        gradient_buffer = backprop_cache.parameter_gradient_views[index+1]
        current_layer = model.layers[index+1]
        
        if typeof(current_layer) <: AbstractParameterisedLayer
            current_partial = backprop!(current_partial, gradient_buffer, inputs, outputs, _inner_layer(current_layer))
        end

        if index > 1
            next_partials = backprop_cache.layer_partials[index - 1]
            current_partial = pullback!(next_partials, current_partial, current_layer)
        end
    end

    return total_loss
end

@generated function _backprop!(backprop_cache::BackpropagationCache, forward_cache::ForwardPassCache, layers::Tuple{Vararg{<:Any,N}}, loss) where {N}
    setup_block = quote 
        current_partial = last(backprop_cache.layer_partials)
        current_partial = pullback!(current_partial, last(forward_cache.layer_outputs), loss)
    end
    
    layer_blocks = map((N-1):-1:1) do i
        quote
            inputs = if $i==1
                forward_cache.input
            elseif typeof(forward_cache.layer_outputs[$i-1]) <: Flatten
                forward_cache.layer_outputs[$i-2]
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

export preallocate_grads, backprop!, MSELoss, LogitCrossEntropyLoss